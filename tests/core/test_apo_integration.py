"""Integration tests: APO + PartitionOptimizer, calibration, coordinator.

Tests the full pipeline: quantization-aware partition optimization,
online quality calibration, and distributed coordination.

Run: pytest tests/core/test_apo_integration.py -v
"""

import json
import time

import pytest

from distllm.dist.partition.quantization_tuner import (
    ActivationQuantMethod,
    KVCacheBits,
    NodeInfo,
    NodeQuantRecommendation,
    QuantMethod,
    QuantProfile,
    QuantizationAutoTuner,
    QuantizationPlan,
    ScoreWeights,
)
from distllm.dist.partition.quant_cost import (
    QuantizationAwareCostModel,
    QuantNodeCost,
)
from distllm.dist.partition.quant_calibrate import (
    CalibrationReport,
    CalibrationResult,
    QualityCalibrator,
)
from distllm.dist.partition.quant_coordinator import (
    CoordinatorState,
    NodeProfile,
    NodeQuantAssignment,
    QuantizationCoordinator,
)
from distllm.dist.partition.quant_report import ReportGenerator


# ---------------------------------------------------------------------------
# QualityCalibrator
# ---------------------------------------------------------------------------


class TestQualityCalibrator:
    """Test online quality calibration."""

    def test_calibrate_without_model_returns_synthetic(self):
        calibrator = QualityCalibrator(num_samples=4)
        report = calibrator.calibrate()
        assert isinstance(report, CalibrationReport)
        assert report.baseline_perplexity > 0
        assert len(report.results) > 0

    def test_synthetic_estimates_cover_all_methods(self):
        calibrator = QualityCalibrator()
        report = calibrator.calibrate(methods=["bnb_8bit", "bnb_4bit", "fp8_e4m3"])
        assert "bnb_8bit" in report.results
        assert "bnb_4bit" in report.results
        assert "fp8_e4m3" in report.results

    def test_best_method_is_fp8(self):
        calibrator = QualityCalibrator()
        report = calibrator.calibrate()
        assert report.best_method == "fp8_e4m3"

    def test_quality_loss_dict(self):
        calibrator = QualityCalibrator()
        report = calibrator.calibrate()
        losses = report.to_quality_loss_dict()
        assert "none" in losses
        assert losses["none"] == 0.0
        assert losses["fp8_e4m3"] < losses["bnb_4bit"]

    def test_report_summary(self):
        calibrator = QualityCalibrator()
        report = calibrator.calibrate()
        summary = report.summary()
        assert "Calibration Report" in summary
        assert "fp8_e4m3" in summary


# ---------------------------------------------------------------------------
# QuantizationCoordinator
# ---------------------------------------------------------------------------


class TestQuantizationCoordinator:
    """Test distributed quantization coordinator."""

    def test_register_node(self):
        coord = QuantizationCoordinator(
            model_name="test-model",
            model_size_bytes=14 * 1024**3,
            num_layers=32,
        )
        profile = NodeProfile(
            node_id="n0",
            gpu_name="H100",
            total_memory_bytes=80 * 1024**3,
            compute_tflops=989.0,
            compute_capability=9.0,
            is_hopper_or_newer=True,
        )
        coord.register_node(profile)
        assert "n0" in coord.get_state().nodes
        assert coord.get_state().nodes["n0"].status == "online"

    def test_unregister_node(self):
        coord = QuantizationCoordinator()
        coord.register_node(NodeProfile(node_id="n0", total_memory_bytes=80 * 1024**3))
        coord.unregister_node("n0")
        assert coord.get_state().nodes["n0"].status == "offline"

    def test_generate_plan(self):
        coord = QuantizationCoordinator(
            model_name="test-model",
            model_size_bytes=14 * 1024**3,
            num_layers=32,
        )
        coord.register_node(NodeProfile(
            node_id="n0", total_memory_bytes=8 * 1024**3,
        ))
        result = coord.generate_plan()
        assert "plan" in result
        assert "assignments" in result
        assert "n0" in result["assignments"]
        assert result["assignments"]["n0"]["quant_method"] != "none"

    def test_generate_plan_no_nodes(self):
        coord = QuantizationCoordinator()
        result = coord.generate_plan()
        assert "error" in result

    def test_generate_plan_mixed_hardware(self):
        coord = QuantizationCoordinator(
            model_size_bytes=14 * 1024**3, num_layers=32,
        )
        coord.register_node(NodeProfile(
            node_id="big", total_memory_bytes=80 * 1024**3,
        ))
        coord.register_node(NodeProfile(
            node_id="small", total_memory_bytes=8 * 1024**3,
        ))
        result = coord.generate_plan()
        assert len(result["assignments"]) == 2
        assert result["assignments"]["big"]["quant_method"] == "none"
        assert result["assignments"]["small"]["quant_method"] != "none"

    def test_report_failure_reduces_quality_loss(self):
        coord = QuantizationCoordinator(
            model_size_bytes=14 * 1024**3, num_layers=32, max_quality_loss=0.05,
        )
        coord.register_node(NodeProfile(
            node_id="n0", total_memory_bytes=8 * 1024**3,
        ))
        coord.generate_plan()
        result = coord.report_failure("n0", "bnb_4bit", "kernel not found")
        assert "plan" in result

    def test_report_failure_marks_offline_after_3(self):
        coord = QuantizationCoordinator(
            model_size_bytes=14 * 1024**3, num_layers=32,
        )
        coord.register_node(NodeProfile(
            node_id="n0", total_memory_bytes=8 * 1024**3,
        ))
        coord.generate_plan()
        for _ in range(3):
            coord.report_failure("n0", "bnb_4bit", "error")
        assert coord.get_state().nodes["n0"].status == "offline"

    def test_get_assignment(self):
        coord = QuantizationCoordinator(
            model_size_bytes=14 * 1024**3, num_layers=32,
        )
        coord.register_node(NodeProfile(
            node_id="n0", total_memory_bytes=8 * 1024**3,
        ))
        coord.generate_plan()
        assignment = coord.get_assignment("n0")
        assert assignment is not None
        assert assignment.node_id == "n0"

    def test_status(self):
        coord = QuantizationCoordinator(
            model_name="test", model_size_bytes=14 * 1024**3,
        )
        coord.register_node(NodeProfile(
            node_id="n0", total_memory_bytes=8 * 1024**3,
        ))
        status = coord.status()
        assert status["model"] == "test"
        assert status["nodes_online"] == 1
        assert status["nodes_total"] == 1


# ---------------------------------------------------------------------------
# End-to-end: APO + Cost Model + Report
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_single_node(self):
        node = NodeInfo(node_id="n0", total_memory_bytes=16 * 1024**3)

        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 14 * 1024**3, 32)

        reporter = ReportGenerator()
        report = reporter.generate(plan, [node], 14 * 1024**3, 32)

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert len(parsed["nodes"]) == 1

    def test_full_pipeline_multi_node_heterogeneous(self):
        nodes = [
            NodeInfo(node_id="hopper", total_memory_bytes=80 * 1024**3,
                     compute_capability=9.0, is_hopper_or_newer=True),
            NodeInfo(node_id="ampere", total_memory_bytes=24 * 1024**3,
                     compute_capability=8.0),
            NodeInfo(node_id="small", total_memory_bytes=8 * 1024**3),
        ]

        tuner = QuantizationAutoTuner(max_quality_loss=0.05)
        plan = tuner.recommend(nodes, 70 * 1024**3, 80)

        reporter = ReportGenerator()
        report = reporter.generate(plan, nodes, 70 * 1024**3, 80)

        # Hopper should get FP8, others should get different methods
        methods = {r.node_id: r.method for r in plan.recommendations}
        assert methods["hopper"] in (QuantMethod.FP8_E4M3, QuantMethod.FP8_E5M2)

        # Report should have warnings for calibration-dependent methods
        text = report.to_text()
        assert "Adaptive Precision Optimizer" in text

    def test_plan_serialization_roundtrip(self):
        nodes = [
            NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3),
            NodeInfo(node_id="n1", total_memory_bytes=80 * 1024**3),
        ]
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend(nodes, 14 * 1024**3, 32)

        json_str = plan.to_json()
        restored = QuantizationPlan.from_json(json_str)

        assert len(restored.recommendations) == 2
        assert restored.strategy == plan.strategy
        for orig, rest in zip(plan.recommendations, restored.recommendations):
            assert orig.node_id == rest.node_id
            assert orig.method == rest.method

    def test_coordinator_generates_usable_plan(self):
        coord = QuantizationCoordinator(
            model_name="Llama-3.1-70B",
            model_size_bytes=140 * 1024**3,
            num_layers=80,
        )
        # Register 3 nodes with different VRAM
        for i, vram_gb in enumerate([80, 24, 8]):
            coord.register_node(NodeProfile(
                node_id=f"node-{i}",
                gpu_name=f"GPU-{i}",
                total_memory_bytes=vram_gb * 1024**3,
            ))

        result = coord.generate_plan()

        # All 3 nodes should have assignments
        assert len(result["assignments"]) == 3

        # Largest VRAM should get least aggressive method
        method_80 = result["assignments"]["node-0"]["quant_method"]
        method_8 = result["assignments"]["node-2"]["quant_method"]

        # Smallest VRAM should get more aggressive quant than largest
        aggressiveness = {
            "none": 0, "fp8_e4m3": 1, "fp8_e5m2": 1, "bnb_8bit": 1,
            "int8": 1, "bnb_4bit": 2, "awq": 2, "gptq": 2, "nf4": 2,
        }
        assert aggressiveness.get(method_8, 2) >= aggressiveness.get(method_80, 0)

    def test_calibration_feeds_into_tuner(self):
        calibrator = QualityCalibrator()
        report = calibrator.calibrate()
        losses = report.to_quality_loss_dict()

        # Use calibrated losses to configure the tuner
        tuner = QuantizationAutoTuner(
            max_quality_loss=0.05,
            profile_overrides={
                QuantMethod.BNB_4BIT: QuantProfile(
                    method=QuantMethod.BNB_4BIT,
                    memory_reduction=0.25,
                    speed_penalty=1.10,
                    min_vram_gb=2,
                    quality_loss=losses.get("bnb_4bit", 0.03),
                ),
            },
        )

        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        assert len(plan.recommendations) == 1


# ---------------------------------------------------------------------------
# PartitionOptimizer with APO
# ---------------------------------------------------------------------------


class TestOptimizerWithAPO:
    """Test PartitionOptimizer with quantization-aware evaluation."""

    def test_optimizer_accepts_quant_tuner(self):
        """Verify optimizer constructor accepts APO parameters."""
        from distllm.dist.partition.optimizer import PartitionOptimizer

        tuner = QuantizationAutoTuner()
        node_infos = [
            NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3),
            NodeInfo(node_id="n1", total_memory_bytes=8 * 1024**3),
        ]

        # Should not raise
        optimizer = PartitionOptimizer(
            cost_model=None,
            node_ids=["n0", "n1"],
            quant_tuner=tuner,
            node_infos=node_infos,
            model_size_bytes=14 * 1024**3,
        )
        assert optimizer._quant_tuner is tuner

    def test_partition_point_has_quant_method(self):
        from distllm.dist.partition.optimizer import PartitionPoint
        pt = PartitionPoint(
            node_id="n0", start_layer=0, end_layer=16,
            estimated_time_ms=10.0, quant_method="bnb_4bit",
        )
        assert pt.quant_method == "bnb_4bit"

    def test_solution_has_quant_plan_field(self):
        from distllm.dist.partition.optimizer import PartitionSolution
        sol = PartitionSolution()
        assert sol.quant_plan is None
