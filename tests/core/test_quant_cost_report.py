"""Tests: QuantizationAwareCostModel and QuantizationReport.

Tests: cost model with quantization applied, report generation,
comparison with/without quantization, serialization.

Run: pytest tests/core/test_quant_cost_report.py -v
"""

import pytest

from distllm.dist.partition.quantization_tuner import (
    ActivationQuantMethod,
    KVCacheBits,
    NodeInfo,
    NodeQuantRecommendation,
    QuantMethod,
    QuantizationAutoTuner,
    QuantizationPlan,
)
from distllm.dist.partition.quant_cost import (
    QuantizationAwareCostModel,
    QuantNodeCost,
)
from distllm.dist.partition.quant_report import (
    ConflictWarning,
    NodeReport,
    QuantizationReport,
    ReportGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeNodeCost:
    """Minimal NodeCost stand-in for testing."""
    def __init__(self, node_id, start, end, compute_ms=10.0, mem_bytes=4*1024**3, avail=8*1024**3, fits=True, comm_ms=1.0):
        self.node_id = node_id
        self.start_layer = start
        self.end_layer = end
        self.compute_time_ms = compute_ms
        self.communication_time_ms = comm_ms
        self.total_time_ms = compute_ms + comm_ms
        self.memory_bytes = mem_bytes
        self.memory_available_bytes = avail
        self.fits_in_memory = fits


class FakeBaseCostModel:
    """Stub cost model for testing QuantizationAwareCostModel."""
    def evaluate(self, node_id, start, end, batch_size=1, seq_len=4096):
        return FakeNodeCost(node_id, start, end)


# ---------------------------------------------------------------------------
# QuantizationAwareCostModel
# ---------------------------------------------------------------------------


class TestQuantizationAwareCostModel:
    """Test quantization-aware cost model."""

    def test_no_quant_returns_base_cost(self):
        base = FakeBaseCostModel()
        model = QuantizationAwareCostModel(base)
        cost = model.evaluate_with_quant("n0", 0, 32, quant_recommendation=None)
        assert cost.compute_time_ms == 10.0
        assert cost.weight_quant_method == "none"

    def test_quant_reduces_memory(self):
        base = FakeBaseCostModel()
        model = QuantizationAwareCostModel(base)
        rec = NodeQuantRecommendation(
            node_id="n0",
            method=QuantMethod.BNB_4BIT,
            memory_bytes_without_quant=4 * 1024**3,
            memory_bytes_with_quant=1 * 1024**3,
            memory_savings_bytes=3 * 1024**3,
            memory_savings_pct=75.0,
            speed_penalty=1.1,
            quality_loss=0.03,
            reason="test",
        )
        cost = model.evaluate_with_quant("n0", 0, 32, quant_recommendation=rec)
        assert cost.memory_bytes < cost.base_memory_bytes
        assert cost.weight_quant_method == "bnb_4bit"

    def test_quant_increases_compute_time(self):
        base = FakeBaseCostModel()
        model = QuantizationAwareCostModel(base)
        rec = NodeQuantRecommendation(
            node_id="n0",
            method=QuantMethod.BNB_4BIT,
            memory_bytes_without_quant=4 * 1024**3,
            memory_bytes_with_quant=1 * 1024**3,
            memory_savings_bytes=3 * 1024**3,
            memory_savings_pct=75.0,
            speed_penalty=1.5,
            quality_loss=0.03,
            reason="test",
        )
        cost = model.evaluate_with_quant("n0", 0, 32, quant_recommendation=rec)
        assert cost.compute_time_ms > 10.0

    def test_activation_quant_reduces_communication(self):
        base = FakeBaseCostModel()
        model = QuantizationAwareCostModel(base)
        rec = NodeQuantRecommendation(
            node_id="n0",
            method=QuantMethod.BNB_8BIT,
            memory_bytes_without_quant=4 * 1024**3,
            memory_bytes_with_quant=2 * 1024**3,
            memory_savings_bytes=2 * 1024**3,
            memory_savings_pct=50.0,
            speed_penalty=1.05,
            quality_loss=0.01,
            reason="test",
            activation_quant=ActivationQuantMethod.INT8,
        )
        cost = model.evaluate_with_quant("n0", 0, 32, quant_recommendation=rec)
        # INT8 halves bandwidth but adds overhead
        assert cost.communication_time_ms < 1.0 + 0.2  # base + overhead

    def test_evaluate_partition_with_quant(self):
        base = FakeBaseCostModel()
        model = QuantizationAwareCostModel(base)
        partition = [("n0", 0, 16), ("n1", 16, 32)]
        plan = QuantizationPlan(
            recommendations=[
                NodeQuantRecommendation(
                    node_id="n0", method=QuantMethod.BNB_8BIT,
                    memory_bytes_without_quant=2 * 1024**3,
                    memory_bytes_with_quant=1 * 1024**3,
                    memory_savings_bytes=1 * 1024**3,
                    memory_savings_pct=50.0, speed_penalty=1.05,
                    quality_loss=0.01, reason="test",
                ),
            ],
        )
        costs = model.evaluate_partition_with_quant(partition, plan)
        assert len(costs) == 2
        assert costs[0].weight_quant_method == "bnb_8bit"
        assert costs[1].weight_quant_method == "none"  # no recommendation for n1

    def test_compare_with_without_quant(self):
        base = FakeBaseCostModel()
        model = QuantizationAwareCostModel(base)
        partition = [("n0", 0, 16), ("n1", 16, 32)]
        plan = QuantizationPlan(
            recommendations=[
                NodeQuantRecommendation(
                    node_id="n0", method=QuantMethod.BNB_4BIT,
                    memory_bytes_without_quant=2 * 1024**3,
                    memory_bytes_with_quant=512 * 1024**2,
                    memory_savings_bytes=1536 * 1024**2,
                    memory_savings_pct=75.0, speed_penalty=1.1,
                    quality_loss=0.03, reason="test",
                ),
            ],
        )
        comparison = model.compare_with_without_quant(partition, plan)
        assert "without_quant" in comparison
        assert "with_quant" in comparison


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class TestReportGenerator:
    """Test cluster-wide report generation."""

    def test_generate_report(self):
        tuner = QuantizationAutoTuner()
        nodes = [
            NodeInfo(node_id="big", total_memory_bytes=80 * 1024**3),
            NodeInfo(node_id="small", total_memory_bytes=8 * 1024**3),
        ]
        plan = tuner.recommend(nodes, 14 * 1024**3, 32)
        reporter = ReportGenerator()
        report = reporter.generate(plan, nodes, 14 * 1024**3, 32)

        assert isinstance(report, QuantizationReport)
        assert len(report.nodes) == 2
        assert report.summary["num_nodes"] == 2

    def test_report_to_json(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        reporter = ReportGenerator()
        report = reporter.generate(plan, [node], 14 * 1024**3, 32)

        json_str = report.to_json()
        assert '"strategy"' in json_str
        assert '"nodes"' in json_str

    def test_report_to_text(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        reporter = ReportGenerator()
        report = reporter.generate(plan, [node], 14 * 1024**3, 32)

        text = report.to_text()
        assert "Adaptive Precision Optimizer" in text
        assert "n0" in text

    def test_report_detects_calibration_warning(self):
        tuner = QuantizationAutoTuner(max_quality_loss=0.05)
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        reporter = ReportGenerator()
        report = reporter.generate(plan, [node], 14 * 1024**3, 32)

        # If GPTQ/AWQ was selected, should have a calibration warning
        for nr in report.nodes:
            if nr.method in ("gptq", "awq"):
                assert any("calibration" in w.lower() for w in nr.warnings)

    def test_empty_plan_report(self):
        plan = QuantizationPlan(strategy="No nodes")
        reporter = ReportGenerator()
        report = reporter.generate(plan)
        assert report.summary["num_nodes"] == 0


# ---------------------------------------------------------------------------
# NodeReport
# ---------------------------------------------------------------------------


class TestNodeReport:
    """Test NodeReport dataclass."""

    def test_to_dict(self):
        nr = NodeReport(
            node_id="n0",
            method="bnb_4bit",
            method_display="BitsAndBytes NF4",
            memory_without_quant_gb=14.0,
            memory_with_quant_gb=3.5,
            memory_savings_gb=10.5,
            memory_savings_pct=75.0,
            speed_penalty=1.1,
            quality_loss=0.03,
            reason="test",
            activation_quant="none",
            kv_cache_bits="none",
        )
        d = nr.to_dict()
        assert d["node_id"] == "n0"
        assert d["method"] == "bnb_4bit"
