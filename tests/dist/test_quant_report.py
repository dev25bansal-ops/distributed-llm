"""Tests for quant_report module — ReportGenerator, NodeReport, QuantizationReport."""

from __future__ import annotations

import json

import pytest

from distllm.dist.partition.quant_report import (
    ConflictWarning,
    NodeReport,
    QuantizationReport,
    ReportGenerator,
)
from distllm.dist.partition.quantization_tuner import (
    ActivationQuantMethod,
    KVCacheBits,
    NodeInfo,
    NodeQuantRecommendation,
    QuantizationPlan,
    QuantMethod,
)


# ---------------------------------------------------------------------------
# NodeReport
# ---------------------------------------------------------------------------

class TestNodeReport:
    """Test NodeReport dataclass."""

    def test_creation_defaults(self) -> None:
        report = NodeReport(
            node_id="node-0",
            method="none",
            method_display="FP16 (no quantization)",
            memory_without_quant_gb=10.0,
            memory_with_quant_gb=10.0,
            memory_savings_gb=0.0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="Fits in VRAM",
            activation_quant="none",
            kv_cache_bits="none",
        )
        assert report.node_id == "node-0"
        assert report.warnings == []
        assert report.method == "none"
        assert report.memory_savings_pct == 0.0

    def test_to_dict(self) -> None:
        report = NodeReport(
            node_id="node-1",
            method="bnb_4bit",
            method_display="BitsAndBytes NF4",
            memory_without_quant_gb=16.0,
            memory_with_quant_gb=4.0,
            memory_savings_gb=12.0,
            memory_savings_pct=75.0,
            speed_penalty=1.1,
            quality_loss=0.03,
            reason="Saves memory",
            activation_quant="none",
            kv_cache_bits="none",
            warnings=["Tight VRAM"],
        )
        d = report.to_dict()
        assert d["node_id"] == "node-1"
        assert d["memory_savings_gb"] == 12.0
        assert d["warnings"] == ["Tight VRAM"]

    def test_with_warnings(self) -> None:
        report = NodeReport(
            node_id="node-2",
            method="fp8_e4m3",
            method_display="FP8 E4M3",
            memory_without_quant_gb=20.0,
            memory_with_quant_gb=10.0,
            memory_savings_gb=10.0,
            memory_savings_pct=50.0,
            speed_penalty=0.9,
            quality_loss=0.005,
            reason="Hopper GPU",
            activation_quant="fp8_e4m3",
            kv_cache_bits="fp8",
            warnings=["Needs cc>=9.0"],
        )
        assert len(report.warnings) == 1
        assert report.warnings[0] == "Needs cc>=9.0"

    def test_negative_savings(self) -> None:
        """Edge case: negative savings (method uses more memory)."""
        report = NodeReport(
            node_id="node-neg",
            method="none",
            method_display="FP16",
            memory_without_quant_gb=10.0,
            memory_with_quant_gb=12.0,
            memory_savings_gb=-2.0,
            memory_savings_pct=-20.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="Overhead",
            activation_quant="none",
            kv_cache_bits="none",
        )
        assert report.memory_savings_gb == -2.0


# ---------------------------------------------------------------------------
# ConflictWarning
# ---------------------------------------------------------------------------

class TestConflictWarning:
    """Test ConflictWarning dataclass."""

    def test_creation(self) -> None:
        cw = ConflictWarning(node_id="node-0", severity="warning", message="Low VRAM")
        assert cw.node_id == "node-0"
        assert cw.severity == "warning"
        assert cw.message == "Low VRAM"

    def test_all_severities(self) -> None:
        for sev in ("info", "warning", "error"):
            cw = ConflictWarning(node_id="n1", severity=sev, message="test")
            assert cw.severity == sev

    def test_empty_message(self) -> None:
        cw = ConflictWarning(node_id="n1", severity="info", message="")
        assert cw.message == ""


# ---------------------------------------------------------------------------
# QuantizationReport
# ---------------------------------------------------------------------------

class TestQuantizationReport:
    """Test QuantizationReport dataclass."""

    @pytest.fixture
    def sample_report(self) -> QuantizationReport:
        node = NodeReport(
            node_id="node-0",
            method="none",
            method_display="FP16 (no quantization)",
            memory_without_quant_gb=10.0,
            memory_with_quant_gb=10.0,
            memory_savings_gb=0.0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="Fits",
            activation_quant="none",
            kv_cache_bits="none",
        )
        return QuantizationReport(
            strategy="Uniform none",
            nodes=[node],
            summary={"num_nodes": 1},
            baseline_comparison={
                "without_quant": {"oom_nodes": 2, "total_memory_gb": 80.0},
            },
        )

    def test_empty(self) -> None:
        report = QuantizationReport(strategy="empty")
        assert report.strategy == "empty"
        assert report.nodes == []
        assert report.conflicts == []
        assert report.summary == {}
        assert report.baseline_comparison == {}

    def test_to_dict(self, sample_report: QuantizationReport) -> None:
        d = sample_report.to_dict()
        assert d["strategy"] == "Uniform none"
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["node_id"] == "node-0"
        assert d["conflicts"] == []
        assert d["baseline_comparison"]["without_quant"]["oom_nodes"] == 2

    def test_to_json(self, sample_report: QuantizationReport) -> None:
        raw = sample_report.to_json()
        parsed = json.loads(raw)
        assert parsed["strategy"] == "Uniform none"
        assert len(parsed["nodes"]) == 1

    def test_to_json_indent(self, sample_report: QuantizationReport) -> None:
        raw = sample_report.to_json(indent=4)
        assert "    " in raw  # 4-space indent

    def test_to_text_summary_present(self, sample_report: QuantizationReport) -> None:
        text = sample_report.to_text()
        assert "Uniform none" in text
        assert "node-0" in text
        assert "No Quantization" in text
        assert "OOM nodes" in text

    def test_to_text_empty(self) -> None:
        report = QuantizationReport(strategy="empty")
        text = report.to_text()
        assert "Strategy: empty" in text
        assert "Summary:" not in text  # no summary dict -> skipped

    def test_to_text_with_conflicts(self) -> None:
        conflict = ConflictWarning(node_id="node-0", severity="error", message="OOM risk")
        report = QuantizationReport(strategy="test", conflicts=[conflict])
        text = report.to_text()
        assert "OOM risk" in text
        assert "ERROR" in text

    def test_to_text_with_activation_and_kv(self) -> None:
        node = NodeReport(
            node_id="node-0",
            method="gptq",
            method_display="GPTQ 4-bit",
            memory_without_quant_gb=20.0,
            memory_with_quant_gb=5.0,
            memory_savings_gb=15.0,
            memory_savings_pct=75.0,
            speed_penalty=1.0,
            quality_loss=0.02,
            reason="GPTQ method",
            activation_quant="int8",
            kv_cache_bits="int8",
        )
        report = QuantizationReport(strategy="hybrid", nodes=[node])
        text = report.to_text()
        assert "Activation quant" in text
        assert "KV cache" in text

    def test_to_text_no_summary_no_baseline(self) -> None:
        """Report with only strategy and nodes — no summary/baseline_comparison."""
        node = NodeReport(
            node_id="n1",
            method="none",
            method_display="FP16",
            memory_without_quant_gb=8.0,
            memory_with_quant_gb=8.0,
            memory_savings_gb=0.0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="ok",
            activation_quant="none",
            kv_cache_bits="none",
        )
        report = QuantizationReport(strategy="basic", nodes=[node])
        text = report.to_text()
        assert "basic" in text
        assert "Comparison vs No Quantization" not in text


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class TestReportGenerator:
    """Test ReportGenerator class."""

    @pytest.fixture
    def generator(self) -> ReportGenerator:
        return ReportGenerator()

    @pytest.fixture
    def recommendation(self) -> NodeQuantRecommendation:
        return NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.BNB_4BIT,
            memory_bytes_without_quant=20_000_000_000,
            memory_bytes_with_quant=5_000_000_000,
            memory_savings_bytes=15_000_000_000,
            memory_savings_pct=75.0,
            speed_penalty=1.1,
            quality_loss=0.03,
            reason="Fits VRAM with 4-bit",
            activation_quant=ActivationQuantMethod.NONE,
            kv_cache_bits=KVCacheBits.NONE,
        )

    @pytest.fixture
    def plan(self, recommendation: NodeQuantRecommendation) -> QuantizationPlan:
        return QuantizationPlan(
            recommendations=[recommendation],
            strategy="Uniform bnb_4bit",
            total_memory_saved_bytes=15_000_000_000,
            avg_quality_loss=0.03,
        )

    # -- Basic generation --------------------------------------------------

    def test_generate_minimal(self, generator: ReportGenerator, plan: QuantizationPlan) -> None:
        """Generate without any node info."""
        report = generator.generate(plan)
        assert report.strategy == "Uniform bnb_4bit"
        assert len(report.nodes) == 1
        n = report.nodes[0]
        assert n.node_id == "node-0"
        assert n.method == "bnb_4bit"
        # memory_without_quant_gb = 20e9 / 1024^3
        assert n.memory_without_quant_gb == pytest.approx(18.63, rel=0.01)
        # No warnings because node info is None
        assert n.warnings == []

    def test_generate_with_node_info(self, generator: ReportGenerator, plan: QuantizationPlan) -> None:
        """Generate with NodeInfo objects."""
        node = NodeInfo(node_id="node-0", total_memory_bytes=32 * 1024**3)
        report = generator.generate(plan, nodes=[node])
        n = report.nodes[0]
        assert n.memory_savings_pct == 75.0

    def test_generate_with_node_dict(self, generator: ReportGenerator, plan: QuantizationPlan) -> None:
        """Generate with raw dict node info (from_dict path)."""
        node_dict = {"node_id": "node-0", "total_memory_bytes": 32 * 1024**3}
        report = generator.generate(plan, nodes=[node_dict])
        assert len(report.nodes) == 1
        assert report.nodes[0].node_id == "node-0"

    def test_generate_empty_plan(self, generator: ReportGenerator) -> None:
        """Generate with an empty plan (no recommendations)."""
        plan = QuantizationPlan(strategy="empty")
        report = generator.generate(plan)
        assert report.strategy == "empty"
        assert report.nodes == []
        assert report.summary["num_nodes"] == 0
        assert report.summary["methods_used"] == []

    def test_generate_multiple_nodes(self, generator: ReportGenerator) -> None:
        """Generate with multiple recommendations."""
        recs = [
            NodeQuantRecommendation(
                node_id=f"node-{i}",
                method=QuantMethod.NONE,
                memory_bytes_without_quant=10_000_000_000,
                memory_bytes_with_quant=10_000_000_000,
                memory_savings_bytes=0,
                memory_savings_pct=0.0,
                speed_penalty=1.0,
                quality_loss=0.0,
                reason=f"Node {i} fits",
                activation_quant=ActivationQuantMethod.NONE,
                kv_cache_bits=KVCacheBits.NONE,
            )
            for i in range(3)
        ]
        plan = QuantizationPlan(recommendations=recs, strategy="uniform")
        report = generator.generate(plan)
        assert len(report.nodes) == 3
        assert report.summary["num_nodes"] == 3
        assert report.summary["avg_quality_loss"] == 0.0
        assert report.summary["methods_used"] == ["none"]

    def test_generate_summary_model_size(self, generator: ReportGenerator, plan: QuantizationPlan) -> None:
        """Summary includes model_size_gb and num_layers."""
        report = generator.generate(plan, model_size_bytes=20_000_000_000, num_layers=32)
        s = report.summary
        assert s["model_size_gb"] == pytest.approx(18.63, rel=0.01)
        assert s["num_layers"] == 32

    def test_generate_conflicts_recorded(self, generator: ReportGenerator) -> None:
        """Conflicts from node warnings are recorded on the report."""
        rec = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.FP8_E4M3,
            memory_bytes_without_quant=20_000_000_000,
            memory_bytes_with_quant=10_000_000_000,
            memory_savings_bytes=10_000_000_000,
            memory_savings_pct=50.0,
            speed_penalty=0.9,
            quality_loss=0.005,
            reason="FP8",
            activation_quant=ActivationQuantMethod.NONE,
            kv_cache_bits=KVCacheBits.NONE,
        )
        node = NodeInfo(node_id="node-0", compute_capability=8.9, total_memory_bytes=40 * 1024**3)
        plan = QuantizationPlan(recommendations=[rec], strategy="fp8")
        report = generator.generate(plan, nodes=[node])
        assert len(report.conflicts) >= 1

    # -- Warning checks ----------------------------------------------------

    def test_warning_fp8_non_hopper(self, generator: ReportGenerator) -> None:
        """FP8 on non-Hopper should generate a warning."""
        rec = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.FP8_E4M3,
            memory_bytes_without_quant=20_000_000_000,
            memory_bytes_with_quant=10_000_000_000,
            memory_savings_bytes=10_000_000_000,
            memory_savings_pct=50.0,
            speed_penalty=0.9,
            quality_loss=0.005,
            reason="FP8 on Ada",
            activation_quant=ActivationQuantMethod.NONE,
            kv_cache_bits=KVCacheBits.NONE,
        )
        node = NodeInfo(node_id="node-0", compute_capability=8.9, total_memory_bytes=40 * 1024**3)
        plan = QuantizationPlan(recommendations=[rec], strategy="fp8")
        report = generator.generate(plan, nodes=[node])
        n = report.nodes[0]
        assert any("Hopper" in w for w in n.warnings)

    def test_warning_high_quality_loss(self, generator: ReportGenerator) -> None:
        """Quality loss > 0.03 should generate a warning."""
        rec = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.BNB_4BIT,
            memory_bytes_without_quant=20_000_000_000,
            memory_bytes_with_quant=5_000_000_000,
            memory_savings_bytes=15_000_000_000,
            memory_savings_pct=75.0,
            speed_penalty=1.1,
            quality_loss=0.05,
            reason="Aggressive",
            activation_quant=ActivationQuantMethod.NONE,
            kv_cache_bits=KVCacheBits.NONE,
        )
        node = NodeInfo(node_id="node-0", compute_capability=9.0, total_memory_bytes=40 * 1024**3)
        plan = QuantizationPlan(recommendations=[rec], strategy="aggressive")
        report = generator.generate(plan, nodes=[node])
        n = report.nodes[0]
        assert any("Quality loss" in w for w in n.warnings)

    def test_warning_tight_vram(self, generator: ReportGenerator) -> None:
        """VRAM utilization > 90% should generate a warning."""
        rec = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.NONE,
            memory_bytes_without_quant=19_000_000_000,
            memory_bytes_with_quant=19_000_000_000,
            memory_savings_bytes=0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="Tight",
            activation_quant=ActivationQuantMethod.NONE,
            kv_cache_bits=KVCacheBits.NONE,
        )
        node = NodeInfo(node_id="node-0", total_memory_bytes=20_000_000_000)
        plan = QuantizationPlan(recommendations=[rec], strategy="tight")
        report = generator.generate(plan, nodes=[node])
        n = report.nodes[0]
        assert any("VRAM utilization" in w for w in n.warnings)

    def test_warning_calibration(self, generator: ReportGenerator) -> None:
        """GPTQ and AWQ methods should warn about calibration requirement."""
        for method in (QuantMethod.GPTQ, QuantMethod.AWQ):
            rec = NodeQuantRecommendation(
                node_id="node-0",
                method=method,
                memory_bytes_without_quant=20_000_000_000,
                memory_bytes_with_quant=5_000_000_000,
                memory_savings_bytes=15_000_000_000,
                memory_savings_pct=75.0,
                speed_penalty=1.0,
                quality_loss=0.02,
                reason="Calibration needed",
                activation_quant=ActivationQuantMethod.NONE,
                kv_cache_bits=KVCacheBits.NONE,
            )
            node = NodeInfo(node_id="node-0", total_memory_bytes=40 * 1024**3)
            plan = QuantizationPlan(recommendations=[rec], strategy=method.value)
            report = generator.generate(plan, nodes=[node])
            n = report.nodes[0]
            assert any("calibration" in w.lower() for w in n.warnings)

    def test_no_warning_none_method(self, generator: ReportGenerator) -> None:
        """NONE method with low quality loss and loose VRAM -> no warnings."""
        rec = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.NONE,
            memory_bytes_without_quant=10_000_000_000,
            memory_bytes_with_quant=10_000_000_000,
            memory_savings_bytes=0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="Fits",
            activation_quant=ActivationQuantMethod.NONE,
            kv_cache_bits=KVCacheBits.NONE,
        )
        node = NodeInfo(node_id="node-0", total_memory_bytes=80 * 1024**3)
        plan = QuantizationPlan(recommendations=[rec], strategy="none")
        report = generator.generate(plan, nodes=[node])
        assert report.nodes[0].warnings == []

    # -- _method_display ---------------------------------------------------

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            (QuantMethod.NONE, "FP16 (no quantization)"),
            (QuantMethod.BNB_8BIT, "BitsAndBytes INT8"),
            (QuantMethod.BNB_4BIT, "BitsAndBytes NF4"),
            (QuantMethod.GPTQ, "GPTQ 4-bit"),
            (QuantMethod.AWQ, "AWQ 4-bit"),
            (QuantMethod.FP8_E4M3, "FP8 E4M3"),
            (QuantMethod.FP8_E5M2, "FP8 E5M2"),
            (QuantMethod.INT8, "INT8"),
            (QuantMethod.NF4, "NF4"),
        ],
    )
    def test_method_display(
        self,
        generator: ReportGenerator,
        method: QuantMethod,
        expected: str,
    ) -> None:
        assert generator._method_display(method) == expected

    # -- Edge cases --------------------------------------------------------

    def test_generate_no_nodes_list(self, generator: ReportGenerator, plan: QuantizationPlan) -> None:
        """Pass nodes=None explicitly."""
        report = generator.generate(plan, nodes=None)
        assert report.nodes[0].warnings == []

    def test_generate_empty_nodes_list(self, generator: ReportGenerator, plan: QuantizationPlan) -> None:
        """Pass empty nodes list explicitly."""
        report = generator.generate(plan, nodes=[])
        assert report.nodes[0].warnings == []

    def test_generate_zero_model_size(self, generator: ReportGenerator) -> None:
        """Zero model size should not cause division errors."""
        rec = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.NONE,
            memory_bytes_without_quant=0,
            memory_bytes_with_quant=0,
            memory_savings_bytes=0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="Zero size",
            activation_quant=ActivationQuantMethod.NONE,
            kv_cache_bits=KVCacheBits.NONE,
        )
        plan = QuantizationPlan(recommendations=[rec], strategy="zero")
        report = generator.generate(plan, model_size_bytes=0, nodes=[NodeInfo(node_id="node-0")])
        assert report.summary["total_savings_pct"] == 0.0
        assert report.summary["avg_quality_loss"] == 0.0

    def test_generate_different_methods_in_summary(self, generator: ReportGenerator) -> None:
        """methods_used in summary contains sorted unique methods."""
        recs = [
            NodeQuantRecommendation(
                node_id=f"node-{i}",
                method=method,
                memory_bytes_without_quant=10_000_000_000,
                memory_bytes_with_quant=2_500_000_000,
                memory_savings_bytes=7_500_000_000,
                memory_savings_pct=75.0,
                speed_penalty=1.0,
                quality_loss=0.02,
                reason="test",
                activation_quant=ActivationQuantMethod.NONE,
                kv_cache_bits=KVCacheBits.NONE,
            )
            for i, method in enumerate([QuantMethod.INT8, QuantMethod.NF4, QuantMethod.INT8])
        ]
        plan = QuantizationPlan(recommendations=recs, strategy="multi")
        report = generator.generate(plan)
        assert report.summary["methods_used"] == ["int8", "nf4"]

    def test_generate_vram_accumulation(self, generator: ReportGenerator) -> None:
        """Total VRAM in summary sums across all provided NodeInfos."""
        rec = NodeQuantRecommendation(
            node_id="node-0",
            method=QuantMethod.NONE,
            memory_bytes_without_quant=0,
            memory_bytes_with_quant=0,
            memory_savings_bytes=0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="test",
            activation_quant=ActivationQuantMethod.NONE,
            kv_cache_bits=KVCacheBits.NONE,
        )
        nodes = [
            NodeInfo(node_id="node-0", total_memory_bytes=16 * 1024**3),
            NodeInfo(node_id="node-1", total_memory_bytes=24 * 1024**3),
        ]
        plan = QuantizationPlan(recommendations=[rec], strategy="vram_test")
        report = generator.generate(plan, nodes=nodes)
        assert report.summary["total_vram_gb"] == pytest.approx(40.0, rel=0.01)
