"""Tests for distllm.dist.partition.quantization_tuner.

Uses only real objects from the module (zero mocks). Tests the public API
surface: enums, data models, SensitivityAnalyzer, QuantizationAutoTuner,
select_for_node, and non-GPU methods of AutoMixedPrecisionPipeline.

GPU-dependent functions (profile_layer_precision, assign_mixed_precision)
are tested with skipif when torch.cuda is unavailable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import pytest

from distllm.dist.partition.quantization_tuner import (
    ACTIVATION_PROFILES,
    KV_PROFILES,
    QUANT_PROFILES,
    ActivationQuantMethod,
    AutoMixedPrecisionPipeline,
    KVCacheBits,
    LayerPrecisionProfile,
    LayerPrecisionResult,
    LayerQuantPlan,
    MixedPrecisionPlan,
    NodeInfo,
    NodeQuantRecommendation,
    QuantMethod,
    QuantProfile,
    QuantizationAutoTuner,
    QuantizationPlan,
    ScoreWeights,
    SensitivityAnalyzer,
    assign_mixed_precision,
    profile_layer_precision,
    select_for_node,
)

# ---------------------------------------------------------------------------
# Optional torch import for GPU-dependent tests
# ---------------------------------------------------------------------------
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ===========================================================================
# Enums
# ===========================================================================


class TestQuantMethod:
    """QuantMethod enum."""

    def test_values(self) -> None:
        assert QuantMethod.NONE.value == "none"
        assert QuantMethod.BNB_8BIT.value == "bnb_8bit"
        assert QuantMethod.BNB_4BIT.value == "bnb_4bit"
        assert QuantMethod.GPTQ.value == "gptq"
        assert QuantMethod.AWQ.value == "awq"
        assert QuantMethod.FP8_E4M3.value == "fp8_e4m3"
        assert QuantMethod.FP8_E5M2.value == "fp8_e5m2"
        assert QuantMethod.INT8.value == "int8"
        assert QuantMethod.NF4.value == "nf4"

    def test_all_members_unique(self) -> None:
        values = [m.value for m in QuantMethod]
        assert len(values) == len(set(values))

    def test_from_string(self) -> None:
        assert QuantMethod("none") is QuantMethod.NONE
        assert QuantMethod("int8") is QuantMethod.INT8

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            QuantMethod("invalid_quant")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            QuantMethod("")


class TestActivationQuantMethod:
    """ActivationQuantMethod enum."""

    def test_values(self) -> None:
        assert ActivationQuantMethod.NONE.value == "none"
        assert ActivationQuantMethod.INT8.value == "int8"
        assert ActivationQuantMethod.FP8_E4M3.value == "fp8_e4m3"

    def test_all_members_unique(self) -> None:
        values = [m.value for m in ActivationQuantMethod]
        assert len(values) == len(set(values))

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ActivationQuantMethod("fp16")


class TestKVCacheBits:
    """KVCacheBits enum."""

    def test_values(self) -> None:
        assert KVCacheBits.NONE.value == "none"
        assert KVCacheBits.FP8.value == "fp8"
        assert KVCacheBits.INT8.value == "int8"
        assert KVCacheBits.INT4.value == "int4"

    def test_all_members_unique(self) -> None:
        values = [m.value for m in KVCacheBits]
        assert len(values) == len(set(values))

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            KVCacheBits("int2")


# ===========================================================================
# NodeInfo (Pydantic model)
# ===========================================================================


class TestNodeInfo:
    """NodeInfo Pydantic model."""

    def test_defaults(self) -> None:
        node = NodeInfo(node_id="node-0")
        assert node.node_id == "node-0"
        assert node.device_type == "cuda"
        assert node.total_memory_bytes == 8 * 1024**3
        assert node.compute_capability is None
        assert node.gpu_name is None
        assert node.bandwidth_gbps is None
        assert node.num_layers_assigned is None
        assert node.is_hopper_or_newer is False

    def test_custom_values(self) -> None:
        node = NodeInfo(
            node_id="gpu-1",
            device_type="rocm",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
            gpu_name="A100",
            bandwidth_gbps=2000.0,
            num_layers_assigned=16,
            is_hopper_or_newer=False,
        )
        assert node.node_id == "gpu-1"
        assert node.device_type == "rocm"
        assert node.compute_capability == 8.0
        assert node.num_layers_assigned == 16
        assert node.is_hopper_or_newer is False
        assert node.gpu_name == "A100"

    def test_hopper_true_when_cc_9_or_above(self) -> None:
        node = NodeInfo(node_id="h100", compute_capability=9.0, is_hopper_or_newer=True)
        assert node.is_hopper_or_newer is True

    def test_from_dict(self) -> None:
        d: dict = {
            "node_id": "node-2",
            "total_memory_bytes": 40 * 1024**3,
            "compute_capability": 9.0,
            "gpu_name": "H100",
            "num_layers_assigned": 8,
        }
        node = NodeInfo.from_dict(d)
        assert node.node_id == "node-2"
        assert node.compute_capability == 9.0
        assert node.gpu_name == "H100"
        assert node.num_layers_assigned == 8
        # from_dict does not set is_hopper_or_newer; it is False by default
        assert node.is_hopper_or_newer is False

    def test_from_dict_ignores_extra_keys(self) -> None:
        d: dict = {
            "node_id": "n",
            "total_memory_bytes": 16 * 1024**3,
            "extra_field": "should_be_ignored",
        }
        node = NodeInfo.from_dict(d)
        assert node.node_id == "n"
        assert node.total_memory_bytes == 16 * 1024**3

    def test_from_dict_minimal(self) -> None:
        node = NodeInfo.from_dict({"node_id": "n"})
        assert node.node_id == "n"
        assert node.total_memory_bytes == 8 * 1024**3

    def test_from_dict_negative_memory_raises(self) -> None:
        with pytest.raises(Exception):
            NodeInfo.from_dict({"node_id": "n", "total_memory_bytes": -1})

    def test_node_id_required(self) -> None:
        with pytest.raises(Exception):
            NodeInfo()  # type: ignore[call-arg]

    def test_serialize_roundtrip(self) -> None:
        node = NodeInfo(
            node_id="r1",
            total_memory_bytes=16 * 1024**3,
            compute_capability=8.9,
            gpu_name="A100",
        )
        d = node.model_dump()
        restored = NodeInfo(**d)
        assert restored.node_id == node.node_id
        assert restored.total_memory_bytes == node.total_memory_bytes
        assert restored.compute_capability == node.compute_capability

    def test_from_gpu_profile(self) -> None:
        """Test NodeInfo.from_gpu_profile with a GPUProfile-like object."""
        from dataclasses import dataclass

        @dataclass
        class FakeGPUProfile:
            total_memory_bytes: int = 80 * 1024**3
            name: str = "A100"
            memory_bandwidth_gbps: float = 2000.0
            compute_capability: float = 8.0

        profile = FakeGPUProfile()
        node = NodeInfo.from_gpu_profile(profile, node_id="n0", num_layers_assigned=8)
        assert node.node_id == "n0"
        assert node.total_memory_bytes == 80 * 1024**3
        assert node.gpu_name == "A100"
        assert node.bandwidth_gbps == 2000.0
        assert node.compute_capability == 8.0
        assert node.num_layers_assigned == 8
        assert node.is_hopper_or_newer is False

    def test_from_gpu_profile_hopper(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class FakeGPUProfile:
            total_memory_bytes: int = 80 * 1024**3
            name: str = "H100"
            memory_bandwidth_gbps: float = 3000.0
            compute_capability: float = 9.0

        profile = FakeGPUProfile()
        node = NodeInfo.from_gpu_profile(profile, node_id="h100")
        assert node.is_hopper_or_newer is True
        assert node.gpu_name == "H100"


# ===========================================================================
# ScoreWeights
# ===========================================================================


class TestScoreWeights:
    """ScoreWeights dataclass."""

    def test_defaults(self) -> None:
        w = ScoreWeights()
        assert w.headroom == 0.5
        assert w.quality == 0.3
        assert w.speed == 0.2

    def test_normalized(self) -> None:
        n = ScoreWeights(headroom=1.0, quality=1.0, speed=1.0).normalized()
        assert math.isclose(n.headroom, 1 / 3)
        assert math.isclose(n.quality, 1 / 3)
        assert math.isclose(n.speed, 1 / 3)

    def test_normalized_zero_total(self) -> None:
        n = ScoreWeights(headroom=0.0, quality=0.0, speed=0.0).normalized()
        assert math.isclose(n.headroom, 1 / 3)
        assert math.isclose(n.quality, 1 / 3)
        assert math.isclose(n.speed, 1 / 3)

    def test_normalized_one_dominant(self) -> None:
        n = ScoreWeights(headroom=1.0, quality=0.0, speed=0.0).normalized()
        assert math.isclose(n.headroom, 1.0)
        assert math.isclose(n.quality, 0.0)
        assert math.isclose(n.speed, 0.0)

    def test_normalized_returns_new_instance(self) -> None:
        w = ScoreWeights()
        n = w.normalized()
        assert n is not w

    def test_asdict(self) -> None:
        d = asdict(ScoreWeights(headroom=0.7, quality=0.2, speed=0.1))
        assert d == {"headroom": 0.7, "quality": 0.2, "speed": 0.1}


# ===========================================================================
# QuantProfile and constant profile registries
# ===========================================================================


class TestQuantProfile:
    """QuantProfile dataclass and QUANT_PROFILES registry."""

    def test_all_methods_have_profiles(self) -> None:
        for method in QuantMethod:
            assert method in QUANT_PROFILES, f"Missing profile for {method}"

    def test_none_profile(self) -> None:
        p = QUANT_PROFILES[QuantMethod.NONE]
        assert p.memory_reduction == 1.0
        assert p.speed_penalty == 1.0
        assert p.min_vram_gb == 0
        assert p.quality_loss == 0.0

    def test_fp8_e4m3_requires_hopper(self) -> None:
        p = QUANT_PROFILES[QuantMethod.FP8_E4M3]
        assert p.min_compute_capability == 9.0

    def test_gptq_requires_calibration(self) -> None:
        p = QUANT_PROFILES[QuantMethod.GPTQ]
        assert p.requires_calibration is True

    def test_profile_invariants(self) -> None:
        for method, profile in QUANT_PROFILES.items():
            assert 0.0 <= profile.quality_loss <= 1.0, f"{method} quality_loss={profile.quality_loss}"
            assert profile.memory_reduction > 0, f"{method} memory_reduction must be positive"
            assert profile.memory_reduction <= 1.0, f"{method} memory_reduction must be <= 1.0"
            assert profile.speed_penalty > 0, f"{method} speed_penalty must be positive"
            assert profile.min_vram_gb >= 0, f"{method} min_vram_gb must be >= 0"

    def test_none_supports_all_hardware(self) -> None:
        p = QUANT_PROFILES[QuantMethod.NONE]
        for hw in ("cuda", "rocm", "mps", "xpu", "cpu"):
            assert hw in p.supported_hardware

    def test_bnb_only_cuda_rocm(self) -> None:
        p = QUANT_PROFILES[QuantMethod.BNB_8BIT]
        assert "cuda" in p.supported_hardware
        assert "rocm" in p.supported_hardware
        assert "cpu" not in p.supported_hardware

    def test_awq_only_cuda(self) -> None:
        p = QUANT_PROFILES[QuantMethod.AWQ]
        assert p.supported_hardware == ["cuda"]


class TestActivationProfiles:
    """ACTIVATION_PROFILES registry."""

    def test_all_methods_present(self) -> None:
        for method in ActivationQuantMethod:
            assert method in ACTIVATION_PROFILES

    def test_none_profile(self) -> None:
        p = ACTIVATION_PROFILES[ActivationQuantMethod.NONE]
        assert p["bandwidth_reduction"] == 1.0
        assert p["quality_loss"] == 0.0
        assert p["overhead_ms"] == 0.0

    def test_int8_reduces_bandwidth(self) -> None:
        p = ACTIVATION_PROFILES[ActivationQuantMethod.INT8]
        assert p["bandwidth_reduction"] == 0.5

    def test_fp8_quality_loss(self) -> None:
        p = ACTIVATION_PROFILES[ActivationQuantMethod.FP8_E4M3]
        assert p["quality_loss"] == 0.003
        assert p["overhead_ms"] == 0.05


class TestKVProfiles:
    """KV_PROFILES registry."""

    def test_all_methods_present(self) -> None:
        for method in KVCacheBits:
            assert method in KV_PROFILES

    def test_int4_has_highest_quality_loss(self) -> None:
        assert KV_PROFILES[KVCacheBits.INT4]["quality_loss"] == 0.02

    def test_none_no_loss(self) -> None:
        assert KV_PROFILES[KVCacheBits.NONE]["quality_loss"] == 0.0

    def test_memory_reductions(self) -> None:
        assert KV_PROFILES[KVCacheBits.NONE]["memory_reduction"] == 1.0
        assert KV_PROFILES[KVCacheBits.FP8]["memory_reduction"] == 0.5
        assert KV_PROFILES[KVCacheBits.INT8]["memory_reduction"] == 0.5
        assert KV_PROFILES[KVCacheBits.INT4]["memory_reduction"] == 0.25


# ===========================================================================
# LayerQuantPlan
# ===========================================================================


class TestLayerQuantPlan:
    """LayerQuantPlan dataclass."""

    def test_minimal(self) -> None:
        plan = LayerQuantPlan(
            layer_idx=0,
            layer_type="attention",
            weight_dtype="float16",
            activation_dtype="float16",
        )
        assert plan.layer_idx == 0
        assert plan.sensitivity_score == 0.0
        assert plan.compression_ratio == 1.0

    def test_summary(self) -> None:
        plan = LayerQuantPlan(
            layer_idx=3,
            layer_type="mlp",
            weight_dtype="int8",
            activation_dtype="float16",
            sensitivity_score=0.3,
            compression_ratio=2.0,
        )
        s = plan.summary()
        assert "L3" in s
        assert "mlp" in s
        assert "int8" in s
        assert "0.30" in s

    def test_custom_values(self) -> None:
        plan = LayerQuantPlan(
            layer_idx=5,
            layer_type="norm",
            weight_dtype="nf4",
            activation_dtype="float16",
            sensitivity_score=0.95,
            compression_ratio=4.0,
        )
        assert plan.layer_type == "norm"
        assert plan.sensitivity_score == 0.95
        assert plan.compression_ratio == 4.0


# ===========================================================================
# MixedPrecisionPlan
# ===========================================================================


class TestMixedPrecisionPlan:
    """MixedPrecisionPlan dataclass."""

    def test_empty(self) -> None:
        plan = MixedPrecisionPlan()
        assert plan.plans == []
        assert plan.overall_compression_ratio == 1.0
        assert plan.num_layers == 0

    def test_summary(self) -> None:
        plans = [
            LayerQuantPlan(0, "attention", "float16", "float16"),
            LayerQuantPlan(1, "mlp", "int8", "float16"),
        ]
        plan = MixedPrecisionPlan(
            plans=plans, overall_compression_ratio=1.5, num_layers=2
        )
        s = plan.summary()
        assert "2 layers" in s
        assert "1.5x" in s
        assert "L0" in s
        assert "L1" in s

    def test_with_plans(self) -> None:
        plans = [
            LayerQuantPlan(i, "attention" if i % 2 == 0 else "mlp", "float16", "float16")
            for i in range(4)
        ]
        plan = MixedPrecisionPlan(plans=plans, num_layers=4)
        assert len(plan.plans) == 4
        assert plan.num_layers == 4


# ===========================================================================
# NodeQuantRecommendation
# ===========================================================================


class TestNodeQuantRecommendation:
    """NodeQuantRecommendation dataclass."""

    @staticmethod
    def _make(
        node_id: str = "n0",
        method: QuantMethod = QuantMethod.NONE,
    ) -> NodeQuantRecommendation:
        return NodeQuantRecommendation(
            node_id=node_id,
            method=method,
            memory_bytes_without_quant=1000,
            memory_bytes_with_quant=1000,
            memory_savings_bytes=0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="test",
        )

    def test_minimal(self) -> None:
        rec = self._make()
        assert rec.node_id == "n0"
        assert rec.activation_quant is ActivationQuantMethod.NONE
        assert rec.kv_cache_bits is KVCacheBits.NONE
        assert rec.mixed_precision_plan is None

    def test_to_dict(self) -> None:
        rec = NodeQuantRecommendation(
            node_id="n0",
            method=QuantMethod.INT8,
            memory_bytes_without_quant=1000,
            memory_bytes_with_quant=500,
            memory_savings_bytes=500,
            memory_savings_pct=50.0,
            speed_penalty=1.02,
            quality_loss=0.01,
            reason="test",
            activation_quant=ActivationQuantMethod.INT8,
            kv_cache_bits=KVCacheBits.INT4,
        )
        d = rec.to_dict()
        assert d["node_id"] == "n0"
        assert d["method"] == "int8"
        assert d["activation_quant"] == "int8"
        assert d["kv_cache_bits"] == "int4"
        assert d["mixed_precision_plan"] is None

    def test_to_dict_with_mixed_precision(self) -> None:
        mp = MixedPrecisionPlan(
            plans=[LayerQuantPlan(0, "attention", "float16", "float16")],
            num_layers=1,
        )
        rec = NodeQuantRecommendation(
            node_id="n0",
            method=QuantMethod.NONE,
            memory_bytes_without_quant=1000,
            memory_bytes_with_quant=1000,
            memory_savings_bytes=0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="test",
            mixed_precision_plan=mp,
        )
        d = rec.to_dict()
        assert d["mixed_precision_plan"] is not None
        assert d["mixed_precision_plan"]["num_layers"] == 1
        assert d["mixed_precision_plan"]["overall_compression_ratio"] == 1.0
        assert len(d["mixed_precision_plan"]["plans"]) == 1

    def test_to_dict_with_activation_and_kv(self) -> None:
        rec = NodeQuantRecommendation(
            node_id="n0",
            method=QuantMethod.NF4,
            memory_bytes_without_quant=2000,
            memory_bytes_with_quant=500,
            memory_savings_bytes=1500,
            memory_savings_pct=75.0,
            speed_penalty=1.08,
            quality_loss=0.025,
            reason="aggressive",
            activation_quant=ActivationQuantMethod.FP8_E4M3,
            kv_cache_bits=KVCacheBits.INT8,
        )
        d = rec.to_dict()
        assert d["method"] == "nf4"
        assert d["activation_quant"] == "fp8_e4m3"
        assert d["kv_cache_bits"] == "int8"
        assert d["memory_savings_pct"] == 75.0


# ===========================================================================
# QuantizationPlan
# ===========================================================================


class TestQuantizationPlan:
    """QuantizationPlan dataclass."""

    @staticmethod
    def _make_rec(
        node_id: str = "n0", method: QuantMethod = QuantMethod.NONE
    ) -> NodeQuantRecommendation:
        return NodeQuantRecommendation(
            node_id=node_id,
            method=method,
            memory_bytes_without_quant=1000,
            memory_bytes_with_quant=1000,
            memory_savings_bytes=0,
            memory_savings_pct=0.0,
            speed_penalty=1.0,
            quality_loss=0.0,
            reason="test",
        )

    def test_empty(self) -> None:
        plan = QuantizationPlan()
        assert plan.recommendations == []
        assert plan.strategy == ""
        assert plan.total_memory_saved_bytes == 0
        assert plan.avg_quality_loss == 0.0
        assert plan.methods_used == set()
        assert plan.timestamp > 0

    def test_methods_used(self) -> None:
        recs = [
            self._make_rec("n0", QuantMethod.INT8),
            self._make_rec("n1", QuantMethod.NF4),
        ]
        plan = QuantizationPlan(recommendations=recs)
        assert plan.methods_used == {QuantMethod.INT8, QuantMethod.NF4}

    def test_methods_used_empty(self) -> None:
        plan = QuantizationPlan()
        assert plan.methods_used == set()

    def test_methods_used_deduplicates(self) -> None:
        recs = [
            self._make_rec("n0", QuantMethod.INT8),
            self._make_rec("n1", QuantMethod.INT8),
        ]
        plan = QuantizationPlan(recommendations=recs)
        assert plan.methods_used == {QuantMethod.INT8}

    def test_summary(self) -> None:
        recs = [self._make_rec("n0", QuantMethod.INT8)]
        plan = QuantizationPlan(
            recommendations=recs, strategy="test", total_memory_saved_bytes=500
        )
        s = plan.summary()
        assert "test" in s
        assert "n0" in s
        assert "int8" in s

    def test_summary_with_activation_and_kv(self) -> None:
        rec = NodeQuantRecommendation(
            node_id="n0",
            method=QuantMethod.INT8,
            memory_bytes_without_quant=1000,
            memory_bytes_with_quant=500,
            memory_savings_bytes=500,
            memory_savings_pct=50.0,
            speed_penalty=1.02,
            quality_loss=0.01,
            reason="test",
            activation_quant=ActivationQuantMethod.INT8,
            kv_cache_bits=KVCacheBits.INT4,
        )
        plan = QuantizationPlan(recommendations=[rec], strategy="hybrid")
        s = plan.summary()
        assert "activation: int8" in s
        assert "kv_cache: int4" in s

    def test_to_dict(self) -> None:
        recs = [self._make_rec("n0", QuantMethod.INT8)]
        plan = QuantizationPlan(
            recommendations=recs,
            strategy="test",
            total_memory_saved_bytes=500,
            avg_quality_loss=0.01,
        )
        d = plan.to_dict()
        assert d["strategy"] == "test"
        assert d["total_memory_saved_bytes"] == 500
        assert d["avg_quality_loss"] == 0.01
        assert len(d["recommendations"]) == 1

    def test_to_json(self) -> None:
        recs = [self._make_rec("n0", QuantMethod.NONE)]
        plan = QuantizationPlan(recommendations=recs, strategy="none")
        j = plan.to_json()
        parsed = json.loads(j)
        assert parsed["strategy"] == "none"
        assert len(parsed["recommendations"]) == 1

    def test_from_dict_roundtrip(self) -> None:
        recs = [self._make_rec("n0", QuantMethod.INT8)]
        original = QuantizationPlan(
            recommendations=recs,
            strategy="test",
            total_memory_saved_bytes=500,
        )
        d = original.to_dict()
        restored = QuantizationPlan.from_dict(d)
        assert restored.strategy == original.strategy
        assert restored.total_memory_saved_bytes == original.total_memory_saved_bytes
        assert restored.avg_quality_loss == original.avg_quality_loss
        assert len(restored.recommendations) == len(original.recommendations)
        assert restored.recommendations[0].method == original.recommendations[0].method

    def test_from_json_roundtrip(self) -> None:
        recs = [self._make_rec("n0", QuantMethod.NF4)]
        original = QuantizationPlan(recommendations=recs, strategy="nf4")
        j = original.to_json()
        restored = QuantizationPlan.from_json(j)
        assert restored.strategy == "nf4"
        assert restored.recommendations[0].method == QuantMethod.NF4
        assert restored.recommendations[0].node_id == "n0"

    def test_from_dict_empty(self) -> None:
        plan = QuantizationPlan.from_dict({})
        assert plan.recommendations == []
        assert plan.strategy == ""
        assert plan.total_memory_saved_bytes == 0

    def test_from_dict_with_activation_and_kv(self) -> None:
        data: dict = {
            "strategy": "hybrid",
            "total_memory_saved_bytes": 1500,
            "avg_quality_loss": 0.01,
            "timestamp": 1000.0,
            "recommendations": [
                {
                    "node_id": "n0",
                    "method": "int8",
                    "memory_bytes_without_quant": 2000,
                    "memory_bytes_with_quant": 1000,
                    "memory_savings_bytes": 1000,
                    "memory_savings_pct": 50.0,
                    "speed_penalty": 1.02,
                    "quality_loss": 0.01,
                    "reason": "test",
                    "activation_quant": "int8",
                    "kv_cache_bits": "int4",
                }
            ],
        }
        plan = QuantizationPlan.from_dict(data)
        assert len(plan.recommendations) == 1
        rec = plan.recommendations[0]
        assert rec.method == QuantMethod.INT8
        assert rec.activation_quant == ActivationQuantMethod.INT8
        assert rec.kv_cache_bits == KVCacheBits.INT4


# ===========================================================================
# SensitivityAnalyzer
# ===========================================================================


class TestSensitivityAnalyzer:
    """SensitivityAnalyzer layer classification and dtype recommendation."""

    def setup_method(self) -> None:
        self.analyzer = SensitivityAnalyzer()

    # --- classify_layer ---

    def test_classify_attention_q_proj(self) -> None:
        assert self.analyzer.classify_layer("model.layers.0.self_attn.q_proj") == "attention"

    def test_classify_attention_k_proj(self) -> None:
        assert self.analyzer.classify_layer("model.layers.0.self_attn.k_proj") == "attention"

    def test_classify_attention_v_proj(self) -> None:
        assert self.analyzer.classify_layer("model.layers.0.self_attn.v_proj") == "attention"

    def test_classify_attention_explicit(self) -> None:
        assert self.analyzer.classify_layer("attention") == "attention"

    def test_classify_attention_output(self) -> None:
        assert self.analyzer.classify_layer("model.attn.output") == "attention"

    def test_classify_mlp(self) -> None:
        assert self.analyzer.classify_layer("model.layers.0.mlp") == "mlp"

    def test_classify_mlp_gate_proj(self) -> None:
        assert self.analyzer.classify_layer("model.layers.1.mlp.gate_proj") == "mlp"

    def test_classify_mlp_up_proj(self) -> None:
        assert self.analyzer.classify_layer("model.layers.2.mlp.up_proj") == "mlp"

    def test_classify_mlp_down_proj(self) -> None:
        assert self.analyzer.classify_layer("down_proj") == "mlp"

    def test_classify_embed(self) -> None:
        assert self.analyzer.classify_layer("model.embed_tokens") == "embed"

    def test_classify_embed_short(self) -> None:
        assert self.analyzer.classify_layer("embed") == "embed"

    def test_classify_lm_head(self) -> None:
        assert self.analyzer.classify_layer("lm_head") == "lm_head"

    def test_classify_output(self) -> None:
        assert self.analyzer.classify_layer("model.output") == "lm_head"

    def test_classify_norm(self) -> None:
        assert self.analyzer.classify_layer("model.norm") == "norm"

    def test_classify_ln(self) -> None:
        assert self.analyzer.classify_layer("model.layers.0.ln_1") == "norm"

    def test_classify_layernorm(self) -> None:
        assert self.analyzer.classify_layer("model.layers.0.post_attention_layernorm") == "norm"

    def test_classify_input_layernorm(self) -> None:
        assert self.analyzer.classify_layer("input_layernorm") == "norm"

    def test_classify_unknown_falls_back_to_mlp(self) -> None:
        assert self.analyzer.classify_layer("some_random_layer") == "mlp"

    def test_classify_empty_string_falls_back_to_mlp(self) -> None:
        assert self.analyzer.classify_layer("") == "mlp"

    def test_classify_attention_precedence_before_embed(self) -> None:
        """'attn' is checked before 'embed' alphabetically? No, order is embed, lm_head, norm, attn."""
        # Actually the check order in the source is:
        # embed first -> lm_head -> norm -> attention -> mlp
        # A name containing both 'embed' and 'attn' would match 'embed' first
        assert self.analyzer.classify_layer("embed_attn_layer") == "embed"

    # --- recommend_dtype ---

    def test_recommend_dtype_high_sensitivity(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("embed")
        assert dtype == "float16"
        assert sens == 0.9

    def test_recommend_dtype_medium_sensitivity(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("attention")
        assert dtype == "int8"
        assert sens == 0.6

    def test_recommend_dtype_low_sensitivity(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("mlp")
        assert dtype == "nf4"
        assert sens == 0.3

    def test_recommend_dtype_norm(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("norm")
        assert dtype == "float16"
        assert sens == 0.95

    def test_recommend_dtype_with_override(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("mlp", sensitivity_override=0.9)
        assert dtype == "float16"
        assert sens == 0.9

    def test_recommend_dtype_boundary_0_8(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("any", sensitivity_override=0.8)
        assert dtype == "float16"
        assert sens == 0.8

    def test_recommend_dtype_just_below_0_8(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("any", sensitivity_override=0.79)
        assert dtype == "int8"
        assert sens == 0.79

    def test_recommend_dtype_boundary_0_5(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("any", sensitivity_override=0.5)
        assert dtype == "int8"
        assert sens == 0.5

    def test_recommend_dtype_just_below_0_5(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("any", sensitivity_override=0.49)
        assert dtype == "nf4"
        assert sens == 0.49

    def test_recommend_dtype_zero_sensitivity(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("any", sensitivity_override=0.0)
        assert dtype == "nf4"
        assert sens == 0.0

    def test_recommend_dtype_unknown_layer_type(self) -> None:
        dtype, sens = self.analyzer.recommend_dtype("unknown")
        # Falls back to LAYER_SENSITIVITY.get(layer_type, 0.5)
        assert sens == 0.5
        assert dtype == "int8"

    # --- build_mixed_precision_plan ---

    def test_build_mixed_precision_plan(self) -> None:
        plan = self.analyzer.build_mixed_precision_plan(num_layers=4, layers_per_node=4)
        assert plan.num_layers == 4
        assert len(plan.plans) == 4
        # Even layers default to "attention" (int8), odd to "mlp" (nf4)
        assert plan.plans[0].weight_dtype == "int8"
        assert plan.plans[1].weight_dtype == "nf4"
        assert plan.plans[2].weight_dtype == "int8"
        assert plan.plans[3].weight_dtype == "nf4"
        assert plan.overall_compression_ratio > 1.0

    def test_build_mixed_precision_plan_zero_layers(self) -> None:
        plan = self.analyzer.build_mixed_precision_plan(num_layers=0, layers_per_node=0)
        assert plan.num_layers == 0
        assert plan.plans == []
        assert plan.overall_compression_ratio == 0.0

    def test_build_mixed_precision_plan_single_layer(self) -> None:
        plan = self.analyzer.build_mixed_precision_plan(num_layers=1, layers_per_node=1)
        assert plan.num_layers == 1
        assert len(plan.plans) == 1
        # Even index (0) defaults to "attention"
        assert plan.plans[0].weight_dtype == "int8"

    def test_build_mixed_precision_plan_custom_fn(self) -> None:
        def layer_type_fn(i: int) -> str:
            if i == 0:
                return "embed"
            if i == 1:
                return "attention"
            return "mlp"

        plan = self.analyzer.build_mixed_precision_plan(
            num_layers=3, layers_per_node=3, layer_type_fn=layer_type_fn
        )
        assert plan.plans[0].weight_dtype == "float16"  # embed
        assert plan.plans[1].weight_dtype == "int8"  # attention
        assert plan.plans[2].weight_dtype == "nf4"  # mlp

    def test_build_mixed_precision_plan_compression_ratio(self) -> None:
        plan = self.analyzer.build_mixed_precision_plan(num_layers=1, layers_per_node=1)
        # int8 -> compression_ratio = 2.0, avg = 2.0 / 1 = 2.0
        assert plan.overall_compression_ratio == 2.0

    def test_layer_sensitivity_constant_values(self) -> None:
        assert SensitivityAnalyzer.LAYER_SENSITIVITY["embed"] == 0.9
        assert SensitivityAnalyzer.LAYER_SENSITIVITY["lm_head"] == 0.9
        assert SensitivityAnalyzer.LAYER_SENSITIVITY["norm"] == 0.95
        assert SensitivityAnalyzer.LAYER_SENSITIVITY["attention"] == 0.6
        assert SensitivityAnalyzer.LAYER_SENSITIVITY["mlp"] == 0.3


# ===========================================================================
# QuantizationAutoTuner
# ===========================================================================


class TestQuantizationAutoTuner:
    """QuantizationAutoTuner — core quantization recommendation engine."""

    @staticmethod
    def _make_node(
        node_id: str = "n0",
        memory_bytes: int = 16 * 1024**3,
        cc: float | None = 8.0,
        device: str = "cuda",
        layers: int | None = None,
    ) -> NodeInfo:
        return NodeInfo(
            node_id=node_id,
            total_memory_bytes=memory_bytes,
            compute_capability=cc,
            device_type=device,
            num_layers_assigned=layers,
            gpu_name="TestGPU",
        )

    # --- constructor ---

    def test_default_constructor(self) -> None:
        tuner = QuantizationAutoTuner()
        assert tuner is not None

    def test_constructor_custom_max_quality_loss(self) -> None:
        tuner = QuantizationAutoTuner(max_quality_loss=0.01)
        assert tuner is not None

    def test_constructor_prefer_speed(self) -> None:
        tuner = QuantizationAutoTuner(prefer_speed=True)
        assert tuner is not None

    def test_constructor_require_calibration(self) -> None:
        tuner = QuantizationAutoTuner(require_calibration=True)
        assert tuner is not None

    def test_constructor_custom_weights(self) -> None:
        w = ScoreWeights(headroom=0.8, quality=0.1, speed=0.1)
        tuner = QuantizationAutoTuner(weights=w)
        assert tuner is not None

    def test_constructor_profile_overrides(self) -> None:
        overrides = {
            QuantMethod.INT8: QuantProfile(
                method=QuantMethod.INT8,
                memory_reduction=0.1,
                speed_penalty=1.0,
                min_vram_gb=0,
                quality_loss=0.001,
            )
        }
        tuner = QuantizationAutoTuner(profile_overrides=overrides)
        assert tuner is not None

    # --- recommend with empty / edge cases ---

    def test_recommend_empty_nodes(self) -> None:
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([], 1000, 1)
        assert plan.strategy == "No nodes provided"
        assert plan.recommendations == []
        assert plan.total_memory_saved_bytes == 0

    def test_recommend_single_node_no_quant_needed(self) -> None:
        """80 GB VRAM, 10 GB model -> no quantization needed."""
        node = self._make_node(memory_bytes=80 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32)
        assert plan.recommendations[0].method is QuantMethod.NONE

    def test_recommend_single_node_quant_needed(self) -> None:
        """4 GB VRAM, 10 GB model -> quantization required."""
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32)
        assert plan.recommendations[0].method is not QuantMethod.NONE
        assert plan.recommendations[0].memory_savings_bytes > 0

    def test_recommend_max_quality_loss_strict(self) -> None:
        """Very strict quality loss may force fallback or no quant candidates."""
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner(max_quality_loss=0.001)
        plan = tuner.recommend([node], 10_000_000_000, 32)
        rec = plan.recommendations[0]
        # Either the method fits within quality budget or it falls back to NONE
        if rec.method is not QuantMethod.NONE:
            assert rec.quality_loss <= 0.001

    def test_recommend_with_profile_overrides(self) -> None:
        """Override INT8 to be extremely aggressive -> should be selected."""
        node = self._make_node(memory_bytes=4 * 1024**3)
        overrides = {
            QuantMethod.INT8: QuantProfile(
                method=QuantMethod.INT8,
                memory_reduction=0.1,
                speed_penalty=1.0,
                min_vram_gb=0,
                quality_loss=0.001,
            )
        }
        tuner = QuantizationAutoTuner(profile_overrides=overrides)
        plan = tuner.recommend([node], 10_000_000_000, 32)
        assert plan.recommendations[0].method is QuantMethod.INT8

    def test_recommend_prefer_speed_vs_normal(self) -> None:
        """Both normal and speed-preferring tuners produce valid results."""
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner_normal = QuantizationAutoTuner(prefer_speed=False)
        tuner_fast = QuantizationAutoTuner(prefer_speed=True)
        plan_normal = tuner_normal.recommend([node], 10_000_000_000, 32)
        plan_fast = tuner_fast.recommend([node], 10_000_000_000, 32)
        assert len(plan_normal.recommendations) == 1
        assert len(plan_fast.recommendations) == 1

    def test_recommend_require_calibration_excludes_gptq_awq(self) -> None:
        node = self._make_node(memory_bytes=8 * 1024**3)
        tuner = QuantizationAutoTuner(require_calibration=True)
        plan = tuner.recommend([node], 20_000_000_000, 32)
        method = plan.recommendations[0].method
        assert method not in (QuantMethod.GPTQ, QuantMethod.AWQ)

    def test_recommend_custom_weights(self) -> None:
        node = self._make_node(memory_bytes=4 * 1024**3)
        weights = ScoreWeights(headroom=1.0, quality=0.0, speed=0.0)
        tuner = QuantizationAutoTuner(weights=weights)
        plan = tuner.recommend([node], 10_000_000_000, 32)
        assert len(plan.recommendations) == 1

    def test_recommend_with_dict_nodes(self) -> None:
        nodes: list = [
            {"node_id": "n0", "total_memory_bytes": 4 * 1024**3, "num_layers_assigned": 4}
        ]
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend(nodes, 10_000_000_000, 32)
        assert len(plan.recommendations) == 1

    # --- multiple nodes ---

    def test_recommend_multiple_nodes(self) -> None:
        nodes = [
            self._make_node("n0", memory_bytes=80 * 1024**3),  # Large
            self._make_node("n1", memory_bytes=4 * 1024**3),  # Small
        ]
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend(nodes, 10_000_000_000, 32)
        assert len(plan.recommendations) == 2
        assert plan.recommendations[0].method is QuantMethod.NONE
        assert plan.recommendations[1].method is not QuantMethod.NONE

    def test_recommend_three_nodes(self) -> None:
        nodes = [
            self._make_node("n0", memory_bytes=80 * 1024**3),
            self._make_node("n1", memory_bytes=16 * 1024**3),
            self._make_node("n2", memory_bytes=4 * 1024**3),
        ]
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend(nodes, 10_000_000_000, 32)
        assert len(plan.recommendations) == 3

    # --- strategy description ---

    def test_strategy_description_no_nodes(self) -> None:
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([], 1000, 1)
        assert plan.strategy == "No nodes provided"

    def test_strategy_description_no_quant(self) -> None:
        node = self._make_node(memory_bytes=80 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 1_000_000_000, 32)
        assert "No quantization needed" in plan.strategy

    def test_strategy_description_uniform(self) -> None:
        nodes = [
            self._make_node("n0", memory_bytes=4 * 1024**3),
            self._make_node("n1", memory_bytes=4 * 1024**3),
        ]
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend(nodes, 10_000_000_000, 32)
        # Should be either Uniform or Hybrid depending on node count
        assert isinstance(plan.strategy, str)
        assert len(plan.strategy) > 0

    # --- total memory saved ---

    def test_total_memory_saved_with_quant(self) -> None:
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32)
        assert plan.total_memory_saved_bytes > 0

    def test_total_memory_saved_without_quant(self) -> None:
        node = self._make_node(memory_bytes=80 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 1_000_000_000, 32)
        assert plan.total_memory_saved_bytes == 0

    def test_avg_quality_loss_non_negative(self) -> None:
        nodes = [
            self._make_node("n0", memory_bytes=4 * 1024**3),
            self._make_node("n1", memory_bytes=80 * 1024**3),
        ]
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend(nodes, 10_000_000_000, 32)
        assert plan.avg_quality_loss >= 0.0

    # --- activation quantization ---

    def test_activation_quant_low_bandwidth(self) -> None:
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32, inter_node_bandwidth_gbps=10)
        rec = plan.recommendations[0]
        assert rec.activation_quant is not ActivationQuantMethod.NONE

    def test_activation_quant_high_bandwidth(self) -> None:
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32, inter_node_bandwidth_gbps=200)
        rec = plan.recommendations[0]
        assert rec.activation_quant is ActivationQuantMethod.NONE

    def test_activation_quant_none_when_bandwidth_none(self) -> None:
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32, inter_node_bandwidth_gbps=None)
        rec = plan.recommendations[0]
        assert rec.activation_quant is ActivationQuantMethod.NONE

    def test_activation_quant_moderate_bandwidth_uses_fp8(self) -> None:
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32, inter_node_bandwidth_gbps=50)
        rec = plan.recommendations[0]
        # Moderate bandwidth: 25 <= 50 < 100 -> FP8_E4M3
        # (Only if quality loss permits, but FP8 has very low loss)
        assert rec.activation_quant in (ActivationQuantMethod.FP8_E4M3, ActivationQuantMethod.INT8)

    # --- KV cache recommendation ---

    def test_kv_cache_tight_vram(self) -> None:
        """Very tight VRAM -> aggressive KV cache compression."""
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=2 * 1024**3,
            compute_capability=8.0,
            num_layers_assigned=4,
        )
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32)
        rec = plan.recommendations[0]
        assert rec.kv_cache_bits in (KVCacheBits.INT4, KVCacheBits.INT8)

    def test_kv_cache_abundant_vram(self) -> None:
        """Abundant spare VRAM -> no KV cache compression."""
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
            num_layers_assigned=4,
        )
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 1_000_000_000, 32)
        rec = plan.recommendations[0]
        assert rec.kv_cache_bits is KVCacheBits.NONE

    # --- fallback hardware ---

    def test_fallback_on_unsupported_hardware(self) -> None:
        """CPU device with limited VRAM -> NONE fallback."""
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=2 * 1024**3,
            device_type="cpu",
            num_layers_assigned=4,
        )
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32)
        rec = plan.recommendations[0]
        assert rec.method is QuantMethod.NONE

    def test_fallback_on_unsupported_hardware_rocm(self) -> None:
        """ROCM device with very tight VRAM -> BNB_4BIT fallback."""
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=1 * 1024**3,
            device_type="rocm",
            compute_capability=8.0,
            num_layers_assigned=4,
        )
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32)
        rec = plan.recommendations[0]
        # ROCM -> fallback checks BNB_4BIT, but VRAM may be too low even for that
        assert rec.method is QuantMethod.BNB_4BIT or rec.method is QuantMethod.NONE

    # --- mixed precision plan in recommendation ---

    def test_mixed_precision_plan_in_rec(self) -> None:
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 32)
        rec = plan.recommendations[0]
        assert rec.mixed_precision_plan is not None
        assert rec.mixed_precision_plan.num_layers == 32
        assert len(rec.mixed_precision_plan.plans) == 32

    def test_mixed_precision_plan_uses_num_layers(self) -> None:
        node = self._make_node(memory_bytes=4 * 1024**3)
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 10_000_000_000, 8)
        rec = plan.recommendations[0]
        assert rec.mixed_precision_plan is not None
        assert rec.mixed_precision_plan.num_layers == 8
        assert len(rec.mixed_precision_plan.plans) == 8

    # --- legacy interface ---

    def test_recommend_legacy(self) -> None:
        nodes: list[dict] = [
            {
                "node_id": "n0",
                "total_memory_bytes": 4 * 1024**3,
                "num_layers_assigned": 4,
            }
        ]
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend_legacy(nodes, 10_000_000_000, 32)
        assert len(plan.recommendations) == 1

    # --- compute capability filtering ---

    def test_compute_capability_too_low_for_fp8(self) -> None:
        """CC 7.0 cannot use FP8 methods requiring CC >= 9.0."""
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=16 * 1024**3,
            compute_capability=7.0,
            num_layers_assigned=4,
        )
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([node], 20_000_000_000, 32)
        rec = plan.recommendations[0]
        assert rec.method not in (QuantMethod.FP8_E4M3, QuantMethod.FP8_E5M2)


# ===========================================================================
# select_for_node convenience function
# ===========================================================================


class TestSelectForNode:
    """select_for_node convenience function."""

    def test_select_no_quant(self) -> None:
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=80 * 1024**3,
            compute_capability=8.0,
        )
        method = select_for_node(node, 1_000_000_000)
        assert method is QuantMethod.NONE

    def test_select_with_quant(self) -> None:
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=4 * 1024**3,
            compute_capability=8.0,
        )
        method = select_for_node(node, 10_000_000_000)
        assert method is not QuantMethod.NONE

    def test_select_from_dict(self) -> None:
        node: dict = {
            "node_id": "n0",
            "total_memory_bytes": 4 * 1024**3,
            "compute_capability": 8.0,
        }
        method = select_for_node(node, 10_000_000_000)
        assert isinstance(method, QuantMethod)

    def test_select_strict_quality(self) -> None:
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=4 * 1024**3,
            compute_capability=8.0,
        )
        method = select_for_node(node, 10_000_000_000, max_quality_loss=0.001)
        assert isinstance(method, QuantMethod)

    def test_select_tiny_model(self) -> None:
        node = NodeInfo(
            node_id="n0",
            total_memory_bytes=4 * 1024**3,
            compute_capability=8.0,
        )
        method = select_for_node(node, 1_000_000)  # 1 MB model
        assert method is QuantMethod.NONE


# ===========================================================================
# LayerPrecisionProfile and LayerPrecisionResult (dataclasses used by
# profile_layer_precision, tested without GPU)
# ===========================================================================


class TestLayerPrecisionProfile:
    """LayerPrecisionProfile dataclass."""

    def test_create(self) -> None:
        prof = LayerPrecisionProfile(
            layer_idx=0,
            layer_name="model.layers.0",
            precision="fp16",
            runtime_ms=1.5,
            memory_bytes=1000000,
            peak_memory_bytes=2000000,
        )
        assert prof.layer_idx == 0
        assert prof.precision == "fp16"
        assert prof.runtime_ms == 1.5

    def test_zero_values(self) -> None:
        prof = LayerPrecisionProfile(
            layer_idx=0,
            layer_name="test",
            precision="int8",
            runtime_ms=0.0,
            memory_bytes=0,
            peak_memory_bytes=0,
        )
        assert prof.runtime_ms == 0.0
        assert prof.memory_bytes == 0


class TestLayerPrecisionResult:
    """LayerPrecisionResult dataclass."""

    def test_empty(self) -> None:
        result = LayerPrecisionResult(
            layer_idx=0,
            layer_name="test",
            layer_type="attention",
            profiles={},
        )
        assert result.recommended_precision == "fp16"
        assert result.fp16 is None
        assert result.fp8 is None
        assert result.int8 is None

    def test_profiles(self) -> None:
        fp16 = LayerPrecisionProfile(0, "test", "fp16", 1.0, 100, 200)
        int8 = LayerPrecisionProfile(0, "test", "int8", 2.0, 50, 100)
        result = LayerPrecisionResult(
            layer_idx=0,
            layer_name="test",
            layer_type="mlp",
            profiles={"fp16": fp16, "int8": int8},
        )
        assert result.fp16 is fp16
        assert result.int8 is int8
        assert result.fp8 is None

    def test_best_precision_default(self) -> None:
        fp16 = LayerPrecisionProfile(0, "test", "fp16", 1.0, 100, 200)
        int8 = LayerPrecisionProfile(0, "test", "int8", 2.0, 50, 100)
        result = LayerPrecisionResult(
            layer_idx=0,
            layer_name="test",
            layer_type="mlp",
            profiles={"fp16": fp16, "int8": int8},
        )
        # No constraints: lower memory wins -> int8
        assert result.best_precision() == "int8"

    def test_best_precision_with_memory_budget(self) -> None:
        fp16 = LayerPrecisionProfile(0, "test", "fp16", 1.0, 100, 200)
        int8 = LayerPrecisionProfile(0, "test", "int8", 2.0, 50, 100)
        result = LayerPrecisionResult(
            layer_idx=0,
            layer_name="test",
            layer_type="mlp",
            profiles={"fp16": fp16, "int8": int8},
        )
        # Budget 75 -> only int8 (50 <= 75) qualifies
        assert result.best_precision(memory_budget_bytes=75) == "int8"

    def test_best_precision_with_runtime_constraint(self) -> None:
        fp16 = LayerPrecisionProfile(0, "test", "fp16", 1.0, 100, 200)
        int8 = LayerPrecisionProfile(0, "test", "int8", 2.0, 50, 100)
        result = LayerPrecisionResult(
            layer_idx=0,
            layer_name="test",
            layer_type="mlp",
            profiles={"fp16": fp16, "int8": int8},
        )
        # Max runtime 1.5ms -> only fp16 (1.0 <= 1.5) qualifies
        assert result.best_precision(max_runtime_ms=1.5) == "fp16"

    def test_best_precision_all_disqualified(self) -> None:
        fp16 = LayerPrecisionProfile(0, "test", "fp16", 5.0, 100, 200)
        int8 = LayerPrecisionProfile(0, "test", "int8", 10.0, 50, 100)
        result = LayerPrecisionResult(
            layer_idx=0,
            layer_name="test",
            layer_type="mlp",
            profiles={"fp16": fp16, "int8": int8},
        )
        # Max runtime 1ms -> both disqualified -> fallback to "fp16"
        assert result.best_precision(max_runtime_ms=1.0) == "fp16"

    def test_best_precision_tiebreak_by_runtime(self) -> None:
        fp16 = LayerPrecisionProfile(0, "test", "fp16", 5.0, 50, 200)
        int8 = LayerPrecisionProfile(0, "test", "int8", 2.0, 50, 100)
        result = LayerPrecisionResult(
            layer_idx=0,
            layer_name="test",
            layer_type="mlp",
            profiles={"fp16": fp16, "int8": int8},
        )
        # Same memory (50) -> tiebreak by lower runtime -> int8 (2.0 < 5.0)
        assert result.best_precision() == "int8"


# ===========================================================================
# profile_layer_precision (GPU-dependent)
# ===========================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestProfileLayerPrecision:
    """profile_layer_precision function (requires torch)."""

    def test_requires_torch_module(self) -> None:
        """The function requires a torch.nn.Module as first argument."""
        with pytest.raises(AttributeError):
            profile_layer_precision(
                model="not_a_module",  # type: ignore[arg-type]
                layer_idx=0,
                layer_name="test",
                sample_input=None,  # type: ignore[arg-type]
            )


# ===========================================================================
# assign_mixed_precision (GPU-dependent)
# ===========================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestAssignMixedPrecision:
    """assign_mixed_precision function (requires torch)."""

    def test_requires_valid_module(self) -> None:
        """The function requires a torch.nn.Module with submodules."""
        model = torch.nn.Linear(10, 10)  # No submodule tree -> will fail
        sample = torch.randn(1, 10)
        with pytest.raises(AttributeError):
            assign_mixed_precision(model, num_layers=1, sample_input=sample)


# ===========================================================================
# AutoMixedPrecisionPipeline (non-GPU methods)
# ===========================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
class TestAutoMixedPrecisionPipeline:
    """AutoMixedPrecisionPipeline — tests without GPU dependency."""

    @staticmethod
    def _make_plan() -> MixedPrecisionPlan:
        plans_list = [
            LayerQuantPlan(0, "attention", "float16", "float16"),
            LayerQuantPlan(1, "mlp", "int8", "float16"),
            LayerQuantPlan(2, "attention", "float16", "float16"),
        ]
        return MixedPrecisionPlan(plans=plans_list, overall_compression_ratio=1.33, num_layers=3)

    def test_parse_dtype(self) -> None:
        assert AutoMixedPrecisionPipeline._parse_dtype("float16") == torch.float16
        assert AutoMixedPrecisionPipeline._parse_dtype("float32") == torch.float32
        assert AutoMixedPrecisionPipeline._parse_dtype("bfloat16") == torch.bfloat16
        assert AutoMixedPrecisionPipeline._parse_dtype("int8") == torch.int8
        assert AutoMixedPrecisionPipeline._parse_dtype("nf4") == torch.float16

    def test_parse_dtype_unknown_falls_back(self) -> None:
        assert AutoMixedPrecisionPipeline._parse_dtype("unknown") == torch.float16

    def test_parse_dtype_empty_string(self) -> None:
        assert AutoMixedPrecisionPipeline._parse_dtype("") == torch.float16

    def test_init_and_dtype_map(self) -> None:
        plan = self._make_plan()
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator=None,  # type: ignore[arg-type]
            precision_plan=plan,
            model=None,
            device="cpu",
        )
        assert pipeline.get_dtype_for_layer(0) == torch.float16
        assert pipeline.get_dtype_for_layer(1) == torch.int8
        assert pipeline.get_dtype_for_layer(2) == torch.float16

    def test_get_dtype_for_layer_unknown(self) -> None:
        plan = self._make_plan()
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator=None,  # type: ignore[arg-type]
            precision_plan=plan,
            device="cpu",
        )
        # Layer index 99 not in plan -> falls back to float16
        assert pipeline.get_dtype_for_layer(99) == torch.float16

    def test_cast_to_layer_precision(self) -> None:
        plan = self._make_plan()
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator=None,  # type: ignore[arg-type]
            precision_plan=plan,
            device="cpu",
        )
        tensor = torch.randn(1, 10, dtype=torch.float32)
        cast = pipeline.cast_to_layer_precision(tensor, layer_idx=1)  # int8 layer
        assert cast.dtype == torch.int8

    def test_cast_to_layer_precision_noop_when_same_dtype(self) -> None:
        plan = self._make_plan()
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator=None,  # type: ignore[arg-type]
            precision_plan=plan,
            device="cpu",
        )
        tensor = torch.randn(1, 10, dtype=torch.float16)
        cast = pipeline.cast_to_layer_precision(tensor, layer_idx=0)  # float16 layer
        assert cast.dtype == torch.float16
        assert cast is tensor  # Should be the same object

    def test_summary(self) -> None:
        plan = self._make_plan()
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator=None,  # type: ignore[arg-type]
            precision_plan=plan,
            device="cpu",
        )
        s = pipeline.summary()
        assert "3 layers" in s
        assert "L  0" in s
        assert "L  1" in s
        assert "L  2" in s
        assert "1.3x" in s

    def test_orchestrator_property(self) -> None:
        plan = self._make_plan()
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator="mock_orch",  # type: ignore[arg-type]
            precision_plan=plan,
            device="cpu",
        )
        assert pipeline.orchestrator == "mock_orch"

    def test_precision_plan_property(self) -> None:
        plan = self._make_plan()
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator=None,  # type: ignore[arg-type]
            precision_plan=plan,
            device="cpu",
        )
        assert pipeline.precision_plan is plan

    def test_empty_plan(self) -> None:
        plan = MixedPrecisionPlan()
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator=None,  # type: ignore[arg-type]
            precision_plan=plan,
            device="cpu",
        )
        assert pipeline.get_dtype_for_layer(0) == torch.float16

    def test_apply_to_model_weights(self) -> None:
        """Apply precision plan to a simple model."""
        # Use a float-only plan to avoid RuntimeError with int params
        plan = MixedPrecisionPlan(
            plans=[
                LayerQuantPlan(0, "attention", "float16", "float16"),
                LayerQuantPlan(1, "mlp", "float16", "float16"),
                LayerQuantPlan(2, "attention", "float16", "float16"),
            ],
            overall_compression_ratio=1.0,
            num_layers=3,
        )
        pipeline = AutoMixedPrecisionPipeline(
            orchestrator=None,  # type: ignore[arg-type]
            precision_plan=plan,
            device="cpu",
        )

        # Create a model with 3 layer submodules
        class TestLayer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(10, 10))

        class TestModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.Module()
                for i in range(3):
                    self.model.layers.add_module(str(i), TestLayer())

        model = TestModel()
        # The source walks model.layers.N via getattr, so the test model
        # must have model.layers.0, model.layers.1, etc. as submodules
        # reachable via chained getattr calls.
        result = pipeline.apply_to_model_weights(model)  # type: ignore[arg-type]
        assert result is model
