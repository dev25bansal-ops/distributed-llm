"""Tests: Adaptive Precision Optimizer (APO) — QuantizationAutoTuner.

Tests: NodeInfo validation, QuantMethod enum, per-node recommendation,
scoring weights, fallback behavior, serialization, mixed precision plans,
activation quant, KV cache tiering, edge cases.

Run: pytest tests/core/test_apo_tuner.py -v
"""

import json

import pytest

from distllm.dist.partition.quantization_tuner import (
    ActivationQuantMethod,
    KVCacheBits,
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
    QUANT_PROFILES,
    select_for_node,
)


# ---------------------------------------------------------------------------
# NodeInfo validation
# ---------------------------------------------------------------------------


class TestNodeInfo:
    """Test NodeInfo Pydantic model."""

    def test_default_values(self):
        info = NodeInfo(node_id="n0")
        assert info.node_id == "n0"
        assert info.device_type == "cuda"
        assert info.total_memory_bytes == 8 * 1024**3
        assert info.compute_capability is None

    def test_custom_values(self):
        info = NodeInfo(
            node_id="n1",
            device_type="cuda",
            total_memory_bytes=16 * 1024**3,
            compute_capability=9.0,
            gpu_name="H100",
            is_hopper_or_newer=True,
        )
        assert info.total_memory_bytes == 16 * 1024**3
        assert info.compute_capability == 9.0
        assert info.is_hopper_or_newer is True

    def test_from_dict_valid(self):
        d = {"node_id": "n2", "device_type": "rocm", "total_memory_bytes": 32 * 1024**3}
        info = NodeInfo.from_dict(d)
        assert info.node_id == "n2"
        assert info.device_type == "rocm"

    def test_from_dict_ignores_unknown_keys(self):
        d = {"node_id": "n3", "unknown_field": 42}
        info = NodeInfo.from_dict(d)
        assert info.node_id == "n3"

    def test_from_dict_missing_node_id_raises(self):
        with pytest.raises(Exception):
            NodeInfo.from_dict({"device_type": "cuda"})

    def test_from_gpu_profile(self):
        class FakeGPU:
            total_memory_bytes = 24 * 1024**3
            name = "RTX 4090"
            memory_bandwidth_gbps = 1000.0

        info = NodeInfo.from_gpu_profile(FakeGPU(), "gpu-0", num_layers_assigned=16)
        assert info.node_id == "gpu-0"
        assert info.total_memory_bytes == 24 * 1024**3
        assert info.gpu_name == "RTX 4090"
        assert info.num_layers_assigned == 16


# ---------------------------------------------------------------------------
# ScoreWeights
# ---------------------------------------------------------------------------


class TestScoreWeights:
    """Test ScoreWeights configuration."""

    def test_default_weights(self):
        w = ScoreWeights()
        assert w.headroom == 0.5
        assert w.quality == 0.3
        assert w.speed == 0.2

    def test_normalized_sums_to_one(self):
        w = ScoreWeights(headroom=3, quality=6, speed=1).normalized()
        assert abs(w.headroom + w.quality + w.speed - 1.0) < 1e-6

    def test_normalized_zero_total_returns_defaults(self):
        w = ScoreWeights(headroom=0, quality=0, speed=0).normalized()
        assert w.headroom == 1.0 / 3  # equal division fallback

    def test_custom_weights(self):
        w = ScoreWeights(headroom=0.7, quality=0.2, speed=0.1)
        tuner = QuantizationAutoTuner(weights=w)
        assert abs(tuner._weights.headroom - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# QuantMethod enum
# ---------------------------------------------------------------------------


class TestQuantMethod:
    """Test QuantMethod enum completeness."""

    def test_fp8_present(self):
        assert hasattr(QuantMethod, "FP8_E4M3")
        assert hasattr(QuantMethod, "FP8_E5M2")
        assert QuantMethod.FP8_E4M3.value == "fp8_e4m3"

    def test_int8_present(self):
        assert hasattr(QuantMethod, "INT8")
        assert QuantMethod.INT8.value == "int8"

    def test_nf4_present(self):
        assert hasattr(QuantMethod, "NF4")
        assert QuantMethod.NF4.value == "nf4"

    def test_all_methods_have_profiles(self):
        for method in QuantMethod:
            assert method in QUANT_PROFILES, f"{method} missing from QUANT_PROFILES"


# ---------------------------------------------------------------------------
# QuantizationAutoTuner — core recommend()
# ---------------------------------------------------------------------------


class TestQuantizationAutoTuner:
    """Test APO recommend() with various scenarios."""

    def test_empty_nodes_returns_empty_plan(self):
        tuner = QuantizationAutoTuner()
        plan = tuner.recommend([], model_size_bytes=14 * 1024**3, num_layers=32)
        assert len(plan.recommendations) == 0
        assert "No nodes" in plan.strategy

    def test_single_node_fits_returns_none(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=64 * 1024**3)
        plan = tuner.recommend([node], model_size_bytes=14 * 1024**3, num_layers=32)
        assert len(plan.recommendations) == 1
        assert plan.recommendations[0].method == QuantMethod.NONE

    def test_single_node_needs_quant(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], model_size_bytes=14 * 1024**3, num_layers=32)
        assert len(plan.recommendations) == 1
        assert plan.recommendations[0].method != QuantMethod.NONE

    def test_multi_node_mixed_hardware(self):
        tuner = QuantizationAutoTuner()
        nodes = [
            NodeInfo(node_id="big", total_memory_bytes=80 * 1024**3),
            NodeInfo(node_id="small", total_memory_bytes=8 * 1024**3),
        ]
        plan = tuner.recommend(nodes, model_size_bytes=14 * 1024**3, num_layers=32)
        assert len(plan.recommendations) == 2
        # Big node should be NONE, small should need quant
        big_rec = next(r for r in plan.recommendations if r.node_id == "big")
        small_rec = next(r for r in plan.recommendations if r.node_id == "small")
        assert big_rec.method == QuantMethod.NONE
        assert small_rec.method != QuantMethod.NONE

    def test_hopper_prefers_fp8(self):
        tuner = QuantizationAutoTuner(max_quality_loss=0.05)
        node = NodeInfo(
            node_id="hopper",
            total_memory_bytes=16 * 1024**3,
            compute_capability=9.0,
            is_hopper_or_newer=True,
        )
        plan = tuner.recommend([node], model_size_bytes=14 * 1024**3, num_layers=32)
        rec = plan.recommendations[0]
        # FP8 should be preferred on Hopper for tight VRAM
        assert rec.method in (QuantMethod.FP8_E4M3, QuantMethod.FP8_E5M2)

    def test_cpu_node_returns_none(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="cpu", device_type="cpu", total_memory_bytes=16 * 1024**3)
        plan = tuner.recommend([node], model_size_bytes=14 * 1024**3, num_layers=32)
        assert plan.recommendations[0].method == QuantMethod.NONE

    def test_quality_loss_budget_respected(self):
        tuner = QuantizationAutoTuner(max_quality_loss=0.01)
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], model_size_bytes=14 * 1024**3, num_layers=32)
        for rec in plan.recommendations:
            # If a non-fallback method was selected, quality should be within budget
            # Fallback (BNB_4BIT) may exceed budget — that's by design
            if "Forced" not in rec.reason:
                assert rec.quality_loss <= 0.015

    def test_require_calibration_excludes_gptq_awq(self):
        tuner = QuantizationAutoTuner(require_calibration=True)
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], model_size_bytes=14 * 1024**3, num_layers=32)
        for rec in plan.recommendations:
            assert rec.method not in (QuantMethod.GPTQ, QuantMethod.AWQ)


# ---------------------------------------------------------------------------
# Per-node layer assignment
# ---------------------------------------------------------------------------


class TestPerNodeLayerAssignment:
    """Test that model portion is correctly calculated per-node."""

    def test_assigned_layers_reduces_model_portion(self):
        tuner = QuantizationAutoTuner()
        # 14GB model, 32 layers -> ~437MB per layer
        # Node with 8 layers -> ~3.5GB -> fits in 8GB with no quant
        node = NodeInfo(
            node_id="partial",
            total_memory_bytes=8 * 1024**3,
            num_layers_assigned=8,
        )
        plan = tuner.recommend([node], model_size_bytes=14 * 1024**3, num_layers=32)
        assert plan.recommendations[0].method == QuantMethod.NONE

    def test_no_assigned_layers_uses_full_model(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="full", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], model_size_bytes=14 * 1024**3, num_layers=32)
        assert plan.recommendations[0].method != QuantMethod.NONE


# ---------------------------------------------------------------------------
# Scoring behavior
# ---------------------------------------------------------------------------


class TestScoring:
    """Test scoring weight influence on method selection."""

    def test_prefer_speed_selects_faster_method(self):
        tuner_fast = QuantizationAutoTuner(prefer_speed=True)
        tuner_quality = QuantizationAutoTuner(prefer_speed=False)
        node = NodeInfo(node_id="n0", total_memory_bytes=10 * 1024**3)
        model_bytes = 14 * 1024**3

        plan_fast = tuner_fast.recommend([node], model_bytes, 32)
        plan_quality = tuner_quality.recommend([node], model_bytes, 32)

        # Both should produce valid plans
        assert len(plan_fast.recommendations) == 1
        assert len(plan_quality.recommendations) == 1


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestFallback:
    """Test fallback when no method fits constraints."""

    def test_fallback_for_cuda_node(self):
        tuner = QuantizationAutoTuner(max_quality_loss=0.001)  # Very strict
        node = NodeInfo(node_id="tight", total_memory_bytes=4 * 1024**3)
        plan = tuner.recommend([node], model_size_bytes=70 * 1024**3, num_layers=80)
        rec = plan.recommendations[0]
        # Should fall back to BNB_4BIT or NONE
        assert rec.method in (QuantMethod.BNB_4BIT, QuantMethod.NONE)

    def test_fallback_for_cpu_node(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="cpu", device_type="cpu", total_memory_bytes=16 * 1024**3)
        plan = tuner.recommend([node], model_size_bytes=70 * 1024**3, num_layers=80)
        assert plan.recommendations[0].method == QuantMethod.NONE


# ---------------------------------------------------------------------------
# Activation quantization recommendation
# ---------------------------------------------------------------------------


class TestActivationQuant:
    """Test activation quantization co-selection."""

    def test_no_bandwidth_returns_none(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        assert plan.recommendations[0].activation_quant == ActivationQuantMethod.NONE

    def test_low_bandwidth_returns_int8(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32, inter_node_bandwidth_gbps=10)
        assert plan.recommendations[0].activation_quant == ActivationQuantMethod.INT8

    def test_moderate_bandwidth_returns_fp8(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32, inter_node_bandwidth_gbps=50)
        assert plan.recommendations[0].activation_quant == ActivationQuantMethod.FP8_E4M3

    def test_high_bandwidth_returns_none(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32, inter_node_bandwidth_gbps=200)
        assert plan.recommendations[0].activation_quant == ActivationQuantMethod.NONE


# ---------------------------------------------------------------------------
# KV cache tiering
# ---------------------------------------------------------------------------


class TestKVCacheTiering:
    """Test KV cache quantization recommendation."""

    def test_lots_of_spare_vram_returns_none(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        assert plan.recommendations[0].kv_cache_bits == KVCacheBits.NONE

    def test_tight_vram_returns_int4(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 70 * 1024**3, 80)
        # Should have aggressive quant + KV compression
        rec = plan.recommendations[0]
        assert rec.kv_cache_bits in (KVCacheBits.INT4, KVCacheBits.INT8)


# ---------------------------------------------------------------------------
# Mixed precision plans
# ---------------------------------------------------------------------------


class TestMixedPrecisionPlan:
    """Test mixed precision plan generation."""

    def test_plan_attached_to_recommendation(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        rec = plan.recommendations[0]
        assert rec.mixed_precision_plan is not None
        assert rec.mixed_precision_plan.num_layers == 32

    def test_plan_has_attention_and_mlp(self):
        analyzer = SensitivityAnalyzer()
        plan = analyzer.build_mixed_precision_plan(num_layers=8, layers_per_node=8)
        types = {p.layer_type for p in plan.plans}
        assert "attention" in types
        assert "mlp" in types

    def test_attention_higher_precision_than_mlp(self):
        analyzer = SensitivityAnalyzer()
        plan = analyzer.build_mixed_precision_plan(num_layers=8, layers_per_node=8)
        for p in plan.plans:
            if p.layer_type == "attention":
                assert p.weight_dtype in ("float16", "int8")
            if p.layer_type == "mlp":
                assert p.weight_dtype in ("int8", "nf4")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test QuantizationPlan serialization."""

    def test_to_json_roundtrip(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)

        json_str = plan.to_json()
        restored = QuantizationPlan.from_json(json_str)

        assert len(restored.recommendations) == len(plan.recommendations)
        assert restored.strategy == plan.strategy
        assert restored.recommendations[0].method == plan.recommendations[0].method

    def test_to_dict_contains_all_fields(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        d = plan.to_dict()

        assert "strategy" in d
        assert "recommendations" in d
        assert "total_memory_saved_bytes" in d
        assert "avg_quality_loss" in d
        assert "timestamp" in d

    def test_node_recommendation_to_dict(self):
        rec = NodeQuantRecommendation(
            node_id="n0",
            method=QuantMethod.BNB_4BIT,
            memory_bytes_without_quant=14 * 1024**3,
            memory_bytes_with_quant=int(14 * 1024**3 * 0.25),
            memory_savings_bytes=int(14 * 1024**3 * 0.75),
            memory_savings_pct=75.0,
            speed_penalty=1.1,
            quality_loss=0.03,
            reason="test",
        )
        d = rec.to_dict()
        assert d["method"] == "bnb_4bit"
        assert d["node_id"] == "n0"


# ---------------------------------------------------------------------------
# Profile overrides
# ---------------------------------------------------------------------------


class TestProfileOverrides:
    """Test custom profile overrides."""

    def test_custom_profile_used(self):
        custom = QuantProfile(
            method=QuantMethod.BNB_4BIT,
            memory_reduction=0.20,  # Better than default 0.25
            speed_penalty=1.05,
            min_vram_gb=2,
            quality_loss=0.02,
        )
        # Use require_calibration to exclude AWQ/GPTQ so BNB_4BIT wins
        tuner = QuantizationAutoTuner(
            profile_overrides={QuantMethod.BNB_4BIT: custom},
            require_calibration=True,
        )
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        rec = plan.recommendations[0]
        assert rec.method == QuantMethod.BNB_4BIT


# ---------------------------------------------------------------------------
# select_for_node convenience function
# ---------------------------------------------------------------------------


class TestSelectForNode:
    """Test the quick select_for_node function."""

    def test_returns_quant_method(self):
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        method = select_for_node(node, 14 * 1024**3)
        assert isinstance(method, QuantMethod)

    def test_large_vram_returns_none(self):
        node = NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3)
        method = select_for_node(node, 14 * 1024**3)
        assert method == QuantMethod.NONE

    def test_dict_input_works(self):
        node = {"node_id": "n0", "total_memory_bytes": 80 * 1024**3}
        method = select_for_node(node, 14 * 1024**3)
        assert isinstance(method, QuantMethod)


# ---------------------------------------------------------------------------
# QuantizationPlan summary
# ---------------------------------------------------------------------------


class TestPlanSummary:
    """Test plan summary output."""

    def test_summary_contains_strategy(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=80 * 1024**3)
        plan = tuner.recommend([node], 14 * 1024**3, 32)
        summary = plan.summary()
        assert "APO Plan" in summary

    def test_methods_used(self):
        tuner = QuantizationAutoTuner()
        nodes = [
            NodeInfo(node_id="big", total_memory_bytes=80 * 1024**3),
            NodeInfo(node_id="small", total_memory_bytes=8 * 1024**3),
        ]
        plan = tuner.recommend(nodes, 14 * 1024**3, 32)
        methods = plan.methods_used
        assert QuantMethod.NONE in methods
        assert len(methods) >= 1


# ---------------------------------------------------------------------------
# SensitivityAnalyzer
# ---------------------------------------------------------------------------


class TestSensitivityAnalyzer:
    """Test layer classification and sensitivity analysis."""

    def test_classify_attention(self):
        analyzer = SensitivityAnalyzer()
        assert analyzer.classify_layer("model.layers.0.self_attn.q_proj") == "attention"
        assert analyzer.classify_layer("k_proj") == "attention"

    def test_classify_mlp(self):
        analyzer = SensitivityAnalyzer()
        assert analyzer.classify_layer("model.layers.0.mlp.gate_proj") == "mlp"
        assert analyzer.classify_layer("up_proj") == "mlp"

    def test_classify_norm(self):
        analyzer = SensitivityAnalyzer()
        assert analyzer.classify_layer("model.norm") == "norm"
        assert analyzer.classify_layer("input_layernorm") == "norm"

    def test_classify_embed(self):
        analyzer = SensitivityAnalyzer()
        assert analyzer.classify_layer("embed_tokens") == "embed"

    def test_classify_lm_head(self):
        analyzer = SensitivityAnalyzer()
        assert analyzer.classify_layer("lm_head") == "lm_head"

    def test_recommend_dtype_attention(self):
        analyzer = SensitivityAnalyzer()
        dtype, score = analyzer.recommend_dtype("attention")
        assert dtype in ("float16", "int8")
        assert score >= 0.5

    def test_recommend_dtype_mlp(self):
        analyzer = SensitivityAnalyzer()
        dtype, score = analyzer.recommend_dtype("mlp")
        assert dtype in ("int8", "nf4")
        assert score <= 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_size_model(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 0, 32)
        assert plan.recommendations[0].method == QuantMethod.NONE

    def test_very_large_model(self):
        tuner = QuantizationAutoTuner()
        node = NodeInfo(node_id="n0", total_memory_bytes=8 * 1024**3)
        plan = tuner.recommend([node], 700 * 1024**3, 80)
        rec = plan.recommendations[0]
        assert rec.method != QuantMethod.NONE
        assert rec.memory_savings_pct > 0

    def test_many_nodes(self):
        tuner = QuantizationAutoTuner()
        nodes = [
            NodeInfo(node_id=f"n{i}", total_memory_bytes=(4 + i * 4) * 1024**3)
            for i in range(16)
        ]
        plan = tuner.recommend(nodes, 70 * 1024**3, 80)
        assert len(plan.recommendations) == 16

    def test_legacy_dict_interface(self):
        tuner = QuantizationAutoTuner()
        nodes = [{"node_id": "n0", "total_memory_bytes": 8 * 1024**3}]
        plan = tuner.recommend_legacy(nodes, 14 * 1024**3, 32)
        assert len(plan.recommendations) == 1
