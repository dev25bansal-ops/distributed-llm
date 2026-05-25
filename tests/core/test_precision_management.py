"""Tests: FP8 engine, adaptive precision, quantization selector, precision boundaries.

Tests: quantize_tensor roundtrip, FP8Linear matmul, FP8ModelPatcher, FP8Engine,
SensitivityAnalyzer, AdaptivePrecisionEngine, MixedPrecisionPlan,
QuantizationAutoTuner, PrecisionBoundary conversion.

Run: pytest tests/core/test_precision_management.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from distllm.core.fp8_engine import (
    FP8_AVAILABLE,
    quantize_tensor,
    dequantize_tensor,
    quantize_kv_fp8,
    dequantize_kv_fp8,
    FP8Tensor,
    FP8Linear,
    FP8Attention,
    FP8ModelPatcher,
    FP8Engine,
    FP8Scheme,
)
from distllm.core.adaptive_precision import (
    LayerPrecision,
    PrecisionPlan,
    SensitivityAnalyzer,
    AdaptivePrecisionEngine,
)
from distllm.core.quantization_selector import (
    NodeVRAMInfo,
    select_for_node,
    LayerQuantPlan,
    MixedPrecisionPlan,
    QuantizationAutoTuner,
    SimulatedInt8Linear,
    SimulatedNF4Linear,
)
from distllm.core.precision_boundary import PrecisionBoundary


# ===========================================================================
# 1. FP8 Engine Tests
# ===========================================================================


class TestFP8TensorQuantization:
    """Tests for FP8 tensor quantize/dequantize."""

    def test_quantize_tensor_returns_fp8tensor(self):
        t = torch.randn(4, 16)
        qt = quantize_tensor(t, FP8Scheme.E4M3)
        assert isinstance(qt, FP8Tensor)
        assert qt.data.shape == t.shape
        assert qt.scale.numel() == 1
        assert qt.scale > 0

    def test_dequantize_roundtrip_shape(self):
        t = torch.randn(4, 16)
        qt = quantize_tensor(t, FP8Scheme.E4M3)
        restored = dequantize_tensor(qt)
        assert restored.shape == t.shape

    def test_dequantize_roundtrip_close(self):
        t = torch.randn(4, 16) * 0.1
        qt = quantize_tensor(t, FP8Scheme.E4M3)
        restored = dequantize_tensor(qt)
        err = (t - restored).abs().mean().item()
        assert err < 0.5

    def test_quantize_tensor_preserves_dtype(self):
        t = torch.randn(4, 16).double()
        qt = quantize_tensor(t, FP8Scheme.E4M3)
        assert qt.original_dtype == torch.double

    def test_quantize_zero_tensor(self):
        t = torch.zeros(4, 16)
        qt = quantize_tensor(t, FP8Scheme.E4M3)
        restored = dequantize_tensor(qt)
        assert restored.abs().sum().item() == 0.0

    def test_quantize_kv_fp8_roundtrip(self):
        t = torch.randn(8, 16, 64) * 0.1
        fp8_data, scale = quantize_kv_fp8(t, FP8Scheme.E4M3)
        restored = dequantize_kv_fp8(fp8_data, scale)
        assert restored.shape == t.shape
        err = (t - restored).abs().mean().item()
        assert err < 0.5

    def test_e5m2_scheme(self):
        t = torch.randn(4, 16) * 0.1
        qt = quantize_tensor(t, FP8Scheme.E5M2)
        restored = dequantize_tensor(qt)
        assert restored.shape == t.shape

    def test_small_tensor_preserves_sign(self):
        t = torch.tensor([[0.5, -0.3, 0.0]], dtype=torch.float32)
        qt = quantize_tensor(t, FP8Scheme.E4M3)
        restored = dequantize_tensor(qt)
        assert restored[0, 0].item() > 0
        assert restored[0, 1].item() < 0


class TestFP8Linear:
    """Tests for FP8Linear layer."""

    def test_init_defaults(self):
        layer = FP8Linear(32, 16)
        assert layer.in_features == 32
        assert layer.out_features == 16
        assert layer.weight.shape == (16, 32)
        assert layer.bias is not None

    def test_no_bias(self):
        layer = FP8Linear(32, 16, bias=False)
        assert layer.bias is None

    def test_fp16_forward(self):
        layer = FP8Linear(32, 16)
        x = torch.randn(4, 32, dtype=torch.float16)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_to_fp8_and_back(self):
        layer = FP8Linear(32, 16)
        layer.to_fp8()
        assert layer._fp8_enabled is True or FP8_AVAILABLE is False
        layer.to_fp16()
        assert layer._fp8_enabled is False

    def test_fp8_via_fallback_path(self):
        layer = FP8Linear(32, 16, quantize_activations=False)
        layer._has_scaled_mm = False
        layer.to_fp8()
        x = torch.randn(4, 32, dtype=torch.float16)
        out = layer(x)
        assert out.shape == (4, 16)

    def test_fp16_and_fp8_outputs_not_degenerate(self):
        layer = FP8Linear(32, 16)
        nn.init.normal_(layer.weight, std=0.02)
        x = torch.randn(4, 32, dtype=torch.float16)
        out_fp16 = layer(x)
        layer.to_fp8()
        layer._has_scaled_mm = False
        out_fp8 = layer(x)
        assert out_fp16.abs().mean().item() > 0
        assert out_fp8.abs().mean().item() > 0


class TestFP8ModelPatcher:
    """Tests for FP8ModelPatcher."""

    def test_patch_simple_model(self):
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        patcher = FP8ModelPatcher()
        patcher.patch_model(model)
        assert isinstance(model[0], FP8Linear)
        assert isinstance(model[2], FP8Linear)
        assert patcher.patched_count == 2

    def test_patched_model_forward(self):
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.Linear(16, 8),
        )
        patcher = FP8ModelPatcher()
        patcher.patch_model(model)
        x = torch.randn(2, 32)
        out = model(x)
        assert out.shape == (2, 8)

    def test_unpatch_restores_linear(self):
        model = nn.Sequential(nn.Linear(32, 16))
        patcher = FP8ModelPatcher()
        patcher.patch_model(model)
        patcher.unpatch_model(model)
        assert isinstance(model[0], nn.Linear)
        assert patcher.patched_count == 0

    def test_patch_nested_model(self):
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 16)
                self.block = nn.Sequential(
                    nn.Linear(16, 16),
                    nn.Linear(16, 8),
                )

        model = NestedModel()
        patcher = FP8ModelPatcher()
        patcher.patch_model(model)
        assert isinstance(model.fc1, FP8Linear)
        assert isinstance(model.block[0], FP8Linear)
        assert isinstance(model.block[1], FP8Linear)

    def test_non_linear_layers_unchanged(self):
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        patcher = FP8ModelPatcher()
        patcher.patch_model(model)
        assert isinstance(model[0], FP8Linear)
        assert isinstance(model[1], nn.LayerNorm)

    def test_patched_weights_preserved(self):
        model = nn.Sequential(nn.Linear(32, 16))
        orig_weight = model[0].weight.data.clone()
        patcher = FP8ModelPatcher()
        patcher.patch_model(model)
        restored = model[0].weight.data
        assert torch.allclose(orig_weight, restored, atol=0.5)


class TestFP8Engine:
    """Tests for FP8Engine."""

    def test_init_defaults(self):
        engine = FP8Engine()
        assert engine.scheme == FP8Scheme.E4M3
        assert engine.quantize_activations is True
        assert engine.kv_cache_fp8 is True
        assert engine._is_prepared is False

    def test_prepare_simple_model(self):
        model = nn.Sequential(nn.Linear(32, 16))
        engine = FP8Engine()
        prepared = engine.prepare_model(model)
        assert isinstance(prepared[0], FP8Linear)
        assert engine._is_prepared is True

    def test_revert_model(self):
        model = nn.Sequential(nn.Linear(32, 16))
        engine = FP8Engine()
        engine.prepare_model(model)
        engine.revert_model(model)
        assert isinstance(model[0], nn.Linear)
        assert engine._is_prepared is False

    def test_estimate_savings(self):
        model = nn.Sequential(nn.Linear(32, 16))
        engine = FP8Engine()
        engine.prepare_model(model)
        savings = engine.estimate_savings(model)
        assert savings["total_params"] > 0
        assert savings["savings_pct"] > 0
        assert savings["patched_layers"] >= 1


# ===========================================================================
# 2. Adaptive Precision Tests
# ===========================================================================


class TestLayerPrecision:
    """Tests for LayerPrecision and PrecisionPlan dataclasses."""

    def test_layer_precision_defaults(self):
        lp = LayerPrecision(layer_name="test", layer_type="mlp")
        assert lp.layer_name == "test"
        assert lp.layer_type == "mlp"
        assert lp.sensitivity_score == 0.0
        assert lp.supports_int8 is False

    def test_precision_plan_defaults(self):
        plan = PrecisionPlan()
        assert plan.layer_precisions == []
        assert plan.total_memory_original_mb == 0.0

    def test_precision_plan_with_layers(self):
        lp = LayerPrecision(layer_name="a", layer_type="mlp")
        plan = PrecisionPlan(layer_precisions=[lp], total_memory_original_mb=100.0)
        assert len(plan.layer_precisions) == 1
        assert plan.total_memory_original_mb == 100.0


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer."""

    def test_classify_attention_layer(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.Linear(32, 32)
        result = analyzer._classify_layer("model.layers.0.self_attn.q_proj", mod)
        assert result == "attention"

    def test_classify_mlp_layer(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.Linear(32, 32)
        result = analyzer._classify_layer("model.layers.0.mlp.gate_proj", mod)
        assert result == "mlp"

    def test_classify_norm_layer(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.LayerNorm(32)
        result = analyzer._classify_layer("model.layers.0.norm", mod)
        assert result == "norm"

    def test_classify_embedding(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.Embedding(100, 32)
        result = analyzer._classify_layer("embed_tokens", mod)
        assert result == "embed"

    def test_classify_lm_head(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.Linear(32, 100)
        result = analyzer._classify_layer("lm_head", mod)
        assert result == "lm_head"

    def test_classify_fallback_to_mlp(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.Linear(32, 32)
        result = analyzer._classify_layer("custom_layer", mod)
        assert result == "mlp"

    def test_analyze_layer_norm_returns_fp32(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.LayerNorm(32)
        inp = torch.randn(2, 32)
        out = mod(inp)
        lp = analyzer.analyze_layer("norm", mod, inp, out)
        assert lp.recommended_dtype == torch.float32
        assert lp.sensitivity_score >= 0.5

    def test_analyze_layer_mlp_returns_fp16_or_int8(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.Linear(32, 32)
        nn.init.normal_(mod.weight, std=0.02)
        inp = torch.randn(4, 32)
        out = mod(inp)
        lp = analyzer.analyze_layer("mlp.fc1", mod, inp, out)
        assert lp.recommended_dtype in (torch.float16, torch.int8)

    def test_sensitivity_score_range(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.Linear(32, 32)
        inp = torch.randn(4, 32)
        out = mod(inp)
        lp = analyzer.analyze_layer("layer", mod, inp, out)
        assert 0.0 <= lp.sensitivity_score <= 1.0

    def test_high_outlier_ratio_increases_sensitivity(self):
        analyzer = SensitivityAnalyzer()
        mod = nn.Linear(32, 32)
        with torch.no_grad():
            mod.weight[0] = 100.0
        inp = torch.randn(4, 32)
        out = mod(inp)
        lp = analyzer.analyze_layer("layer", mod, inp, out)
        assert lp.sensitivity_score > 0.4


class TestAdaptivePrecisionEngine:
    """Tests for AdaptivePrecisionEngine."""

    def test_profile_model_returns_plan(self):
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.Linear(16, 8),
        )
        engine = AdaptivePrecisionEngine(calibration_samples=4)
        x = torch.randn(2, 32)
        plan = engine.profile_model(model, x)
        assert isinstance(plan, PrecisionPlan)
        assert len(plan.layer_precisions) > 0

    def test_profile_model_no_input(self):
        model = nn.Sequential(nn.Linear(32, 16))
        engine = AdaptivePrecisionEngine()
        plan = engine.profile_model(model)
        assert isinstance(plan, PrecisionPlan)
        assert len(plan.layer_precisions) >= 1

    def test_apply_precision_converts_layers(self):
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        engine = AdaptivePrecisionEngine()
        x = torch.randn(2, 32)
        engine.profile_model(model, x)
        converted = engine.apply_precision(model)
        assert converted >= 0

    def test_apply_without_profile_returns_zero(self):
        model = nn.Sequential(nn.Linear(32, 16))
        engine = AdaptivePrecisionEngine()
        converted = engine.apply_precision(model)
        assert converted == 0

    def test_plan_stored_after_profile(self):
        model = nn.Sequential(nn.Linear(32, 16))
        engine = AdaptivePrecisionEngine()
        assert engine.plan is None
        engine.profile_model(model, torch.randn(2, 32))
        assert engine.plan is not None

    def test_report_contains_plan_info(self):
        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.Linear(16, 8),
        )
        engine = AdaptivePrecisionEngine()
        engine.profile_model(model, torch.randn(2, 32))
        report = engine.report()
        assert "Adaptive Precision Report" in report
        assert "Original:" in report
        assert "Optimized:" in report


# ===========================================================================
# 3. Quantization Selector — Advanced Tests
# ===========================================================================


class TestMixedPrecisionPlan:
    """Tests for MixedPrecisionPlan and LayerQuantPlan dataclasses."""

    def test_layer_quant_plan_defaults(self):
        plan = LayerQuantPlan(layer_idx=0, layer_type="attention",
                               weight_dtype="float16", activation_dtype="float16")
        assert plan.layer_idx == 0
        assert plan.layer_type == "attention"
        assert plan.sensitivity_score == 0.0

    def test_mixed_precision_plan_empty(self):
        plan = MixedPrecisionPlan()
        assert plan.plans == []
        assert plan.overall_compression_ratio == 1.0
        assert plan.num_layers == 0

    def test_mixed_precision_plan_summary(self):
        plans = [
            LayerQuantPlan(0, "attention", "int8", "float16", sensitivity_score=0.3, compression_ratio=2.0),
            LayerQuantPlan(0, "mlp", "nf4", "float16", sensitivity_score=0.1, compression_ratio=4.0),
        ]
        plan = MixedPrecisionPlan(plans=plans, overall_compression_ratio=3.0, num_layers=1)
        summary = plan.summary()
        assert "1 layers" in summary
        assert "int8" in summary
        assert "nf4" in summary

    def test_compression_ratio_reflects_dtype(self):
        plans = [
            LayerQuantPlan(0, "attention", "float16", "float16", compression_ratio=1.0),
            LayerQuantPlan(0, "mlp", "int8", "float16", compression_ratio=2.0),
            LayerQuantPlan(1, "mlp", "nf4", "float16", compression_ratio=4.0),
        ]
        ratios = [p.compression_ratio for p in plans]
        assert ratios == [1.0, 2.0, 4.0]


class TestSimulatedInt8Linear:
    """Tests for SimulatedInt8Linear."""

    def test_forward_output_shape(self):
        mod = nn.Linear(32, 16)
        sim = SimulatedInt8Linear(mod)
        x = torch.randn(4, 32)
        out = sim(x)
        assert out.shape == (4, 16)

    def test_forward_no_nan(self):
        mod = nn.Linear(32, 16)
        sim = SimulatedInt8Linear(mod)
        x = torch.randn(4, 32)
        out = sim(x)
        assert not torch.isnan(out).any()


class TestSimulatedNF4Linear:
    """Tests for SimulatedNF4Linear."""

    def test_forward_output_shape(self):
        mod = nn.Linear(32, 16)
        sim = SimulatedNF4Linear(mod)
        x = torch.randn(4, 32)
        out = sim(x)
        assert out.shape == (4, 16)

    def test_forward_no_nan(self):
        mod = nn.Linear(32, 16)
        sim = SimulatedNF4Linear(mod)
        x = torch.randn(4, 32)
        out = sim(x)
        assert not torch.isnan(out).any()


class TestQuantizationAutoTuner:
    """Tests for QuantizationAutoTuner methods."""

    def test_recommend_no_results_returns_none(self):
        tuner = QuantizationAutoTuner()
        result = tuner.recommend([])
        assert result == "none"

    def test_recommend_fp16_only_returns_none(self):
        tuner = QuantizationAutoTuner()
        from distllm.core.quantization_selector import QuantProfileResult
        results = [
            QuantProfileResult(method="none", tokens_per_sec=100.0, speedup_vs_fp16=1.0),
        ]
        result = tuner.recommend(results, max_perplexity_delta=0.5)
        assert result == "none"

    def test_recommend_picks_fastest_within_delta(self):
        tuner = QuantizationAutoTuner()
        from distllm.core.quantization_selector import QuantProfileResult
        results = [
            QuantProfileResult(method="none", tokens_per_sec=100.0, speedup_vs_fp16=1.0),
            QuantProfileResult(method="bnb_8bit", tokens_per_sec=200.0,
                               perplexity_delta=0.3, speedup_vs_fp16=2.0),
            QuantProfileResult(method="bnb_4bit", tokens_per_sec=300.0,
                               perplexity_delta=0.8, speedup_vs_fp16=3.0),
        ]
        result = tuner.recommend(results, max_perplexity_delta=0.5)
        assert result == "bnb_8bit"

    def test_recommend_no_candidates_returns_none(self):
        tuner = QuantizationAutoTuner()
        from distllm.core.quantization_selector import QuantProfileResult
        results = [
            QuantProfileResult(method="none", tokens_per_sec=100.0, speedup_vs_fp16=1.0),
            QuantProfileResult(method="bnb_4bit", tokens_per_sec=300.0,
                               perplexity_delta=1.0, speedup_vs_fp16=3.0),
        ]
        result = tuner.recommend(results, max_perplexity_delta=0.5)
        assert result == "none"

    def test_recommend_respects_min_speedup(self):
        tuner = QuantizationAutoTuner()
        from distllm.core.quantization_selector import QuantProfileResult
        results = [
            QuantProfileResult(method="none", tokens_per_sec=100.0, speedup_vs_fp16=1.0),
            QuantProfileResult(method="bnb_8bit", tokens_per_sec=110.0,
                               perplexity_delta=0.1, speedup_vs_fp16=1.1),
            QuantProfileResult(method="bnb_4bit", tokens_per_sec=300.0,
                               perplexity_delta=0.4, speedup_vs_fp16=3.0),
        ]
        result = tuner.recommend(results, max_perplexity_delta=0.5, min_speedup=2.0)
        assert result == "bnb_4bit"

    def test_compression_ratio_values(self):
        tuner = QuantizationAutoTuner()
        assert tuner._compression_ratio("float16") == 1.0
        assert tuner._compression_ratio("int8") == 2.0
        assert tuner._compression_ratio("nf4") == 4.0
        assert tuner._compression_ratio("fp8") == 2.0
        assert tuner._compression_ratio("unknown") == 1.0

    def test_compute_outlier_ratio_uniform_weights(self):
        tuner = QuantizationAutoTuner()
        mod = nn.Linear(32, 16)
        nn.init.constant_(mod.weight, 0.5)
        ratio = tuner._compute_outlier_ratio(mod)
        assert ratio == 0.0

    def test_compute_outlier_ratio_with_outliers(self):
        tuner = QuantizationAutoTuner()
        mod = nn.Linear(32, 16)
        with torch.no_grad():
            mod.weight[0] = 100.0
        ratio = tuner._compute_outlier_ratio(mod)
        assert ratio > 0.0

    def test_profile_layer_sensitivity_no_model(self):
        tuner = QuantizationAutoTuner()
        scores = tuner.profile_layer_sensitivity()
        assert scores == []

    def test_apply_noise_and_restore(self):
        mod = nn.Linear(32, 16)
        orig = mod.weight.data.clone()
        QuantizationAutoTuner._apply_noise(mod, noise_level=0.1)
        assert not torch.allclose(mod.weight.data, orig, atol=1e-6)
        QuantizationAutoTuner._restore_weights(mod)
        assert torch.allclose(mod.weight.data, orig)


class TestSelectForNodeExtended:
    """Extended tests for select_for_node covering hardware scenarios."""

    def test_hopper_gpu_fp8_selected(self):
        info = NodeVRAMInfo(
            device_type="cuda", available_memory=9e9, compute_capability=9.0,
        )
        result = select_for_node(info, model_size_bytes=5e9)
        assert result == "fp8"

    def test_non_hopper_gpu_skips_fp8(self):
        info = NodeVRAMInfo(
            device_type="cuda", available_memory=9e9, compute_capability=8.0,
        )
        result = select_for_node(info, model_size_bytes=5.1e9)
        assert result == "bnb_8bit"

    def test_bnb_4bit_range(self):
        info = NodeVRAMInfo(
            device_type="cuda", available_memory=5e9, compute_capability=8.0,
        )
        result = select_for_node(info, model_size_bytes=4.5e9)
        assert result == "bnb_4bit"

    def test_awq_range(self):
        info = NodeVRAMInfo(
            device_type="cuda", available_memory=4e9, compute_capability=8.0,
        )
        result = select_for_node(info, model_size_bytes=3e9)
        assert result == "awq"


# ===========================================================================
# 4. Precision Boundary Tests
# ===========================================================================


class TestPrecisionBoundary:
    """Tests for PrecisionBoundary."""

    def test_convert_same_dtype_unchanged(self):
        t = torch.randn(4, 16, dtype=torch.float16)
        result = PrecisionBoundary.convert_precision(t, torch.float16, torch.float16)
        assert result.dtype == torch.float16
        assert torch.allclose(t, result)

    def test_convert_fp16_to_fp32(self):
        t = torch.randn(4, 16, dtype=torch.float16)
        result = PrecisionBoundary.convert_precision(t, torch.float16, torch.float32)
        assert result.dtype == torch.float32

    def test_convert_fp32_to_fp16(self):
        t = torch.randn(4, 16, dtype=torch.float32)
        result = PrecisionBoundary.convert_precision(t, torch.float32, torch.float16)
        assert result.dtype == torch.float16

    def test_convert_int8_to_fp16(self):
        t = torch.randint(-128, 127, (4, 16), dtype=torch.int8)
        result = PrecisionBoundary.convert_precision(t, torch.int8, torch.float16)
        assert result.dtype == torch.float16
        assert result.shape == t.shape

    def test_get_boundary_dtype_both_fp16(self):
        dtype = PrecisionBoundary.get_boundary_dtype("float16", "float16")
        assert dtype == torch.float16

    def test_get_boundary_dtype_int8_and_fp16(self):
        dtype = PrecisionBoundary.get_boundary_dtype("int8", "float16")
        assert dtype == torch.float16

    def test_get_boundary_dtype_int4_and_fp32(self):
        dtype = PrecisionBoundary.get_boundary_dtype("int4", "float32")
        assert dtype == torch.float32

    def test_get_boundary_dtype_fp8_and_bf16(self):
        dtype = PrecisionBoundary.get_boundary_dtype("float8_e4m3fn", "bfloat16")
        assert dtype == torch.bfloat16

    def test_prepare_for_transfer_returns_metadata(self):
        t = torch.randn(4, 16, dtype=torch.float16)
        converted, metadata = PrecisionBoundary.prepare_for_transfer(t, "int8", "float16")
        assert isinstance(converted, torch.Tensor)
        assert isinstance(metadata, dict)
        assert metadata["src_precision"] == "int8"
        assert metadata["dst_precision"] == "float16"
        assert "boundary_dtype" in metadata
        assert "shape" in metadata

    def test_prepare_for_transfer_converts_dtype(self):
        t = torch.randint(-128, 127, (4, 16), dtype=torch.int8)
        converted, _ = PrecisionBoundary.prepare_for_transfer(t, "int8", "float16")
        assert converted.dtype == torch.float16

    def test_prepare_for_transfer_int4_to_float32(self):
        t = torch.randint(-8, 7, (4, 16), dtype=torch.int8)
        converted, _ = PrecisionBoundary.prepare_for_transfer(t, "int4", "float32")
        assert converted.dtype == torch.float32

    def test_convert_different_shapes(self):
        shapes = [(4, 16), (2, 8, 32), (1,)]
        for shape in shapes:
            t = torch.randn(*shape, dtype=torch.float16)
            result = PrecisionBoundary.convert_precision(t, torch.float16, torch.float32)
            assert result.shape == shape

    def test_conversion_roundtrip_preserves_values(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        to_fp16 = PrecisionBoundary.convert_precision(t, torch.float32, torch.float16)
        back = PrecisionBoundary.convert_precision(to_fp16, torch.float16, torch.float32)
        assert torch.allclose(t, back, atol=1e-3)
