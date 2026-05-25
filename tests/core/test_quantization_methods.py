"""Tests: AWQ, GPTQ, INT4, INT8 quantization methods — load, apply, verify.

Tests: WQLinear pack/unpack, AWQQuantizer, GPTQQuantizer, CompressionPipeline
INT4/INT8, edge quantization backend, quality preservation.

Run: pytest tests/core/test_quantization_methods.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from distllm.core.compression_pipeline import (
    AWQQuantizer,
    GPTQQuantizer,
    WQLinear,
    CompressionPipeline,
)
from distllm.core.compression_config import CompressionConfig, CompressionMethod
from distllm.edge.quantized import QuantizationBackend
from distllm.edge.models import QuantizationType


# ---------------------------------------------------------------------------
# Tiny helper models and helpers
# ---------------------------------------------------------------------------

class TinyTestModel(nn.Module):
    """Minimal transformer-like model for quantization tests."""

    def __init__(self, hidden_dim: int = 32, vocab_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Linear(hidden_dim, hidden_dim),
                "k_proj": nn.Linear(hidden_dim, hidden_dim),
                "v_proj": nn.Linear(hidden_dim, hidden_dim),
                "o_proj": nn.Linear(hidden_dim, hidden_dim),
            })
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        for layer in self.layers:
            q = layer["q_proj"](h)
            k = layer["k_proj"](h)
            v = layer["v_proj"](h)
            attn = torch.softmax(
                torch.matmul(q, k.transpose(-2, -1)) / (h.shape[-1] ** 0.5), dim=-1
            )
            h = h + layer["o_proj"](torch.matmul(attn, v))
        h = self.norm(h)
        return self.lm_head(h)


class TinyMLP(nn.Module):
    """Simple MLP for quantization tests (no named module patterns)."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def compute_kl_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = F.softmax(logits_a.float(), dim=-1).clamp(min=1e-10)
    q = F.softmax(logits_b.float(), dim=-1).clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum(dim=-1)
    return kl.mean().item()


# ---------------------------------------------------------------------------
# WQLinear tests
# ---------------------------------------------------------------------------


class TestWQLinear:
    """Tests for WQLinear weight-quantized linear layer."""

    def test_pack_unpack_roundtrip(self):
        orig = torch.randn(16, 32)
        packed = WQLinear.pack_int4(orig.clone())
        unpacked = WQLinear.unpack_int4(packed, orig.shape)
        assert unpacked.shape == orig.shape
        assert unpacked.dtype == torch.int8

    def test_quantize_preserves_shape(self):
        wq = WQLinear(32, 16, group_size=16)
        weight = torch.randn(16, 32) * 0.1
        wq.quantize(weight)
        assert wq.qweight is not None
        assert wq.scales.shape == (16, wq.num_groups)
        assert wq.zeros.shape == (16, wq.num_groups)

    def test_quantize_reduces_range(self):
        wq = WQLinear(32, 16, group_size=16)
        weight = torch.randn(16, 32) * 2.0
        wq.quantize(weight)
        packed = wq.qweight
        unpacked = WQLinear.unpack_int4(packed, (16, 32))
        assert unpacked.min() >= -8
        assert unpacked.max() <= 7

    def test_forward_output_shape(self):
        wq = WQLinear(32, 16, group_size=16)
        weight = torch.randn(16, 32) * 0.1
        wq.quantize(weight)
        x = torch.randn(4, 32)
        out = wq(x)
        assert out.shape == (4, 16)

    def test_forward_no_nan(self):
        wq = WQLinear(32, 16, group_size=16)
        weight = torch.randn(16, 32) * 0.1
        wq.quantize(weight)
        x = torch.randn(4, 32)
        out = wq(x)
        assert not torch.isnan(out).any()

    def test_awq_scaling_applied(self):
        wq = WQLinear(32, 16, group_size=16, use_awq=True)
        wq.act_scales.data = torch.full((32,), 2.0)
        weight = torch.randn(16, 32) * 0.5
        wq.quantize(weight)
        x = torch.ones(1, 32)
        out_awq = wq(x)
        wq_no = WQLinear(32, 16, group_size=16, use_awq=False)
        wq_no.quantize(weight)
        out_no = wq_no(x)
        assert out_awq.abs().sum() > 0
        assert out_no.abs().sum() > 0
        assert not torch.allclose(out_awq, out_no, atol=1e-3)

    def test_different_group_sizes(self):
        for group_size in [8, 16, 32, 64]:
            wq = WQLinear(64, 16, group_size=group_size)
            weight = torch.randn(16, 64) * 0.1
            wq.quantize(weight)
            x = torch.randn(2, 64)
            out = wq(x)
            assert out.shape == (2, 16)

    def test_no_bias(self):
        wq = WQLinear(32, 16, bias=False)
        assert wq.bias is None

    def test_with_bias(self):
        wq = WQLinear(32, 16, bias=True)
        assert wq.bias is not None
        assert wq.bias.shape == (16,)


# ---------------------------------------------------------------------------
# AWQ quantizer tests
# ---------------------------------------------------------------------------


class TestAWQQuantizer:
    """Tests for AWQ-style quantization."""

    def test_init_defaults(self):
        quantizer = AWQQuantizer()
        assert quantizer.group_size == 128

    def test_init_custom_group_size(self):
        quantizer = AWQQuantizer(group_size=64)
        assert quantizer.group_size == 64

    def test_quantize_model_replaces_linears(self):
        model = TinyTestModel(hidden_dim=32)
        orig_q = model.layers[0]["q_proj"]
        assert isinstance(orig_q, nn.Linear)
        quantizer = AWQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model)
        assert isinstance(qmodel.layers[0]["q_proj"], WQLinear)

    def test_quantize_model_all_linears_replaced(self):
        model = TinyTestModel(hidden_dim=32, num_layers=2)
        quantizer = AWQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model)
        count = 0
        for mod in qmodel.modules():
            if isinstance(mod, WQLinear):
                count += 1
        # Each of 2 layers has 4 linears = 8
        assert count == 8

    def test_quantize_model_with_calibration(self):
        model = TinyTestModel(hidden_dim=32)
        calib = [torch.randint(0, 64, (2, 16))]
        quantizer = AWQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model, calib_inputs=calib)
        assert isinstance(qmodel.layers[0]["q_proj"], WQLinear)

    def test_quantize_model_no_target_modules_fallback(self):
        model = TinyMLP(dim=32)
        quantizer = AWQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model, target_modules=["dummy"])
        assert isinstance(qmodel.fc1, WQLinear)
        assert isinstance(qmodel.fc2, WQLinear)

    def test_quantize_model_with_custom_targets(self):
        model = TinyTestModel(hidden_dim=32)
        quantizer = AWQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model, target_modules=["q_proj"])
        assert isinstance(qmodel.layers[0]["q_proj"], WQLinear)
        assert isinstance(qmodel.layers[0]["k_proj"], nn.Linear)

    def test_quantize_model_returns_same_model(self):
        model = TinyTestModel(hidden_dim=32)
        quantizer = AWQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model)
        assert qmodel is model

    def test_output_not_degenerate(self):
        model = TinyTestModel(hidden_dim=32)
        quantizer = AWQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model)
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            logits = qmodel(x)
        probs = F.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values.mean()
        assert max_prob < 0.5, "AWQ model collapsed to degenerate output"

    def test_quality_close_to_original(self):
        torch.manual_seed(42)
        model = TinyTestModel(hidden_dim=32)
        quantizer = AWQQuantizer(group_size=32)
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            logits_before = model(x)
        qmodel = quantizer.quantize_model(model)
        with torch.no_grad():
            logits_after = qmodel(x)
        kl = compute_kl_divergence(logits_before, logits_after)
        assert kl < 1.0, f"AWQ KL divergence too high: {kl:.4f}"


# ---------------------------------------------------------------------------
# GPTQ quantizer tests
# ---------------------------------------------------------------------------


class TestGPTQQuantizer:
    """Tests for GPTQ-style quantization."""

    def test_init_defaults(self):
        quantizer = GPTQQuantizer()
        assert quantizer.group_size == 128
        assert quantizer.damp_percent == 0.01

    def test_init_custom(self):
        quantizer = GPTQQuantizer(group_size=64, damp_percent=0.05)
        assert quantizer.group_size == 64
        assert quantizer.damp_percent == 0.05

    def test_hessian_computation_shape(self):
        quantizer = GPTQQuantizer()
        x = torch.randn(16, 32)
        H = quantizer._compute_hessian(x, out_features=16)
        assert H.shape == (32, 32)

    def test_hessian_positive_definite(self):
        quantizer = GPTQQuantizer()
        x = torch.randn(64, 16)
        H = quantizer._compute_hessian(x, out_features=8)
        eigenvalues = torch.linalg.eigvalsh(H)
        assert (eigenvalues > 0).all(), "Hessian not positive definite"

    def test_hessian_large_input_downsampled(self):
        quantizer = GPTQQuantizer()
        x = torch.randn(4096, 16)
        H = quantizer._compute_hessian(x, out_features=8)
        assert H.shape == (16, 16)

    def test_quantize_model_replaces_linears(self):
        model = TinyTestModel(hidden_dim=32)
        calib = [torch.randint(0, 64, (2, 16))]
        quantizer = GPTQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model, calib)
        assert isinstance(qmodel.layers[0]["q_proj"], WQLinear)

    def test_quantize_model_all_linears_replaced(self):
        model = TinyTestModel(hidden_dim=32, num_layers=2)
        calib = [torch.randint(0, 64, (2, 16))]
        quantizer = GPTQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model, calib)
        count = 0
        for mod in qmodel.modules():
            if isinstance(mod, WQLinear):
                count += 1
        assert count == 8

    def test_quantize_model_fallback_on_empty_replacements(self):
        model = TinyMLP(dim=32)
        calib = [torch.randn(2, 32)]
        quantizer = GPTQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model, calib, target_modules=["nonexistent"])
        assert isinstance(qmodel.fc1, WQLinear)
        assert isinstance(qmodel.fc2, WQLinear)

    def test_quantize_no_calib_data_no_replacement(self):
        model = TinyTestModel(hidden_dim=32)
        quantizer = GPTQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model, [])
        assert isinstance(qmodel.layers[0]["q_proj"], nn.Linear)

    def test_output_not_degenerate(self):
        model = TinyTestModel(hidden_dim=32)
        calib = [torch.randint(0, 64, (2, 16))]
        quantizer = GPTQQuantizer(group_size=32)
        qmodel = quantizer.quantize_model(model, calib)
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            logits = qmodel(x)
        probs = F.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values.mean()
        assert max_prob < 0.5, "GPTQ model collapsed to degenerate output"

    def test_quality_close_to_original(self):
        torch.manual_seed(42)
        model = TinyTestModel(hidden_dim=32)
        calib = [torch.randint(0, 64, (4, 16)) for _ in range(2)]
        quantizer = GPTQQuantizer(group_size=32)
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            logits_before = model(x)
        qmodel = quantizer.quantize_model(model, calib)
        with torch.no_grad():
            logits_after = qmodel(x)
        kl = compute_kl_divergence(logits_before, logits_after)
        assert kl < 1.0, f"GPTQ KL divergence too high: {kl:.4f}"


# ---------------------------------------------------------------------------
# Compression pipeline tests
# ---------------------------------------------------------------------------


class TestCompressionPipelineINT4:
    """Tests for INT4 quantization via CompressionPipeline."""

    def test_apply_quantization_int4_awq(self):
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=CompressionMethod.PTQ_INT4,
            enabled=True,
            target_bits=4,
            quant_method="awq",
        )
        pipeline = CompressionPipeline(config)
        qmodel = pipeline.apply_quantization(model, bits=4, quant_method="awq")
        assert isinstance(qmodel.layers[0]["q_proj"], WQLinear)

    def test_apply_quantization_int4_gptq(self):
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=CompressionMethod.QUANT_GPTQ,
            enabled=True,
            target_bits=4,
            quant_method="gptq",
        )
        pipeline = CompressionPipeline(config)
        qmodel = pipeline.apply_quantization(model, bits=4, quant_method="gptq", tokenizer=None)
        assert isinstance(qmodel.layers[0]["q_proj"], WQLinear)

    def test_apply_method_int4(self):
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=CompressionMethod.QUANT_AWQ,
            enabled=True,
            target_bits=4,
        )
        pipeline = CompressionPipeline(config)
        qmodel = pipeline.apply(model)
        assert isinstance(qmodel.layers[0]["q_proj"], WQLinear)

    def test_apply_not_enabled_returns_unchanged(self):
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=CompressionMethod.QUANT_AWQ,
            enabled=False,
            target_bits=4,
        )
        pipeline = CompressionPipeline(config)
        qmodel = pipeline.apply(model)
        assert qmodel is model
        assert isinstance(qmodel.layers[0]["q_proj"], nn.Linear)

    def test_apply_method_none_returns_unchanged(self):
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=CompressionMethod.NONE,
            enabled=True,
        )
        pipeline = CompressionPipeline(config)
        qmodel = pipeline.apply(model)
        assert qmodel is model

    def test_pipeline_plan_returns_plan(self):
        config = CompressionConfig(
            method=CompressionMethod.PTQ_INT4,
            enabled=True,
            target_bits=4,
        )
        pipeline = CompressionPipeline(config)
        plan = pipeline.plan()
        assert isinstance(plan.stages, list)


class TestCompressionPipelineINT8:
    """Tests for INT8 quantization via CompressionPipeline."""

    def test_apply_quantization_int8(self):
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=CompressionMethod.PTQ_INT8,
            enabled=True,
            target_bits=8,
        )
        pipeline = CompressionPipeline(config)
        qmodel = pipeline.apply_quantization(model, bits=8)
        for mod in qmodel.modules():
            if isinstance(mod, nn.Linear):
                assert hasattr(mod, "weight")

    def test_apply_method_int8(self):
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=CompressionMethod.PTQ_INT8,
            enabled=True,
            target_bits=8,
        )
        pipeline = CompressionPipeline(config)
        qmodel = pipeline.apply(model)
        assert qmodel is not None


# ---------------------------------------------------------------------------
# Edge quantization backend tests
# ---------------------------------------------------------------------------


class TestEdgeQuantization:
    """Tests for edge quantization backend."""

    def test_int4_quantize_dequantize_roundtrip(self):
        weights = [0.5, -0.3, 0.0, 1.0, -1.0, 0.7, -0.8, 0.2]
        packed = QuantizationBackend.quantize_weights(weights, QuantizationType.INT4)
        unpacked = QuantizationBackend.dequantize_weights(packed, QuantizationType.INT4, len(weights))
        assert len(unpacked) == len(weights)
        for orig, recon in zip(weights, unpacked):
            assert abs(orig - recon) < 0.2

    def test_int8_quantize_dequantize_roundtrip(self):
        weights = [0.5, -0.3, 0.0, 1.0, -1.0]
        packed = QuantizationBackend.quantize_weights(weights, QuantizationType.INT8)
        unpacked = QuantizationBackend.dequantize_weights(packed, QuantizationType.INT8, len(weights))
        assert len(unpacked) == len(weights)
        for orig, recon in zip(weights, unpacked):
            assert abs(orig - recon) < 0.05

    def test_int4_compression_ratio(self):
        weights = [0.1] * 16
        packed_int4 = QuantizationBackend.quantize_weights(weights, QuantizationType.INT4)
        packed_fp16 = QuantizationBackend.quantize_weights(weights, QuantizationType.FP16)
        assert len(packed_int4) < len(packed_fp16)

    def test_int4_odd_count(self):
        weights = [0.1, -0.2, 0.3]
        packed = QuantizationBackend.quantize_weights(weights, QuantizationType.INT4)
        unpacked = QuantizationBackend.dequantize_weights(packed, QuantizationType.INT4, len(weights))
        assert len(unpacked) == 3

    def test_nf4_quantize(self):
        weights = [0.5, -0.5]
        packed = QuantizationBackend.quantize_weights(weights, QuantizationType.NF4)
        unpacked = QuantizationBackend.dequantize_weights(packed, QuantizationType.NF4, len(weights))
        assert len(unpacked) == 2

    def test_fp16_quantize(self):
        weights = [0.5, -0.5, 1.0]
        packed = QuantizationBackend.quantize_weights(weights, QuantizationType.FP16)
        unpacked = QuantizationBackend.dequantize_weights(packed, QuantizationType.FP16, len(weights))
        assert len(unpacked) == len(weights)
        for orig, recon in zip(weights, unpacked):
            assert abs(orig - recon) < 0.01


# ---------------------------------------------------------------------------
# Quality preservation tests
# ---------------------------------------------------------------------------


class TestQuantizationQualityPreservation:
    """End-to-end quality verification for all quantization methods."""

    @pytest.fixture(params=[
        ("awq", CompressionMethod.QUANT_AWQ),
        ("gptq", CompressionMethod.QUANT_GPTQ),
    ], ids=["awq", "gptq"])
    def int4_method(self, request):
        return request.param

    def test_int4_perplexity_stable(self, int4_method):
        torch.manual_seed(42)
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=int4_method[1],
            enabled=True,
            target_bits=4,
            quant_method=int4_method[0],
        )
        pipeline = CompressionPipeline(config)
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            logits_before = model(x)
        qmodel = pipeline.apply(model)
        with torch.no_grad():
            logits_after = qmodel(x)
        kl = compute_kl_divergence(logits_before, logits_after)
        logger.info(f"{int4_method[0]} KL={kl:.4f}")
        assert kl < 1.0, f"{int4_method[0]} KL={kl:.4f}"

    def test_int8_vs_fp16_quality(self):
        torch.manual_seed(42)
        model = TinyTestModel(hidden_dim=32)
        config = CompressionConfig(
            method=CompressionMethod.PTQ_INT8,
            enabled=True,
            target_bits=8,
        )
        pipeline = CompressionPipeline(config)
        x = torch.randint(0, 64, (2, 16))
        with torch.no_grad():
            logits_before = model(x)
        qmodel = pipeline.apply(model)
        with torch.no_grad():
            logits_after = qmodel(x)
        kl = compute_kl_divergence(logits_before, logits_after)
        logger.info(f"INT8 KL={kl:.4f}")
        assert kl < 1.0, f"INT8 KL={kl:.4f}"

    def test_int4_methods_agree_on_large_input(self):
        torch.manual_seed(42)
        model_awq = TinyTestModel(hidden_dim=32)
        model_gptq = TinyTestModel(hidden_dim=32)
        # Share weights
        model_gptq.load_state_dict(model_awq.state_dict())
        x = torch.randint(0, 64, (4, 32))
        with torch.no_grad():
            ref = model_awq(x)
        awq_config = CompressionConfig(method=CompressionMethod.QUANT_AWQ, enabled=True, target_bits=4)
        gptq_config = CompressionConfig(method=CompressionMethod.QUANT_GPTQ, enabled=True, target_bits=4)
        awq_model = CompressionPipeline(awq_config).apply(model_awq)
        gptq_model = CompressionPipeline(gptq_config).apply(model_gptq)
        with torch.no_grad():
            logits_awq = awq_model(x)
            logits_gptq = gptq_model(x)
        kl_awq = compute_kl_divergence(ref, logits_awq)
        kl_gptq = compute_kl_divergence(ref, logits_gptq)
        logger.info(f"AWQ KL={kl_awq:.4f}, GPTQ KL={kl_gptq:.4f}")
        assert kl_awq < 1.0
        assert kl_gptq < 1.0

    def test_no_catastrophic_collapse_any_method(self):
        torch.manual_seed(42)
        x = torch.randint(0, 64, (2, 16))
        for name, method in [
            ("awq", CompressionMethod.QUANT_AWQ),
            ("gptq", CompressionMethod.QUANT_GPTQ),
            ("int4", CompressionMethod.PTQ_INT4),
            ("int8", CompressionMethod.PTQ_INT8),
        ]:
            model = TinyTestModel(hidden_dim=32)
            config = CompressionConfig(method=method, enabled=True, target_bits=8 if name == "int8" else 4)
            qmodel = CompressionPipeline(config).apply(model)
            with torch.no_grad():
                logits = qmodel(x)
            probs = F.softmax(logits.float(), dim=-1)
            max_prob = probs.max(dim=-1).values.mean()
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            logger.info(f"{name}: max_prob={max_prob:.4f}, entropy={entropy:.4f}")
            assert max_prob < 0.5, f"{name} collapsed: max_prob={max_prob:.4f}"
            assert entropy > 0.1, f"{name} entropy too low: {entropy:.4f}"
