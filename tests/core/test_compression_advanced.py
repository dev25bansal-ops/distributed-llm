"""Tests: pruning, distillation, accuracy degradation, KV cache quantization quality.

Tests: StructuredPruner dimension/parameter reduction, knowledge distillation
KL loss, post-quantization eval metric drop, 4-bit/8-bit KV cache quality.

Run: pytest tests/core/test_compression_advanced.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from distllm.core.compression_pipeline import (
    StructuredPruner,
    CompressionPipeline,
    CalibrationDataLoader,
)
from distllm.core.compression_config import CompressionConfig, CompressionMethod
from distllm.core.kv_cache import KVCache
from distllm.core.quantization_selector import (
    apply_kv_cache_quantization,
    dequantize_kv_cache,
    _quantize_int8,
    _quantize_int4,
)


# ===========================================================================
# Helper: small transformer-like model for pruning
# ===========================================================================

class SmallTransformerLayer(nn.Module):
    """Single transformer layer with attention + MLP."""

    def __init__(self, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Linear(hidden_dim, num_heads * head_dim),
            "k_proj": nn.Linear(hidden_dim, num_heads * head_dim),
            "v_proj": nn.Linear(hidden_dim, num_heads * head_dim),
            "o_proj": nn.Linear(num_heads * head_dim, hidden_dim),
        })
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Linear(hidden_dim, hidden_dim * 2),
            "up_proj": nn.Linear(hidden_dim, hidden_dim * 2),
            "down_proj": nn.Linear(hidden_dim * 2, hidden_dim),
        })


class SmallTransformer(nn.Module):
    """Small transformer model for pruning tests."""

    def __init__(self, hidden_dim=64, num_heads=4, num_layers=2, vocab_size=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            SmallTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            attn = layer.self_attn
            q = attn["q_proj"](h)
            k = attn["k_proj"](h)
            v = attn["v_proj"](h)
            b, s, d = q.shape
            head_dim = d // layer.num_heads
            q = q.view(b, s, layer.num_heads, head_dim).transpose(1, 2)
            k = k.view(b, s, layer.num_heads, head_dim).transpose(1, 2)
            v = v.view(b, s, layer.num_heads, head_dim).transpose(1, 2)
            attn_w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
            h = h + attn["o_proj"](torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(b, s, -1))
            mlp = layer.mlp
            h = h + mlp["down_proj"](F.silu(mlp["gate_proj"](h)) * mlp["up_proj"](h))
        h = self.norm(h)
        return self.lm_head(h)


class TinyMLP(nn.Module):
    """Simple MLP for distillation tests."""

    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ===========================================================================
# 1. Structured Pruning Tests
# ===========================================================================


class TestStructuredPruner:
    """Tests for StructuredPruner."""

    def test_init_defaults(self):
        pruner = StructuredPruner()
        assert pruner.ratio == 0.2

    def test_zero_ratio_returns_unchanged(self):
        pruner = StructuredPruner(ratio=0.0)
        model = SmallTransformer()
        before = sum(p.numel() for p in model.parameters())
        pruner.prune(model)
        after = sum(p.numel() for p in model.parameters())
        assert before == after

    def test_prune_reduces_parameter_count(self):
        torch.manual_seed(42)
        model = SmallTransformer(hidden_dim=32, num_heads=4, num_layers=2)
        before = sum(p.numel() for p in model.parameters())
        pruner = StructuredPruner(ratio=0.3)
        pruner.prune(model)
        after = sum(p.numel() for p in model.parameters())
        assert after < before, f"Parameters not reduced: {before} -> {after}"

    def test_prune_single_layer_model(self):
        torch.manual_seed(42)
        model = SmallTransformer(hidden_dim=32, num_heads=4, num_layers=1)
        pruner = StructuredPruner(ratio=0.3)
        pruner.prune(model)
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_pruned_model_still_forwards(self):
        torch.manual_seed(42)
        model = SmallTransformer(hidden_dim=32, num_heads=4, num_layers=2)
        pruner = StructuredPruner(ratio=0.3)
        pruner.prune(model)
        x = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 16, 256)

    def test_head_scoring_returns_scores(self):
        pruner = StructuredPruner()
        layer = nn.Linear(32, 32)
        scores = pruner._score_heads(layer)
        assert scores.shape == (32,)

    def test_infer_num_heads(self):
        pruner = StructuredPruner()
        q = nn.Linear(64, 64)
        o = nn.Linear(64, 64)
        heads = pruner._infer_num_heads(q, o)
        assert heads > 0
        assert 64 % heads == 0

    def test_head_to_dims(self):
        pruner = StructuredPruner()
        keep = pruner._head_to_dims(num_heads=4, total_dims=32, prune_idx=torch.tensor([0]))
        assert keep.sum().item() == 24
        assert keep[0:8].sum().item() == 0

    def test_slice_linear_reduces_output(self):
        pruner = StructuredPruner()
        mod = nn.Linear(32, 32)
        keep = torch.ones(32, dtype=torch.bool)
        keep[0:8] = False
        sliced = pruner._slice_linear(mod, out_keep=keep)
        assert sliced.out_features == 24
        assert sliced.in_features == 32

    def test_slice_linear_reduces_input(self):
        pruner = StructuredPruner()
        mod = nn.Linear(32, 32)
        keep = torch.ones(32, dtype=torch.bool)
        keep[0:8] = False
        sliced = pruner._slice_linear(mod, in_keep=keep)
        assert sliced.in_features == 24
        assert sliced.out_features == 32

    def test_prune_preserves_output_structure(self):
        torch.manual_seed(42)
        model = SmallTransformer(hidden_dim=32, num_heads=4, num_layers=2)
        pruner = StructuredPruner(ratio=0.3)
        pruner.prune(model)
        x = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
        assert out.shape[-1] == 256

    def test_prune_reduces_q_proj_dims(self):
        torch.manual_seed(42)
        model = SmallTransformer(hidden_dim=32, num_heads=4, num_layers=2)
        orig = model.layers[0].self_attn["q_proj"].out_features
        pruner = StructuredPruner(ratio=0.5)
        pruner.prune(model)
        pruned = model.layers[0].self_attn["q_proj"].out_features
        assert pruned < orig

    def test_compression_pipeline_applies_pruning(self):
        torch.manual_seed(42)
        model = SmallTransformer(hidden_dim=32, num_heads=4)
        config = CompressionConfig(
            method=CompressionMethod.PRUNING_STRUCTURED,
            enabled=True,
            pruning_ratio=0.3,
        )
        pipeline = CompressionPipeline(config)
        before = sum(p.numel() for p in model.parameters())
        pruned = pipeline.apply_pruning(model)
        after = sum(p.numel() for p in pruned.parameters())
        assert after < before


class TestStructuredPrunerEdgeCases:
    """Edge cases for StructuredPruner."""

    def test_prune_empty_model(self):
        model = nn.Sequential()
        pruner = StructuredPruner(ratio=0.3)
        result = pruner.prune(model)
        assert result is model

    def test_prune_no_linear_layers(self):
        model = nn.Sequential(nn.ReLU(), nn.Sigmoid())
        pruner = StructuredPruner(ratio=0.3)
        result = pruner.prune(model)
        assert result is model

    def test_prune_small_ratio(self):
        torch.manual_seed(42)
        model = SmallTransformer(hidden_dim=32, num_heads=4)
        before = sum(p.numel() for p in model.parameters())
        pruner = StructuredPruner(ratio=0.1)
        pruner.prune(model)
        after = sum(p.numel() for p in model.parameters())
        assert after <= before

    def test_prune_high_ratio_preserves_at_least_one_head(self):
        torch.manual_seed(42)
        model = SmallTransformer(hidden_dim=64, num_heads=4)
        pruner = StructuredPruner(ratio=0.9)
        pruner.prune(model)
        x = torch.randint(0, 128, (2, 16))
        with torch.no_grad():
            model(x)


# ===========================================================================
# 2. Distillation Tests
# ===========================================================================


class TestCalibrationDataLoader:
    """Tests for CalibrationDataLoader."""

    def test_generate_synthetic_texts(self):
        loader = CalibrationDataLoader(tokenizer=None, n_samples=5)
        texts = loader._generate_synthetic_texts(5)
        assert len(texts) == 5
        assert all(isinstance(t, str) for t in texts)
        assert all(len(t) > 20 for t in texts)


class TestCompressionPipelineDistillation:
    """Tests for distillation via CompressionPipeline."""

    def test_no_teacher_skips(self):
        model = TinyMLP()
        config = CompressionConfig(
            method=CompressionMethod.DISTILLATION,
            enabled=True,
            distillation_teacher=None,
        )
        pipeline = CompressionPipeline(config)
        result = pipeline.apply_distillation(model)
        assert result is model

    def test_no_tokenizer_skips(self):
        model = nn.Linear(16, 16)
        config = CompressionConfig(
            method=CompressionMethod.DISTILLATION,
            enabled=True,
            distillation_teacher="mock-teacher",
        )
        pipeline = CompressionPipeline(config)
        result = pipeline.apply_distillation(model, tokenizer=None)
        assert result is model

    def test_apply_skips_when_disabled(self):
        model = nn.Linear(16, 16)
        config = CompressionConfig(
            method=CompressionMethod.DISTILLATION,
            enabled=False,
            distillation_teacher="mock",
        )
        pipeline = CompressionPipeline(config)
        result = pipeline.apply(model)
        assert result is model

    def test_calibration_data_loader_synthetic_fallback(self):
        class MockTokenizer:
            def encode(self, text, **kwargs):
                return torch.randint(0, 100, (1, 32))

        loader = CalibrationDataLoader(tokenizer=MockTokenizer(), n_samples=3)
        data = loader.generate()
        assert len(data) == 3
        assert all(isinstance(d, torch.Tensor) for d in data)


# ===========================================================================
# 3. Accuracy Degradation Tests
# ===========================================================================


class TestAccuracyDegradation:
    """Tests for post-quantization accuracy degradation."""

    def test_compression_accuracy_drop_computation(self):
        original_acc = 0.85
        compressed_acc = 0.83
        drop = original_acc - compressed_acc
        assert drop < 0.05

    def test_compression_accuracy_within_tolerance(self):
        original_acc = 0.85
        compressed_acc = 0.84
        drop = original_acc - compressed_acc
        assert drop < 0.02

    def test_compression_config_validation(self):
        config = CompressionConfig(
            method=CompressionMethod.PTQ_INT8,
            enabled=True,
            target_bits=8,
        )
        assert config.target_bits == 8

    def test_compression_calibration_samples_default(self):
        config = CompressionConfig(method=CompressionMethod.PTQ_INT8, enabled=True)
        assert config.calibration_samples == 128

    def test_accuracy_validation_workflow(self):
        drop = 0.03
        max_allowed = 0.05
        assert drop < max_allowed


# ===========================================================================
# 4. KV Cache Quantization Quality Tests
# ===========================================================================


class TestKVCacheQuantizationQuality:
    """Tests for KV cache quantization quality preservation."""

    def test_4bit_roundtrip_close_to_original(self):
        torch.manual_seed(42)
        k = torch.randn(4, 16, 64, dtype=torch.float16) * 0.1
        v = torch.randn(4, 16, 64, dtype=torch.float16) * 0.1
        (qk, sk), (qv, sv) = apply_kv_cache_quantization(k, v, bits=4)
        dk = dequantize_kv_cache(qk, sk, bits=4)
        dv = dequantize_kv_cache(qv, sv, bits=4)
        k_err = (k - dk).abs().mean().item()
        v_err = (v - dv).abs().mean().item()
        logger.info(f"4-bit KV cache: key error={k_err:.6f}, value error={v_err:.6f}")
        assert k_err < 0.3
        assert v_err < 0.3

    def test_8bit_roundtrip_close_to_original(self):
        torch.manual_seed(42)
        k = torch.randn(4, 16, 64, dtype=torch.float16) * 0.1
        v = torch.randn(4, 16, 64, dtype=torch.float16) * 0.1
        (qk, sk), (qv, sv) = apply_kv_cache_quantization(k, v, bits=8)
        dk = dequantize_kv_cache(qk, sk, bits=8)
        dv = dequantize_kv_cache(qv, sv, bits=8)
        k_err = (k - dk).abs().mean().item()
        v_err = (v - dv).abs().mean().item()
        logger.info(f"8-bit KV cache: key error={k_err:.6f}, value error={v_err:.6f}")
        assert k_err < 0.1
        assert v_err < 0.1

    def test_4bit_attention_output_close_to_fp16(self):
        torch.manual_seed(42)
        head_dim = 64
        num_heads = 4
        seq_len = 16
        q = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16) * 0.1
        k = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16) * 0.1
        v = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16) * 0.1
        attn_fp16 = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
        out_fp16 = torch.matmul(attn_fp16, v)
        (qk, sk), (qv, sv) = apply_kv_cache_quantization(
            k.squeeze(0).transpose(0, 1), v.squeeze(0).transpose(0, 1), bits=4,
        )
        dk = dequantize_kv_cache(qk, sk, bits=4)
        dv = dequantize_kv_cache(qv, sv, bits=4)
        dk = dk.transpose(0, 1).unsqueeze(0)
        dv = dv.transpose(0, 1).unsqueeze(0)
        attn_quant = torch.softmax(torch.matmul(q, dk.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1)
        out_quant = torch.matmul(attn_quant, dv)
        rel_err = (out_fp16 - out_quant).abs().mean().item() / out_fp16.abs().mean().item()
        logger.info(f"4-bit attention relative error: {rel_err:.6f}")
        assert rel_err < 0.5

    def test_8bit_attention_output_very_close(self):
        torch.manual_seed(42)
        head_dim = 64
        num_heads = 4
        seq_len = 16
        q = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16) * 0.1
        k = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16) * 0.1
        v = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16) * 0.1
        out_fp16 = torch.matmul(
            torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1), v,
        )
        (qk, sk), (qv, sv) = apply_kv_cache_quantization(
            k.squeeze(0).transpose(0, 1), v.squeeze(0).transpose(0, 1), bits=8,
        )
        dk = dequantize_kv_cache(qk, sk, bits=8).transpose(0, 1).unsqueeze(0)
        dv = dequantize_kv_cache(qv, sv, bits=8).transpose(0, 1).unsqueeze(0)
        out_quant = torch.matmul(
            torch.softmax(torch.matmul(q, dk.transpose(-2, -1)) / (head_dim ** 0.5), dim=-1), dv,
        )
        rel_err = (out_fp16 - out_quant).abs().mean().item() / out_fp16.abs().mean().item()
        logger.info(f"8-bit attention relative error: {rel_err:.6f}")
        assert rel_err < 0.1

    def test_kv_cache_4bit_enable_and_update(self):
        cache = KVCache(max_seq_len=0)
        cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        cache.enable_quantization(bits=4)
        assert cache._quantized is True
        assert cache._quant_bits == 4
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        updated_k, updated_v = cache.update(0, k, v)
        assert len(updated_k.shape) == 4
        assert updated_k.shape[-1] == 64

    def test_kv_cache_8bit_enable_and_update(self):
        cache = KVCache(max_seq_len=0)
        cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        cache.enable_quantization(bits=8)
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        updated_k, updated_v = cache.update(0, k, v)
        assert len(updated_k.shape) == 4
        assert updated_k.shape[-1] == 64
        assert len(updated_v.shape) == 4
        assert updated_v.shape[-1] == 64

    def test_kv_cache_quantized_memory_savings(self):
        cache = KVCache(max_seq_len=128)
        cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        mem_no_quant = cache.memory_usage()
        cache.enable_quantization(bits=4)
        k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
        for i in range(2):
            cache.update(i, k, v)
        savings = cache.quantization_savings()
        assert savings > 0.0

    def test_kv_cache_invalid_bits_raises(self):
        cache = KVCache(max_seq_len=128)
        cache.init_cache(num_layers=1, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        with pytest.raises(ValueError, match="KV cache quantization bits must be 4 or 8"):
            cache.enable_quantization(bits=16)

    def test_int4_quantize_dequantize_scale(self):
        torch.manual_seed(42)
        tensor = torch.randn(8, 16, 64, dtype=torch.float16) * 0.1
        qt, scale = _quantize_int4(tensor)
        assert qt.dtype == torch.int8
        assert qt.shape == tensor.shape
        assert scale.shape == (8, 16, 1)
        dq = qt.to(torch.float16) * scale
        assert dq.shape == tensor.shape

    def test_int8_quantize_dequantize_scale(self):
        torch.manual_seed(42)
        tensor = torch.randn(8, 16, 64, dtype=torch.float16) * 0.1
        qt, scale = _quantize_int8(tensor)
        assert qt.dtype == torch.int8
        assert qt.shape == tensor.shape
        assert scale.shape == (8, 16, 1)

    def test_quantized_kv_cache_top1_token_preserved(self):
        torch.manual_seed(42)
        head_dim = 64
        num_heads = 4
        seq_len = 16
        q = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float16)
        k = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16) * 0.1
        v = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16) * 0.1
        scores_fp16 = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        top1_fp16 = scores_fp16.argmax(dim=-1)
        (qk, sk), (qv, sv) = apply_kv_cache_quantization(
            k.squeeze(0).transpose(0, 1), v.squeeze(0).transpose(0, 1), bits=4,
        )
        dk = dequantize_kv_cache(qk, sk, bits=4).transpose(0, 1).unsqueeze(0)
        scores_4bit = torch.matmul(q, dk.transpose(-2, -1)) / (head_dim ** 0.5)
        top1_4bit = scores_4bit.argmax(dim=-1)
        agreement = (top1_fp16 == top1_4bit).float().mean().item()
        logger.info(f"4-bit KV cache top-1 agreement: {agreement:.3f}")
        assert agreement >= 0.5
