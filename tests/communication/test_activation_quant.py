"""Tests for activation quantization in serializers.py (INT8, INT4, FP8 paths)."""

import pytest
import torch

from distllm.communication.serializers import (
    set_activation_quant,
    quantize_activation,
    dequantize_activation,
)


class TestActivationQuantization:
    def setup_method(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)

    def test_disabled_returns_original(self):
        set_activation_quant(enabled=False)
        t = torch.randn(4, 64, dtype=torch.float16)
        q, s = quantize_activation(t)
        assert q is t
        assert s is None

    def test_int8_quant_shape(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)
        t = torch.randn(4, 64, dtype=torch.float16)
        q, s = quantize_activation(t)
        assert q.shape == t.shape
        assert q.dtype == torch.int8
        assert s is not None and s.numel() == 1

    def test_int8_roundtrip(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)
        t = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
        q, s = quantize_activation(t)
        dq = dequantize_activation(q, s, torch.float16)
        assert dq.dtype == torch.float16
        assert dq.shape == t.shape
        error = (t - dq).abs().mean().item()
        assert error < 0.1, f"INT8 roundtrip error too high: {error}"

    def test_int8_negative_values(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)
        t = torch.tensor([[-4.0, -2.0, 0.0, 2.0, 4.0]], dtype=torch.float16)
        q, s = quantize_activation(t)
        dq = dequantize_activation(q, s, torch.float16)
        error = (t - dq).abs().mean().item()
        assert error < 0.2

    def test_int8_zero_tensor(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)
        t = torch.zeros(4, 64, dtype=torch.float16)
        q, s = quantize_activation(t)
        assert s is not None
        dq = dequantize_activation(q, s, torch.float16)
        assert dq.abs().max().item() < 0.01

    def test_int4_quant_shape(self):
        set_activation_quant(enabled=True, bits=4, use_fp8=False)
        t = torch.randn(4, 32, dtype=torch.float16)
        q, s = quantize_activation(t)
        assert q.shape == t.shape
        assert q.dtype == torch.int8  # INT4 stored as int8
        assert s is not None and s.dim() == 1

    def test_int4_roundtrip(self):
        set_activation_quant(enabled=True, bits=4, use_fp8=False)
        t = torch.randn(4, 32, dtype=torch.float16)
        q, s = quantize_activation(t)
        dq = dequantize_activation(q, s, torch.float16)
        assert dq.shape == t.shape
        error = (t - dq).abs().mean().item()
        assert error < 0.5, f"INT4 roundtrip error too high: {error}"

    def test_dequantize_no_scale(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)
        t = torch.randn(4, dtype=torch.float32)
        result = dequantize_activation(t, None, torch.float16)
        assert result.dtype == torch.float16
        assert result.shape == t.shape

    def test_dequantize_different_dtype(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)
        t = torch.randn(4, dtype=torch.float32)
        q, s = quantize_activation(t)
        dq = dequantize_activation(q, s, torch.bfloat16)
        assert dq.dtype == torch.bfloat16

    def test_large_tensor_int8(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)
        t = torch.randn(128, 128, dtype=torch.float16)
        q, s = quantize_activation(t)
        assert q.shape == t.shape
        dq = dequantize_activation(q, s, torch.float16)
        error = (t - dq).abs().mean().item()
        assert error < 0.1

    def test_large_tensor_int4(self):
        set_activation_quant(enabled=True, bits=4, use_fp8=False)
        # Use size divisible by group_size (32)
        t = torch.randn(8, 64, dtype=torch.float16)
        q, s = quantize_activation(t)
        assert q.shape == t.shape
        dq = dequantize_activation(q, s, torch.float16)
        assert dq.shape == t.shape

    def test_fp8_path_when_available(self):
        """FP8 path should be used when use_fp8=True and CUDA is available."""
        has_fp8 = hasattr(torch, "float8_e4m3fn") and torch.cuda.is_available()
        set_activation_quant(enabled=True, bits=8, use_fp8=True)
        t = torch.randn(4, 64, dtype=torch.float16)
        q, s = quantize_activation(t)
        if has_fp8:
            assert q.dtype == torch.float8_e4m3fn
            assert s is None
        else:
            assert s is not None

    def test_fp8_dequant(self):
        has_fp8 = hasattr(torch, "float8_e4m3fn") and torch.cuda.is_available()
        if not has_fp8:
            pytest.skip("FP8 not available")

        set_activation_quant(enabled=True, bits=8, use_fp8=True)
        t = torch.randn(4, 64, dtype=torch.float16)
        q, s = quantize_activation(t)
        dq = dequantize_activation(q, s, torch.float16)
        assert dq.dtype == torch.float16
        assert dq.shape == t.shape

    def test_mixed_configs(self):
        set_activation_quant(enabled=True, bits=8, use_fp8=False)
        t = torch.randn(4, 4, dtype=torch.float16)
        q1, s1 = quantize_activation(t)
        assert q1.dtype == torch.int8

        set_activation_quant(enabled=True, bits=4, use_fp8=False)
        t_large = torch.randn(8, 64, dtype=torch.float16)
        q2, s2 = quantize_activation(t_large)
        assert q2.dtype == torch.int8
        assert s2.numel() > 1

        set_activation_quant(enabled=False)
        q3, s3 = quantize_activation(t)
        assert q3 is t
        assert s3 is None
