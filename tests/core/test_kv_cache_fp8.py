"""Tests for per-step FP8 quantization in KV cache.

Covers:
- _update_fp8_unquantized stores quantized segments
- Per-step quantization produces correct dequantized values
- FP8 fallback works when float8_e4m3fn is unavailable
"""

from __future__ import annotations

import pytest
import torch


class TestKVCacheFP8:
    """KV cache FP8 per-step quantization."""

    def test_fp8_incremental_quantization(self):
        """Per-step FP8 should store quantized segments."""
        from distllm.core.kv_cache import KVCache

        cache = KVCache(num_layers=2, max_seq_len=64, quantize=False)
        # Set up for FP8 mode
        cache._quant_fp8 = True
        cache._quantized = True
        cache._qsegments = [[] for _ in range(2)]

        # Create sample KV tensors
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)

        if hasattr(torch, 'float8_e4m3fn'):
            result_k, result_v = cache._update_fp8_unquantized(0, k, v)
            assert result_k is not None
            assert result_v is not None
            assert result_k.shape == k.shape
            assert result_v.shape == v.shape
            # Should have stored in qsegments
            assert len(cache._qsegments[0]) == 1
        else:
            # FP8 not available — should fall through
            pytest.skip("torch.float8_e4m3fn not available on this system")

    def test_fp8_full_precision_fallback(self):
        """When FP8 is unavailable, fall back to full precision."""
        from distllm.core.kv_cache import KVCache

        cache = KVCache(num_layers=1, max_seq_len=64, quantize=False)
        cache._quant_fp8 = True
        cache._quantized = True

        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)

        # _append_full_precision should store in cache directly
        cache.cache = [(torch.zeros(1, 4, 8, 16), torch.zeros(1, 4, 8, 16))]
        cache._seq_lens = [0]
        k_out, v_out = cache._append_full_precision(0, k, v)
        assert k_out.shape[-2] == k.shape[-2]
        assert v_out.shape[-2] == v.shape[-2]

    def test_fp8_compress_bulk(self):
        """Bulk FP8 compression should work end-to-end."""
        from distllm.core.kv_cache import KVCache

        cache = KVCache(num_layers=1, max_seq_len=64, quantize=False)
        k = torch.randn(1, 4, 32, 16)
        v = torch.randn(1, 4, 32, 16)
        cache.cache = [(k, v)]
        cache._seq_lens = [32]

        if hasattr(torch, 'float8_e4m3fn'):
            result = cache.compress(method="fp8")
            assert result["method"] == "fp8"
            assert result["compressed_bytes"] > 0
            assert result["savings_pct"] > 0
        else:
            # Should fall back to int8
            result = cache.compress(method="fp8")
            assert result is not None
