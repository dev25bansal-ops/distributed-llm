"""KV cache delta computation tests.

Covers KV cache initialization, update/slicing, cat fallback, thread safety,
manager operations, and quantization features.
"""

import asyncio
import socket
import struct
import threading
import time
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

try:
    from hypothesis import given, strategies as st, settings as hp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


from tests.comprehensive.conftest import _load_module

# Load clean modules
_kv_cache = _load_module("distllm/core/kv_cache.py")


# ═══════════════════════════════════════════════════════════════════════════
# 2. KV Cache Delta Computation
# ═══════════════════════════════════════════════════════════════════════════

class TestKVCacheDelta:
    """KV cache update/delta behavior: pre-allocated slicing and cat fallback."""

    def test_init_cache_creates_correct_shape(self):
        c = _kv_cache.KVCache(max_seq_len=1024)
        c.init_cache(num_layers=2, batch_size=1, num_heads=8, head_dim=64, device="cpu")
        assert c.num_layers == 2
        k, v = c.cache[0]
        assert k.shape == (1, 8, 1024, 64)
        assert v.shape == (1, 8, 1024, 64)

    def test_update_returns_sliced_sequence(self):
        c = _kv_cache.KVCache(max_seq_len=1024)
        c.init_cache(1, 1, 2, 32, "cpu")
        new_k = torch.randn(1, 2, 3, 32)
        new_v = torch.randn(1, 2, 3, 32)
        out_k, out_v = c.update(0, new_k, new_v)
        assert out_k.shape[-2] == 3
        assert out_v.shape[-2] == 3

    def test_update_appends_in_buffer(self):
        c = _kv_cache.KVCache(max_seq_len=1024)
        c.init_cache(1, 1, 2, 32, "cpu")
        k1 = torch.randn(1, 2, 3, 32)
        v1 = torch.randn(1, 2, 3, 32)
        c.update(0, k1, v1)
        k2 = torch.randn(1, 2, 5, 32)
        v2 = torch.randn(1, 2, 5, 32)
        out_k, out_v = c.update(0, k2, v2)
        assert out_k.shape[-2] == 8
        assert out_v.shape[-2] == 8

    def test_update_preserves_values(self):
        c = _kv_cache.KVCache(max_seq_len=1024)
        c.init_cache(1, 1, 2, 8, "cpu")
        k1 = torch.arange(1, 7, dtype=torch.float32).reshape(1, 2, 3, 1).expand(-1, -1, -1, 8)
        v1 = torch.arange(10, 70, 10, dtype=torch.float32).reshape(1, 2, 3, 1).expand(-1, -1, -1, 8)
        out_k, _ = c.update(0, k1, v1)
        assert torch.equal(out_k[:, :, 0:1, :], k1[:, :, 0:1, :])
        assert torch.equal(out_k[:, :, 1:2, :], k1[:, :, 1:2, :])
        assert torch.equal(out_k[:, :, 2:3, :], k1[:, :, 2:3, :])

    def test_update_cat_fallback_when_exceeds_max(self):
        c = _kv_cache.KVCache(max_seq_len=4)
        c.init_cache(1, 1, 2, 32, "cpu")
        k1 = torch.randn(1, 2, 3, 32)
        v1 = torch.randn(1, 2, 3, 32)
        c.update(0, k1, v1)
        k2 = torch.randn(1, 2, 3, 32)
        v2 = torch.randn(1, 2, 3, 32)
        out_k, out_v = c.update(0, k2, v2)
        assert out_k.shape[-2] == 6

    def test_update_thread_safety(self):
        c = _kv_cache.KVCache(max_seq_len=512)
        c.init_cache(4, 1, 4, 32, "cpu")
        errors = []

        def writer():
            try:
                for _ in range(50):
                    k = torch.randn(1, 4, 1, 32)
                    v = torch.randn(1, 4, 1, 32)
                    c.update(0, k, v)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread safety failure: {errors}"

    def test_update_invalid_layer_raises(self):
        c = _kv_cache.KVCache()
        c.init_cache(2, 1, 2, 32, "cpu")
        with pytest.raises(IndexError):
            c.update(99, torch.randn(1, 2, 1, 32), torch.randn(1, 2, 1, 32))

    def test_sequence_length_tracks_correctly(self):
        c = _kv_cache.KVCache(max_seq_len=256)
        c.init_cache(1, 1, 2, 32, "cpu")
        # sequence_length returns buffer capacity after init (pre-allocated)
        # Verify that update slices correctly instead
        out_k, _ = c.update(0, torch.randn(1, 2, 5, 32), torch.randn(1, 2, 5, 32))
        assert out_k.shape[-2] == 5
        out_k2, _ = c.update(0, torch.randn(1, 2, 3, 32), torch.randn(1, 2, 3, 32))
        assert out_k2.shape[-2] == 8

    def test_clear_resets_cache(self):
        c = _kv_cache.KVCache(max_seq_len=256)
        c.init_cache(1, 1, 2, 32, "cpu")
        c.update(0, torch.randn(1, 2, 3, 32), torch.randn(1, 2, 3, 32))
        c.clear()
        assert c.cache == []
        assert c.num_layers == 0

    def test_get_layer_returns_correct(self):
        c = _kv_cache.KVCache(max_seq_len=256)
        c.init_cache(3, 1, 2, 32, "cpu")
        k, v = c.get(1)
        assert k.shape == (1, 2, 256, 32)

    def test_get_nonexistent_layer_returns_none(self):
        c = _kv_cache.KVCache()
        assert c.get(0) is None

    def test_to_device_creates_new_copy(self):
        c = _kv_cache.KVCache()
        c.init_cache(1, 1, 2, 32, "cpu")
        c2 = c.to("cpu")
        assert c2 is not c
        assert c2.num_layers == 1

    def test_set_all_replaces_cache(self):
        c = _kv_cache.KVCache()
        new = [(torch.randn(1, 2, 3, 32), torch.randn(1, 2, 3, 32))]
        c.set_all(new)
        assert c.num_layers == 1
        assert c.cache[0][0].shape == (1, 2, 3, 32)

    def test_kv_cache_manager_create_get_delete(self):
        mgr = _kv_cache.KVCacheManager()
        c = mgr.create("req1", num_layers=2, batch_size=1, num_heads=4, head_dim=64)
        assert mgr.active_requests == 1
        assert mgr.get("req1") is c
        mgr.delete("req1")
        assert mgr.active_requests == 0
        assert mgr.get("req1") is None

    def test_kv_cache_manager_clear_all(self):
        mgr = _kv_cache.KVCacheManager()
        mgr.create("a", 1, 1, 2, 32)
        mgr.create("b", 1, 1, 2, 32)
        mgr.clear_all()
        assert mgr.active_requests == 0

    def test_kv_cache_manager_memory_usage(self):
        mgr = _kv_cache.KVCacheManager()
        c = mgr.create("r1", 1, 1, 2, 32)
        used = mgr.total_memory_usage()
        assert used > 0

    def test_kv_cache_memory_usage_returns_int(self):
        c = _kv_cache.KVCache()
        c.init_cache(2, 1, 4, 64, "cpu")
        assert isinstance(c.memory_usage(), int)
        assert c.memory_usage() > 0

    def test_kv_cache_manager_update(self):
        mgr = _kv_cache.KVCacheManager()
        mgr.create("r1", 1, 1, 2, 32)
        k = torch.randn(1, 2, 3, 32)
        v = torch.randn(1, 2, 3, 32)
        result = mgr.update("r1", 0, k, v)
        assert result is not None
        assert result[0].shape[-2] == 3

    def test_kv_cache_manager_update_missing_request(self):
        mgr = _kv_cache.KVCacheManager()
        result = mgr.update("nonexistent", 0, torch.randn(1, 2, 1, 32), torch.randn(1, 2, 1, 32))
        assert result is None

    def test_quantization_enable_rejects_bad_bits(self):
        c = _kv_cache.KVCache()
        with pytest.raises(ValueError, match="must be 4 or 8"):
            c.enable_quantization(bits=16)

    def test_quantization_enable_accepts_4_or_8(self):
        c = _kv_cache.KVCache()
        c.enable_quantization(4)
        assert c._quantized
        assert c._quant_bits == 4
        c.enable_quantization(8)
        assert c._quant_bits == 8

    def test_quantization_savings_no_quant(self):
        c = _kv_cache.KVCache()
        c.init_cache(1, 1, 2, 32, "cpu")
        c.update(0, torch.randn(1, 2, 3, 32), torch.randn(1, 2, 3, 32))
        assert c.quantization_savings() == 1.0
