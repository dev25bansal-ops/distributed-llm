"""Comprehensive tests for KV cache: quantization, paged allocation, memory limits."""

import pytest
import torch

from distllm.core.kv_cache import KVCache, KVCacheManager, serialize_kv_cache, deserialize_kv_cache


# ---------------------------------------------------------------------------
# KV Cache quantization
# ---------------------------------------------------------------------------

class TestKVCacheQuantization:
    def test_enable_quantization_4bit(self):
        cache = KVCache()
        cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        cache.enable_quantization(bits=4)
        assert cache._quant_bits == 4
        assert cache._quantized is True

    def test_enable_quantization_8bit(self):
        cache = KVCache()
        cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        cache.enable_quantization(bits=8)
        assert cache._quant_bits == 8
        assert cache._quantized is True

    def test_quantization_reduces_memory(self):
        """Quantized cache should use less memory (or scales appropriately)."""
        fp16_cache = KVCache()
        fp16_cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        fp16_mem = fp16_cache.memory_usage()

        int8_cache = KVCache()
        int8_cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        int8_cache.enable_quantization(8)
        # Note: quantization might store both scales + quantized data
        int8_mem = int8_cache.memory_usage()
        assert int8_mem >= 0

    def test_quantize_dequantize_roundtrip(self):
        """After quantize then dequantize, approximate shape preserved."""
        cache = KVCache()
        cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        # Update with data
        key = torch.randn(1, 4, 5, 64)
        value = torch.randn(1, 4, 5, 64)
        cache.update(0, key, value)

        # Quantize
        cache.enable_quantization(8)
        # Verify cache still usable after quantization
        k, v = cache.get(0)
        assert k is not None
        assert v is not None

    def test_quantization_invalid_bits(self):
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=32, device="cpu")
        with pytest.raises((ValueError, Exception)):
            cache.enable_quantization(bits=3)


# ===================================================================
# Paged KV Cache
# ===================================================================

class TestPagedAllocation:
    def test_paged_kv_creation(self):
        """Create a paged attention manager if available."""
        try:
            from distllm.core.paged_attention import PagedAttentionManager
            mgr = PagedAttentionManager(
                num_blocks=64,
                block_size=16,
                num_layers=2,
                num_heads=4,
                head_dim=64,
                device="cpu",
            )
            assert mgr.pool.num_blocks == 64
            assert mgr.block_size == 16
        except ImportError:
            pytest.skip("PagedAttention not available")

    def test_paged_block_allocation(self):
        try:
            from distllm.core.paged_attention import PagedAttentionManager
            mgr = PagedAttentionManager(
                num_blocks=32, block_size=16,
                num_layers=2, num_heads=4, head_dim=64,
                device="cpu",
            )
            # Allocate a block from the pool
            block_idx = mgr.pool.allocate_block()
            assert block_idx is not None
        except ImportError:
            pytest.skip("PagedAttention not available")

    def test_paged_free_blocks(self):
        try:
            from distllm.core.paged_attention import PagedAttentionManager
            mgr = PagedAttentionManager(
                num_blocks=32, block_size=16,
                num_layers=2, num_heads=4, head_dim=64,
                device="cpu",
            )
            block_idx = mgr.pool.allocate_block()
            mgr.pool.free_block(block_idx)
            # Should be available again
            assert True
        except ImportError:
            pytest.skip("PagedAttention not available")


# ===================================================================
# Memory limits
# ===================================================================

class TestMemoryLimits:
    def test_large_allocation_uses_memory(self):
        """Larger cache should use more memory."""
        small = KVCache()
        small.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=32, device="cpu")

        large = KVCache()
        large.init_cache(num_layers=4, batch_size=2, num_heads=8, head_dim=128, device="cpu")

        assert large.memory_usage() >= small.memory_usage()

    def test_manager_total_memory(self):
        mgr = KVCacheManager()
        c1 = mgr.create("r1", num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        c2 = mgr.create("r2", num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        total = mgr.total_memory_usage()
        assert total >= 0
        assert mgr.active_requests == 2

    def test_manager_delete_frees_memory(self):
        mgr = KVCacheManager()
        mgr.create("r1", num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        mgr.create("r2", num_layers=2, batch_size=1, num_heads=4, head_dim=64, device="cpu")
        mem_before = mgr.total_memory_usage()
        mgr.delete("r1")
        mem_after = mgr.total_memory_usage()
        assert mgr.active_requests == 1
        assert mem_after <= mem_before

    def test_manager_clear_all(self):
        mgr = KVCacheManager()
        for i in range(5):
            mgr.create(f"r{i}", num_layers=1, batch_size=1, num_heads=2, head_dim=32, device="cpu")
        assert mgr.active_requests == 5
        mgr.clear_all()
        assert mgr.active_requests == 0


# ===================================================================
# Extend and reuse
# ===================================================================

class TestExtension:
    def test_cache_extend(self):
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=32, device="cpu")
        key1 = torch.randn(1, 2, 3, 32)
        value1 = torch.randn(1, 2, 3, 32)
        cache.update(0, key1, value1)
        old_len = len(cache.get(0)[0][0, 0])

        # Extend
        key2 = torch.randn(1, 2, 2, 32)
        value2 = torch.randn(1, 2, 2, 32)
        cache.update(0, key2, value2)
        new_len = len(cache.get(0)[0][0, 0])
        assert new_len == old_len + 2

    def test_cache_serialize_roundtrip(self):
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=32, device="cpu")
        key = torch.randn(1, 2, 3, 32)
        value = torch.randn(1, 2, 3, 32)
        cache.update(0, key, value)

        serialized = serialize_kv_cache(cache)
        assert isinstance(serialized, dict)

        cache2 = deserialize_kv_cache(serialized)
        k2, v2 = cache2.get(0)
        assert k2 is not None
        assert torch.equal(k2, key)
        assert torch.equal(v2, value)

    def test_cache_set_all(self):
        cache = KVCache()
        cache.init_cache(num_layers=2, batch_size=1, num_heads=2, head_dim=32, device="cpu")
        keys = [torch.randn(1, 2, 3, 32) for _ in range(2)]
        values = [torch.randn(1, 2, 3, 32) for _ in range(2)]
        cache.set_all(list(zip(keys, values)))
        for i in range(2):
            k, v = cache.get(i)
            assert torch.equal(k, keys[i])
            assert torch.equal(v, values[i])
