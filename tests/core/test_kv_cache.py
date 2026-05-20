"""KV Cache tests for cache operations, memory management, and edge cases.

Tests:
- KVCache: init_cache, update, get, get_all, set_all, to, sequence_length, clear, memory_usage
- KVCacheManager: create, get, update, delete, clear_all, active_requests, total_memory_usage
- Serialization: serialize_kv_cache, deserialize_kv_cache
- Edge cases: no batch_size validation, set_all with empty list, sequence_length on empty cache

Run: pytest tests/core/test_kv_cache.py -v
"""

import pytest
import torch

from distllm.core.kv_cache import (
    KVCache,
    KVCacheManager,
    deserialize_kv_cache,
    serialize_kv_cache,
)

# ============================================================
# KVCache Tests
# ============================================================


class TestKVCacheInit:
    """Tests for KVCache initialization."""

    def test_init_cache_creates_tensors(self):
        """init_cache should create zero tensors for each layer."""
        cache = KVCache()
        cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=8, device="cpu")

        assert len(cache.cache) == 2
        for k, v in cache.cache:
            assert k.shape == (1, 4, 1, 8)  # (batch, heads, seq_len, head_dim)
            assert v.shape == (1, 4, 1, 8)
            assert k.device.type == "cpu"

    def test_init_cache_sets_num_layers(self):
        """num_layers should be set after init."""
        cache = KVCache()
        cache.init_cache(num_layers=5, batch_size=1, num_heads=4, head_dim=8, device="cpu")
        assert cache.num_layers == 5

    def test_init_cache_multiple_batches(self):
        """Should support batch_size > 1."""
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=4, num_heads=2, head_dim=16, device="cpu")
        k, v = cache.cache[0]
        assert k.shape[0] == 4


class TestKVCacheUpdate:
    """Tests for KVCache update operations."""

    def test_update_appends_to_cache(self):
        """update should concatenate new key/value states."""
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=4, device="cpu")

        new_k = torch.randn(1, 2, 3, 4)  # seq_len=3
        new_v = torch.randn(1, 2, 3, 4)

        k, v = cache.update(0, new_k, new_v)

        assert k.shape == (1, 2, 3, 4)
        assert v.shape == (1, 2, 3, 4)
        assert torch.equal(k, new_k)

    def test_update_accumulates(self):
        """Multiple updates should accumulate sequence length."""
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=4, device="cpu")

        cache.update(0, torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
        cache.update(0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))

        k, v = cache.cache[0]
        assert k.shape == (1, 2, 5, 4)  # 2 + 3 = 5

    def test_update_invalid_layer_index(self):
        """Should raise IndexError for out-of-range layer."""
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=4, device="cpu")

        with pytest.raises(IndexError):
            cache.update(5, torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))


class TestKVCacheGet:
    """Tests for KVCache retrieval."""

    def test_get_valid_layer(self, sample_kv_cache):
        """get should return (k, v) for valid layer index."""
        k, v = sample_kv_cache.get(0)
        assert k is not None
        assert v is not None
        assert isinstance(k, torch.Tensor)

    def test_get_invalid_layer(self, sample_kv_cache):
        """get should return None for invalid layer index."""
        result = sample_kv_cache.get(100)
        assert result is None

    def test_get_all(self, sample_kv_cache):
        """get_all should return all layer caches."""
        all_cache = sample_kv_cache.get_all()
        assert len(all_cache) == 2
        assert all(isinstance(item, tuple) for item in all_cache)


class TestKVCacheSetAll:
    """Tests for KVCache set_all operations."""

    def test_set_all_replaces_cache(self):
        """set_all should replace entire cache."""
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=4, device="cpu")

        new_layers = [
            (torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)),
            (torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)),
        ]
        cache.set_all(new_layers)

        assert len(cache.cache) == 2
        assert cache.num_layers == 2

    def test_set_all_empty_list(self):
        """set_all with empty list should result in empty cache.

        Edge case: no validation, just accepts empty list.
        """
        cache = KVCache()
        cache.init_cache(num_layers=2, batch_size=1, num_heads=2, head_dim=4, device="cpu")

        cache.set_all([])

        assert len(cache.cache) == 0
        assert cache.num_layers == 0


class TestKVCacheToDevice:
    """Tests for KVCache device movement."""

    def test_to_creates_new_cache(self, sample_kv_cache):
        """to should return a new KVCache instance."""
        new_cache = sample_kv_cache.to("cpu")

        assert isinstance(new_cache, KVCache)
        assert new_cache is not sample_kv_cache

    def test_to_preserves_data(self, sample_kv_cache):
        """to should preserve cache data."""
        new_cache = sample_kv_cache.to("cpu")

        assert len(new_cache.cache) == len(sample_kv_cache.cache)
        assert new_cache.num_layers == sample_kv_cache.num_layers


class TestKVCacheSequenceLength:
    """Tests for sequence_length property."""

    def test_sequence_length_empty(self):
        """sequence_length should return 0 for empty cache."""
        cache = KVCache()
        assert cache.sequence_length == 0

    def test_sequence_length_after_update(self):
        """sequence_length should return seq_len dimension."""
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=4, device="cpu")

        cache.update(0, torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4))

        assert cache.sequence_length == 7

    def test_sequence_length_assumes_cache_exists(self):
        """sequence_length assumes cache[0][0] exists.

        Edge case: if cache is empty, returns 0 (handled by early return).
        """
        cache = KVCache()
        # Empty cache - should not crash
        assert cache.sequence_length == 0


class TestKVCacheClear:
    """Tests for cache clearing."""

    def test_clear_resets_cache(self, sample_kv_cache):
        """clear should empty the cache."""
        sample_kv_cache.clear()

        assert len(sample_kv_cache.cache) == 0
        assert sample_kv_cache.num_layers == 0


class TestKVCacheMemoryUsage:
    """Tests for memory_usage calculation."""

    def test_memory_usage_empty(self):
        """memory_usage should return 0 for empty cache."""
        cache = KVCache()
        assert cache.memory_usage() == 0

    def test_memory_usage_after_update(self):
        """memory_usage should return total bytes."""
        cache = KVCache()
        cache.init_cache(num_layers=1, batch_size=1, num_heads=2, head_dim=4, device="cpu")

        cache.update(0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))

        usage = cache.memory_usage()
        # k: 1*2*3*4 = 24 elements, v: 24 elements, each float32 = 4 bytes
        # total = 48 * 4 = 192 bytes
        assert usage == 192


# ============================================================
# KVCacheManager Tests
# ============================================================


class TestKVCacheManager:
    """Tests for KVCacheManager operations."""

    def test_create_cache(self):
        """create should create and store a KVCache."""
        manager = KVCacheManager()
        cache = manager.create(
            "req-1", num_layers=2, batch_size=1, num_heads=4, head_dim=8, device="cpu"
        )

        assert isinstance(cache, KVCache)
        assert manager.active_requests == 1

    def test_get_cache(self, kv_cache_manager):
        """get should return cache for known request."""
        cache = kv_cache_manager.get("req-0")
        assert cache is not None
        assert isinstance(cache, KVCache)

    def test_get_unknown_request(self, kv_cache_manager):
        """get should return None for unknown request."""
        cache = kv_cache_manager.get("unknown")
        assert cache is None

    def test_update_cache(self, kv_cache_manager):
        """update should update the underlying KVCache."""
        new_k = torch.randn(1, 4, 2, 8)
        new_v = torch.randn(1, 4, 2, 8)

        result = kv_cache_manager.update("req-0", 0, new_k, new_v)

        assert result is not None
        k, v = result
        assert k.shape == (1, 4, 2, 8)

    def test_update_unknown_request(self, kv_cache_manager):
        """update should return None for unknown request."""
        k = torch.randn(1, 4, 1, 8)
        v = torch.randn(1, 4, 1, 8)
        result = kv_cache_manager.update("unknown", 0, k, v)
        assert result is None

    def test_delete_cache(self, kv_cache_manager):
        """delete should remove the cache."""
        assert kv_cache_manager.active_requests == 3

        kv_cache_manager.delete("req-0")

        assert kv_cache_manager.active_requests == 2
        assert kv_cache_manager.get("req-0") is None

    def test_delete_unknown_request(self, kv_cache_manager):
        """delete should handle unknown request gracefully."""
        kv_cache_manager.delete("unknown")  # Should not raise
        assert kv_cache_manager.active_requests == 3

    def test_clear_all(self, kv_cache_manager):
        """clear_all should remove all caches."""
        kv_cache_manager.clear_all()
        assert kv_cache_manager.active_requests == 0
        assert kv_cache_manager.caches == {}

    def test_active_requests(self, kv_cache_manager):
        """active_requests should return count of cached requests."""
        assert kv_cache_manager.active_requests == 3

    def test_total_memory_usage(self, kv_cache_manager):
        """total_memory_usage should sum all cache usages."""
        total = kv_cache_manager.total_memory_usage()
        assert total >= 0


# ============================================================
# Serialization Tests
# ============================================================


class TestSerializeKVCache:
    """Tests for KV cache serialization."""

    def test_serialize_returns_dict(self, sample_kv_cache):
        """serialize_kv_cache should return a dict with 'layers' key."""
        data = serialize_kv_cache(sample_kv_cache)

        assert isinstance(data, dict)
        assert "layers" in data
        assert len(data["layers"]) == 2

    def test_serialize_preserves_tensor_data(self, sample_kv_cache):
        """Serialized tensors should match original."""
        data = serialize_kv_cache(sample_kv_cache)

        # Our sample cache was initialized but not updated, so tensors are empty
        # Let's update first
        sample_kv_cache.update(0, torch.randn(1, 4, 3, 8), torch.randn(1, 4, 3, 8))
        data = serialize_kv_cache(sample_kv_cache)

        assert data["layers"][0]["key"].shape[2] == 3  # seq_len=3


class TestDeserializeKVCache:
    """Tests for KV cache deserialization."""

    def test_deserialize_creates_cache(self, sample_kv_cache):
        """deserialize_kv_cache should create a KVCache from data."""
        data = serialize_kv_cache(sample_kv_cache)
        new_cache = deserialize_kv_cache(data)

        assert isinstance(new_cache, KVCache)
        assert new_cache.num_layers == sample_kv_cache.num_layers

    def test_roundtrip(self):
        """Serialize -> deserialize should preserve data."""
        cache = KVCache()
        cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=8, device="cpu")
        cache.update(0, torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8))

        data = serialize_kv_cache(cache)
        new_cache = deserialize_kv_cache(data)

        assert len(new_cache.cache) == 2
        assert new_cache.cache[0][0].shape == cache.cache[0][0].shape
