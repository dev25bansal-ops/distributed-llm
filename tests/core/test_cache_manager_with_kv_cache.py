"""Integration test: CacheManager with KVCache end-to-end.

Tests the full lifecycle: create KV cache → store in CacheManager →
lookup → verify data integrity.
"""

import torch

from distllm.core.cache_manager import CacheManager
from distllm.core.kv_cache import KVCache, KVCacheManager


class TestCacheManagerWithKVCache:
    """End-to-end integration tests."""

    def test_create_store_lookup_roundtrip(self):
        """Create KV cache, store in CacheManager, retrieve."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)

        # Create a KV cache
        cache = KVCache()
        cache.init_cache(num_layers=4, batch_size=1, num_heads=8, head_dim=64, device="cpu")

        # Store in CacheManager
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        cm.store_prefix(tokens, cache)

        # Retrieve
        match_len, result = cm.lookup_prefix(tokens)
        assert match_len == 16
        assert result is cache

    def test_kv_cache_manager_create_and_cache(self):
        """KVCacheManager creates cache, store in CacheManager, retrieve."""
        kv_mgr = KVCacheManager()
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)

        # Create via KVCacheManager
        cache = kv_mgr.create(
            request_id="req-1",
            num_layers=4,
            batch_size=1,
            num_heads=8,
            head_dim=64,
            device="cpu",
        )

        # Store in CacheManager
        tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
        cm.store_prefix(tokens, cache)

        # Retrieve and verify it's the same cache
        match_len, result = cm.lookup_prefix(tokens)
        assert match_len == 16
        assert result is cache

    def test_kv_cache_update_then_cache(self):
        """Update KV cache, then store in CacheManager."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)

        cache = KVCache()
        cache.init_cache(num_layers=4, batch_size=1, num_heads=8, head_dim=64, device="cpu")

        # Update some layers
        for layer in range(4):
            new_k = torch.randn(1, 8, 10, 64)
            new_v = torch.randn(1, 8, 10, 64)
            cache.update(layer, new_k, new_v)

        # Store in CacheManager
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        cm.store_prefix(tokens, cache)

        # Retrieve and verify data is intact
        match_len, result = cm.lookup_prefix(tokens)
        assert match_len == 16
        assert result is cache
        assert result.sequence_length == 10

    def test_multiple_caches_different_prefixes(self):
        """Store multiple caches with different prefixes."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)

        caches = {}
        for i in range(5):
            cache = KVCache()
            cache.init_cache(num_layers=2, batch_size=1, num_heads=4, head_dim=32, device="cpu")
            tokens = [i, i + 100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            cm.store_prefix(tokens, cache)
            caches[i] = (tokens, cache)

        # Verify each cache is retrievable
        for i, (tokens, expected_cache) in caches.items():
            match_len, result = cm.lookup_prefix(tokens)
            assert match_len == 16
            assert result is expected_cache

    def test_cache_slice_then_store(self):
        """Slice a KV cache, store the slice in CacheManager."""
        cm = CacheManager(prefix_cache_enabled=True, prefix_cache_min_prefix_len=2)

        cache = KVCache()
        cache.init_cache(num_layers=8, batch_size=1, num_heads=8, head_dim=64, device="cpu")

        # Slice layers 2-5
        sliced = cache.slice(2, 5)
        assert sliced.num_layers == 3

        # Store slice in CacheManager
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        cm.store_prefix(tokens, sliced)

        match_len, result = cm.lookup_prefix(tokens)
        assert match_len == 16
        assert result is sliced
        assert result.num_layers == 3

    def test_eviction_score_tracks_access(self):
        """KVCacheManager eviction score should track access patterns."""
        kv_mgr = KVCacheManager()

        cache1 = kv_mgr.create("req-1", 4, 1, 8, 64)
        cache2 = kv_mgr.create("req-2", 4, 1, 8, 64)

        # Access req-1 more
        for _ in range(10):
            kv_mgr.get("req-1")
        kv_mgr.get("req-2")

        score1 = kv_mgr.eviction_score("req-1")
        score2 = kv_mgr.eviction_score("req-2")

        # req-1 has been accessed more recently and more frequently
        assert score1 >= score2

    def test_evict_lowest_score(self):
        """evict_lowest_score should remove the least valuable cache."""
        kv_mgr = KVCacheManager()

        kv_mgr.create("req-1", 4, 1, 8, 64)
        kv_mgr.create("req-2", 4, 1, 8, 64)

        # Access req-1 more
        for _ in range(10):
            kv_mgr.get("req-1")

        evicted = kv_mgr.evict_lowest_score()
        assert evicted == "req-2"
        assert kv_mgr.get("req-2") is None
        assert kv_mgr.get("req-1") is not None
