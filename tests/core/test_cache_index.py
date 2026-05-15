"""Tests for prefix-hash-based KV cache index."""

import pytest
from distllm.core.cache_index import CacheIndex


class TestCacheIndex:
    """Test CacheIndex basic operations."""

    def test_store_and_lookup(self):
        idx = CacheIndex()
        idx.store("h123", "node-1", "ref-1")
        assert idx.lookup("h123") == "node-1"

    def test_lookup_miss(self):
        idx = CacheIndex()
        assert idx.lookup("h999") is None

    def test_store_multiple_nodes_same_hash(self):
        idx = CacheIndex()
        idx.store("h123", "node-1", "ref-1")
        idx.store("h123", "node-2", "ref-2")
        nodes = idx.lookup_all("h123")
        assert set(nodes) == {"node-1", "node-2"}

    def test_remove_single_node(self):
        idx = CacheIndex()
        idx.store("h123", "node-1", "ref-1")
        idx.store("h123", "node-2", "ref-2")
        idx.remove("h123", node_id="node-1")
        assert idx.lookup("h123") == "node-2"

    def test_remove_all_replicas(self):
        idx = CacheIndex()
        idx.store("h123", "node-1", "ref-1")
        idx.store("h123", "node-2", "ref-2")
        idx.remove("h123")
        assert idx.lookup("h123") is None

    def test_remove_nonexistent(self):
        idx = CacheIndex()
        idx.remove("h999")  # Should not raise

    def test_get_ref(self):
        idx = CacheIndex()
        idx.store("h123", "node-1", "ref-1")
        assert idx.get_ref("h123") == "ref-1"
        assert idx.get_ref("h999") is None

    def test_clear(self):
        idx = CacheIndex()
        idx.store("h123", "node-1", "ref-1")
        idx.store("h456", "node-2", "ref-2")
        idx.clear()
        assert idx.lookup("h123") is None
        assert idx.lookup("h456") is None


class TestCacheIndexHashing:
    """Test token hashing."""

    def test_same_tokens_same_hash(self):
        idx = CacheIndex()
        h1 = idx.index_tokens([1, 2, 3])
        h2 = idx.index_tokens([1, 2, 3])
        assert h1 == h2

    def test_different_tokens_different_hash(self):
        idx = CacheIndex()
        h1 = idx.index_tokens([1, 2, 3])
        h2 = idx.index_tokens([4, 5, 6])
        assert h1 != h2

    def test_empty_tokens(self):
        idx = CacheIndex()
        h = idx.index_tokens([])
        assert h.startswith("h")


class TestCacheIndexStats:
    """Test statistics tracking."""

    def test_stats_empty(self):
        idx = CacheIndex()
        stats = idx.stats()
        assert stats["total_entries"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0

    def test_stats_with_entries(self):
        idx = CacheIndex()
        idx.store("h1", "node-1", "ref-1")
        idx.store("h2", "node-2", "ref-2")
        stats = idx.stats()
        assert stats["total_entries"] == 2
        assert stats["unique_nodes"] == 2

    def test_hit_miss_tracking(self):
        idx = CacheIndex()
        idx.store("h1", "node-1", "ref-1")
        idx.lookup("h1")  # hit
        idx.lookup("h99")  # miss
        stats = idx.stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
