"""Tests for prefix-hash-based KV cache index."""

import threading

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


class TestCacheIndexGossip:
    """Multi-node / gossip-style scenarios."""

    def test_lookup_returns_one_of_replicas(self):
        idx = CacheIndex()
        idx.store("h1", "node-a", "ref-a")
        idx.store("h1", "node-b", "ref-b")
        node = idx.lookup("h1")
        assert node in ("node-a", "node-b")

    def test_lookup_all_returns_all_replicas(self):
        idx = CacheIndex()
        idx.store("h1", "node-a", "ref-a")
        idx.store("h1", "node-b", "ref-b")
        assert set(idx.lookup_all("h1")) == {"node-a", "node-b"}

    def test_lookup_all_empty(self):
        idx = CacheIndex()
        assert idx.lookup_all("h_miss") == []

    def test_lookup_is_deterministic_within_process(self):
        """next(iter(set)) returns the same element for the same set."""
        idx = CacheIndex()
        idx.store("h1", "node-a", "ref-a")
        idx.store("h1", "node-b", "ref-b")
        result = idx.lookup("h1")
        for _ in range(50):
            assert idx.lookup("h1") == result

    def test_duplicate_store_does_not_replicate(self):
        idx = CacheIndex()
        idx.store("h1", "node-a", "ref-a")
        idx.store("h1", "node-a", "ref-a")
        assert len(idx.lookup_all("h1")) == 1

    def test_remove_last_node_removes_hash(self):
        idx = CacheIndex()
        idx.store("h1", "node-a", "ref-a")
        idx.remove("h1", node_id="node-a")
        assert idx.lookup("h1") is None
        assert "h1" not in idx._index

    def test_remove_one_keeps_other_node(self):
        idx = CacheIndex()
        idx.store("h1", "node-a", "ref-a")
        idx.store("h1", "node-b", "ref-b")
        idx.remove("h1", node_id="node-a")
        assert idx.lookup("h1") == "node-b"

    def test_clear_resets_stats(self):
        idx = CacheIndex()
        idx.store("h1", "node-a", "ref-a")
        idx.lookup("h1")
        idx.lookup("h_miss")
        idx.clear()
        stats = idx.stats()
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        assert stats["total_entries"] == 0

    def test_overlapping_nodes_multiple_hashes(self):
        idx = CacheIndex()
        idx.store("h1", "shared-node", "ref-a")
        idx.store("h2", "shared-node", "ref-b")
        assert idx.lookup("h1") == "shared-node"
        assert idx.lookup("h2") == "shared-node"
        idx.remove("h1")
        assert idx.lookup("h1") is None
        assert idx.lookup("h2") == "shared-node"
        assert "shared-node" in idx.lookup_all("h2")


class TestCacheIndexConcurrent:
    """Concurrent access (CacheIndex is thread-unsafe; these verify no hard
    crashes but are not expected to be race-free)."""

    def test_concurrent_store_no_crash(self):
        idx = CacheIndex()

        def store(i):
            for j in range(500):
                idx.store(f"h{i}", f"node-{j}", f"ref-{j}")

        threads = [threading.Thread(target=store, args=(i,), daemon=True) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        stats = idx.stats()
        assert stats["total_entries"] <= 4

    def test_concurrent_mixed_ops_no_crash(self):
        idx = CacheIndex()

        def worker(i):
            for j in range(200):
                h = f"h{j % 50}"
                if j % 3 == 0:
                    idx.store(h, f"node-{i}", f"ref-{j}")
                elif j % 3 == 1:
                    idx.lookup(h)
                else:
                    idx.lookup_all(h)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert "total_entries" in idx.stats()
