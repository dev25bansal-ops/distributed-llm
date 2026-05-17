"""Tests for RadixTreeCache (trie-based prefix cache)."""

import pytest
import torch

from distllm.core.radix_tree_cache import RadixTreeCache, RadixNode


class TestRadixNode:
    def test_insert_and_lookup(self):
        root = RadixNode()
        root.insert([1, 2, 3], "kv_data")
        matched, kv = root.lookup([1, 2, 3, 4, 5])
        assert matched == 3
        assert kv == "kv_data"

    def test_lookup_partial(self):
        root = RadixNode()
        root.insert([1, 2, 3], "data_a")
        root.insert([1, 2, 3, 4], "data_b")
        matched, kv = root.lookup([1, 2, 3, 4, 5])
        assert matched == 4
        assert kv == "data_b"

    def test_lookup_miss(self):
        root = RadixNode()
        root.insert([5, 6], "data")
        matched, kv = root.lookup([1, 2, 3])
        assert matched == 0

    def test_find_shared_prefix(self):
        root = RadixNode()
        root.insert([1, 2, 3, 4], "data")
        shared = root.find_shared_prefix([1, 2, 3, 9, 9])
        assert shared == 3  # [1, 2, 3] match even though no KV at node 3

    def test_evict_lru(self):
        root = RadixNode()
        root.insert([1], "a")
        root.insert([2], "b")
        root.insert([3], "c")
        evicted = root.evict_lru(max_entries=2)
        assert evicted >= 1
        assert root._count_entries() <= 2

    def test_evict_path(self):
        root = RadixNode()
        root.insert([1, 2, 3], "data")
        assert root._evict_path([1, 2, 3]) is True
        assert root._evict_path([1, 2, 3]) is False

    def test_clear(self):
        root = RadixNode()
        root.insert([1, 2], "a")
        root.insert([3, 4], "b")
        root.clear()
        assert root._count_entries() == 0

    def test_stats(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        root.insert([1, 2, 4], "b")
        stats = root.stats()
        assert stats["total_entries"] == 2
        assert stats["max_depth"] >= 3
        assert stats["total_nodes"] >= 5


class TestRadixTreeCache:
    def test_lookup_empty(self):
        c = RadixTreeCache(min_prefix_len=2)
        matched, kv = c.lookup([1, 2, 3])
        assert matched == 0

    def test_lookup_short_sequence(self):
        c = RadixTreeCache(min_prefix_len=5)
        matched, kv = c.lookup([1, 2])
        assert matched == 0

    def test_store_and_lookup_hit(self):
        c = RadixTreeCache(min_prefix_len=2)
        kv = torch.randn(1, 4, 8)
        c.store([1, 2, 3, 4], kv)
        matched, result = c.lookup([1, 2, 3, 4, 5])
        assert matched == 4
        assert result is kv

    def test_store_and_lookup_longer_prefix(self):
        c = RadixTreeCache(min_prefix_len=2)
        kv_a = torch.randn(1, 4, 8)
        kv_b = torch.randn(1, 4, 8)
        c.store([1, 2, 3], kv_a)
        c.store([1, 2, 3, 4], kv_b)

        # Should return the longest match
        matched, result = c.lookup([1, 2, 3, 4, 5])
        assert matched == 4
        assert result is kv_b

    def test_miss_different_sequence(self):
        c = RadixTreeCache(min_prefix_len=2)
        c.store([1, 2, 3], torch.randn(1, 4, 8))
        matched, kv = c.lookup([5, 6, 7])
        assert matched == 0

    def test_find_shared_prefix(self):
        c = RadixTreeCache(min_prefix_len=2)
        c.store([1, 2, 3, 4], torch.randn(1, 4, 8))
        shared = c.find_shared_prefix([1, 2, 3, 9, 9])
        assert shared == 3

    def test_evict_specific(self):
        c = RadixTreeCache(min_prefix_len=1)
        c.store([1, 2], "data")
        assert c.evict([1, 2]) is True
        assert c.evict([1, 2]) is False

    def test_clear(self):
        c = RadixTreeCache(min_prefix_len=1)
        c.store([1, 2], "a")
        c.store([3, 4], "b")
        c.clear()
        assert c.stats()["prefix_cache_entries"] == 0
        assert c.hit_rate == 0.0

    def test_hit_rate(self):
        c = RadixTreeCache(min_prefix_len=2)
        kv = torch.randn(1, 4, 8)
        c.store([1, 2, 3], kv)
        c.lookup([1, 2, 3, 4])  # hit
        c.lookup([5, 6, 7])     # miss
        assert c.hit_rate > 0.3

    def test_adjust_memory_budget(self):
        c = RadixTreeCache(min_prefix_len=1, memory_budget_bytes=1_000_000)
        c.adjust_memory_budget(500)
        assert c._memory_budget == 500

    def test_stats_keys(self):
        c = RadixTreeCache(min_prefix_len=1)
        keys = c.stats().keys()
        assert "prefix_cache_entries" in keys
        assert "radix_tree_nodes" in keys
        assert "radix_tree_max_depth" in keys
        assert "prefix_cache_memory_bytes" in keys
        assert "prefix_cache_hit_rate" in keys

    def test_capacity_eviction(self):
        c = RadixTreeCache(max_entries=2, min_prefix_len=1, memory_budget_bytes=10**9)
        c.store([1], "a")
        c.store([2], "b")
        c.store([3], "c")
        # With memory budget high, eviction only by count
        c._evict_until_fit()
        assert c.stats()["prefix_cache_entries"] <= 2

    def test_memory_budget_eviction_large_entry(self):
        c = RadixTreeCache(min_prefix_len=1, memory_budget_bytes=100)
        huge = torch.randn(20, 20, dtype=torch.float32)
        c.store([1], {"x": huge})
        # Store again — may evict if over budget
        c.store([2], {"x": huge})
        # Should still have at least one entry
        assert c.stats()["prefix_cache_entries"] >= 1
