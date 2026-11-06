"""Tests for RadixTreeCache (trie-based prefix cache)."""

import pytest
import torch

from distllm.core.radix_tree_cache import RadixTreeCache, RadixNode


class TestRadixNodeInsertionStructure:
    """Verifies that insert creates the correct tree topology."""

    def test_insert_single_sequence_creates_path(self):
        root = RadixNode()
        root.insert([1, 2, 3], "data")
        assert 1 in root.children
        assert root.children[1].token == 1
        assert 2 in root.children[1].children
        assert root.children[1].children[2].token == 2
        assert 3 in root.children[1].children[2].children
        leaf = root.children[1].children[2].children[3]
        assert leaf.token == 3
        assert leaf.kv_data == "data"

    def test_insert_two_branches_share_prefix(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        root.insert([1, 2, 4], "b")
        shared = root.children[1].children[2]
        assert 3 in shared.children
        assert shared.children[3].kv_data == "a"
        assert 4 in shared.children
        assert shared.children[4].kv_data == "b"
        assert shared.kv_data is None

    def test_insert_nested_prefixes(self):
        root = RadixNode()
        root.insert([1, 2], "short")
        root.insert([1, 2, 3, 4], "long")
        mid = root.children[1].children[2]
        assert mid.kv_data == "short"
        assert 3 in mid.children
        deep = mid.children[3].children[4]
        assert deep.kv_data == "long"

    def test_insert_overwrite_same_key(self):
        root = RadixNode()
        root.insert([1], "old")
        root.insert([1], "new")
        assert root.children[1].kv_data == "new"

    def test_insert_empty_sequence_stores_at_root(self):
        root = RadixNode()
        root.insert([], "data")
        assert root.kv_data == "data"

    def test_insert_updates_size(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        node3 = root.children[1].children[2].children[3]
        # node3's subtree holds exactly its own entry
        assert node3.size == 1
        root.insert([1, 2, 4], "b")
        # node2's subtree now holds both entries
        assert root.children[1].children[2].size == 2

    def test_insert_large_sequence(self):
        root = RadixNode()
        tokens = list(range(100))
        root.insert(tokens, "big")
        node = root
        for tok in tokens:
            node = node.children[tok]
        assert node.kv_data == "big"


class TestRadixNodeSearch:
    """Verifies longest prefix match and shared prefix search."""

    def test_lookup_deepest_non_leaf(self):
        root = RadixNode()
        root.insert([1, 2], "a")
        root.insert([1, 2, 3, 4], "b")
        matched, kv = root.lookup([1, 2, 3, 4, 5, 6])
        assert matched == 4
        assert kv == "b"

    def test_lookup_mid_path_no_kv(self):
        root = RadixNode()
        root.insert([1, 2, 3, 4], "a")
        matched, kv = root.lookup([1, 2, 3])
        assert matched == 0

    def test_lookup_across_branches(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        root.insert([1, 2, 4], "b")
        matched, kv = root.lookup([1, 2, 4, 5])
        assert matched == 3
        assert kv == "b"

    def test_lookup_full_exact_match(self):
        root = RadixNode()
        root.insert([1, 2, 3], "exact")
        matched, kv = root.lookup([1, 2, 3])
        assert matched == 3
        assert kv == "exact"

    def test_lookup_no_match(self):
        root = RadixNode()
        root.insert([5, 6, 7], "a")
        matched, kv = root.lookup([1, 2, 3])
        assert matched == 0

    def test_lookup_partial_path_no_kv_before_end(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        root.insert([1, 2, 3, 4, 5], "b")
        matched, kv = root.lookup([1, 2, 3, 4, 5, 6])
        assert matched == 5
        assert kv == "b"

    def test_find_shared_prefix_full_match(self):
        root = RadixNode()
        root.insert([1, 2, 3, 4], "a")
        assert root.find_shared_prefix([1, 2, 3, 4]) == 4

    def test_find_shared_prefix_partial_match(self):
        root = RadixNode()
        root.insert([1, 2, 3, 4], "a")
        assert root.find_shared_prefix([1, 2, 9, 9]) == 2

    def test_find_shared_prefix_no_match(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        assert root.find_shared_prefix([9, 9, 9]) == 0

    def test_find_shared_prefix_empty_trie(self):
        root = RadixNode()
        assert root.find_shared_prefix([1, 2, 3]) == 0

    def test_touch_on_lookup_hit(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        before = root.children[1].children[2].children[3].last_access
        root.lookup([1, 2, 3, 4])
        after = root.children[1].children[2].children[3].last_access
        assert after >= before

    def test_no_touch_on_lookup_miss(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        before = root.children[1].children[2].children[3].last_access
        root.lookup([5, 6, 7])
        after = root.children[1].children[2].children[3].last_access
        assert after == before


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
        c = RadixTreeCache(min_prefix_len=1, memory_budget_bytes=2000)
        huge = torch.randn(20, 20, dtype=torch.float32)
        entry_bytes = huge.element_size() * huge.numel()  # ~1600
        assert entry_bytes > 1000
        c.store([1], {"x": huge})
        c.store([2], {"x": huge})
        assert c.stats()["prefix_cache_entries"] == 1
        assert c.stats()["prefix_cache_memory_bytes"] <= 2000


class TestRadixNodeEviction:
    """Verifies leaf-node LRU eviction order and shared-prefix survival."""

    def test_evict_lru_removes_oldest_leaf(self):
        root = RadixNode()
        root.insert([1], "a")
        root.insert([2], "b")
        root.children[1].last_access = 100.0
        root.children[2].last_access = 200.0
        evicted = root.evict_lru(max_entries=1)
        assert evicted == 1
        assert root.children[1].kv_data is None
        assert root.children[2].kv_data == "b"

    def test_evict_lru_removes_newest_when_oldest_already_gone(self):
        root = RadixNode()
        root.insert([1], "a")
        root.insert([2], "b")
        root.children[1].last_access = 100.0
        root.children[2].last_access = 200.0
        evicted = root.evict_lru(max_entries=1)
        assert evicted == 1
        _, _ = root.lookup([1])
        assert root.children[1].kv_data is None

    def test_evict_lru_no_op_when_under_capacity(self):
        root = RadixNode()
        root.insert([1], "a")
        root.insert([2], "b")
        evicted = root.evict_lru(max_entries=10)
        assert evicted == 0
        assert root._count_entries() == 2

    def test_evict_lru_zero_max_clears_all(self):
        root = RadixNode()
        root.insert([1], "a")
        root.insert([2], "b")
        evicted = root.evict_lru(max_entries=0)
        assert evicted == 2
        assert root._count_entries() == 0

    def test_evict_lru_empty_tree(self):
        root = RadixNode()
        evicted = root.evict_lru(max_entries=1)
        assert evicted == 0

    def test_shared_prefix_survives_child_eviction(self):
        root = RadixNode()
        root.insert([1, 2, 3], "left")
        root.insert([1, 2, 4], "right")
        node3 = root.children[1].children[2].children[3]
        node4 = root.children[1].children[2].children[4]
        node3.last_access = 100.0
        node4.last_access = 200.0
        evicted = root.evict_lru(max_entries=1)
        assert evicted == 1
        assert node3.kv_data is None
        assert node4.kv_data == "right"
        assert 3 in root.children[1].children[2].children
        assert 4 in root.children[1].children[2].children

    def test_evict_lru_leaf_size_zeroed(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        node3 = root.children[1].children[2].children[3]
        node3.last_access = 100.0
        root.evict_lru(max_entries=0)
        assert node3.kv_data is None
        assert node3.size == 0

    def test_evict_lru_multi_pass_removes_multiple(self):
        root = RadixNode()
        root.insert([1], "a")
        root.insert([2], "b")
        root.insert([3], "c")
        root.children[1].last_access = 100.0
        root.children[2].last_access = 200.0
        root.children[3].last_access = 300.0
        evicted = root.evict_lru(max_entries=1)
        assert evicted == 2
        assert root._count_entries() == 1

    def test_evict_path_non_leaf_no_op(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        root.insert([1, 2, 4], "b")
        result = root._evict_path([1, 2])
        assert result is False
        assert root.children[1].children[2].kv_data is None

    def test_evict_path_unknown_path(self):
        root = RadixNode()
        root.insert([1, 2, 3], "a")
        result = root._evict_path([9])
        assert result is False

    def test_radix_tree_cache_evicts_lru_leaf_by_memory(self):
        c = RadixTreeCache(min_prefix_len=1, memory_budget_bytes=10**9)
        c.store([1], {"x": torch.randn(5, 5)})
        c.store([2], {"x": torch.randn(5, 5)})
        c._root.children[1].last_access = 100.0
        c._root.children[2].last_access = 200.0
        # Budget = 150 fits one entry (5*5*4=100 bytes) but not both (200 bytes)
        c.adjust_memory_budget(150)
        _, kv = c.lookup([2])
        assert kv is not None
        _, kv = c.lookup([1])
        assert kv is None
