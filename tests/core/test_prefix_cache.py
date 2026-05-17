"""Tests for PrefixCache (hash-based LRU)."""

import pytest
import torch

from distllm.core.prefix_cache import PrefixCache


class TestPrefixCache:
    def test_lookup_miss_empty(self):
        c = PrefixCache(min_prefix_len=3)
        matched, kv = c.lookup([1, 2, 3])
        assert matched == 0
        assert kv is None

    def test_lookup_short_sequence(self):
        c = PrefixCache(min_prefix_len=5)
        matched, kv = c.lookup([1, 2, 3])
        assert matched == 0

    def test_store_and_lookup_hit(self):
        c = PrefixCache(min_prefix_len=3)
        kv_data = {"key": torch.randn(1, 4, 8), "value": torch.randn(1, 4, 8)}
        c.store([1, 2, 3, 4, 5], kv_data)

        matched, kv = c.lookup([1, 2, 3, 4, 5, 6])
        assert matched == 5
        assert kv is not None

    def test_store_and_lookup_partial_prefix(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3, 4], {"k": torch.randn(1, 4, 8)})

        matched, kv = c.lookup([1, 2, 3, 4, 5])
        assert matched == 4

    def test_lookup_miss_different_tokens(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3], {"k": torch.randn(1, 4, 8)})

        matched, kv = c.lookup([1, 9, 3])
        # prefix [1] is too short (< min_prefix_len=2)
        assert matched == 0

    def test_hash_collision_detected(self):
        c = PrefixCache(min_prefix_len=2)
        # Different sequences that happen to have same hash — verified by token check
        c.store([1, 2, 3], {"k": torch.randn(1, 4, 8)})
        # A completely different sequence should miss
        matched, kv = c.lookup([0, 0, 0, 0, 0])
        assert matched == 0

    def test_evict_lru_on_capacity(self):
        c = PrefixCache(max_entries=2, min_prefix_len=1)
        c.store([1], {"a": torch.randn(1, 4, 8)})
        c.store([2], {"b": torch.randn(1, 4, 8)})
        c.store([3], {"c": torch.randn(1, 4, 8)})
        assert c.stats()["prefix_cache_entries"] == 2

    def test_memory_budget_eviction(self):
        c = PrefixCache(min_prefix_len=1, memory_budget_bytes=100)
        huge = torch.randn(10, 10, dtype=torch.float32)
        entry_bytes = huge.element_size() * huge.numel()
        c.store([1], {"x": huge})
        # Budget exceeded, store should evict
        c.store([2], {"y": huge})
        # At least one entry should remain
        assert c.stats()["prefix_cache_entries"] >= 1

    def test_evict_specific(self):
        c = PrefixCache(min_prefix_len=1)
        c.store([1, 2, 3], {"k": "v"})
        assert c.evict([1, 2, 3]) is True
        assert c.evict([1, 2, 3]) is False
        matched, kv = c.lookup([1, 2, 3])
        assert matched == 0

    def test_clear(self):
        c = PrefixCache(min_prefix_len=1)
        c.store([1], {"k": "v"})
        c.clear()
        assert c.stats()["prefix_cache_entries"] == 0
        assert c.hit_rate == 0.0

    def test_hit_rate(self):
        c = PrefixCache(min_prefix_len=2)
        kv = {"k": torch.randn(1, 4, 8)}
        c.store([1, 2, 3], kv)
        c.lookup([1, 2, 3, 4])  # hit
        c.lookup([5, 6, 7])     # miss
        assert c.hit_rate > 0.3

    def test_adjust_memory_budget(self):
        c = PrefixCache(min_prefix_len=1, memory_budget_bytes=1_000_000)
        c.adjust_memory_budget(500)
        assert c._memory_budget == 500

    def test_stats_keys(self):
        c = PrefixCache(min_prefix_len=1)
        keys = c.stats().keys()
        assert "prefix_cache_entries" in keys
        assert "prefix_cache_hit_rate" in keys
        assert "prefix_cache_memory_bytes" in keys
        assert "prefix_cache_memory_budget" in keys

    def test_update_existing_entry(self):
        c = PrefixCache(min_prefix_len=2)
        kv1 = {"k": "old"}
        kv2 = {"k": "new"}
        c.store([1, 2, 3], kv1)
        c.store([1, 2, 3], kv2)
        matched, kv = c.lookup([1, 2, 3, 4])
        assert matched == 3
        assert kv["k"] == "new"
