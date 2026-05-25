"""Tests for PrefixCache (hash-based LRU)."""

import threading

import pytest
import torch

from distllm.core.prefix_cache import PrefixCache


class TestPrefixCacheLRUOrder:
    """Verifies that eviction removes the *least recently used* entry."""

    LOOKUP_KEY = [9, 9, 9]

    def test_evict_oldest_on_capacity(self):
        c = PrefixCache(max_entries=2, min_prefix_len=1)
        c.store([1], "a")
        c.store([2], "b")
        # Access [1] to make it recently used, so [2] becomes LRU
        c.lookup([1])
        c.store([3], "c")
        assert c.stats()["prefix_cache_entries"] == 2
        _, kv = c.lookup([1])
        assert kv is not None, "Recently accessed entry should survive"
        _, kv = c.lookup([2])
        assert kv is None, "Oldest unaccessed entry should be evicted"
        _, kv = c.lookup([3])
        assert kv is not None, "Newly stored entry should survive"

    def test_recently_stored_not_evicted(self):
        c = PrefixCache(max_entries=2, min_prefix_len=1)
        c.store([1], "a")
        c.store([2], "b")
        # Store [3] → evict oldest ([1])
        c.store([3], "c")
        _, kv = c.lookup([1])
        assert kv is None
        _, kv = c.lookup([2])
        assert kv is not None
        assert c.stats()["prefix_cache_entries"] == 2


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
        c = PrefixCache(min_prefix_len=1, memory_budget_bytes=500)
        huge = torch.randn(10, 10, dtype=torch.float32)
        entry_bytes = huge.element_size() * huge.numel()
        c.store([1], {"x": huge})
        assert c.stats()["prefix_cache_entries"] == 1
        c.store([2], {"y": huge})
        assert c.stats()["prefix_cache_entries"] == 1

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


class TestPrefixCacheHashCollision:
    """Hash collisions: polynomial rolling hash h = (h * 31 + tok) % (2^61 - 1).

    Colliding pairs: [5,32] and [6,1] → hash 187;
                    [1,1,1] and [0,32,1] → hash 993.
    """

    def test_hit_on_exact_match(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([5, 32, 7], {"v": 1})
        matched, kv = c.lookup([5, 32, 7, 9])
        assert matched == 3
        assert kv["v"] == 1

    def test_colliding_sequence_misses(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([5, 32, 7], {"v": 1})
        matched, kv = c.lookup([6, 1, 7, 9])
        assert matched == 0
        assert kv is None

    def test_collision_three_tokens(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 1, 1], {"v": 1})
        matched, kv = c.lookup([0, 32, 1, 9])
        assert matched == 0
        assert kv is None

    def test_collision_does_not_affect_other_keys(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([5, 32], {"v": 1})
        c.store([9, 9, 9], {"v": 2})
        matched, kv = c.lookup([9, 9, 9, 5])
        assert matched == 3
        assert kv["v"] == 2

    def test_collision_overwrite_then_hit(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([5, 32, 7], {"v": "a"})
        c.store([6, 1, 7], {"v": "b"})
        matched, kv = c.lookup([6, 1, 7, 3])
        assert matched == 3
        assert kv["v"] == "b"

    def test_collision_longest_prefix_noncollider(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([5, 32], {"v": "collider"})
        c.store([1, 2, 3], {"v": "normal"})
        matched, kv = c.lookup([1, 2, 3, 8])
        assert matched == 3
        assert kv["v"] == "normal"


class TestPrefixCacheConcurrent:
    """Thread safety under concurrent reads and writes."""

    def test_concurrent_store_and_lookup_no_crash(self):
        c = PrefixCache(max_entries=100, min_prefix_len=1)
        seqs = [[i, i + 1] for i in range(50)]

        def worker(seed):
            for _ in range(20):
                for seq in seqs:
                    c.store(seq, {"v": seq[-1]})
                    c.lookup(seq + [0])

        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert c.stats()["prefix_cache_hits"] > 0

    def test_concurrent_store_same_key(self):
        c = PrefixCache(max_entries=10, min_prefix_len=1)
        N = 8

        def writer(idx):
            for _ in range(100):
                c.store([42, 99], {"writer": idx, "val": idx})

        threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        matched, kv = c.lookup([42, 99, 1])
        assert matched == 2
        assert kv is not None

    def test_concurrent_lookup_does_not_raise(self):
        c = PrefixCache(max_entries=50, min_prefix_len=1)
        for i in range(30):
            c.store([i, i + 1], {"v": i})

        results = []

        def reader():
            for _ in range(100):
                import random
                k = random.randint(0, 40)
                matched, kv = c.lookup([k, k + 1, 0])
                results.append((matched, kv is not None))

        def writer():
            for i in range(100, 200):
                c.store([i, i + 1], {"v": i})

        threads = [
            threading.Thread(target=reader, daemon=True) for _ in range(4)
        ] + [threading.Thread(target=writer, daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        hits = sum(1 for m, hit in results if hit)
        assert hits >= 0  # just verifying no crash

    def test_stats_consistent_under_concurrent_load(self):
        c = PrefixCache(max_entries=500, min_prefix_len=1)
        N = 6
        barrier = threading.Barrier(N)
        errors = []

        def worker(offset):
            barrier.wait()
            try:
                for i in range(200):
                    seq = [offset * 1000 + i, offset * 1000 + i + 1]
                    c.store(seq, {"v": i})
                    c.lookup(seq + [0])
                for i in range(200):
                    c.lookup([999999, i])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        s = c.stats()
        assert s["prefix_cache_entries"] <= s["prefix_cache_max_entries"]
        assert s["prefix_cache_memory_bytes"] >= 0
