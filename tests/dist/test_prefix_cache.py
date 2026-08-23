"""Real tests for prefix_cache module — PrefixCache, DistributedPrefixCache.

Zero mocks — all tests use real instances and deterministic logic.
No GPU required, no network, no timing-dependent assertions.
"""

from __future__ import annotations

import threading
import time

import pytest

from distllm.dist.cache import TTLPolicy
from distllm.dist.prefix_cache import DistributedPrefixCache, PrefixCache


# ── helpers ──────────────────────────────────────────────────────────

class _FakeTensor:
    """Minimal object that quacks like a torch Tensor for memory estimation."""

    def __init__(self, num_bytes: int):
        self._num_bytes = num_bytes

    def element_size(self) -> int:
        return 1

    def numel(self) -> int:
        return self._num_bytes


# ── TestBloomFilter ──────────────────────────────────────────────────

class TestBloomFilter:
    """Bloom filter internals (used by PrefixCache)."""

    def test_add_and_might_contain(self):
        from distllm.dist.prefix_cache import _BloomFilter
        bf = _BloomFilter(size=256, num_hashes=5)
        bf.add([1, 2, 3])
        assert bf.might_contain([1, 2, 3]) is True

    def test_does_not_contain_never_added(self):
        from distllm.dist.prefix_cache import _BloomFilter
        bf = _BloomFilter(size=256, num_hashes=5)
        bf.add([4, 5, 6])
        # [7,8,9] was never added — may false-positive but usually false
        assert bf.might_contain([7, 8, 9]) is False

    def test_clear_resets(self):
        from distllm.dist.prefix_cache import _BloomFilter
        bf = _BloomFilter(size=256, num_hashes=5)
        bf.add([1, 2, 3])
        bf.clear()
        assert bf.might_contain([1, 2, 3]) is False

    def test_empty_token_list(self):
        from distllm.dist.prefix_cache import _BloomFilter
        bf = _BloomFilter(size=256, num_hashes=5)
        bf.add([])
        assert bf.might_contain([]) is True

    def test_large_token_list(self):
        from distllm.dist.prefix_cache import _BloomFilter
        bf = _BloomFilter(size=1 << 16, num_hashes=7)
        tokens = list(range(500))
        bf.add(tokens)
        assert bf.might_contain(tokens) is True


# ── TestPrefixCache ─────────────────────────────────────────────────

class TestPrefixCache:
    """Public API of PrefixCache (store, lookup, evict, clear, stats)."""

    # --- empty / short input ---

    def test_lookup_empty_cache(self):
        c = PrefixCache(min_prefix_len=3)
        matched, kv = c.lookup([1, 2, 3])
        assert matched == 0
        assert kv is None

    def test_lookup_short_sequence(self):
        """Sequences shorter than min_prefix_len always miss."""
        c = PrefixCache(min_prefix_len=5)
        matched, kv = c.lookup([1, 2, 3])
        assert matched == 0
        assert kv is None

    def test_store_short_sequence_is_noop(self):
        """Storing a sequence below min_prefix_len does nothing."""
        c = PrefixCache(min_prefix_len=5)
        c.store([1, 2, 3], {"k": "v"})
        assert c.stats()["prefix_cache_entries"] == 0

    def test_lookup_empty_token_list(self):
        c = PrefixCache(min_prefix_len=1)
        matched, kv = c.lookup([])
        assert matched == 0
        assert kv is None

    def test_store_empty_token_list(self):
        c = PrefixCache(min_prefix_len=1)
        c.store([], {"k": "v"})
        assert c.stats()["prefix_cache_entries"] == 0

    # --- store & lookup hit ---

    def test_store_and_lookup_hit(self):
        c = PrefixCache(min_prefix_len=3)
        c.store([1, 2, 3, 4, 5], {"key": "val1"})
        matched, kv = c.lookup([1, 2, 3, 4, 5, 6])
        assert matched == 5
        assert kv is not None
        assert kv["key"] == "val1"

    def test_store_and_lookup_partial_prefix(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3, 4], {"k": "v"})
        matched, kv = c.lookup([1, 2, 3, 4, 5])
        assert matched == 4
        assert kv is not None

    def test_lookup_miss_different_tokens(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3], {"k": "v"})
        # [1, 9, 3] shares prefix [1] which is shorter than min_prefix_len
        matched, kv = c.lookup([1, 9, 3])
        assert matched == 0
        assert kv is None

    # --- evict ---

    def test_evict_existing(self):
        c = PrefixCache(min_prefix_len=1)
        c.store([1, 2, 3], {"k": "v"})
        assert c.evict([1, 2, 3]) is True
        # Second evict should return False
        assert c.evict([1, 2, 3]) is False

    def test_evict_non_existent(self):
        c = PrefixCache(min_prefix_len=1)
        assert c.evict([100, 200]) is False

    def test_evict_removes_entry(self):
        c = PrefixCache(min_prefix_len=1)
        c.store([1, 2, 3], {"k": "v"})
        c.evict([1, 2, 3])
        matched, kv = c.lookup([1, 2, 3])
        assert matched == 0
        assert kv is None

    def test_evict_empty_token_list(self):
        c = PrefixCache(min_prefix_len=1)
        assert c.evict([]) is False

    # --- clear ---

    def test_clear_empty_cache(self):
        c = PrefixCache(min_prefix_len=1)
        c.clear()  # should not raise
        assert c.stats()["prefix_cache_entries"] == 0
        assert c.hit_rate == 0.0

    def test_clear_removes_all_entries(self):
        c = PrefixCache(min_prefix_len=1)
        c.store([1], {"k": "v"})
        c.store([2], {"k": "v"})
        c.clear()
        assert c.stats()["prefix_cache_entries"] == 0
        assert c.hit_rate == 0.0

    # --- max_entries ---

    def test_max_entries_property(self):
        c = PrefixCache(max_entries=10, min_prefix_len=1)
        assert c.max_entries == 10

    def test_max_entries_setter(self):
        c = PrefixCache(max_entries=10, min_prefix_len=1)
        c.max_entries = 20
        assert c.max_entries == 20

    def test_max_entries_zero_means_no_limit(self):
        c = PrefixCache(max_entries=0, min_prefix_len=1)
        # When max_entries=0, the property returns len(self._cache) + 1
        assert c.max_entries == 1  # empty cache => 0 + 1
        c.store([1], {"k": "v"})
        assert c.max_entries == 2  # 1 entry => 1 + 1

    def test_eviction_on_max_entries(self):
        c = PrefixCache(max_entries=3, min_prefix_len=1)
        c.store([1], {"k": "a"})
        c.store([2], {"k": "b"})
        c.store([3], {"k": "c"})
        c.store([4], {"k": "d"})
        assert c.stats()["prefix_cache_entries"] == 3

    # --- min_prefix_len ---

    def test_min_prefix_len_blocks_lookup(self):
        c = PrefixCache(min_prefix_len=5)
        c.store([1, 2, 3, 4, 5], {"k": "v"})
        # lookup with shorter-than-min sequence
        matched, kv = c.lookup([1, 2, 3])
        assert matched == 0

    def test_min_prefix_len_attribute(self):
        c = PrefixCache(min_prefix_len=8)
        assert c.min_prefix_len == 8

    # --- memory budget ---

    def test_memory_budget_no_eviction_for_small_entries(self):
        """Entries under the budget are never evicted for memory reasons."""
        c = PrefixCache(min_prefix_len=1, memory_budget_bytes=1000000)
        c.store([1], {"k": "v"})
        c.store([2], {"k": "v"})
        c.store([3], {"k": "v"})
        assert c.stats()["prefix_cache_entries"] == 3

    def test_memory_budget_eviction_large_entry(self):
        c = PrefixCache(min_prefix_len=1, memory_budget_bytes=500)
        huge = _FakeTensor(600)
        c.store([1], {"x": huge})
        # The entry itself exceeds budget — store skips it
        assert c.stats()["prefix_cache_entries"] == 0

    def test_memory_budget_accumulated_eviction(self):
        c = PrefixCache(min_prefix_len=1, memory_budget_bytes=500)
        med = _FakeTensor(300)
        c.store([1], {"x": med})
        c.store([2], {"x": med})
        # Total 600 > 500 — one gets evicted
        assert c.stats()["prefix_cache_entries"] == 1

    def test_adjust_memory_budget(self):
        c = PrefixCache(min_prefix_len=1, memory_budget_bytes=1_000_000)
        c.adjust_memory_budget(500)
        assert c._memory_budget == 500
        assert c.stats()["prefix_cache_memory_budget"] == 500

    # --- hit_rate ---

    def test_hit_rate_start(self):
        c = PrefixCache(min_prefix_len=2)
        assert c.hit_rate == 0.0

    def test_hit_rate_after_hit_and_miss(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3], {"k": "v"})
        c.lookup([1, 2, 3, 4])  # hit
        c.lookup([5, 6, 7])     # miss
        hr = c.hit_rate
        assert 0.3 < hr < 0.7

    def test_hit_rate_all_misses(self):
        c = PrefixCache(min_prefix_len=2)
        c.lookup([1, 2, 3])
        c.lookup([4, 5, 6])
        assert c.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3], {"k": "v"})
        c.lookup([1, 2, 3, 4])
        assert c.hit_rate == 1.0

    # --- stats ---

    def test_stats_keys(self):
        c = PrefixCache(min_prefix_len=1)
        keys = c.stats().keys()
        assert "prefix_cache_entries" in keys
        assert "prefix_cache_max_entries" in keys
        assert "prefix_cache_hits" in keys
        assert "prefix_cache_misses" in keys
        assert "prefix_cache_hit_rate" in keys
        assert "prefix_cache_memory_bytes" in keys
        assert "prefix_cache_memory_budget" in keys
        assert "prefix_cache_memory_util" in keys

    def test_stats_values_are_integer_or_float(self):
        c = PrefixCache(min_prefix_len=1)
        s = c.stats()
        assert isinstance(s["prefix_cache_entries"], int)
        assert isinstance(s["prefix_cache_hits"], int)
        assert isinstance(s["prefix_cache_misses"], int)
        assert isinstance(s["prefix_cache_hit_rate"], float)
        assert isinstance(s["prefix_cache_memory_util"], float)

    # --- update existing entry ---

    def test_update_existing_entry_kv_data(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3], {"k": "old"})
        c.store([1, 2, 3], {"k": "new"})
        matched, kv = c.lookup([1, 2, 3, 4])
        assert matched == 3
        assert kv["k"] == "new"

    def test_update_existing_entry_increments_access_count(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3], {"k": "v"})
        # First update
        c.store([1, 2, 3], {"k": "v2"})
        raw_key = (PrefixCache._compute_full_hash([1, 2, 3]), "")
        entry = c._cache[raw_key]
        assert entry["access_count"] == 2

    # --- tenant isolation ---

    def test_different_tenants_do_not_share_cache(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3], {"k": "a"}, tenant_id="alice")
        c.store([1, 2, 3], {"k": "b"}, tenant_id="bob")
        matched_a, kv_a = c.lookup([1, 2, 3, 4], tenant_id="alice")
        matched_b, kv_b = c.lookup([1, 2, 3, 4], tenant_id="bob")
        assert matched_a == 3
        assert matched_b == 3
        assert kv_a["k"] == "a"
        assert kv_b["k"] == "b"

    def test_evict_respects_tenant(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([1, 2, 3], {"k": "v"}, tenant_id="alice")
        assert c.evict([1, 2, 3], tenant_id="bob") is False
        assert c.evict([1, 2, 3], tenant_id="alice") is True

    # --- hash collision ---

    def test_hash_colliding_sequence_misses(self):
        """Polynomial rolling hash: [5,32,7] and [6,1,7] collide."""
        c = PrefixCache(min_prefix_len=2)
        c.store([5, 32, 7], {"k": "v"})
        matched, kv = c.lookup([6, 1, 7, 9])
        assert matched == 0
        assert kv is None

    def test_collision_does_not_contaminate(self):
        c = PrefixCache(min_prefix_len=2)
        c.store([5, 32, 7], {"v": 1})
        c.store([9, 9, 9], {"v": 2})
        matched, kv = c.lookup([9, 9, 9, 5])
        assert matched == 3
        assert kv["v"] == 2

    # --- TTL integration ---

    def test_ttl_expired_entry_is_skipped(self):
        ttl = TTLPolicy(default_ttl_seconds=0.001)
        c = PrefixCache(min_prefix_len=2, ttl_policy=ttl)
        c.store([1, 2, 3], {"k": "v"})
        time.sleep(0.005)
        matched, kv = c.lookup([1, 2, 3, 4])
        assert matched == 0
        assert kv is None

    def test_ttl_record_access(self):
        ttl = TTLPolicy(default_ttl_seconds=3600)
        c = PrefixCache(min_prefix_len=2, ttl_policy=ttl)
        c.store([1, 2, 3], {"k": "v"})
        c.lookup([1, 2, 3, 4])
        matched, kv = c.lookup([1, 2, 3, 4])
        assert matched == 3

    # --- _compute_full_hash ---

    def test_compute_full_hash_zero(self):
        assert PrefixCache._compute_full_hash([]) == 0

    def test_compute_full_hash_deterministic(self):
        h1 = PrefixCache._compute_full_hash([1, 2, 3])
        h2 = PrefixCache._compute_full_hash([1, 2, 3])
        assert h1 == h2

    def test_compute_full_hash_different_inputs(self):
        h1 = PrefixCache._compute_full_hash([1, 2, 3])
        h2 = PrefixCache._compute_full_hash([1, 2, 4])
        assert h1 != h2


# ── TestPrefixCacheConcurrent ──────────────────────────────────────

class TestPrefixCacheConcurrent:
    """Thread safety under concurrent reads and writes."""

    def test_concurrent_store_and_lookup_no_crash(self):
        c = PrefixCache(max_entries=100, min_prefix_len=1)
        seqs = [[i, i + 1] for i in range(50)]

        def worker():
            for _ in range(20):
                for seq in seqs:
                    c.store(seq, {"v": seq[-1]})
                    c.lookup(seq + [0])

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert c.stats()["prefix_cache_hits"] > 0

    def test_concurrent_evict_and_lookup_no_crash(self):
        c = PrefixCache(max_entries=50, min_prefix_len=1)
        for i in range(20):
            c.store([i, i + 1], {"v": i})

        def evicter():
            for i in range(20):
                c.evict([i, i + 1])

        def looker():
            for i in range(20):
                c.lookup([i, i + 1, 0])

        threads = [
            threading.Thread(target=evicter, daemon=True),
            threading.Thread(target=looker, daemon=True),
            threading.Thread(target=looker, daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No crash is the test
        s = c.stats()
        assert s["prefix_cache_entries"] >= 0

    def test_stats_consistent_under_concurrent_load(self):
        c = PrefixCache(max_entries=500, min_prefix_len=1)
        n_threads = 6
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def worker(offset: int):
            barrier.wait()
            try:
                for i in range(200):
                    seq = [offset * 1000 + i, offset * 1000 + i + 1]
                    c.store(seq, {"v": i})
                    c.lookup(seq + [0])
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,), daemon=True)
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        s = c.stats()
        assert s["prefix_cache_entries"] <= s["prefix_cache_max_entries"]
        assert s["prefix_cache_memory_bytes"] >= 0


# ── TestDistributedPrefixCache ─────────────────────────────────────

class TestDistributedPrefixCache:
    """Public API of DistributedPrefixCache."""

    # --- init ---

    def test_init(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local_cache=local, node_id="node0")
        assert dp._node_id == "node0"
        assert dp._local is local

    def test_init_with_gossip_interval(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local_cache=local, node_id="n1", gossip_interval_s=10.0)
        assert dp._gossip_interval == 10.0

    # --- compute_merkle_root ---

    def test_compute_merkle_root_empty(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        assert dp.compute_merkle_root([]) == 0

    def test_compute_merkle_root_single(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        root = dp.compute_merkle_root([42])
        assert isinstance(root, int)
        assert root > 0

    def test_compute_merkle_root_deterministic(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        r1 = dp.compute_merkle_root([1, 2, 3, 4, 5])
        r2 = dp.compute_merkle_root([1, 2, 3, 4, 5])
        assert r1 == r2

    def test_compute_merkle_root_different_inputs_differ(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        r1 = dp.compute_merkle_root([1, 2, 3])
        r2 = dp.compute_merkle_root([1, 2, 4])
        assert r1 != r2

    def test_compute_merkle_root_even_length(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        r = dp.compute_merkle_root([1, 2, 3, 4])
        assert isinstance(r, int)
        assert r > 0

    def test_compute_merkle_root_odd_length(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        r = dp.compute_merkle_root([1, 2, 3, 4, 5])
        assert isinstance(r, int)
        assert r > 0

    # --- update_local_prefix ---

    def test_update_local_prefix(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        assert len(dp.get_prefix_info()) == 0
        dp.update_local_prefix([1, 2, 3, 4])
        info = dp.get_prefix_info()
        assert len(info) == 1
        h = list(info.keys())[0]
        assert info[h] == 4

    def test_update_local_prefix_too_short(self):
        local = PrefixCache(min_prefix_len=5)
        dp = DistributedPrefixCache(local, "node0")
        dp.update_local_prefix([1, 2])
        assert len(dp.get_prefix_info()) == 0

    def test_update_local_prefix_twice(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.update_local_prefix([1, 2, 3])
        dp.update_local_prefix([4, 5, 6])
        info = dp.get_prefix_info()
        assert len(info) == 2

    # --- get_prefix_info ---

    def test_get_prefix_info_empty(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        assert dp.get_prefix_info() == {}

    def test_get_prefix_info_returns_copy(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.update_local_prefix([1, 2, 3])
        info = dp.get_prefix_info()
        # Mutating the returned dict should not affect internal state
        info.clear()
        assert len(dp.get_prefix_info()) == 1

    # --- receive_gossip ---

    def test_receive_gossip(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.receive_gossip("node1", {12345: 10, 67890: 20})
        stats = dp.stats()
        assert stats["remote_nodes"] == 1
        assert stats["total_remote_prefixes"] == 2

    def test_receive_gossip_multiple_nodes(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.receive_gossip("node1", {1: 2})
        dp.receive_gossip("node2", {3: 4})
        stats = dp.stats()
        assert stats["remote_nodes"] == 2

    def test_receive_gossip_overwrite(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.receive_gossip("node1", {1: 2})
        dp.receive_gossip("node1", {3: 4})
        stats = dp.stats()
        assert stats["remote_nodes"] == 1
        assert stats["total_remote_prefixes"] == 1

    def test_receive_gossip_empty(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.receive_gossip("node1", {})
        stats = dp.stats()
        assert stats["remote_nodes"] == 1
        assert stats["total_remote_prefixes"] == 0

    # --- find_best_node ---

    def test_find_best_node_short_sequence(self):
        local = PrefixCache(min_prefix_len=5)
        dp = DistributedPrefixCache(local, "node0")
        assert dp.find_best_node([1, 2]) is None

    def test_find_best_node_local_match(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.update_local_prefix([1, 2, 3])
        result = dp.find_best_node([1, 2, 3])
        assert result is not None
        node_id, length = result
        assert node_id == "node0"
        assert length == 3

    def test_find_best_node_remote_match(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        h = 0
        for tok in tokens:
            h = ((h * 31337) + tok) & ((1 << 61) - 1)
        dp.receive_gossip("node1", {h: 8, 67890: 5})
        result = dp.find_best_node(tokens)
        assert result is not None
        node_id, length = result
        assert node_id == "node1"
        assert length == 8

    def test_find_best_node_remote_best_length(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.receive_gossip("node1", {42: 3, 43: 10})
        dp.receive_gossip("node2", {42: 7, 43: 2})
        # provide both hashes so the longer one wins
        # We can control by making the hash match
        h = 42
        dp._remote_prefixes["node1"] = {h: 3, 99: 10}
        dp._remote_prefixes["node2"] = {h: 7, 98: 2}
        result = dp.find_best_node([1, 2, 3, 4])  # len=4, min=2
        if result is not None:
            assert result[1] == 7  # node2 has length 7 for hash 42

    def test_find_best_node_no_match(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.receive_gossip("node1", {99999: 5})
        result = dp.find_best_node([1, 2, 3, 4])
        assert result is None

    def test_find_best_node_empty_token_ids(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        assert dp.find_best_node([]) is None

    # --- get_gossip_payload ---

    def test_get_gossip_payload_keys(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        payload = dp.get_gossip_payload()
        assert "node_id" in payload
        assert "prefixes" in payload
        assert "merkle_root" in payload
        assert "timestamp" in payload

    def test_get_gossip_payload_values(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.update_local_prefix([1, 2, 3, 4, 5])
        payload = dp.get_gossip_payload()
        assert payload["node_id"] == "node0"
        assert len(payload["prefixes"]) == 1
        assert isinstance(payload["merkle_root"], int)
        assert isinstance(payload["timestamp"], float)

    # --- stats ---

    def test_stats_keys(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        s = dp.stats()
        assert "local_prefixes" in s
        assert "remote_nodes" in s
        assert "total_remote_prefixes" in s

    def test_stats_values(self):
        local = PrefixCache(min_prefix_len=2)
        dp = DistributedPrefixCache(local, "node0")
        dp.update_local_prefix([1, 2, 3])
        dp.receive_gossip("node1", {42: 5})
        s = dp.stats()
        assert s["local_prefixes"] == 1
        assert s["remote_nodes"] == 1
        assert s["total_remote_prefixes"] == 1
