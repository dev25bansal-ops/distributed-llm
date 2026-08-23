"""Real tests for cache modules — TTLPolicy, SemanticGrouping, CacheIndex.

Zero mocks — all tests use real instances and deterministic logic.
"""

from __future__ import annotations

import time

from distllm.dist.cache import TTLPolicy, SemanticGrouping, CacheIndex


class TestTTLPolicy:
    def test_default_ttl(self):
        p = TTLPolicy(default_ttl_seconds=60.0)
        assert p.default_ttl == 60.0

    def test_expired_if_not_stored(self):
        p = TTLPolicy()
        assert p.is_expired(42)

    def test_fresh_not_expired(self):
        p = TTLPolicy(default_ttl_seconds=3600.0)
        p._entry_stored_at[42] = time.time()
        assert not p.is_expired(42)

    def test_stale_expired(self):
        p = TTLPolicy(default_ttl_seconds=1.0)
        p._entry_stored_at[42] = time.time() - 10
        assert p.is_expired(42)

    def test_record_access_refreshes(self):
        p = TTLPolicy(default_ttl_seconds=3600.0)
        p._entry_stored_at[42] = time.time() - 1000
        p.record_access(42)
        assert not p.is_expired(42)

    def test_remove_clears(self):
        p = TTLPolicy(default_ttl_seconds=3600.0)
        p._entry_stored_at[42] = time.time()
        p.remove(42)
        assert p.is_expired(42)

    def test_clear_all(self):
        p = TTLPolicy()
        p._entry_stored_at[1] = time.time()
        p._entry_stored_at[2] = time.time()
        p.clear()
        assert p.is_expired(1)
        assert p.is_expired(2)


class TestSemanticGrouping:
    def test_compute_signature(self):
        sg = SemanticGrouping(num_permutations=32)
        sig = sg.compute_signature([101, 202, 303])
        assert len(sig) == 32
        assert all(v >= 0 for v in sig)

    def test_find_or_create_group(self):
        sg = SemanticGrouping(num_permutations=128, threshold=0.5)
        gid = sg.find_or_create_group([101, 202, 303])
        assert gid.startswith("semantic_")

    def test_get_group_members(self):
        sg = SemanticGrouping(num_permutations=64, threshold=0.1)
        gid = sg.find_or_create_group([101, 202])
        members = sg.get_group_members(gid)
        assert len(members) >= 1

    def test_clear(self):
        sg = SemanticGrouping()
        sg.find_or_create_group([101, 202])
        sg.clear()
        assert len(sg._groups) == 0

    def test_similarity(self):
        sg = SemanticGrouping(num_permutations=128)
        sig_a = sg.compute_signature([101, 202, 303])
        sig_b = sg.compute_signature([101, 202, 303])
        assert sg._similarity(sig_a, sig_b) == 1.0
        sig_c = sg.compute_signature([999, 888, 777])
        assert sg._similarity(sig_a, sig_c) < 1.0


class TestCacheIndex:
    def test_index_tokens_consistent(self):
        ci = CacheIndex()
        assert ci.index_tokens([1, 2, 3]) == ci.index_tokens([1, 2, 3])

    def test_index_tokens_different(self):
        ci = CacheIndex()
        assert ci.index_tokens([1, 2, 3]) != ci.index_tokens([1, 2, 4])

    def test_store_and_lookup(self):
        ci = CacheIndex()
        h = ci.index_tokens([1, 2, 3])
        ci.store(h, "node-1", "ref-1")
        assert ci.lookup(h) == "node-1"

    def test_lookup_miss(self):
        ci = CacheIndex()
        assert ci.lookup("nonexistent") is None

    def test_lookup_all(self):
        ci = CacheIndex()
        h = ci.index_tokens([1, 2, 3])
        ci.store(h, "node-1", "ref-1")
        ci.store(h, "node-2", "ref-2")
        assert len(ci.lookup_all(h)) == 2

    def test_remove_by_hash(self):
        ci = CacheIndex()
        h = ci.index_tokens([1, 2, 3])
        ci.store(h, "node-1", "ref-1")
        ci.remove(h)
        assert ci.lookup(h) is None

    def test_stats(self):
        ci = CacheIndex()
        h = ci.index_tokens([1, 2, 3])
        ci.store(h, "node-1", "ref-1")
        ci.lookup(h)
        ci.lookup("miss")
        s = ci.stats()
        assert s["hit_count"] == 1
        assert s["miss_count"] == 1

    def test_rolling_prefix_hash(self):
        ci = CacheIndex()
        hashes = ci.rolling_prefix_hash([1, 2, 3, 4, 5, 6], window_size=3)
        assert len(hashes) == 2

    def test_longest_prefix_match(self):
        ci = CacheIndex()
        hashes = ci.rolling_prefix_hash([1, 2, 3, 4, 5, 6], window_size=3)
        for h in hashes:
            ci.store(h, "node-1", "ref-1")
        node, matched = ci.longest_prefix_match([1, 2, 3, 4, 5, 6, 7], window_size=3)
        assert node == "node-1"
        assert matched >= 6
