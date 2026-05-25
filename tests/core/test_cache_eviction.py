"""Tests for cache eviction policies (TTLPolicy, SemanticGrouping)."""

import time

from distllm.core.cache_eviction import TTLPolicy, SemanticGrouping


class TestTTLPolicy:
    def test_not_expired_before_ttl(self):
        p = TTLPolicy(default_ttl_seconds=3600)
        p.set_ttl(1, 3600)
        assert not p.is_expired(1)

    def test_expired_after_ttl(self):
        p = TTLPolicy(default_ttl_seconds=0.02)
        p.set_ttl(1, 0.02)
        time.sleep(0.03)
        assert p.is_expired(1)

    def test_expired_uses_custom_now(self):
        p = TTLPolicy(default_ttl_seconds=3600)
        p.set_ttl(1, 3600)
        future = p._entry_stored_at[1] + 4000
        assert p.is_expired(1, now=future)

    def test_is_expired_unknown_key(self):
        p = TTLPolicy()
        assert p.is_expired(999) is True

    def test_get_expired_keys_mixed(self):
        p = TTLPolicy(default_ttl_seconds=3600)
        now = time.time()
        p.set_ttl(1, 0.01)
        p._entry_stored_at[1] = now - 10
        p.set_ttl(2, 3600)
        p._entry_stored_at[2] = now
        expired = p.get_expired_keys([1, 2, 3], now=now)
        assert 1 in expired
        assert 2 not in expired
        assert 3 in expired

    def test_record_access_refreshes(self):
        p = TTLPolicy(default_ttl_seconds=0.05)
        p.set_ttl(1, 0.05)
        p._entry_stored_at[1] = time.time() - 0.1
        p.record_access(1)
        assert not p.is_expired(1)

    def test_record_access_updates_timestamp(self):
        p = TTLPolicy()
        p.set_ttl(1, 10)
        before = p._entry_stored_at[1]
        time.sleep(0.01)
        p.record_access(1)
        assert p._entry_stored_at[1] > before

    def test_remove_stops_tracking(self):
        p = TTLPolicy()
        p.set_ttl(1, 10)
        p.remove(1)
        assert p.is_expired(1)
        assert 1 not in p._entry_stored_at
        assert 1 not in p._entry_ttl

    def test_remove_unknown_safe(self):
        p = TTLPolicy()
        p.remove(999)
        assert p.is_expired(999)

    def test_clear(self):
        p = TTLPolicy()
        p.set_ttl(1, 10)
        p.set_ttl(2, 10)
        p.clear()
        assert len(p._entry_stored_at) == 0
        assert len(p._entry_ttl) == 0

    def test_default_ttl_property(self):
        p = TTLPolicy(default_ttl_seconds=60)
        assert p.default_ttl == 60
        p.default_ttl = 120
        assert p.default_ttl == 120

    def test_default_ttl_used_when_no_custom(self):
        p = TTLPolicy(default_ttl_seconds=0.01)
        p.set_ttl(1, 0.01)
        time.sleep(0.02)
        assert p.is_expired(1)


class TestSemanticGrouping:
    def test_compute_signature_deterministic(self):
        g = SemanticGrouping()
        sig1 = g.compute_signature([1, 2, 3])
        sig2 = g.compute_signature([1, 2, 3])
        assert sig1 == sig2

    def test_compute_signature_differs_for_different_inputs(self):
        g = SemanticGrouping()
        sig1 = g.compute_signature([1, 2, 3])
        sig2 = g.compute_signature([100, 200, 300])
        assert sig1 != sig2

    def test_compute_signature_empty(self):
        g = SemanticGrouping()
        sig = g.compute_signature([])
        assert sig == [0] * 128

    def test_find_or_create_group_new(self):
        g = SemanticGrouping()
        gid = g.find_or_create_group([1, 2, 3])
        assert gid.startswith("semantic_")
        assert g.get_group_id([1, 2, 3]) == gid

    def test_find_or_create_group_similar_joins(self):
        g = SemanticGrouping(num_permutations=128, threshold=0.5)
        gid1 = g.find_or_create_group([1, 2, 3])
        gid2 = g.find_or_create_group([1, 2, 4])
        assert gid1 == gid2

    def test_find_or_create_group_below_threshold(self):
        g = SemanticGrouping(num_permutations=128, threshold=0.99)
        gid1 = g.find_or_create_group([1, 2, 3])
        gid2 = g.find_or_create_group([100, 200, 300])
        assert gid1 != gid2

    def test_get_group_members(self):
        g = SemanticGrouping(num_permutations=128, threshold=0.5)
        gid = g.find_or_create_group([1, 2, 3])
        g.find_or_create_group([1, 2, 4])
        members = g.get_group_members(gid)
        assert len(members) == 2

    def test_get_group_members_unknown(self):
        g = SemanticGrouping()
        assert g.get_group_members("nonexistent") == []

    def test_get_group_id_unknown(self):
        g = SemanticGrouping()
        assert g.get_group_id([9, 9, 9]) is None

    def test_get_group_id_after_find(self):
        g = SemanticGrouping()
        gid = g.find_or_create_group([1, 2, 3])
        assert g.get_group_id([1, 2, 3]) == gid

    def test_clear(self):
        g = SemanticGrouping()
        g.find_or_create_group([1, 2, 3])
        g.clear()
        assert g.get_group_id([1, 2, 3]) is None
        assert g._next_group_id == 0
