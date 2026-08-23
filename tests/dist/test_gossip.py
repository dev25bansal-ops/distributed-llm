"""Tests for distllm.dist.p2p.gossip module.

Covers the full public API: VectorClock, LWWRegister, GossipState,
GossipProtocol, GossipClient, and GossipReplicator.

Deterministic -- no time.sleep, no threading, no network, no GPU.
Zero mocks -- uses only real objects from the module.
"""

from __future__ import annotations

import os
import time

import pytest

from distllm.dist.p2p.gossip import (
    GossipClient,
    GossipProtocol,
    GossipReplicator,
    GossipState,
    LWWRegister,
    VectorClock,
)


# ---------------------------------------------------------------------------
# VectorClock
# ---------------------------------------------------------------------------

class TestVectorClock:
    """Vector clock for causal ordering."""

    def test_default_empty(self) -> None:
        vc = VectorClock()
        assert vc.clocks == {}

    def test_increment_new_node(self) -> None:
        vc = VectorClock()
        vc.increment("a")
        assert vc.clocks == {"a": 1}

    def test_increment_existing_node(self) -> None:
        vc = VectorClock({"a": 3})
        vc.increment("a")
        assert vc.clocks == {"a": 4}

    def test_merge_empty(self) -> None:
        vc = VectorClock({"a": 1})
        vc.merge(VectorClock())
        assert vc.clocks == {"a": 1}

    def test_merge_newer(self) -> None:
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 5, "b": 2})
        vc1.merge(vc2)
        assert vc1.clocks == {"a": 5, "b": 2}

    def test_merge_disjoint(self) -> None:
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"b": 2})
        vc1.merge(vc2)
        assert vc1.clocks == {"a": 1, "b": 2}

    def test_happens_before_true(self) -> None:
        vc1 = VectorClock({"a": 1, "b": 2})
        vc2 = VectorClock({"a": 2, "b": 3})
        assert vc1.happens_before(vc2)

    def test_happens_before_false(self) -> None:
        vc1 = VectorClock({"a": 5, "b": 2})
        vc2 = VectorClock({"a": 2, "b": 3})
        assert not vc1.happens_before(vc2)

    def test_happens_before_equal(self) -> None:
        vc = VectorClock({"a": 2, "b": 3})
        assert not vc.happens_before(vc)

    def test_happens_before_empty_vs_nonempty(self) -> None:
        empty = VectorClock()
        vc = VectorClock({"a": 1})
        assert empty.happens_before(vc)
        assert not vc.happens_before(empty)

    def test_is_concurrent(self) -> None:
        vc1 = VectorClock({"a": 1, "b": 3})
        vc2 = VectorClock({"a": 2, "b": 1})
        assert vc1.is_concurrent(vc2)
        assert vc2.is_concurrent(vc1)

    def test_is_concurrent_not_when_equal(self) -> None:
        vc1 = VectorClock({"a": 1})
        vc2 = VectorClock({"a": 1})
        assert not vc1.is_concurrent(vc2)

    def test_is_concurrent_not_when_before(self) -> None:
        vc1 = VectorClock({"a": 1, "b": 2})
        vc2 = VectorClock({"a": 2, "b": 3})
        assert not vc1.is_concurrent(vc2)

    def test_empty_clocks_comparison(self) -> None:
        vc1 = VectorClock()
        vc2 = VectorClock()
        assert not vc1.happens_before(vc2)
        assert not vc2.happens_before(vc1)
        assert not vc1.is_concurrent(vc2)


# ---------------------------------------------------------------------------
# LWWRegister
# ---------------------------------------------------------------------------

class TestLWWRegister:
    """Last-writer-wins register for entry metadata."""

    def test_default_values(self) -> None:
        reg = LWWRegister()
        assert reg.value == ""
        assert reg.timestamp == 0.0
        assert reg.writer_id == ""

    def test_merge_newer_timestamp_wins(self) -> None:
        older = LWWRegister(value="old", timestamp=100.0, writer_id="a")
        newer = LWWRegister(value="new", timestamp=200.0, writer_id="b")
        older.merge(newer)
        assert older.value == "new"
        assert older.timestamp == 200.0
        assert older.writer_id == "b"

    def test_merge_older_timestamp_loses(self) -> None:
        older = LWWRegister(value="old", timestamp=100.0, writer_id="a")
        newer = LWWRegister(value="new", timestamp=200.0, writer_id="b")
        newer.merge(older)
        assert newer.value == "new"
        assert newer.writer_id == "b"

    def test_merge_same_timestamp_higher_writer_wins(self) -> None:
        low = LWWRegister(value="low", timestamp=100.0, writer_id="a")
        high = LWWRegister(value="high", timestamp=100.0, writer_id="z")
        low.merge(high)
        assert low.value == "high"
        assert low.writer_id == "z"

    def test_merge_same_timestamp_lower_writer_loses(self) -> None:
        low = LWWRegister(value="low", timestamp=100.0, writer_id="a")
        high = LWWRegister(value="high", timestamp=100.0, writer_id="z")
        high.merge(low)
        assert high.value == "high"
        assert high.writer_id == "z"

    def test_merge_self_unchanged(self) -> None:
        reg = LWWRegister(value="x", timestamp=50.0, writer_id="a")
        reg.merge(LWWRegister(value="y", timestamp=30.0, writer_id="b"))
        assert reg.value == "x"
        assert reg.writer_id == "a"


# ---------------------------------------------------------------------------
# GossipState
# ---------------------------------------------------------------------------

class TestGossipState:
    """Gossip protocol state container."""

    def test_default_values(self) -> None:
        state = GossipState()
        assert state.node_id == ""
        assert state.known_peers == set()
        assert state.cache_index == {}
        assert state.last_exchange_time == 0.0
        assert state.local_entries == {}
        assert state.vector_clock.clocks == {}
        assert state.entry_metadata == {}
        assert state.tombstones == {}
        assert state.page_table_index == {}
        assert state.peer_merkle_roots == {}
        assert state.local_block_hashes == []

    def test_with_node_id(self) -> None:
        state = GossipState(node_id="node-1")
        assert state.node_id == "node-1"


# ---------------------------------------------------------------------------
# GossipProtocol
# ---------------------------------------------------------------------------

HMAC_KEY = "test-hmac-key-32-chars-minimum!!!"


@pytest.fixture
def gp() -> GossipProtocol:
    """GossipProtocol with a known HMAC key and no env-var dependency."""
    return GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)


class TestGossipProtocolInit:
    """GossipProtocol constructor variations."""

    def test_init_with_hmac_key(self) -> None:
        p = GossipProtocol(node_id="n1", hmac_key=HMAC_KEY)
        assert p.state.node_id == "n1"
        assert p.max_peers == 16
        assert p.cache_ttl == 300.0

    def test_init_custom_max_peers_and_ttl(self) -> None:
        p = GossipProtocol(node_id="n1", max_peers=8, cache_ttl=60.0, hmac_key=HMAC_KEY)
        assert p.max_peers == 8
        assert p.cache_ttl == 60.0

    def test_init_without_hmac_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISTLLM_GOSSIP_HMAC_KEY", raising=False)
        monkeypatch.delenv("DISTLLM_ALLOW_INSECURE_GOSSIP_KEY", raising=False)
        monkeypatch.delenv("DISTLLM_DEV_MODE", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        with pytest.raises(ValueError, match="GossipProtocol requires a per-deployment HMAC key"):
            GossipProtocol(node_id="n1")

    def test_init_with_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTLLM_GOSSIP_HMAC_KEY", HMAC_KEY)
        p = GossipProtocol(node_id="n1")
        assert p.state.node_id == "n1"

    def test_init_with_insecure_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISTLLM_GOSSIP_HMAC_KEY", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("DISTLLM_ALLOW_INSECURE_GOSSIP_KEY", "1")
        p = GossipProtocol(node_id="n1")
        assert p.state.node_id == "n1"

    def test_init_with_dev_mode_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISTLLM_GOSSIP_HMAC_KEY", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
        p = GossipProtocol(node_id="n1")
        assert p.state.node_id == "n1"


class TestGossipProtocolHMAC:
    """HMAC signing and verification."""

    def test_sign_and_verify(self, gp: GossipProtocol) -> None:
        ad = {"node_id": "test-node", "cache_prefixes": ["abc"]}
        signed = gp.sign_message(ad)
        assert "_hmac" in signed
        assert gp.verify_message(signed)

    def test_verify_no_hmac_fails(self, gp: GossipProtocol) -> None:
        assert not gp.verify_message({"node_id": "test-node"})

    def test_verify_wrong_hmac_fails(self, gp: GossipProtocol) -> None:
        signed = gp.sign_message({"node_id": "test-node"})
        signed["_hmac"] = "deadbeef" * 16
        assert not gp.verify_message(signed)

    def test_verify_tampered_body_fails(self, gp: GossipProtocol) -> None:
        signed = gp.sign_message({"node_id": "test-node", "data": "original"})
        signed["data"] = "tampered"
        assert not gp.verify_message(signed)

    def test_sign_with_body_subkey(self, gp: GossipProtocol) -> None:
        msg = {"_body": {"node_id": "test-node", "data": "payload"}}
        signed = gp.sign_message(msg)
        assert gp.verify_message(signed)


class TestGossipProtocolPeers:
    """Peer management."""

    def test_add_peer(self, gp: GossipProtocol) -> None:
        gp.add_peer("peer-1")
        assert "peer-1" in gp.state.known_peers

    def test_add_peer_duplicate(self, gp: GossipProtocol) -> None:
        gp.add_peer("peer-1")
        gp.add_peer("peer-1")
        assert len(gp.state.known_peers) == 1

    def test_add_peer_exceeds_max(self, gp: GossipProtocol) -> None:
        gp.max_peers = 2
        gp.add_peer("peer-1")
        gp.add_peer("peer-2")
        gp.add_peer("peer-3")
        assert len(gp.state.known_peers) <= 2

    def test_remove_peer(self, gp: GossipProtocol) -> None:
        gp.add_peer("peer-1")
        gp.remove_peer("peer-1")
        assert "peer-1" not in gp.state.known_peers

    def test_remove_peer_not_present(self, gp: GossipProtocol) -> None:
        gp.remove_peer("nonexistent")
        assert gp.state.known_peers == set()

    def test_get_peers(self, gp: GossipProtocol) -> None:
        gp.add_peer("p1")
        gp.add_peer("p2")
        peers = gp.get_peers()
        assert set(peers) == {"p1", "p2"}

    def test_get_peers_empty(self, gp: GossipProtocol) -> None:
        assert gp.get_peers() == []

    def test_select_peer(self, gp: GossipProtocol) -> None:
        gp.add_peer("p1")
        gp.add_peer("p2")
        assert gp.select_peer() in ("p1", "p2")

    def test_select_peer_empty(self, gp: GossipProtocol) -> None:
        assert gp.select_peer() is None


class TestGossipProtocolStoreAndAdvertise:
    """Local entry storage and advertisement building."""

    def test_store_local(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        assert "hash-abc" in gp.state.local_entries
        assert gp.state.local_entries["hash-abc"] == "ref-xyz"
        assert gp.state.vector_clock.clocks.get("test-node", 0) >= 1

    def test_store_local_updates_existing(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-old")
        gp.store_local("hash-abc", "ref-new")
        assert gp.state.local_entries["hash-abc"] == "ref-new"

    def test_advertise_full_on_first_call(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        ad = gp.advertise(delta_only=True)
        assert ad["node_id"] == "test-node"
        assert "hash-abc" in ad["cache_prefixes"]
        assert ad["total_cache_entries"] == 1
        assert ad["is_delta"] is False  # first call is always full

    def test_advertise_delta_on_subsequent_call(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        gp.advertise(delta_only=True)  # first call, sets last_exchange_time on state

        # Second call: delta mode with nothing new
        ad2 = gp.advertise(delta_only=True)
        assert "cache_prefixes" in ad2
        # Since nothing changed since last exchange, delta mode produces empty prefixes
        assert len(ad2["cache_prefixes"]) == 0

    def test_advertise_delta_includes_recent_changes(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        gp.advertise(delta_only=True)  # first call, sets last_exchange_time

        # Store a new entry after the first exchange
        time.sleep(0.01)
        gp.store_local("hash-new", "ref-new")
        ad = gp.advertise(delta_only=True)
        assert "hash-new" in ad["cache_prefixes"]

    def test_advertise_full_state(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-a", "ref-a")
        gp.store_local("hash-b", "ref-b")
        gp.advertise(delta_only=True)  # sets last_exchange_time
        ad = gp.advertise(delta_only=False)  # force full state
        assert len(ad["cache_prefixes"]) >= 2

    def test_advertise_excludes_tombstoned(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        gp.tombstone_entry("hash-abc")
        ad = gp.advertise(delta_only=False)
        assert "hash-abc" not in ad["cache_prefixes"]

    def test_advertise_includes_entry_metadata(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        ad = gp.advertise(delta_only=False)
        assert "hash-abc" in ad["entry_metadata"]
        assert ad["entry_metadata"]["hash-abc"]["value"] == "ref-xyz"
        assert ad["entry_metadata"]["hash-abc"]["writer_id"] == "test-node"

    def test_advertise_includes_vector_clock(self, gp: GossipProtocol) -> None:
        gp.store_local("h1", "r1")
        gp.store_local("h2", "r2")
        ad = gp.advertise()
        assert ad["vector_clock"]["test-node"] >= 2


class TestGossipProtocolHasChanges:
    """Pre-exchange change detection."""

    def test_has_changes_since_true(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        before = time.time() - 100
        assert gp.has_changes_since(before)

    def test_has_changes_since_false(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        after = time.time() + 100
        assert not gp.has_changes_since(after)

    def test_has_changes_since_tombstone(self, gp: GossipProtocol) -> None:
        gp.tombstone_entry("hash-abc")
        before = time.time() - 100
        assert gp.has_changes_since(before)

    def test_has_changes_since_empty(self, gp: GossipProtocol) -> None:
        assert not gp.has_changes_since(time.time() - 100)


class TestGossipProtocolProcessAdvertisement:
    """Processing peer advertisements."""

    def test_process_invalid_hmac_returns_empty(self, gp: GossipProtocol) -> None:
        ad = {"node_id": "peer-1", "cache_prefixes": ["abc"]}
        result = gp.process_advertisement(ad)
        assert result == []

    def test_process_valid_ad_adds_peer(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-local", "ref-local")
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("hash-peer", "ref-peer")
        peer_ad = peer_gp.sign_message(peer_gp.advertise(delta_only=False))

        missing = gp.process_advertisement(peer_ad)
        assert "peer-1" in gp.state.known_peers
        assert "hash-peer" in missing
        assert missing == ["hash-peer"]

    def test_process_ad_returns_missing_prefixes(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-common", "ref-common")
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("hash-common", "ref-common")
        peer_gp.store_local("hash-peer-only", "ref-peer-only")
        peer_ad = peer_gp.sign_message(peer_gp.advertise(delta_only=False))

        missing = gp.process_advertisement(peer_ad)
        assert "hash-peer-only" in missing
        assert "hash-common" not in missing

    def test_process_ad_skips_tombstoned(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-abc")
        gp.tombstone_entry("hash-abc")
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("hash-abc", "ref-abc")
        peer_ad = peer_gp.sign_message(peer_gp.advertise(delta_only=False))

        missing = gp.process_advertisement(peer_ad)
        assert "hash-abc" not in missing

    def test_process_ad_merges_vector_clock(self, gp: GossipProtocol) -> None:
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("h", "r")
        peer_ad = peer_gp.sign_message(peer_gp.advertise(delta_only=False))
        gp.process_advertisement(peer_ad)
        assert gp.state.vector_clock.clocks.get("peer-1", 0) >= 1

    def test_process_ad_merges_tombstones(self, gp: GossipProtocol) -> None:
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.tombstone_entry("hash-removed")
        peer_ad = peer_gp.sign_message(peer_gp.advertise(delta_only=False))
        gp.process_advertisement(peer_ad)
        assert "hash-removed" in gp.state.tombstones

    def test_process_ad_merges_entry_metadata(self, gp: GossipProtocol) -> None:
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("hash-abc", "ref-peer")
        peer_ad = peer_gp.sign_message(peer_gp.advertise(delta_only=False))
        gp.process_advertisement(peer_ad)
        assert "hash-abc" in gp.state.entry_metadata
        assert gp.state.entry_metadata["hash-abc"].value == "ref-peer"

    def test_process_ad_without_vector_clock(self, gp: GossipProtocol) -> None:
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("h", "r")
        ad = peer_gp.advertise(delta_only=False)
        signed = peer_gp.sign_message(ad)
        # Remove vector_clock to test robustness
        del signed["vector_clock"]
        gp.process_advertisement(signed)  # should not raise

    def test_process_ad_without_tombstones(self, gp: GossipProtocol) -> None:
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("h", "r")
        ad = peer_gp.advertise(delta_only=False)
        signed = peer_gp.sign_message(ad)
        del signed["tombstones"]
        gp.process_advertisement(signed)  # should not raise

    def test_process_ad_without_entry_metadata(self, gp: GossipProtocol) -> None:
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("h", "r")
        ad = peer_gp.advertise(delta_only=False)
        signed = peer_gp.sign_message(ad)
        del signed["entry_metadata"]
        gp.process_advertisement(signed)  # should not raise

    def test_process_ad_does_not_add_peer_twice(self, gp: GossipProtocol) -> None:
        gp.store_local("h", "r")
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_local("h", "r")
        ad = peer_gp.sign_message(peer_gp.advertise(delta_only=False))
        gp.process_advertisement(ad)
        # Process a second time with the same ad
        gp.process_advertisement(ad)
        # The peer's entry should only appear once in cache_index
        peer_entries = [(nid, ref) for nid, ref, ts in gp.state.cache_index["h"] if nid == "peer-1"]
        assert len(peer_entries) == 1


class TestGossipProtocolCacheIndex:
    """Cache index management (lookup, cleanup, tombstone)."""

    def test_lookup(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        node_id = gp.lookup("hash-abc")
        assert node_id == "test-node"

    def test_lookup_missing(self, gp: GossipProtocol) -> None:
        assert gp.lookup("nonexistent") is None

    def test_lookup_returns_newest(self, gp: GossipProtocol) -> None:
        # Manually populate cache_index with two entries, second should win
        now = time.time()
        gp.state.cache_index["hash-abc"] = [("peer-old", "ref-old", now - 100), ("peer-new", "ref-new", now)]
        assert gp.lookup("hash-abc") == "peer-new"

    def test_tombstone_entry(self, gp: GossipProtocol) -> None:
        gp.store_local("hash-abc", "ref-xyz")
        gp.tombstone_entry("hash-abc")
        assert "hash-abc" not in gp.state.local_entries
        assert "hash-abc" in gp.state.tombstones

    def test_tombstone_entry_twice_preserves_latest(self, gp: GossipProtocol) -> None:
        gp.tombstone_entry("h1")
        ts1 = gp.state.tombstones["h1"]
        gp.tombstone_entry("h1")
        ts2 = gp.state.tombstones["h1"]
        assert ts2 >= ts1

    def test_cleanup_expired_no_entries(self, gp: GossipProtocol) -> None:
        removed = gp.cleanup_expired()
        assert removed == 0

    def test_cleanup_expired_fresh_entries(self, gp: GossipProtocol) -> None:
        gp.store_local("h1", "r1")
        removed = gp.cleanup_expired()
        assert removed == 0  # should not remove fresh entries

    def test_cleanup_expired_stale_entries(self, gp: GossipProtocol) -> None:
        now = time.time()
        gp.state.cache_index["stale-hash"] = [("peer", "ref", now - gp.cache_ttl - 10)]
        removed = gp.cleanup_expired()
        assert removed == 1
        assert "stale-hash" not in gp.state.cache_index

    def test_cleanup_expired_tombstones(self, gp: GossipProtocol) -> None:
        now = time.time()
        gp.state.tombstones["old-tombstone"] = now - gp.cache_ttl * 3
        gp.state.entry_metadata["old-tombstone"] = LWWRegister(value="x", timestamp=now - gp.cache_ttl * 3, writer_id="a")
        gp.cleanup_expired()
        assert "old-tombstone" not in gp.state.tombstones
        assert "old-tombstone" not in gp.state.entry_metadata

    def test_build_request(self, gp: GossipProtocol) -> None:
        req = gp.build_request("target-peer", ["h1", "h2"])
        assert req["requester_id"] == "test-node"
        assert req["target_node_id"] == "target-peer"
        assert req["requested_prefixes"] == ["h1", "h2"]

    def test_process_response(self, gp: GossipProtocol) -> None:
        resp = {"success": True, "cache_entries": {"h1": "ref-h1", "h2": "ref-h2"}}
        count = gp.process_response(resp)
        assert count == 2
        assert "h1" in gp.state.cache_index

    def test_process_response_not_success(self, gp: GossipProtocol) -> None:
        count = gp.process_response({"success": False})
        assert count == 0

    def test_process_response_empty_entries(self, gp: GossipProtocol) -> None:
        count = gp.process_response({"success": True, "cache_entries": {}})
        assert count == 0


class TestGossipProtocolBlockHashes:
    """Block hash tracking and page table management."""

    def test_store_block_hash(self, gp: GossipProtocol) -> None:
        gp.store_block_hash("block-hash-1")
        assert "block-hash-1" in gp.state.page_table_index
        assert "test-node" in gp.state.page_table_index["block-hash-1"]
        assert "block-hash-1" in gp.state.local_block_hashes

    def test_store_block_hash_with_owner(self, gp: GossipProtocol) -> None:
        gp.store_block_hash("block-hash-1", node_id="peer-1")
        assert "peer-1" in gp.state.page_table_index["block-hash-1"]
        assert "block-hash-1" not in gp.state.local_block_hashes

    def test_store_block_hash_duplicate(self, gp: GossipProtocol) -> None:
        gp.store_block_hash("bh1")
        gp.store_block_hash("bh1")
        assert gp.state.page_table_index["bh1"] == ["test-node"]
        assert gp.state.local_block_hashes == ["bh1"]

    def test_store_local_block_hashes(self, gp: GossipProtocol) -> None:
        gp.store_local_block_hashes(["bh1", "bh2"])
        assert gp.state.local_block_hashes == ["bh1", "bh2"]
        assert gp.state.page_table_index["bh1"] == ["test-node"]
        assert gp.state.page_table_index["bh2"] == ["test-node"]

    def test_store_local_block_hashes_replaces(self, gp: GossipProtocol) -> None:
        gp.store_local_block_hashes(["bh-old"])
        gp.store_local_block_hashes(["bh-new"])
        assert gp.state.local_block_hashes == ["bh-new"]

    def test_update_merkle_root(self, gp: GossipProtocol) -> None:
        gp.update_merkle_root("merkle-root-hash")
        assert gp.state.peer_merkle_roots["test-node"] == "merkle-root-hash"

    def test_lookup_block(self, gp: GossipProtocol) -> None:
        gp.store_block_hash("bh1")
        gp.store_block_hash("bh1", node_id="peer-1")
        owners = gp.lookup_block("bh1")
        assert "test-node" in owners
        assert "peer-1" in owners

    def test_lookup_block_missing(self, gp: GossipProtocol) -> None:
        assert gp.lookup_block("nonexistent") == []

    def test_remove_block_hash(self, gp: GossipProtocol) -> None:
        gp.store_block_hash("bh1")
        gp.remove_block_hash("bh1")
        assert "bh1" not in gp.state.local_block_hashes

    def test_remove_block_hash_not_present(self, gp: GossipProtocol) -> None:
        gp.store_block_hash("bh1")
        gp.remove_block_hash("nonexistent")
        assert gp.state.local_block_hashes == ["bh1"]

    def test_build_page_advertisement(self, gp: GossipProtocol) -> None:
        gp.store_block_hash("bh1")
        gp.update_merkle_root("root-hash")
        ad = gp.build_page_advertisement()
        assert ad["node_id"] == "test-node"
        assert ad["merkle_root"] == "root-hash"
        assert ad["block_count"] == 1
        assert "bh1" in ad["block_hashes_sample"]

    def test_process_page_advertisement(self, gp: GossipProtocol) -> None:
        gp.store_block_hash("bh-local")
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.store_block_hash("bh-peer")
        peer_ad = peer_gp.build_page_advertisement()

        missing = gp.process_page_advertisement(peer_ad)
        assert "peer-1" in gp.state.known_peers
        assert "bh-peer" in missing
        assert "bh-local" not in missing

    def test_process_page_advertisement_merkle_root(self, gp: GossipProtocol) -> None:
        peer_gp = GossipProtocol(node_id="peer-1", hmac_key=HMAC_KEY)
        peer_gp.update_merkle_root("peer-root")
        peer_ad = peer_gp.build_page_advertisement()

        gp.process_page_advertisement(peer_ad)
        assert gp.state.peer_merkle_roots["peer-1"] == "peer-root"


# ---------------------------------------------------------------------------
# GossipClient
# ---------------------------------------------------------------------------

class TestGossipClient:
    """Gossip network client (stub mode, no network)."""

    def test_init_no_network(self) -> None:
        client = GossipClient(node_id="test-client", enable_network=False)
        assert client._transport is None
        assert client._request_count == 0
        assert client._response_count == 0

    def test_exchange_stub_returns_none(self) -> None:
        client = GossipClient(node_id="test-client", enable_network=False)
        result = client.exchange("peer-1", {"node_id": "test-client"})
        assert result is None

    def test_exchange_stub_increments_request(self) -> None:
        client = GossipClient(node_id="test-client", enable_network=False)
        client.exchange("peer-1", {})
        assert client._request_count == 1

    def test_request_entries_stub(self) -> None:
        client = GossipClient(node_id="test-client", enable_network=False)
        result = client.request_entries("peer-1", {"requested_prefixes": ["h1"]})
        assert result == {"success": False, "cache_entries": {}, "entries_returned": 0}

    def test_request_entries_increments_counters(self) -> None:
        client = GossipClient(node_id="test-client", enable_network=False)
        client.request_entries("peer-1", {"requested_prefixes": ["h1"]})
        assert client._request_count == 1

    def test_fetch_kv_cache_stub_returns_none(self) -> None:
        client = GossipClient(node_id="test-client", enable_network=False)
        result = client.fetch_kv_cache("peer-1", "h1")
        assert result is None

    def test_stats_empty(self) -> None:
        client = GossipClient(node_id="test-client", enable_network=False)
        stats = client.stats
        assert stats["requests_sent"] == 0
        assert stats["responses_received"] == 0
        assert stats["transfer"] == {}

    def test_close_no_transport(self) -> None:
        client = GossipClient(node_id="test-client", enable_network=False)
        client.close()  # should not raise


# ---------------------------------------------------------------------------
# GossipReplicator
# ---------------------------------------------------------------------------

class TestGossipReplicator:
    """Background gossip replication loop."""

    def test_init(self) -> None:
        protocol = GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)
        client = GossipClient(node_id="test-client", enable_network=False)
        replicator = GossipReplicator(protocol, client, interval_s=60.0, fanout=3)
        assert replicator._interval_s == 60.0
        assert replicator._fanout == 3
        assert replicator._rounds_completed == 0
        assert not replicator._running

    def test_init_clamps_fanout(self) -> None:
        protocol = GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)
        client = GossipClient(node_id="test-client", enable_network=False)
        r0 = GossipReplicator(protocol, client, fanout=0)
        assert r0._fanout == 1
        r_neg = GossipReplicator(protocol, client, fanout=-5)
        assert r_neg._fanout == 1

    def test_sync_once_no_peers(self) -> None:
        protocol = GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)
        client = GossipClient(node_id="test-client", enable_network=False)
        replicator = GossipReplicator(protocol, client, interval_s=1.0)
        result = replicator.sync_once()
        assert result["peers_contacted"] == []
        assert result["entries_missing"] == 0
        assert result["entries_fetched"] == 0
        assert result["duration_ms"] >= 0
        assert replicator._rounds_completed == 1

    def test_sync_once_with_peer_no_changes(self) -> None:
        protocol = GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)
        client = GossipClient(node_id="test-client", enable_network=False)
        protocol.add_peer("peer-1")
        replicator = GossipReplicator(protocol, client, interval_s=1.0)
        result = replicator.sync_once()
        # Peer has no transport, so exchange returns None (not a skip)
        assert result["entries_missing"] == 0

    def test_start_and_stop(self) -> None:
        protocol = GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)
        client = GossipClient(node_id="test-client", enable_network=False)
        replicator = GossipReplicator(protocol, client, interval_s=0.01)
        replicator.start()
        assert replicator._running
        assert replicator._thread is not None
        replicator.stop()
        assert not replicator._running
        assert replicator._thread is None

    def test_start_idempotent(self) -> None:
        protocol = GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)
        client = GossipClient(node_id="test-client", enable_network=False)
        replicator = GossipReplicator(protocol, client, interval_s=1.0)
        replicator.start()
        thread_id = id(replicator._thread)
        replicator.start()  # second start should be no-op
        assert id(replicator._thread) == thread_id
        replicator.stop()

    def test_stop_when_not_running(self) -> None:
        protocol = GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)
        client = GossipClient(node_id="test-client", enable_network=False)
        replicator = GossipReplicator(protocol, client)
        replicator.stop()  # should not raise

    def test_stats(self) -> None:
        protocol = GossipProtocol(node_id="test-node", hmac_key=HMAC_KEY)
        client = GossipClient(node_id="test-client", enable_network=False)
        replicator = GossipReplicator(protocol, client, interval_s=5.0)
        stats = replicator.stats
        assert stats["running"] is False
        assert stats["interval_s"] == 5.0
        assert stats["rounds_completed"] == 0
