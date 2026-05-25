"""Tests for anti-entropy gossip protocol."""

import time
from unittest.mock import MagicMock

import pytest
from distllm.core.gossip_protocol import GossipProtocol, GossipState


class TestGossipProtocol:
    """Test GossipProtocol basic operations."""

    def test_add_peer(self):
        gp = GossipProtocol("node-1")
        gp.add_peer("node-2")
        assert "node-2" in gp.get_peers()

    def test_add_peer_exceeds_max(self):
        gp = GossipProtocol("node-1", max_peers=2)
        gp.add_peer("node-2")
        gp.add_peer("node-3")
        gp.add_peer("node-4")
        assert len(gp.get_peers()) <= 2

    def test_remove_peer(self):
        gp = GossipProtocol("node-1")
        gp.add_peer("node-2")
        gp.remove_peer("node-2")
        assert "node-2" not in gp.get_peers()

    def test_store_local(self):
        gp = GossipProtocol("node-1")
        gp.store_local("h123", "ref-1")
        assert "h123" in gp.state.local_entries

    def test_select_peer_no_peers(self):
        gp = GossipProtocol("node-1")
        assert gp.select_peer() is None

    def test_select_peer_returns_peer(self):
        gp = GossipProtocol("node-1")
        gp.add_peer("node-2")
        assert gp.select_peer() == "node-2"

    def test_get_peers(self):
        gp = GossipProtocol("node-1")
        gp.add_peer("node-2")
        gp.add_peer("node-3")
        peers = gp.get_peers()
        assert set(peers) == {"node-2", "node-3"}

    def test_lookup_in_cache_index(self):
        gp = GossipProtocol("node-1")
        gp.state.cache_index["h123"] = [("node-2", "ref-2", time.time())]
        assert gp.lookup("h123") == "node-2"

    def test_lookup_not_found(self):
        gp = GossipProtocol("node-1")
        assert gp.lookup("h999") is None


class TestGossipAdvertisement:
    """Test advertisement building and processing."""

    def test_advertise_returns_prefixes(self):
        gp = GossipProtocol("node-1")
        gp.store_local("h1", "ref-1")
        gp.store_local("h2", "ref-2")
        ad = gp.advertise()
        assert ad["node_id"] == "node-1"
        assert ad["total_cache_entries"] == 2
        assert set(ad["cache_prefixes"]) == {"h1", "h2"}

    def test_process_advertisement_finds_missing(self):
        gp = GossipProtocol("node-1")
        gp.store_local("h1", "ref-1")

        peer_ad = {
            "node_id": "node-2",
            "cache_prefixes": ["h1", "h2", "h3"],
            "total_cache_entries": 3,
            "timestamp": time.time(),
        }
        missing = gp.process_advertisement(peer_ad)
        assert set(missing) == {"h2", "h3"}
        assert "node-2" in gp.get_peers()

    def test_process_advertisement_updates_cache_index(self):
        gp = GossipProtocol("node-1")
        peer_ad = {
            "node_id": "node-2",
            "cache_prefixes": ["h1"],
            "total_cache_entries": 1,
            "timestamp": time.time(),
        }
        gp.process_advertisement(peer_ad)
        assert gp.lookup("h1") == "node-2"

    def test_process_advertisement_no_missing(self):
        gp = GossipProtocol("node-1")
        gp.store_local("h1", "ref-1")

        peer_ad = {
            "node_id": "node-2",
            "cache_prefixes": ["h1"],
            "total_cache_entries": 1,
            "timestamp": time.time(),
        }
        missing = gp.process_advertisement(peer_ad)
        assert missing == []


class TestGossipRequestResponse:
    """Test request building and response processing."""

    def test_build_request(self):
        gp = GossipProtocol("node-1")
        req = gp.build_request("node-2", ["h1", "h2"])
        assert req["requester_id"] == "node-1"
        assert req["target_node_id"] == "node-2"
        assert set(req["requested_prefixes"]) == {"h1", "h2"}

    def test_process_response_success(self):
        gp = GossipProtocol("node-1")
        gp.state.cache_index["h1"] = [("node-2", "", time.time())]

        response = {
            "success": True,
            "cache_entries": {"h1": "full-ref-1"},
            "entries_returned": 1,
        }
        count = gp.process_response(response)
        assert count == 1
        # Entry ref should be updated
        entries = gp.state.cache_index["h1"]
        assert any(ref == "full-ref-1" for _, ref, _ in entries)

    def test_process_response_failure(self):
        gp = GossipProtocol("node-1")
        response = {"success": False, "error_message": "not found"}
        count = gp.process_response(response)
        assert count == 0


class TestGossipCleanup:
    """Test expired entry cleanup."""

    def test_cleanup_expired(self):
        gp = GossipProtocol("node-1", cache_ttl=0.1)
        gp.state.cache_index["h1"] = [("node-2", "ref-2", time.time() - 10)]
        gp.state.cache_index["h2"] = [("node-3", "ref-3", time.time())]

        removed = gp.cleanup_expired()
        assert removed == 1
        assert "h1" not in gp.state.cache_index
        assert "h2" in gp.state.cache_index

    def test_cleanup_no_expired(self):
        gp = GossipProtocol("node-1", cache_ttl=300.0)
        gp.state.cache_index["h1"] = [("node-2", "ref-2", time.time())]
        removed = gp.cleanup_expired()
        assert removed == 0
        assert "h1" in gp.state.cache_index


class TestGossipHMAC:
    """Tests for HMAC message signing and verification."""

    def test_sign_and_verify(self):
        gp = GossipProtocol("node-1", hmac_key="test-key")
        msg = {"node_id": "node-1", "cache_prefixes": ["h1", "h2"]}
        signed = gp.sign_message(msg)
        assert "_hmac" in signed
        assert signed["node_id"] == "node-1"
        assert gp.verify_message(signed) is True

    def test_verify_fails_with_wrong_key(self):
        gp1 = GossipProtocol("node-1", hmac_key="key-a")
        gp2 = GossipProtocol("node-2", hmac_key="key-b")
        msg = {"node_id": "node-1", "cache_prefixes": ["h1"]}
        signed = gp1.sign_message(msg)
        assert gp2.verify_message(signed) is False

    def test_verify_fails_no_signature(self):
        gp = GossipProtocol("node-1")
        msg = {"node_id": "node-1"}
        assert gp.verify_message(msg) is False

    def test_verify_tampered_body(self):
        gp = GossipProtocol("node-1", hmac_key="test-key")
        msg = {"node_id": "node-1", "cache_prefixes": ["h1"]}
        signed = gp.sign_message(msg)
        signed["cache_prefixes"] = ["h2"]
        assert gp.verify_message(signed) is False

    def test_sign_with_body_field(self):
        gp = GossipProtocol("node-1", hmac_key="test-key")
        msg = {"_body": {"inner": "data"}, "extra": "field"}
        signed = gp.sign_message(msg)
        assert "_hmac" in signed
        assert gp.verify_message(signed) is True


class TestGossipVectorClock:
    """Tests for CRDT vector clock operations."""

    def test_increment(self):
        from distllm.core.gossip_protocol import VectorClock
        vc = VectorClock()
        vc.increment("node-a")
        assert vc.clocks["node-a"] == 1
        vc.increment("node-a")
        assert vc.clocks["node-a"] == 2

    def test_merge_takes_max(self):
        from distllm.core.gossip_protocol import VectorClock
        vc1 = VectorClock(clocks={"a": 3, "b": 1})
        vc2 = VectorClock(clocks={"a": 2, "b": 5, "c": 1})
        vc1.merge(vc2)
        assert vc1.clocks == {"a": 3, "b": 5, "c": 1}

    def test_happens_before(self):
        from distllm.core.gossip_protocol import VectorClock
        older = VectorClock(clocks={"a": 1, "b": 2})
        newer = VectorClock(clocks={"a": 2, "b": 2})
        assert older.happens_before(newer)
        assert not newer.happens_before(older)

    def test_concurrent_clocks(self):
        from distllm.core.gossip_protocol import VectorClock
        vc1 = VectorClock(clocks={"a": 2, "b": 1})
        vc2 = VectorClock(clocks={"a": 1, "b": 2})
        assert vc1.is_concurrent(vc2)

    def test_not_concurrent_identical(self):
        from distllm.core.gossip_protocol import VectorClock
        vc1 = VectorClock(clocks={"a": 1})
        vc2 = VectorClock(clocks={"a": 1})
        assert not vc1.is_concurrent(vc2)


class TestGossipTombstone:
    """Tests for tombstone-based deletion."""

    def test_tombstone_entry(self):
        gp = GossipProtocol("node-1")
        gp.store_local("h1", "ref-1")
        assert "h1" in gp.state.local_entries
        gp.tombstone_entry("h1")
        assert "h1" not in gp.state.local_entries
        assert "h1" in gp.state.tombstones

    def test_tombstone_excludes_from_advertise(self):
        gp = GossipProtocol("node-1")
        gp.store_local("h1", "ref-1")
        gp.store_local("h2", "ref-2")
        gp.tombstone_entry("h1")
        ad = gp.advertise()
        assert ad["total_cache_entries"] == 1
        assert "h1" not in ad["cache_prefixes"]
        assert "h2" in ad["cache_prefixes"]

    def test_process_advertisement_respects_tombstones(self):
        gp = GossipProtocol("node-1")
        gp.tombstone_entry("h1")
        peer_ad = {
            "node_id": "node-2",
            "cache_prefixes": ["h1", "h2"],
            "total_cache_entries": 2,
            "timestamp": time.time(),
            "tombstones": {},
            "entry_metadata": {},
        }
        missing = gp.process_advertisement(peer_ad)
        assert "h1" not in missing  # Tombstoned locally
        assert "h2" in missing

    def test_tombstone_merge(self):
        gp = GossipProtocol("node-1")
        peer_ad = {
            "node_id": "node-2",
            "cache_prefixes": [],
            "total_cache_entries": 0,
            "timestamp": time.time(),
            "tombstones": {"h1": time.time() + 100},
            "entry_metadata": {},
        }
        gp.process_advertisement(peer_ad)
        assert "h1" in gp.state.tombstones


class TestGossipEndToEnd:
    """End-to-end advertise → exchange → process flow."""

    def test_two_node_exchange(self):
        gp1 = GossipProtocol("node-1")
        gp2 = GossipProtocol("node-2")
        gp1.add_peer("node-2")
        gp2.add_peer("node-1")

        gp1.store_local("h1", "ref-a")
        gp1.store_local("h2", "ref-b")
        gp2.store_local("h3", "ref-c")

        ad1 = gp1.advertise()
        ad2 = gp2.advertise()

        missing1 = gp1.process_advertisement(ad2)
        missing2 = gp2.process_advertisement(ad1)

        assert set(missing1) == {"h3"}
        assert set(missing2) == {"h1", "h2"}

    def test_three_node_convergence(self):
        nodes = [GossipProtocol(f"node-{i}") for i in range(3)]
        for n in nodes:
            for peer in nodes:
                if peer is not n:
                    n.add_peer(peer.state.node_id)

        nodes[0].store_local("h-a", "ref-a")
        nodes[1].store_local("h-b", "ref-b")
        nodes[2].store_local("h-c", "ref-c")

        # Full mesh exchange
        for i, n in enumerate(nodes):
            for j, peer in enumerate(nodes):
                if i != j:
                    ad = peer.advertise()
                    n.process_advertisement(ad)

        # All nodes should know about all entries
        for n in nodes:
            for h in ("h-a", "h-b", "h-c"):
                assert n.lookup(h) is not None

    def test_lww_register_merge(self):
        from distllm.core.gossip_protocol import LWWRegister
        r1 = LWWRegister(value="old", timestamp=100.0, writer_id="a")
        r2 = LWWRegister(value="new", timestamp=200.0, writer_id="b")
        r1.merge(r2)
        assert r1.value == "new"
        assert r1.writer_id == "b"

    def test_lww_register_tiebreak(self):
        from distllm.core.gossip_protocol import LWWRegister
        r1 = LWWRegister(value="a-val", timestamp=100.0, writer_id="a")
        r2 = LWWRegister(value="b-val", timestamp=100.0, writer_id="b")
        r1.merge(r2)
        assert r1.value == "b-val"

    def test_advertise_includes_crdt_state(self):
        gp = GossipProtocol("node-1")
        gp.store_local("h1", "ref-1")
        ad = gp.advertise()
        assert "vector_clock" in ad
        assert "tombstones" in ad
        assert "entry_metadata" in ad

    def test_process_advertisement_merges_vector_clock(self):
        gp = GossipProtocol("node-1")
        gp.state.vector_clock.clocks["node-1"] = 5
        peer_ad = {
            "node_id": "node-2",
            "cache_prefixes": [],
            "total_cache_entries": 0,
            "timestamp": time.time(),
            "vector_clock": {"node-1": 3, "node-2": 7},
            "tombstones": {},
            "entry_metadata": {},
        }
        gp.process_advertisement(peer_ad)
        assert gp.state.vector_clock.clocks["node-1"] == 5  # max(5, 3)
        assert gp.state.vector_clock.clocks["node-2"] == 7

    def test_cache_index_update_from_response(self):
        gp = GossipProtocol("node-1")
        # Simulate: node-2 advertised h1, we don't have it
        gp.state.cache_index["h1"] = [("node-2", "", time.time())]
        resp = {
            "success": True,
            "cache_entries": {"h1": "full-ref-val"},
            "entries_returned": 1,
        }
        count = gp.process_response(resp)
        assert count == 1
        entries = gp.state.cache_index["h1"]
        assert any(ref == "full-ref-val" for _, ref, _ in entries)


class TestGossipReplicator:
    """Tests for GossipReplicator periodic sync loop."""

    def test_init(self):
        from distllm.core.gossip_protocol import GossipReplicator
        gp = GossipProtocol("node-1")
        client = MagicMock()
        rep = GossipReplicator(gp, client, interval_s=10.0)
        assert rep._interval_s == 10.0
        assert not rep._running
        assert rep._rounds_completed == 0

    def test_sync_once_no_peers(self):
        from distllm.core.gossip_protocol import GossipReplicator
        gp = GossipProtocol("node-1")
        client = MagicMock()
        rep = GossipReplicator(gp, client)
        result = rep.sync_once()
        assert result["peer"] is None
        assert result["entries_missing"] == 0

    def test_sync_once_with_peer(self):
        from distllm.core.gossip_protocol import GossipReplicator
        gp = GossipProtocol("node-1")
        gp.store_local("h1", "ref-1")
        gp.add_peer("node-2")
        client = MagicMock()
        client.exchange.return_value = {
            "node_id": "node-2",
            "cache_prefixes": ["h2"],
            "total_cache_entries": 1,
            "timestamp": time.time(),
            "vector_clock": {},
            "tombstones": {},
            "entry_metadata": {},
        }
        client.request_entries.return_value = {
            "success": True,
            "cache_entries": {"h2": "ref-2"},
            "entries_returned": 1,
        }
        rep = GossipReplicator(gp, client)
        result = rep.sync_once()
        assert result["peer"] == "node-2"
        assert result["entries_missing"] == 1
        assert result["entries_fetched"] == 1
        assert result["duration_ms"] >= 0

    def test_sync_once_exchange_fails(self):
        from distllm.core.gossip_protocol import GossipReplicator
        gp = GossipProtocol("node-1")
        gp.add_peer("node-2")
        client = MagicMock()
        client.exchange.return_value = None
        rep = GossipReplicator(gp, client)
        result = rep.sync_once()
        assert result["peer"] == "node-2"
        assert result["entries_missing"] == 0

    def test_start_stop(self):
        from distllm.core.gossip_protocol import GossipReplicator
        gp = GossipProtocol("node-1")
        client = MagicMock()
        rep = GossipReplicator(gp, client, interval_s=0.01)
        rep.start()
        assert rep._running
        import time
        time.sleep(0.05)
        rep.stop()
        assert not rep._running
        assert rep._rounds_completed >= 0

    def test_double_start(self):
        from distllm.core.gossip_protocol import GossipReplicator
        gp = GossipProtocol("node-1")
        client = MagicMock()
        rep = GossipReplicator(gp, client)
        rep.start()
        rep.start()  # Should not raise
        rep.stop()

    def test_stats(self):
        from distllm.core.gossip_protocol import GossipReplicator
        gp = GossipProtocol("node-1")
        client = MagicMock()
        rep = GossipReplicator(gp, client, interval_s=5.0)
        stats = rep.stats
        assert stats["interval_s"] == 5.0
        assert not stats["running"]
