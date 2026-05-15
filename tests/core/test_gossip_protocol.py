"""Tests for anti-entropy gossip protocol."""

import time
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
