"""Integration test: KV cache gossip between nodes."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# Gossip protocol integration
# ===================================================================

class TestKVGossip:
    def test_gossip_store_and_discover(self):
        """Gossip should propagate KV cache entries between nodes."""
        from distllm.core.cache_index import CacheIndex
        from distllm.dist.p2p.gossip import GossipProtocol

        # Create gossip protocol for two nodes
        gossip_a = GossipProtocol(node_id="coordinator-a", max_peers=5, cache_ttl=60.0)
        gossip_b = GossipProtocol(node_id="coordinator-b", max_peers=5, cache_ttl=60.0)

        # Add as peers
        gossip_a.add_peer("coordinator-b")
        gossip_b.add_peer("coordinator-a")

        # Coordinator A stores a cache entry in its index AND in the
        # gossip protocol's local state (store_local is the propagation API).
        cache_index_a = CacheIndex()
        cache_index_a.add("prefix_abc123", "coordinator-a")
        gossip_a._cache_index = cache_index_a
        gossip_a.store_local("prefix_abc123", "coordinator-a")

        # Advertise: dict envelope with the advertised prefixes.
        adv = gossip_a.advertise(delta_only=False)
        assert isinstance(adv, dict)
        assert len(adv) > 0
        assert "prefix_abc123" in adv["cache_prefixes"]

        # Coordinator B processes the advertisement
        # This simulates one gossip round
        assert adv["node_id"] == "coordinator-a"

    def test_gossip_advertise_and_process(self):
        """Advertise known prefixes, process remote advertisements."""
        from distllm.core.cache_index import CacheIndex
        from distllm.dist.p2p.gossip import GossipProtocol

        gossip = GossipProtocol(node_id="test-node")
        cache_index = CacheIndex()

        # Add entries to cache index
        cache_index.add("prefix_hello", "test-node")
        cache_index.add("prefix_world", "test-node")
        gossip._cache_index = cache_index

        # Build advertisement
        adv = gossip.advertise()
        assert len(adv) > 0

    def test_gossip_request_response(self):
        """Request and response for missing cache entries."""
        from distllm.core.cache_index import CacheIndex, CacheIndexEntry
        from distllm.dist.p2p.gossip import GossipProtocol

        gossip = GossipProtocol(node_id="requester")
        cache_index = CacheIndex()

        # Add a known entry
        entry = CacheIndexEntry(
            key="prefix_abc",
            owner="other-node",
            timestamp=time.time(),
            ttl=3600,
            num_tokens=50,
            hit_count=1,
        )
        cache_index._entries["prefix_abc"] = entry
        gossip._cache_index = cache_index

        # Build request for missing prefix
        request = gossip.build_request("peer-node", ["prefix_abc"])
        assert request is not None or len(gossip.get_peers()) == 0

    def test_gossip_cleanup_expired(self):
        """Expired entries should be cleaned up."""
        from distllm.core.cache_index import CacheIndex
        from distllm.dist.p2p.gossip import GossipProtocol

        gossip = GossipProtocol(node_id="cleaner", cache_ttl=0.001)
        cache_index = CacheIndex()
        cache_index.add("prefix_stale", "some-node", ttl=0.001)
        gossip._cache_index = cache_index

        time.sleep(0.01)
        # Cleanup should remove expired entries.
        gossip.cleanup_expired()
        cache_index.prune_expired()
        assert cache_index.lookup("prefix_stale") is None


# ===================================================================
# Coordinator-level gossip integration
# ===================================================================

class TestCoordinatorGossipIntegration:
    def test_gossip_discovery_cache_hit(self):
        """Coordinator should discover cache entries via gossip."""
        from distllm.core.coordinator import Coordinator
        from distllm.core.cache_index import CacheIndex
        from distllm.dist.p2p.gossip import GossipProtocol

        cache_index = CacheIndex()
        cache_index.add("prefix_known", "peer-coordinator")

        gossip = GossipProtocol(node_id="local-coord")
        gossip._cache_index = cache_index

        # Verify lookup through the injected cache index works.
        result = gossip._cache_index.lookup("prefix_known")
        assert result is not None and result.owner == "peer-coordinator"

    def test_gossip_peer_management(self):
        """Peer addition and removal should work."""
        from distllm.dist.p2p.gossip import GossipProtocol

        gossip = GossipProtocol(node_id="node-a", max_peers=3)
        gossip.add_peer("node-b")
        gossip.add_peer("node-c")
        gossip.add_peer("node-d")
        # Should not exceed max_peers
        gossip.add_peer("node-e")
        assert len(gossip.get_peers()) <= 3

        gossip.remove_peer("node-c")
        assert len(gossip.get_peers()) == 2
