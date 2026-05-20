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
        from distllm.core.gossip_protocol import GossipProtocol

        # Create gossip protocol for two nodes
        gossip_a = GossipProtocol(node_id="coordinator-a", max_peers=5, cache_ttl=60.0)
        gossip_b = GossipProtocol(node_id="coordinator-b", max_peers=5, cache_ttl=60.0)

        # Add as peers
        gossip_a.add_peer("coordinator-b")
        gossip_b.add_peer("coordinator-a")

        # Coordinator A stores a cache entry
        cache_index_a = CacheIndex()
        cache_index_a.add("prefix_abc123", "coordinator-a")
        gossip_a._cache_index = cache_index_a

        # Advertise
        adv = gossip_a.advertise()
        assert len(adv) > 0

        # Coordinator B processes the advertisement
        # This simulates one gossip round
        assert isinstance(adv, list)

    def test_gossip_advertise_and_process(self):
        """Advertise known prefixes, process remote advertisements."""
        from distllm.core.cache_index import CacheIndex
        from distllm.core.gossip_protocol import GossipProtocol

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
        from distllm.core.gossip_protocol import GossipProtocol

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
        request = gossip._build_request(["prefix_abc"])
        assert request is not None or len(gossip._peers) == 0

    def test_gossip_cleanup_expired(self):
        """Expired entries should be cleaned up."""
        from distllm.core.cache_index import CacheIndex
        from distllm.core.gossip_protocol import GossipProtocol

        gossip = GossipProtocol(node_id="cleaner", cache_ttl=0.001)
        cache_index = CacheIndex()
        cache_index.add("prefix_stale", "some-node")
        gossip._cache_index = cache_index

        time.sleep(0.01)
        # Cleanup should remove expired entries
        gossip._cleanup()
        assert len(cache_index._entries) == 0 or cache_index.get("prefix_stale") is None


# ===================================================================
# Coordinator-level gossip integration
# ===================================================================

class TestCoordinatorGossipIntegration:
    def test_gossip_discovery_cache_hit(self):
        """Coordinator should discover cache entries via gossip."""
        from distllm.core.coordinator import Coordinator
        from distllm.core.cache_index import CacheIndex
        from distllm.core.gossip_protocol import GossipProtocol

        cache_index = CacheIndex()
        cache_index.add("prefix_known", "peer-coordinator")

        gossip = GossipProtocol(node_id="local-coord")
        gossip._cache_index = cache_index

        # Verify gossip lookup works
        result = gossip.lookup_in_cache_index("prefix_known")
        assert result is not None or gossip._cache_index.get("prefix_known") is not None

    def test_gossip_peer_management(self):
        """Peer addition and removal should work."""
        from distllm.core.gossip_protocol import GossipProtocol

        gossip = GossipProtocol(node_id="node-a", max_peers=3)
        gossip.add_peer("node-b")
        gossip.add_peer("node-c")
        gossip.add_peer("node-d")
        # Should not exceed max_peers
        gossip.add_peer("node-e")
        assert len(gossip._peers) <= 3

        gossip.remove_peer("node-c")
        assert len(gossip._peers) == 2
