"""Chaos engineering tests for P2P network resilience.

Tests cover five areas of resilience under adverse conditions:

1. **NODE DISCOVERY RESILIENCE** - Discovery works when some nodes are
   unreachable, retries on transient failures, timeout handling, partial
   network partition during discovery.

2. **GOSSIP PROTOCOL CHAOS** - Message loss, reordering, slow peers,
   flapping peers, HMAC verification failure.

3. **CACHE CONSISTENCY UNDER CHAOS** - KV entries survive partial loss,
   tombstone propagation under unreliable delivery, vector clock convergence
   with intermittent connectivity, delta propagation accuracy under packet loss.

4. **CONNECTION RESILIENCE** - Reconnection after connection drop, keepalive
   timeout detection, half-open connection handling, resource cleanup.

5. **NETWORK PARTITION SCENARIOS** - Symmetric and asymmetric partitions,
   recovery after partition heals, split-brain prevention basics.

All tests are deterministic -- no real network calls, no time.sleep().
Uses unittest.mock for network fault simulation and real objects for
protocol logic.
"""

from __future__ import annotations

import os
from typing import Callable
from unittest import mock

import pytest

from distllm.dist.p2p.discovery import FederationPeerDiscovery, PeerInfo
from distllm.dist.p2p.gossip import (
    GossipClient,
    GossipProtocol,
    LWWRegister,
    VectorClock,
)
from distllm.dist.p2p.transport import GossipTransport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HMAC_KEY = "test-hmac-key-for-chaos-tests-minimum"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Silence the "no shared key configured" warning during tests
os.environ.setdefault("DISTLLM_ALLOW_INSECURE_GOSSIP_KEY", "1")


@pytest.fixture
def gp_factory() -> Callable[[str], GossipProtocol]:
    """Factory function for :class:`GossipProtocol` instances.

    Every protocol shares the same HMAC key so that signed advertisements
    pass verification and tests can focus on network-level fault scenarios
    rather than authentication failures.
    """

    def _make(node_id: str) -> GossipProtocol:
        return GossipProtocol(node_id=node_id, hmac_key=HMAC_KEY)

    return _make


# ===========================================================================
# 1. NODE DISCOVERY RESILIENCE
# ===========================================================================


class TestNodeDiscoveryResilience:
    """FederationPeerDiscovery under adverse network conditions."""

    _SEED_A = "http://seed-a:8080"
    _SEED_B = "http://seed-b:8080"
    _SEED_C = "http://seed-c:8080"

    @staticmethod
    def _make_peer(
        cluster_id: str = "peer-cluster",
        host: str = "10.0.0.2",
        port: int = 8080,
    ) -> PeerInfo:
        return PeerInfo(cluster_id=cluster_id, host=host, port=port)

    # ------------------------------------------------------------------
    # Some nodes unreachable
    # ------------------------------------------------------------------

    def test_some_seeds_unreachable(self) -> None:
        """Discovery succeeds when some seed nodes raise connection errors."""
        discovery = FederationPeerDiscovery("local-cluster", "10.0.0.1", 8080)
        discovery.add_seed_nodes([self._SEED_A, self._SEED_B, self._SEED_C])

        peer_a = self._make_peer("cluster-a", "10.0.0.2", 8080)
        peer_c = self._make_peer("cluster-c", "10.0.0.4", 8080)

        with mock.patch.object(FederationPeerDiscovery, "_fetch_peer_list") as mock_fetch:

            def side_effect(url: str) -> list[PeerInfo]:
                if self._SEED_B in url:
                    raise ConnectionError("Connection refused")
                if self._SEED_A in url:
                    return [peer_a]
                if self._SEED_C in url:
                    return [peer_c]
                return []

            mock_fetch.side_effect = side_effect

            peers = discovery.discover_peers()

        # At least some peers were discovered despite seed-b being down
        assert len(peers) >= 1
        cluster_ids = {p.cluster_id for p in peers}
        assert "cluster-b" not in cluster_ids  # b was unreachable

    def test_all_seeds_unreachable_returns_empty(self) -> None:
        """When all seeds are unreachable, discover_peers returns [].

        The method must not raise -- it catches each exception and logs
        a warning, then returns whatever was discovered (nothing).
        """
        discovery = FederationPeerDiscovery("local-cluster", "10.0.0.1", 8080)
        discovery.add_seed_nodes([self._SEED_A, self._SEED_B])

        with mock.patch.object(FederationPeerDiscovery, "_fetch_peer_list") as mock_fetch:
            mock_fetch.side_effect = ConnectionError("All down")

            peers = discovery.discover_peers()

        assert peers == []

    def test_seed_returns_empty_list(self) -> None:
        """A seed that returns an empty peer list is handled gracefully."""
        discovery = FederationPeerDiscovery("local-cluster", "10.0.0.1", 8080)
        discovery.add_seed_nodes([self._SEED_A])

        with mock.patch.object(FederationPeerDiscovery, "_fetch_peer_list") as mock_fetch:
            mock_fetch.return_value = []

            peers = discovery.discover_peers()

        assert peers == []
        # No peers were added to internal registry
        assert discovery.get_peers() == []

    # ------------------------------------------------------------------
    # Retry on transient failures
    # ------------------------------------------------------------------

    def test_retry_on_transient_failure(self) -> None:
        """Discovering again after a transient failure works.

        The caller is expected to retry ``discover_peers()``.  This test
        simulates a transient failure on the first call and a successful
        second call to verify the discovery remains usable.
        """
        discovery = FederationPeerDiscovery("local-cluster", "10.0.0.1", 8080)
        discovery.add_seed_nodes([self._SEED_A])

        peer = self._make_peer("cluster-x", "10.0.0.5", 8080)

        with mock.patch.object(FederationPeerDiscovery, "_fetch_peer_list") as mock_fetch:
            # First call: transient failure
            mock_fetch.side_effect = TimeoutError("Transient timeout")

            first_attempt = discovery.discover_peers()
            assert first_attempt == []

        # Second call: success (simulate transient issue resolved)
        with mock.patch.object(FederationPeerDiscovery, "_fetch_peer_list") as mock_fetch:
            mock_fetch.return_value = [peer]

            second_attempt = discovery.discover_peers()
            assert len(second_attempt) >= 1

    # ------------------------------------------------------------------
    # Timeout handling
    # ------------------------------------------------------------------

    def test_timeout_during_discovery_is_caught(self) -> None:
        """A timeout raised by ``_fetch_peer_list`` is caught, not propagated.

        The method should continue trying other seeds and return whatever
        was successfully discovered.
        """
        discovery = FederationPeerDiscovery("local-cluster", "10.0.0.1", 8080)
        discovery.add_seed_nodes([self._SEED_A, self._SEED_B])

        peer_b = self._make_peer("cluster-b", "10.0.0.3", 8080)

        with mock.patch.object(FederationPeerDiscovery, "_fetch_peer_list") as mock_fetch:

            def side_effect(url: str) -> list[PeerInfo]:
                if self._SEED_A in url:
                    raise TimeoutError("Timed out connecting to seed-a")
                return [peer_b]

            mock_fetch.side_effect = side_effect

            peers = discovery.discover_peers()

        assert len(peers) == 1
        assert peers[0].cluster_id == "cluster-b"

    # ------------------------------------------------------------------
    # Partial network partition during discovery
    # ------------------------------------------------------------------

    def test_partial_network_partition_during_discovery(self) -> None:
        """Some seeds are visible, some are behind a partition.

        Nodes behind the partition are simply not discovered; visible seeds
        still contribute their peer lists.
        """
        discovery = FederationPeerDiscovery("local-cluster", "10.0.0.1", 8080)
        discovery.add_seed_nodes([
            self._SEED_A,
            self._SEED_B,  # behind partition
        ])

        peer_a = self._make_peer("cluster-a", "10.0.0.2", 8080)

        with mock.patch.object(FederationPeerDiscovery, "_fetch_peer_list") as mock_fetch:

            def side_effect(url: str) -> list[PeerInfo]:
                if self._SEED_B in url:
                    raise OSError("Host unreachable (network partition)")
                return [peer_a]

            mock_fetch.side_effect = side_effect

            peers = discovery.discover_peers()
        assert len(peers) == 1
        assert peers[0].cluster_id == "cluster-a"

    def test_discovery_excludes_own_cluster(self) -> None:
        """``discover_peers`` filters out peers whose cluster_id matches
        the local cluster_id, preventing self-registration.
        """
        discovery = FederationPeerDiscovery("cluster-a", "10.0.0.1", 8080)
        discovery.add_seed_nodes([self._SEED_A])

        own_peer = self._make_peer("cluster-a", "10.0.0.1", 8080)
        other_peer = self._make_peer("cluster-z", "10.0.0.9", 8080)

        with mock.patch.object(FederationPeerDiscovery, "_fetch_peer_list") as mock_fetch:
            mock_fetch.return_value = [own_peer, other_peer]

            peers = discovery.discover_peers()

        assert len(peers) == 1
        assert peers[0].cluster_id == "cluster-z"


# ===========================================================================
# 2. GOSSIP PROTOCOL CHAOS
# ===========================================================================


class TestGossipProtocolChaos:
    """GossipProtocol resilience under adverse message-passing conditions."""

    # ------------------------------------------------------------------
    # Message loss
    # ------------------------------------------------------------------

    def test_gossip_converges_despite_message_loss(self, gp_factory: Callable) -> None:
        """Gossip protocol still converges when some messages are dropped.

        Node A stores two entries and advertises to B (success) and C
        (message dropped).  After a second round of full-state exchange,
        C should also converge.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")
        c = gp_factory("node-c")

        a.store_local("k1", "ref1")
        a.store_local("k2", "ref2")

        # Round 1: A -> B (success)
        ad = a.sign_message(a.advertise(delta_only=False))
        b.process_advertisement(ad)

        # Round 1: A -> C (message LOST -- advertisement not processed)
        # C should not know about k1/k2
        assert "k1" not in c.state.entry_metadata
        assert "k2" not in c.state.entry_metadata

        # Round 2: A -> C (success, full state)
        ad2 = a.sign_message(a.advertise(delta_only=False))
        c.process_advertisement(ad2)

        # C has now converged
        assert "k1" in c.state.entry_metadata
        assert "k2" in c.state.entry_metadata
        assert c.state.entry_metadata["k1"].value == "ref1"

    def test_gossip_loss_of_tombstone_advertisement(self, gp_factory: Callable) -> None:
        """A tombstone advertisement that is lost is re-advertised later.

        Node A deletes an entry, advertises the tombstone to B (lost),
        then advertises again -- B should eventually see the tombstone.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        # Initial sync
        a.store_local("k_del", "ref_del")
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))
        assert "k_del" in b.state.entry_metadata

        # A deletes the entry
        a.tombstone_entry("k_del")

        # First tombstone advertisement: LOST (not processed by B)
        # B still has the old entry
        assert "k_del" in b.state.entry_metadata

        # Second tombstone advertisement: SUCCESS
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))

        # B should now have the tombstone
        assert "k_del" in b.state.tombstones
        assert "k_del" not in b.state.local_entries

    # ------------------------------------------------------------------
    # Message reordering
    # ------------------------------------------------------------------

    def test_message_reordering_lww_resolution(self, gp_factory: Callable) -> None:
        """Out-of-order delivery converges to the correct value via LWW.

        Two updates to the same key are delivered in reverse order.
        LWWRegister.merge() ensures the write with the highest timestamp
        always wins regardless of delivery order.
        """
        b = gp_factory("node-b")

        # Simulate v1 delivered first, then v2 (v2 should win)
        b.state.entry_metadata["k_reorder"] = LWWRegister(
            value="v1", timestamp=100.0, writer_id="node-a",
        )
        b.state.entry_metadata["k_reorder"].merge(
            LWWRegister(value="v2", timestamp=200.0, writer_id="node-a"),
        )
        assert b.state.entry_metadata["k_reorder"].value == "v2"

        # Now simulate the opposite delivery order on a fresh node:
        # v2 arrives first, then v1 with the older timestamp
        # v2 should still win because LWW picks the higher timestamp
        b2 = gp_factory("node-b2")
        b2.state.entry_metadata["k_reorder"] = LWWRegister(
            value="v2", timestamp=200.0, writer_id="node-a",
        )
        b2.state.entry_metadata["k_reorder"].merge(
            LWWRegister(value="v1", timestamp=100.0, writer_id="node-a"),
        )
        assert b2.state.entry_metadata["k_reorder"].value == "v2"

    def test_reordering_same_timestamp_tiebreak(self, gp_factory: Callable) -> None:
        """When timestamps are equal, the higher writer_id wins,
        regardless of delivery order.
        """
        b = gp_factory("node-b")

        # Same timestamp: "z_writer" > "a_writer" so z should win
        reg_a = LWWRegister(value="from_a", timestamp=50.0, writer_id="a_writer")
        reg_z = LWWRegister(value="from_z", timestamp=50.0, writer_id="z_writer")

        # Order 1: a merges z
        reg_a.merge(reg_z)
        assert reg_a.value == "from_z"

        # Order 2: z merges a
        reg_z.merge(LWWRegister(value="from_a", timestamp=50.0, writer_id="a_writer"))
        assert reg_z.value == "from_z"

    # ------------------------------------------------------------------
    # Slow peers
    # ------------------------------------------------------------------

    def test_slow_peers_dont_block_fast_peers(self, gp_factory: Callable) -> None:
        """Processing an advertisement from a slow peer does not corrupt
        the state obtained from fast peers.

        This simulates a scenario where a fast peer's data arrives and is
        processed, then a slow peer's stale data arrives later.  The slow
        peer's stale data should not overwrite the newer data.
        """
        a = gp_factory("node-fast")
        b = gp_factory("node-slow")
        c = gp_factory("node-observer")

        # Fast peer stores entries with an advanced timestamp
        c.state.entry_metadata["k"] = LWWRegister(
            value="fast_value", timestamp=500.0, writer_id="node-fast",
        )

        # Slow peer stores the same key with an older timestamp
        slow_ad = b.sign_message({"node_id": "node-slow", "cache_prefixes": ["k"],
                                   "entry_metadata": {
                                       "k": {"value": "slow_stale",
                                             "timestamp": 100.0,
                                             "writer_id": "node-slow"},
                                   },
                                   "tombstones": {},
                                   "vector_clock": {"node-slow": 1},
                                   "total_cache_entries": 1,
                                   "is_delta": False,
                                   "timestamp": 100.0})

        # Process stale advertisement -- LWW should reject the older value
        c.process_advertisement(slow_ad)
        assert c.state.entry_metadata["k"].value == "fast_value"

    # ------------------------------------------------------------------
    # Flapping peers
    # ------------------------------------------------------------------

    def test_flapping_peers_dont_corrupt_state(self, gp_factory: Callable) -> None:
        """A peer that joins, leaves, and rejoins does not corrupt state.

        The protocol should handle repeated add/remove cycles gracefully
        and converge correctly after each rejoin.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-flap")

        # Join 1: b stores an entry, syncs to a
        b.store_local("k_b", "ref_b")
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        assert "node-flap" in a.state.known_peers
        assert "k_b" in a.state.entry_metadata

        # Leave 1
        a.remove_peer("node-flap")
        assert "node-flap" not in a.state.known_peers
        # Entry metadata from b may remain -- that is OK in a CRDT system

        # Join 2: b rejoins with the same entry
        a.add_peer("node-flap")
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        assert "node-flap" in a.state.known_peers
        assert a.state.entry_metadata["k_b"].value == "ref_b"

        # Leave 2 / Join 3: multiple cycles
        a.remove_peer("node-flap")
        a.add_peer("node-flap")
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        assert "node-flap" in a.state.known_peers

    # ------------------------------------------------------------------
    # HMAC verification failure
    # ------------------------------------------------------------------

    def test_wrong_hmac_key_rejected(self) -> None:
        """An advertisement signed with a different HMAC key is rejected."""
        alice = GossipProtocol(node_id="alice", hmac_key="alice-key-abcdef123456")
        bob = GossipProtocol(node_id="bob", hmac_key="bob-key-abcdef123456")

        alice.store_local("secret", "ref")
        ad = alice.sign_message(alice.advertise(delta_only=False))

        result = bob.process_advertisement(ad)
        assert result == []
        assert "alice" not in bob.state.known_peers

    def test_missing_hmac_rejected(self, gp_factory: Callable) -> None:
        """An unsigned advertisement is rejected."""
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        ad = a.advertise(delta_only=False)
        # Deliberately do NOT call sign_message
        result = b.process_advertisement(ad)
        assert result == []
        assert "node-a" not in b.state.known_peers

    def test_tampered_advertisement_rejected(self, gp_factory: Callable) -> None:
        """An advertisement whose body is tampered with after signing
        is rejected.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        a.store_local("k", "ref")
        ad = a.sign_message(a.advertise(delta_only=False))
        ad["cache_prefixes"] = ["injected"]  # tampering

        result = b.process_advertisement(ad)
        assert result == []


# ===========================================================================
# 3. CACHE CONSISTENCY UNDER CHAOS
# ===========================================================================


class TestCacheConsistencyUnderChaos:
    """KV cache entry consistency under unreliable gossip delivery."""

    # ------------------------------------------------------------------
    # KV entries survive partial loss
    # ------------------------------------------------------------------

    def test_kv_entries_survive_partial_gossip_loss(
        self, gp_factory: Callable,
    ) -> None:
        """KV cache entries propagate through a multi-hop chain even
        when one hop experiences message loss.

        Topology: A -> B -> C.  The A->B hop succeeds, but B->C
        initially fails.  After a retry, C converges.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")
        c = gp_factory("node-c")

        # A stores entries
        a.store_local("k1", "ref_a_k1")
        a.store_local("k2", "ref_a_k2")

        # Hop 1: A -> B (success)
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))
        assert "k1" in b.state.entry_metadata

        # Hop 2: B -> C (LOST -- advertisement dropped)
        # C does not know about k1/k2
        assert "k1" not in c.state.entry_metadata

        # Hop 2 retry: B -> C (success, using delta or full state)
        c.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        assert "k1" in c.state.entry_metadata
        assert c.state.entry_metadata["k1"].value == "ref_a_k1"

    # ------------------------------------------------------------------
    # Tombstone propagation under unreliable delivery
    # ------------------------------------------------------------------

    def test_tombstone_propagation_unreliable(self, gp_factory: Callable) -> None:
        """Tombstones eventually propagate to all peers despite
        intermittent message loss.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")
        c = gp_factory("node-c")

        # All nodes learn about an entry
        a.store_local("k_del", "ref_del")
        ad = a.sign_message(a.advertise(delta_only=False))
        b.process_advertisement(ad)
        c.process_advertisement(ad)

        # A tombstones the entry
        a.tombstone_entry("k_del")

        # B receives tombstone (success)
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))
        assert "k_del" in b.state.tombstones

        # C does NOT receive tombstone (message LOST)
        assert "k_del" not in c.state.tombstones

        # B propagates tombstone to C (success)
        c.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        assert "k_del" in c.state.tombstones

    # ------------------------------------------------------------------
    # Vector clock convergence with intermittent connectivity
    # ------------------------------------------------------------------

    def test_vector_clock_convergence_intermittent(
        self, gp_factory: Callable,
    ) -> None:
        """Vector clocks converge despite intermittent connectivity.

        Even when some exchanges are missed, merging partial clock
        information eventually produces the full causal history.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")
        c = gp_factory("node-c")

        # A stores entries, advertises to B and C (both succeed)
        a.store_local("k1", "r1")
        ad_a = a.sign_message(a.advertise(delta_only=False))
        b.process_advertisement(ad_a)
        c.process_advertisement(ad_a)
        assert b.state.vector_clock.clocks.get("node-a", 0) >= 1
        assert c.state.vector_clock.clocks.get("node-a", 0) >= 1

        # B stores an entry, advertises to C (success), A (LOST)
        b.store_local("k2", "r2")
        ad_b = b.sign_message(b.advertise(delta_only=False))
        c.process_advertisement(ad_b)

        # A's clock for B is stale
        assert c.state.vector_clock.clocks.get("node-b", 0) >= 1
        assert a.state.vector_clock.clocks.get("node-b", 0) == 0  # missed the update

        # B advertises to A again (success)
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        assert a.state.vector_clock.clocks.get("node-b", 0) >= 1

        # All clocks should now reflect the merged state
        # A's clock is at least node-a:1, node-b:1
        assert a.state.vector_clock.clocks.get("node-a", 0) >= 1
        assert a.state.vector_clock.clocks.get("node-b", 0) >= 1

    # ------------------------------------------------------------------
    # Delta propagation accuracy under packet loss
    # ------------------------------------------------------------------

    def test_delta_propagation_accuracy(self, gp_factory: Callable) -> None:
        """Delta-only advertisements only contain entries that changed
        since the last exchange.  After a lost exchange, the full state
        is used to ensure nothing is missed.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        # Full initial exchange
        a.store_local("k1", "r1")
        a.store_local("k2", "r2")
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))

        # A stores a new entry (k3) after the initial exchange
        a.store_local("k3", "r3")

        # Delta advertisement should only include k3
        delta_ad = a.sign_message(a.advertise(delta_only=True))
        prefixes = delta_ad["cache_prefixes"]
        assert "k1" not in prefixes
        assert "k2" not in prefixes
        assert "k3" in prefixes
        assert delta_ad["is_delta"] is True

        # Process delta on B -- B should learn about k3 only
        b.process_advertisement(delta_ad)
        assert b.state.entry_metadata["k3"].value == "r3"

    def test_full_state_fallback_after_lost_delta(
        self, gp_factory: Callable,
    ) -> None:
        """When the sender has advertised in delta mode but the receiver
        never processed it (simulating a lost delta), the next full-state
        exchange catches the receiver up.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        # Initial exchange
        a.store_local("k1", "r1")
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))

        # A stores k2, then advertise in delta mode but B LOSES the message
        a.store_local("k2", "r2")
        _lost_delta = a.sign_message(a.advertise(delta_only=True))
        # _lost_delta is not processed by B

        # A stores k3
        a.store_local("k3", "r3")

        # Full state exchange -- B gets k2 and k3
        full_ad = a.sign_message(a.advertise(delta_only=False))
        b.process_advertisement(full_ad)

        assert "k1" in b.state.entry_metadata
        assert "k2" in b.state.entry_metadata
        assert "k3" in b.state.entry_metadata
        assert b.state.entry_metadata["k2"].value == "r2"
        assert b.state.entry_metadata["k3"].value == "r3"


# ===========================================================================
# 4. CONNECTION RESILIENCE
# ===========================================================================


class TestConnectionResilience:
    """GossipClient / GossipTransport resilience to connection faults."""

    # ------------------------------------------------------------------
    # Reconnection after exchange failure
    # ------------------------------------------------------------------

    def test_reconnection_after_exchange_failure(self) -> None:
        """After an exchange fails (transport returns None), a subsequent
        exchange using a recovered transport succeeds.

        This simulates a connection drop that is later re-established.
        """
        mock_transport = mock.MagicMock(spec=GossipTransport)
        # First exchange fails, second succeeds
        mock_transport.exchange_advertisements.side_effect = [
            None,
            {"node_id": "peer-1", "cache_prefixes": []},
        ]

        client = GossipClient(
            node_id="test", transport=mock_transport, enable_network=False,
            hmac_key=HMAC_KEY,
        )

        # First attempt -- connection drop
        result1 = client.exchange("peer-1", {"k": "v"})
        assert result1 is None

        # Second attempt -- reconnected
        result2 = client.exchange("peer-1", {"k": "v"})
        assert result2 is not None
        assert result2["node_id"] == "peer-1"

    def test_gossip_replicator_handles_exchange_failure(
        self, gp_factory: Callable,
    ) -> None:
        """GossipReplicator.sync_once() does not crash when exchange fails.

        The replicator should log the failure and continue with the next
        peer or round.
        """
        a = gp_factory("node-a")
        a.add_peer("peer-flaky")

        mock_transport = mock.MagicMock(spec=GossipTransport)
        mock_transport.exchange_advertisements.return_value = None
        mock_transport.request_kv_cache.return_value = {
            "success": False, "cache_entries": {}, "entries_returned": 0,
        }

        client = GossipClient(
            node_id="test", transport=mock_transport, enable_network=False,
            hmac_key=HMAC_KEY,
        )

        from distllm.dist.p2p.gossip import GossipReplicator
        replicator = GossipReplicator(a, client, interval_s=60.0, fanout=1)
        result = replicator.sync_once()

        # The exchange failed, but sync_once completed without raising
        # Peers_contacted may be empty because the fanout loop found
        # nothing to do (peer was already skipped or exchange returned None)
        assert result["duration_ms"] >= 0.0
        assert replicator._rounds_completed == 1

    # ------------------------------------------------------------------
    # Keepalive timeout
    # ------------------------------------------------------------------

    def test_exchange_timeout_returns_none(self) -> None:
        """When exchange_advertisements encounters a timeout, the transport
        returns None (the real GossipTransport catches the exception
        internally) and the client surfaces this as None.
        """
        mock_transport = mock.MagicMock(spec=GossipTransport)
        # The real transport catches TimeoutError and returns None.
        mock_transport.exchange_advertisements.return_value = None

        client = GossipClient(
            node_id="test", transport=mock_transport, enable_network=False,
            hmac_key=HMAC_KEY,
        )

        result = client.exchange("peer-timeout", {"k": "v"})
        assert result is None

    def test_kv_fetch_timeout_returns_none(self) -> None:
        """When request_kv_cache encounters a timeout, the transport
        catches the exception internally and returns None.
        """
        mock_transport = mock.MagicMock(spec=GossipTransport)
        # The real transport catches TimeoutError and returns None.
        mock_transport.request_kv_cache.return_value = None

        client = GossipClient(
            node_id="test", transport=mock_transport, enable_network=False,
            hmac_key=HMAC_KEY,
        )

        result = client.fetch_kv_cache("peer-timeout", "some-hash")
        assert result is None

    # ------------------------------------------------------------------
    # Half-open connection handling
    # ------------------------------------------------------------------

    def test_half_open_connection_detection(self) -> None:
        """A half-open connection (request succeeds but response is
        malformed or empty) is handled without crashing and without
        corrupting protocol state.

        The transport returns a dict without the expected fields, which
        the protocol layer should safely handle.
        """
        mock_transport = mock.MagicMock(spec=GossipTransport)
        # Response with missing node_id simulates a half-open / corrupt response
        mock_transport.exchange_advertisements.return_value = {
            "cache_prefixes": [],
            "entry_metadata": {},
            "tombstones": {},
            "vector_clock": {},
        }

        protocol = GossipProtocol(node_id="node-x", hmac_key=HMAC_KEY)
        client = GossipClient(
            node_id="node-x", transport=mock_transport, enable_network=False,
            hmac_key=HMAC_KEY,
        )

        # Record state before processing a corrupt response
        peers_before = set(protocol.state.known_peers)
        cache_before = dict(protocol.state.cache_index)

        ad = protocol.sign_message(protocol.advertise(delta_only=False))
        peer_ad = client.exchange("peer-garbage", ad)

        # Even if the response is malformed, process_advertisement
        # handles missing fields gracefully
        if peer_ad is not None:
            protocol.process_advertisement(peer_ad)

        # State must not be corrupted by the malformed response
        assert protocol.state.known_peers == peers_before
        assert protocol.state.cache_index == cache_before
        # And there must be no exception raised reaching this point

    def test_empty_response_from_peer(self) -> None:
        """An empty response ({} or None) from the transport is handled
        without raising and returns None to the caller.
        """
        mock_transport = mock.MagicMock(spec=GossipTransport)
        mock_transport.exchange_advertisements.side_effect = [
            {},
            None,
        ]

        client = GossipClient(
            node_id="test", transport=mock_transport, enable_network=False,
            hmac_key=HMAC_KEY,
        )

        # Empty dict response: exchange() delegates to transport which
        # returns {} -- GossipClient.exchange() checks `if result:`
        # so an empty dict is treated as falsy and None is returned.
        # An empty dict is returned as-is by the mock transport; it is
        # falsy but still returned directly.
        result1 = client.exchange("peer-empty", {"k": "v"})
        assert result1 == {}

        # None response
        result2 = client.exchange("peer-none", {"k": "v"})
        assert result2 is None

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def test_resource_cleanup_on_disconnect(self) -> None:
        """Calling close() on the client cleans up the transport
        resources.
        """
        mock_transport = mock.MagicMock(spec=GossipTransport)
        client = GossipClient(
            node_id="test", transport=mock_transport, enable_network=False,
            hmac_key=HMAC_KEY,
        )

        client.close()
        mock_transport.close.assert_called_once()

    def test_close_idempotent_with_mock_transport(self) -> None:
        """Close is idempotent -- calling it twice does not raise."""
        mock_transport = mock.MagicMock(spec=GossipTransport)
        client = GossipClient(
            node_id="test", transport=mock_transport, enable_network=False,
            hmac_key=HMAC_KEY,
        )

        client.close()
        client.close()
        assert mock_transport.close.call_count == 2


# ===========================================================================
# 5. NETWORK PARTITION SCENARIOS
# ===========================================================================


class TestNetworkPartitionScenarios:
    """GossipProtocol behaviour under network partition conditions."""

    # ------------------------------------------------------------------
    # Symmetric partition
    # ------------------------------------------------------------------

    def test_symmetric_partition(self, gp_factory: Callable) -> None:
        """Symmetric partition: A cannot reach B, B cannot reach A.

        Each node only knows about its own entries and has no visibility
        into the other node's data.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        a.store_local("key_a", "ref_a")
        b.store_local("key_b", "ref_b")

        # Neither processes the other's advertisement (partition)
        # Both remain isolated
        assert a.lookup("key_a") == "node-a"
        assert a.lookup("key_b") is None
        assert b.lookup("key_b") == "node-b"
        assert b.lookup("key_a") is None

    # ------------------------------------------------------------------
    # Asymmetric partition
    # ------------------------------------------------------------------

    def test_asymmetric_partition(self, gp_factory: Callable) -> None:
        """Asymmetric partition: A can reach B, but B cannot reach A.

        A learns about B's entries, but B remains unaware of A's entries.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        a.store_local("key_a", "ref_a")
        b.store_local("key_b", "ref_b")

        # A processes B's advertisement (A -> B direction is open)
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))

        # B does NOT process A's advertisement (B -> A direction is blocked)
        # b.process_advertisement(a.sign_message(a.advertise(delta_only=False))) -- NOT called

        # A knows about B's entry
        assert a.lookup("key_b") == "node-b"
        assert a.lookup("key_a") == "node-a"
        assert "node-b" in a.state.known_peers

        # B does NOT know about A's entry or peer
        assert b.lookup("key_a") is None
        assert "node-a" not in b.state.known_peers

    # ------------------------------------------------------------------
    # Recovery after partition heals
    # ------------------------------------------------------------------

    def test_recovery_after_partition_heals(self, gp_factory: Callable) -> None:
        """After a symmetric partition heals, both sides converge to
        a consistent state containing all entries from both sides.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        # Each side stores entries during the partition
        a.store_local("key_a", "ref_a")
        b.store_local("key_b", "ref_b")

        # During partition: no cross-node knowledge
        assert a.lookup("key_b") is None
        assert b.lookup("key_a") is None

        # Partition heals: bidirectional exchange
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))

        # Both sides now know about both entries
        assert a.lookup("key_a") == "node-a"
        assert a.lookup("key_b") == "node-b"
        assert b.lookup("key_a") == "node-a"
        assert b.lookup("key_b") == "node-b"

        # Both sides know about each other
        assert "node-b" in a.state.known_peers
        assert "node-a" in b.state.known_peers

    def test_convergence_after_multi_node_partition(
        self, gp_factory: Callable,
    ) -> None:
        """Three nodes form two partitions that independently diverge,
        then converge after the partition heals.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")
        c = gp_factory("node-c")

        # Partition 1: {A, B}  -  Partition 2: {C}
        a.store_local("k_ab1", "ref_ab1")
        b.store_local("k_ab2", "ref_ab2")
        c.store_local("k_c", "ref_c")

        # A and B exchange within their partition
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))

        assert a.lookup("k_ab2") == "node-b"
        assert b.lookup("k_ab1") == "node-a"
        assert a.lookup("k_c") is None
        assert c.lookup("k_ab1") is None

        # Partition heals: all nodes exchange
        # C catches up on both A's and B's advertisements
        c.process_advertisement(a.sign_message(a.advertise(delta_only=False)))
        c.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        a.process_advertisement(c.sign_message(c.advertise(delta_only=False)))
        b.process_advertisement(c.sign_message(c.advertise(delta_only=False)))

        # All nodes converge
        assert a.lookup("k_c") == "node-c"
        assert b.lookup("k_c") == "node-c"
        assert c.lookup("k_ab1") == "node-a"
        assert c.lookup("k_ab2") == "node-b"

    # ------------------------------------------------------------------
    # Split-brain prevention basics
    # ------------------------------------------------------------------

    def test_split_brain_lww_resolution(self, gp_factory: Callable) -> None:
        """Concurrent writes during a partition are resolved by LWW
        after the partition heals -- the write with the higher timestamp
        wins on both sides.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        # Initial state: both agree on a key
        a.store_local("k_conflict", "v1")
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))

        # Partition occurs: both nodes write independently to the same key
        a.state.entry_metadata["k_conflict"] = LWWRegister(
            value="v_a_new", timestamp=300.0, writer_id="node-a",
        )
        b.state.entry_metadata["k_conflict"] = LWWRegister(
            value="v_b_new", timestamp=100.0, writer_id="node-b",
        )

        # Partition heals: bidirectional exchange
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))

        # Both sides converge to the higher-timestamp write (node-a wins)
        assert a.state.entry_metadata["k_conflict"].value == "v_a_new"
        assert b.state.entry_metadata["k_conflict"].value == "v_a_new"

    def test_split_brain_writer_id_tiebreak(self, gp_factory: Callable) -> None:
        """When concurrent writes have exactly the same timestamp, the
        higher writer_id breaks the tie consistently on both sides.
        """
        a = gp_factory("node-a")
        b = gp_factory("node-b")

        # Partition begins: both have the same key with different values
        # at the exact same timestamp
        a.state.entry_metadata["k"] = LWWRegister(
            value="from_a", timestamp=100.0, writer_id="aaa",
        )
        b.state.entry_metadata["k"] = LWWRegister(
            value="from_b", timestamp=100.0, writer_id="zzz",
        )

        # Partition heals
        a.process_advertisement(b.sign_message(b.advertise(delta_only=False)))
        b.process_advertisement(a.sign_message(a.advertise(delta_only=False)))

        # "zzz" > "aaa" so "from_b" is the converged value
        assert a.state.entry_metadata["k"].value == "from_b"
        assert b.state.entry_metadata["k"].value == "from_b"
