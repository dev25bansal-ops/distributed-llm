"""Regression tests for HA leader election with a real heartbeat transport.

Previously ``RayFaultTolerance._election_loop`` never contacted peers, so
after ``election_timeout_s`` every peer was evicted as stale, the quorum
guard was bypassed (``and len(self._peers) > 0``), and *every* coordinator
self-elected as LEADER — split-brain guaranteed.

These tests verify the fixed behavior:
- peers are probed via an outbound heartbeat transport,
- a coordinator never self-elects without a confirmed majority,
- a healthy cluster elects exactly one leader.
"""

from __future__ import annotations

from typing import Any

from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance


def _dead_transport(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """A transport that never reaches any peer."""
    raise ConnectionError("test: no peer reachable")


def _linked_transport(nodes: list[RayFaultTolerance]):
    """Build an in-memory transport that routes heartbeats to peers.

    Each node is given a unique port (via ``_port``) that the transport uses
    to look up the target node and deliver the heartbeat in-process. This
    simulates the peer coordinator's ``POST /api/v1/ha/heartbeat`` handler.
    """
    by_port = {node._port: node for node in nodes}

    def transport(peer_id: str, host: str, port: int, payload: dict[str, Any]) -> dict[str, Any]:
        target = by_port.get(port)
        if target is None:
            raise ConnectionError(f"peer {peer_id} unreachable")
        return target.handle_heartbeat_request(
            payload.get("coordinator_id", ""),
            payload.get("term", 0),
            payload.get("state"),
        )

    return transport


def _node(node_id: str, port: int, **kwargs: Any) -> RayFaultTolerance:
    node = RayFaultTolerance(node_id, **kwargs)
    node._port = port
    return node


class TestHeartbeatTransport:
    def test_healthy_pair_elects_exactly_one_leader(self) -> None:
        """Two reachable coordinators converge to a single leader (lowest ID)."""
        a = _node("coordinator-a", 5001, election_timeout_s=10.0)
        b = _node("coordinator-b", 5002, election_timeout_s=10.0)
        a.add_peer("coordinator-b", "127.0.0.1", 5002)
        b.add_peer("coordinator-a", "127.0.0.1", 5001)

        transport = _linked_transport([a, b])
        a.set_heartbeat_transport(transport)
        b.set_heartbeat_transport(transport)

        for _ in range(10):
            a._run_election_round()
            b._run_election_round()

        leaders = [n for n in (a, b) if n.is_leader()]
        assert len(leaders) == 1, f"expected one leader, got {len(leaders)}"
        assert a.get_leader() == "coordinator-a"
        assert b.get_leader() == "coordinator-a"

    def test_outbound_heartbeat_refreshes_stale_peer(self) -> None:
        """A peer that responds to an outbound probe stays alive."""
        a = _node("coordinator-a", 5001, election_timeout_s=1.0)
        b = _node("coordinator-b", 5002, election_timeout_s=1.0)
        a.add_peer("coordinator-b", "127.0.0.1", 5002)
        b.add_peer("coordinator-a", "127.0.0.1", 5001)

        transport = _linked_transport([a, b])
        a.set_heartbeat_transport(transport)

        # Simulate the peer having been silent past the timeout.
        a._peers["coordinator-b"]["last_seen"] = 0.0
        a._run_election_round()

        assert "coordinator-b" in a._peers  # refreshed by the outbound probe

    def test_higher_term_from_peer_is_adopted(self) -> None:
        """A peer response carrying a higher term is adopted (no stale leader)."""
        a = _node("coordinator-a", 5001)
        a.add_peer("coordinator-b", "127.0.0.1", 5002)
        a._current_term = 1

        def high_term_transport(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "coordinator_id": "coordinator-b",
                "term": 5,
                "state": "follower",
                "leader_id": "coordinator-b",
            }

        a.set_heartbeat_transport(high_term_transport)
        a._probe_peer("coordinator-b", dict(a._peers["coordinator-b"]))

        assert a._current_term == 5
        assert a._state == CoordinatorState.FOLLOWER


class TestFailClosed:
    def test_unreachable_stale_peer_prevents_self_election(self) -> None:
        """A coordinator that cannot confirm a peer must NOT self-elect."""
        n = _node("coordinator-a", 5001, election_timeout_s=1.0)
        n.add_peer("coordinator-b", "127.0.0.1", 5002)
        n._peers["coordinator-b"]["last_seen"] = 0.0  # never confirmed alive
        n.set_heartbeat_transport(_dead_transport)

        n._run_election_round()

        assert not n.is_leader()
        assert n.get_leader() is None
        assert n._state == CoordinatorState.FOLLOWER

    def test_two_node_partition_no_leader(self) -> None:
        """After the only peer in a 2-node cluster dies, the survivor stays
        a follower (it has 1 of 2 — no majority), preventing split-brain."""
        a = _node("coordinator-a", 5001, election_timeout_s=1.0)
        a.add_peer("coordinator-b", "127.0.0.1", 5002)
        a._peers["coordinator-b"]["last_seen"] = 0.0
        a.set_heartbeat_transport(_dead_transport)

        a._run_election_round()
        a._run_election_round()  # after the peer is evicted

        assert not a.is_leader()

    def test_no_quorum_leader_steps_down(self) -> None:
        """A leader in a 3-node cluster steps down when it holds 1 of 3."""
        a = _node("coordinator-a", 5001)
        a._initial_cluster_size = 3
        a._state = CoordinatorState.LEADER
        a._leader_id = "coordinator-a"
        a.add_peer("coordinator-b", "127.0.0.1", 5002)
        a.add_peer("coordinator-c", "127.0.0.1", 5003)
        a._peers["coordinator-b"]["last_seen"] = 0.0
        a._peers["coordinator-c"]["last_seen"] = 0.0
        a.set_heartbeat_transport(_dead_transport)

        a._run_election_round()

        assert a._state == CoordinatorState.FOLLOWER
        assert a._leader_id is None

    def test_surviving_majority_elects_new_leader(self) -> None:
        """In a 3-node cluster, after the primary dies the surviving 2 nodes
        (a majority) elect a new leader instead of both self-electing."""
        a = _node("coordinator-a", 5001, election_timeout_s=1.0)
        b = _node("coordinator-b", 5002, election_timeout_s=1.0)
        c = _node("coordinator-c", 5003, election_timeout_s=1.0)
        for node, peers in (
            (a, [("coordinator-b", 5002), ("coordinator-c", 5003)]),
            (b, [("coordinator-a", 5001), ("coordinator-c", 5003)]),
            (c, [("coordinator-a", 5001), ("coordinator-b", 5002)]),
        ):
            for pid, port in peers:
                node.add_peer(pid, "127.0.0.1", port)
        transport = _linked_transport([a, b, c])
        for node in (a, b, c):
            node.set_heartbeat_transport(transport)

        # Run until stable.
        for _ in range(20):
            for node in (a, b, c):
                node._run_election_round()

        leaders = [n for n in (a, b, c) if n.is_leader()]
        assert len(leaders) == 1, f"expected one leader, got {len(leaders)}"
        assert leaders[0].get_leader() == "coordinator-a"

        # Kill the leader: it stops responding to probes.
        a._peers = {}  # simulate death from a's point of view
        a.set_heartbeat_transport(_dead_transport)
        a._state = CoordinatorState.FOLLOWER
        a._leader_id = None
        # b and c can no longer reach a.
        b.set_heartbeat_transport(_dead_transport)
        c.set_heartbeat_transport(_dead_transport)
        for node in (b, c):
            node._peers["coordinator-a"]["last_seen"] = 0.0
            node._peers["coordinator-a"]["host"] = "127.0.0.1"
            node._peers["coordinator-a"]["port"] = 5001

        for _ in range(10):
            b._run_election_round()
            c._run_election_round()

        # b (lower ID) is now leader, c follows — and c did not self-elect.
        assert b.is_leader()
        assert b.get_leader() == "coordinator-b"
        assert c._state == CoordinatorState.FOLLOWER
        assert c.get_leader() == "coordinator-b"

    def test_single_node_still_self_elects(self) -> None:
        """A truly single-node cluster (no peers ever registered) elects itself."""
        n = _node("coordinator-sole", 5001)
        n._run_election_round()
        assert n.is_leader()
        assert n.get_leader() == "coordinator-sole"
