"""Tests for coordinator HA failover, leader election, state replication,
split-brain prevention, and shutdown recovery.

Uses mocks only — no real Ray or distributed coordination.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


def _dead_transport(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """A heartbeat transport that never reaches any peer.

    These unit tests drive peer liveness purely through inbound
    ``handle_heartbeat_request`` calls, so outbound probes must fail fast
    (without attempting real network I/O).
    """
    raise ConnectionError("test: no peer reachable")


def _mock_cluster_mgr(nodes: dict | None = None) -> MagicMock:
    """A cluster-manager stand-in whose ``nodes`` / ``node_order`` are plain
    attributes.

    ``Coordinator.nodes`` is a read-through property that delegates to the
    cluster manager and ultimately to ``PipelineOrchestrator.nodes`` (which
    has no setter), so tests cannot assign ``coord.nodes = {...}`` directly.
    """
    nodes = nodes if nodes is not None else {}
    mgr = MagicMock()
    mgr.nodes = nodes
    mgr.node_order = list(nodes.keys())
    return mgr


# ── Helpers ──


def _make_coordinator(model_name: str = "test-model", port: int = 50050):
    """Create a Coordinator with minimal config for testing.

    NOTE: Workaround for Coordinator.__init__ referencing _subsystem_mgr
    before it is assigned (lines 119 vs 196). A class-level mock ensures
    attribute lookup succeeds before the instance attribute is set.
    """
    from distllm.core.coordinator import Coordinator
    from distllm.core.coordinator_config import CoordinatorConfig

    had_attr = hasattr(Coordinator, "_subsystem_mgr")
    old = getattr(Coordinator, "_subsystem_mgr", None)
    Coordinator._subsystem_mgr = MagicMock()
    try:
        return Coordinator(config=CoordinatorConfig(model_name=model_name, port=port))
    finally:
        if had_attr:
            Coordinator._subsystem_mgr = old
        else:
            del Coordinator._subsystem_mgr


# =========================================================================
# 1. LEADER ELECTION
# =========================================================================


class TestLeaderElection:
    """Leader election and is_leader / ha_status properties."""

    def test_single_coordinator_always_leader(self) -> None:
        """A coordinator without HA is always the leader."""
        coord = _make_coordinator()
        assert coord.is_leader is True

    def test_enable_ha_initializes_election_with_peers(self) -> None:
        """enable_ha creates RayFaultTolerance and configures peers."""
        coord = _make_coordinator()
        with patch("distllm.core.ha_coordinator.RayFaultTolerance") as mock_ft:
            mock_instance = MagicMock()
            mock_ft.return_value = mock_instance

            coord.enable_ha(
                coordinator_id="test-1",
                peer_coordinators=[("peer-1", "10.0.0.2", 50051)],
                heartbeat_interval_s=1.0,
                election_timeout_s=5.0,
            )

            mock_ft.assert_called_once_with(
                coordinator_id="test-1",
                heartbeat_interval_s=1.0,
                election_timeout_s=5.0,
            )
            mock_instance.add_peer.assert_called_once_with(
                "peer-1", "10.0.0.2", 50051
            )
            mock_instance.start.assert_called_once()

    def test_ha_status_no_ha(self) -> None:
        """ha_status returns enabled=False when HA is not configured."""
        coord = _make_coordinator()
        assert coord.ha_status == {"enabled": False}

    def test_ha_status_with_ha(self) -> None:
        """ha_status returns stats from the election instance."""
        coord = _make_coordinator()
        with patch("distllm.core.ha_coordinator.RayFaultTolerance") as mock_ft:
            mock_instance = MagicMock()
            mock_instance.stats.return_value = {
                "coordinator_id": "test-1",
                "state": "leader",
                "current_term": 1,
                "peers": 2,
            }
            mock_ft.return_value = mock_instance
            coord.enable_ha(coordinator_id="test-1")

            status = coord.ha_status
            assert status["coordinator_id"] == "test-1"
            assert status["state"] == "leader"
            assert status["current_term"] == 1

    def test_leader_election_lowest_id_wins(self) -> None:
        """Among alive peers the lowest coordinator ID becomes leader."""
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        ft = RayFaultTolerance(
            "coordinator-b", election_timeout_s=10.0, heartbeat_transport=_dead_transport
        )
        ft.add_peer("coordinator-a", "10.0.0.1", 50051)

        ft.handle_heartbeat_request("coordinator-a", term=0)
        ft._run_election_round()

        assert ft._state == CoordinatorState.FOLLOWER
        assert ft._leader_id == "coordinator-a"  # Lower ID wins

    def test_leader_election_no_peers(self) -> None:
        """With no peers the coordinator elects itself leader."""
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        ft = RayFaultTolerance("coordinator-sole", election_timeout_s=10.0)
        ft._run_election_round()

        assert ft._state == CoordinatorState.LEADER
        assert ft._leader_id == "coordinator-sole"


# =========================================================================
# 2. STATE REPLICATION
# =========================================================================


class TestStateReplication:
    """State snapshot, apply, and background replication."""

    def test_state_snapshot_with_nodes(self) -> None:
        """state_snapshot captures registered node details."""
        coord = _make_coordinator()

        # C6: snapshot health comes from the canonical ``is_healthy``
        # attribute (what schedulers filter on) — not ``.healthy``.
        mock_node_0 = MagicMock(
            host="10.0.0.1", port=50051, start_layer=0, end_layer=7, is_healthy=True,
        )
        mock_node_1 = MagicMock(
            host="10.0.0.2", port=50052, start_layer=8, end_layer=15, is_healthy=False,
        )
        coord._cluster_mgr = _mock_cluster_mgr({
            "node-0": mock_node_0,
            "node-1": mock_node_1,
        })
        coord.total_layers = 32

        snapshot = coord.state_snapshot()

        assert snapshot["model_name"] == "test-model"
        assert snapshot["total_layers"] == 32
        assert snapshot["node_order"] == ["node-0", "node-1"]
        assert snapshot["nodes"]["node-0"]["host"] == "10.0.0.1"
        assert snapshot["nodes"]["node-1"]["healthy"] is False
        assert "timestamp" in snapshot

    def test_state_snapshot_empty(self) -> None:
        """state_snapshot works with no registered nodes."""
        coord = _make_coordinator()
        snapshot = coord.state_snapshot()
        assert snapshot["model_name"] == "test-model"
        assert snapshot["nodes"] == {}

    def test_apply_state_snapshot_on_standby(self) -> None:
        """Standby coordinator registers nodes from snapshot."""
        coord = _make_coordinator()
        coord._election._is_standby = True

        with patch.object(coord, "manual_register") as mock_register:
            coord.apply_state_snapshot(
                {
                    "nodes": {
                        "node-0": {
                            "host": "10.0.0.1",
                            "port": 50051,
                            "start_layer": 0,
                            "end_layer": 7,
                        },
                    },
                    "node_order": ["node-0"],
                }
            )
            mock_register.assert_called_once_with(
                node_id="node-0",
                host="10.0.0.1",
                port=50051,
                start_layer=0,
                end_layer=7,
            )

    def test_apply_state_snapshot_non_standby_skips(self) -> None:
        """Non-standby coordinator ignores state snapshot."""
        coord = _make_coordinator()
        coord._election._is_standby = False

        with patch.object(coord, "manual_register") as mock_register:
            coord.apply_state_snapshot(
                {
                    "nodes": {
                        "node-0": {
                            "host": "10.0.0.1",
                            "port": 50051,
                            "start_layer": 0,
                            "end_layer": 7,
                        }
                    },
                    "node_order": ["node-0"],
                }
            )
            mock_register.assert_not_called()

    def test_apply_state_snapshot_empty_nodes(self) -> None:
        """apply_state_snapshot handles empty nodes dict gracefully."""
        coord = _make_coordinator()
        coord._election._is_standby = True

        with patch.object(coord, "manual_register") as mock_register:
            coord.apply_state_snapshot({"nodes": {}, "node_order": []})
            mock_register.assert_not_called()

    def test_replication_thread_starts_with_peers(self) -> None:
        """_start_state_replication creates daemon thread when peers exist."""
        coord = _make_coordinator()
        coord._election._replication_peers = ["http://10.0.0.2:8000"]
        coord._running.set()

        with patch.object(threading, "Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            coord._election._start_state_replication()

            mock_thread_cls.assert_called_once_with(
                target=coord._election._replication_loop,
                daemon=True,
                name="state-replication",
            )
            mock_thread.start.assert_called_once()

    def test_replication_does_not_start_without_peers(self) -> None:
        """_start_state_replication does nothing when peer list is empty."""
        coord = _make_coordinator()
        coord._election._replication_peers = []

        with patch.object(threading, "Thread") as mock_thread_cls:
            coord._election._start_state_replication()
            mock_thread_cls.assert_not_called()

    def test_set_replication_peers_and_start(self) -> None:
        """set_replication_peers stores peers and starts replication if running."""
        coord = _make_coordinator()
        coord._running.set()

        with patch.object(coord._election, "_start_state_replication") as mock_start:
            coord.set_replication_peers(["http://10.0.0.2:8000"])
            assert coord._election._replication_peers == ["http://10.0.0.2:8000"]
            mock_start.assert_called_once()

    def test_replication_loop_sends_heartbeat_to_peers(self) -> None:
        """Replication loop pushes heartbeat pings to peers."""
        coord = _make_coordinator()
        coord._election._replication_peers = ["http://10.0.0.2:8000"]

        coord._running = MagicMock(spec=threading.Event)
        coord._running.is_set.side_effect = [True, False]  # Run once

        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(status_code=200)

            with patch("distllm.core.coordinator_election.time.sleep"):
                coord._election._replication_loop()

            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert "/api/v1/ha/snapshot" in args[0]
            assert kwargs["json"]["heartbeat"] is True
            assert kwargs["json"]["node_count"] == 0

    def test_replication_loop_sends_full_snapshot_on_tick_10(self) -> None:
        """Replication loop sends full state snapshot every 10th tick."""
        coord = _make_coordinator()
        coord._election._replication_peers = ["http://10.0.0.2:8000"]

        coord._running = MagicMock(spec=threading.Event)
        coord._running.is_set.side_effect = [True] * 10 + [False]

        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(status_code=200)

            with patch("distllm.core.coordinator_election.time.sleep"):
                coord._election._replication_loop()

            assert mock_client.post.call_count == 10

            heartbeat_count = 0
            snapshot_count = 0
            for ca in mock_client.post.call_args_list:
                _, kwargs = ca
                if "heartbeat" in kwargs.get("json", {}):
                    heartbeat_count += 1
                else:
                    snapshot_count += 1

            assert heartbeat_count == 9
            assert snapshot_count == 1

    def test_replication_handles_peer_failure_gracefully(self) -> None:
        """Replication loop does not crash when peer is unreachable."""
        coord = _make_coordinator()
        coord._election._replication_peers = ["http://10.0.0.2:8000"]

        coord._running = MagicMock(spec=threading.Event)
        coord._running.is_set.side_effect = [True, False]

        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.side_effect = ConnectionError("Connection refused")

            with patch("distllm.core.coordinator_election.time.sleep"):
                coord._election._replication_loop()  # Should not raise

            mock_client.post.assert_called_once()


# =========================================================================
# 3. FAILOVER SCENARIOS
# =========================================================================


class TestFailover:
    """Leader failure detection, standby promotion, and state handoff."""

    def test_leader_failure_detection_via_heartbeat_timeout(self) -> None:
        """Peer that misses heartbeat timeout is marked OFFLINE (not evicted).

        The HA design keeps timed-out peers in the table with ``online=False``
        so they can rejoin and self-heal when heartbeats resume; quorum math
        counts only online peers, which is what prevents split-brain.
        """
        from distllm.core.ha_coordinator import RayFaultTolerance

        ft = RayFaultTolerance(
            "coordinator-b", election_timeout_s=10.0, heartbeat_transport=_dead_transport
        )
        ft.add_peer("coordinator-a", "10.0.0.1", 50051)
        ft.handle_heartbeat_request("coordinator-a", term=0)

        assert "coordinator-a" in ft._peers
        ft._run_election_round()
        assert ft._leader_id == "coordinator-a"

        ft._peers["coordinator-a"]["last_seen"] = 0.0
        ft._run_election_round()

        assert ft._peers["coordinator-a"]["online"] is False
        # The stale leader view is kept until quorum returns (a 2-node
        # cluster has no majority to elect a successor — split-brain safe).

    def test_standby_promotion_to_leader(self) -> None:
        """A standby in a 3-node cluster becomes leader after the current
        leader times out — with a confirmed majority (2 of 3).

        A 2-node cluster cannot fail over (1 of 2 is no majority); promotion
        requires the surviving quorum, which is what prevents split-brain.
        """
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        ft = RayFaultTolerance(
            "coordinator-b", election_timeout_s=10.0, heartbeat_transport=_dead_transport
        )
        ft.add_peer("coordinator-a", "10.0.0.1", 50051)
        ft.add_peer("coordinator-c", "10.0.0.3", 50053)

        # a is the current leader; both peers are alive.
        ft.handle_heartbeat_request("coordinator-a", term=0)
        ft.handle_heartbeat_request("coordinator-c", term=0)
        ft._run_election_round()
        assert ft._state == CoordinatorState.FOLLOWER
        assert ft._leader_id == "coordinator-a"

        # Leader a dies; c remains alive -> b + c = 2 of 3 = quorum.
        ft._peers["coordinator-a"]["last_seen"] = 0.0
        ft._run_election_round()

        assert ft._state == CoordinatorState.LEADER
        assert ft._leader_id == "coordinator-b"

    def test_failover_increments_term(self) -> None:
        """Term increments when a new leader is elected during failover."""
        from distllm.core.ha_coordinator import RayFaultTolerance

        ft = RayFaultTolerance(
            "coordinator-b", heartbeat_transport=_dead_transport
        )
        ft.add_peer("coordinator-a", "10.0.0.1", 50051)
        ft.add_peer("coordinator-c", "10.0.0.3", 50053)

        initial_term = ft._current_term

        ft.handle_heartbeat_request("coordinator-a", term=0)
        ft.handle_heartbeat_request("coordinator-c", term=0)
        ft._run_election_round()
        assert ft._leader_id == "coordinator-a"

        ft._peers["coordinator-a"]["last_seen"] = 0.0
        ft._run_election_round()

        assert ft._current_term == initial_term + 1
        assert ft._voted_for == "coordinator-b"

    def test_state_handoff_during_failover(self) -> None:
        """Replicated state is preserved and available after failover."""
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        leader = RayFaultTolerance("coordinator-a")
        leader.replicate_state("node_count", 5)
        leader.replicate_state("active_models", ["llama", "mistral"])
        saved_state = leader.get_replicated_state()

        standby = RayFaultTolerance(
            "coordinator-b", heartbeat_transport=_dead_transport
        )
        standby.add_peer("coordinator-a", "10.0.0.1", 50051)
        standby.add_peer("coordinator-c", "10.0.0.3", 50053)
        standby.handle_heartbeat_request(
            "coordinator-a", term=0, state=saved_state,
        )
        standby.handle_heartbeat_request("coordinator-c", term=0)

        assert standby.get_replicated_state()["node_count"] == 5
        assert "mistral" in standby.get_replicated_state()["active_models"]

        standby._peers["coordinator-a"]["last_seen"] = 0.0
        standby._run_election_round()

        assert standby._state == CoordinatorState.LEADER
        assert standby.get_replicated_state()["node_count"] == 5

    def test_multiple_standbys_single_leader(self) -> None:
        """Among multiple standbys only one becomes leader after failover."""
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        b = RayFaultTolerance("coordinator-b", heartbeat_transport=_dead_transport)
        c = RayFaultTolerance("coordinator-c", heartbeat_transport=_dead_transport)

        for peer in [("coordinator-a", "10.0.0.1", 50051), ("coordinator-c", "10.0.0.3", 50053)]:
            b.add_peer(*peer)
        for peer in [("coordinator-a", "10.0.0.1", 50051), ("coordinator-b", "10.0.0.2", 50052)]:
            c.add_peer(*peer)

        for ft in (b, c):
            ft.handle_heartbeat_request("coordinator-a", term=0)

        b._run_election_round()
        c._run_election_round()

        assert b._leader_id == "coordinator-a"
        assert c._leader_id == "coordinator-a"

        for ft in (b, c):
            ft._peers["coordinator-a"]["last_seen"] = 0.0

        b._run_election_round()
        c._run_election_round()

        assert b._state == CoordinatorState.LEADER
        assert b._leader_id == "coordinator-b"
        assert c._state == CoordinatorState.FOLLOWER
        assert c._leader_id == "coordinator-b"


# =========================================================================
# 4. SPLIT-BRAIN PREVENTION
# =========================================================================


class TestSplitBrainPrevention:
    """Preventing multiple active leaders in a partitioned cluster."""

    def test_standby_not_leader(self) -> None:
        """Coordinator in follower/candidate state is not the leader."""
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        ft = RayFaultTolerance("test-coord")
        ft._state = CoordinatorState.FOLLOWER
        assert ft.is_leader() is False

        ft._state = CoordinatorState.CANDIDATE
        assert ft.is_leader() is False

        ft._state = CoordinatorState.LEADER
        assert ft.is_leader() is True

    def test_higher_term_stepping_down(self) -> None:
        """Leader steps down when receiving heartbeat with higher term."""
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        ft = RayFaultTolerance("coordinator-a")
        ft._state = CoordinatorState.LEADER
        ft._leader_id = "coordinator-a"
        ft._current_term = 2

        ft.add_peer("coordinator-b", "10.0.0.2", 50051)
        ft.handle_heartbeat_request("coordinator-b", term=5)

        assert ft._state == CoordinatorState.FOLLOWER
        assert ft._current_term == 5
        assert ft._voted_for is None

    def test_overlapping_elections_resolve_via_higher_term(self) -> None:
        """When two nodes claim leadership the higher term steps down lower.

        The heartbeat handler steps down on higher term but does not set
        peer as leader if the peer's ID is higher. The subsequent election
        round elects the lowest ID among alive peers.
        """
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        a = RayFaultTolerance("coordinator-a", heartbeat_transport=_dead_transport)
        a._state = CoordinatorState.LEADER
        a._leader_id = "coordinator-a"
        a._current_term = 3

        b = RayFaultTolerance("coordinator-b", heartbeat_transport=_dead_transport)
        b._state = CoordinatorState.LEADER
        b._leader_id = "coordinator-b"
        b._current_term = 5
        b.add_peer("coordinator-a", "10.0.0.1", 50051)

        a.add_peer("coordinator-b", "10.0.0.2", 50052)
        a.handle_heartbeat_request("coordinator-b", term=5)

        # A steps down due to higher term but B's ID is higher so B is
        # not recorded as leader
        assert a._state == CoordinatorState.FOLLOWER
        assert a._current_term == 5

        # Election round: lowest ID among alive peers wins.  A (lower) is
        # elected and B becomes follower.
        a._run_election_round()
        assert a._state == CoordinatorState.LEADER
        assert a._current_term == 6

        # B's election round: A has lower ID so B follows A
        b._run_election_round()
        assert b._state == CoordinatorState.FOLLOWER
        assert b._leader_id == "coordinator-a"

    def test_lost_quorum_leader_steps_down(self) -> None:
        """Leader steps down when it loses majority (split-brain prevention).

        With initial cluster of 5, quorum is 3.  If only 2 nodes are alive
        (self + 1 peer) the leader must step down to prevent split-brain.
        """
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        ft = RayFaultTolerance(
            "coordinator-e", heartbeat_transport=_dead_transport
        )
        ft._initial_cluster_size = 5  # quorum = 3
        ft._state = CoordinatorState.LEADER
        ft._leader_id = "coordinator-e"

        # Add 4 peers; only coordinator-a stays alive
        for i in range(1, 5):
            pid = f"coordinator-{chr(96 + i)}"  # a, b, c, d
            ft.add_peer(pid, "10.0.0.1", 50051)

        # coordinator-a stays alive (recent last_seen from add_peer)
        # 3 peers are stale
        for stale_pid in ("coordinator-b", "coordinator-c", "coordinator-d"):
            ft._peers[stale_pid]["last_seen"] = 0.0

        ft._run_election_round()

        # After eviction: alive = [coordinator-a, coordinator-e] = 2
        # quorum = 3 => 2 < 3 => no quorum => leader steps down
        assert ft._state == CoordinatorState.FOLLOWER
        assert ft._leader_id is None

    def test_rejoining_node_follows_higher_term(self) -> None:
        """Rejoining node detects it is behind and follows current leader."""
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        ft = RayFaultTolerance("coordinator-b")
        ft._current_term = 2
        ft._state = CoordinatorState.CANDIDATE

        ft.add_peer("coordinator-a", "10.0.0.1", 50051)
        ft.handle_heartbeat_request("coordinator-a", term=5)

        assert ft._current_term == 5
        assert ft._state == CoordinatorState.FOLLOWER

    def test_lower_id_forces_step_down(self) -> None:
        """Leader with higher ID steps down when lower-ID peer sends heartbeat."""
        from distllm.core.ha_coordinator import CoordinatorState, RayFaultTolerance

        ft = RayFaultTolerance("coordinator-b")
        ft._state = CoordinatorState.LEADER
        ft._leader_id = "coordinator-b"
        ft._current_term = 3

        ft.add_peer("coordinator-a", "10.0.0.1", 50051)
        ft.handle_heartbeat_request("coordinator-a", term=3)

        assert ft._state == CoordinatorState.FOLLOWER
        assert ft._leader_id == "coordinator-a"


# =========================================================================
# 5. SHUTDOWN RECOVERY
# =========================================================================


class TestShutdownRecovery:
    """Graceful shutdown, state persistence, and restart recovery."""

    def test_stop_sets_shutting_down_flag(self) -> None:
        """stop() sets _shutting_down flag to reject new requests."""
        coord = _make_coordinator()

        with patch.multiple(
            coord,
            _health_mgr=MagicMock(),
            _pipeline=MagicMock(),
            _resource_mgr=MagicMock(),
            _batch_scheduler=None,
            _cluster_mgr=_mock_cluster_mgr(),
        ), patch.object(coord, "_save_shutdown_state"), patch(
            "torch.cuda.is_available", return_value=False
        ):
            coord.stop(timeout=1.0)

        assert coord._shutting_down is True
        assert coord._running.is_set() is False

    def test_stop_drains_inflight_requests(self) -> None:
        """Stop drains in-flight requests with configurable timeout."""
        coord = _make_coordinator()
        coord._batch_scheduler = MagicMock()
        coord._recovery_manager = MagicMock()
        coord._resource_mgr = MagicMock()
        coord._health_mgr = MagicMock()
        coord._pipeline = MagicMock()
        coord._cluster_mgr = _mock_cluster_mgr()

        active_counts = [3, 3, 2, 1, 0, 0]
        type(coord._batch_scheduler).active_count = PropertyMock(
            side_effect=active_counts
        )

        with patch.multiple(
            coord._health_mgr, stop=MagicMock(),
        ), patch.object(coord, "_save_shutdown_state"), patch(
            "torch.cuda.is_available", return_value=False
        ), patch.object(
            coord._pipeline, "shutdown"
        ), patch(
            "distllm.core.coordinator.time.time",
            side_effect=[0.0, 0.25, 0.5, 0.75, 1.0, 2.0],
        ):
            coord.stop(timeout=5.0)

        assert coord._shutting_down is True

    def test_stop_checkpoints_active_requests(self) -> None:
        """Active requests are checkpointed if unfinished before timeout."""
        coord = _make_coordinator()
        coord._batch_scheduler = MagicMock()
        coord._recovery_manager = MagicMock()
        coord._resource_mgr = MagicMock()
        coord._health_mgr = MagicMock()
        coord._pipeline = MagicMock()
        coord._cluster_mgr = _mock_cluster_mgr()

        type(coord._batch_scheduler).active_count = PropertyMock(return_value=2)
        coord._batch_scheduler.snapshot_active.return_value = [
            ("req-1", MagicMock(prompt_tokens=[1, 2], generated_tokens=[3])),
            ("req-2", MagicMock(prompt_tokens=[4, 5], generated_tokens=[6, 7])),
        ]

        with patch.multiple(
            coord._health_mgr, stop=MagicMock(),
        ), patch.object(coord, "_save_shutdown_state"), patch(
            "torch.cuda.is_available", return_value=False
        ), patch.object(
            coord._pipeline, "shutdown"
        ), patch(
            "distllm.core.coordinator.time.time",
            side_effect=[0.0, 999.0],
        ), patch(
            "time.sleep"
        ):
            coord.stop(timeout=999.0)

        assert coord._recovery_manager.save_checkpoint.call_count == 2
        req_ids = [
            call_args[1]["request_id"]
            for call_args in coord._recovery_manager.save_checkpoint.call_args_list
        ]
        assert "req-1" in req_ids
        assert "req-2" in req_ids

    def test_save_and_load_election_state(self, tmp_path: pytest.TempPathFactory) -> None:
        """Election state persists to disk and reloads correctly."""
        from distllm.core.ha_coordinator import RayFaultTolerance

        state_path = str(tmp_path / "election_state.json")

        ft = RayFaultTolerance("coordinator-1")
        ft._current_term = 7
        ft._voted_for = "coordinator-1"
        ft._initial_cluster_size = 5
        ft.save_election_state(path=state_path)

        assert os.path.exists(state_path)

        ft2 = RayFaultTolerance("coordinator-1")
        ft2.load_election_state(path=state_path)

        assert ft2._current_term == 7
        assert ft2._voted_for == "coordinator-1"
        assert ft2._initial_cluster_size == 5

    def test_shutdown_state_save_content(self, tmp_path: pytest.TempPathFactory) -> None:
        """Shutdown state file contains expected coordinator metadata."""
        coord = _make_coordinator()
        mock_node = MagicMock(host="10.0.0.1", port=50051, healthy=True)
        coord._cluster_mgr = _mock_cluster_mgr({"node-0": mock_node})

        state_dir = str(tmp_path)
        prev = os.environ.get("DISTLLM_DATA_DIR")
        try:
            os.environ["DISTLLM_DATA_DIR"] = state_dir
            with patch("torch.cuda.is_available", return_value=False), patch(
                "distllm.core.coordinator.time.time", return_value=123456.0
            ), patch.multiple(
                coord,
                _health_mgr=MagicMock(),
                _pipeline=MagicMock(),
                _resource_mgr=MagicMock(),
                _batch_scheduler=None,
            ), patch(
                "time.sleep"
            ):
                coord.stop(timeout=1.0)

            state_path = os.path.join(state_dir, "shutdown_state.json")
            assert os.path.exists(state_path)

            with open(state_path) as f:
                data = json.load(f)

            assert data["model_name"] == "test-model"
            assert data["shutdown_time"] == 123456.0
            assert "node-0" in data["nodes"]
            assert data["nodes"]["node-0"]["host"] == "10.0.0.1"
        finally:
            if prev is not None:
                os.environ["DISTLLM_DATA_DIR"] = prev
            else:
                os.environ.pop("DISTLLM_DATA_DIR", None)
