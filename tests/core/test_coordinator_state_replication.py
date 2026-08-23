"""Tests for continuous coordinator state replication between HA peers.

Covers:
- state_snapshot() produces correct state
- apply_state_snapshot() updates standby coordinator
- _replication_loop pushes snapshots periodically
- set_replication_peers() configures replication targets
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


class TestCoordinatorStateReplication:
    _active_patchers: list

    @pytest.fixture(autouse=True)
    def _stop_patchers(self):
        self._active_patchers = []
        yield
        for p in self._active_patchers:
            p.stop()
    """HA coordinator state replication."""

    def test_state_snapshot_structure(self):
        """state_snapshot should return correct structure."""
        from distllm.core.coordinator import Coordinator
        from distllm.core.coordinator_config import CoordinatorConfig

        coord = Coordinator(config=CoordinatorConfig(model_name="test-model"))
        snapshot = coord.state_snapshot()

        assert "model_name" in snapshot
        assert "nodes" in snapshot
        assert "node_order" in snapshot
        assert "timestamp" in snapshot
        assert snapshot["model_name"] == "test-model"

    def test_apply_state_snapshot_standby_only(self):
        """apply_state_snapshot should only work on standby coordinators."""
        from distllm.core.coordinator import Coordinator
        from distllm.core.coordinator_config import CoordinatorConfig

        coord = Coordinator(config=CoordinatorConfig(model_name="test"))
        coord._election._is_standby = False
        with patch.object(coord, 'manual_register') as mock_register:
            coord.apply_state_snapshot({
                "nodes": {"node-0": {"host": "10.0.0.1", "port": 50051, "start_layer": 0, "end_layer": 15}},
                "node_order": ["node-0"],
            })
            # Non-standby should not register nodes
            mock_register.assert_not_called()

    def test_apply_state_snapshot_on_standby(self):
        """Standby coordinator should register nodes from snapshot."""
        from distllm.core.coordinator import Coordinator
        from distllm.core.coordinator_config import CoordinatorConfig

        coord = Coordinator(config=CoordinatorConfig(model_name="test"))
        coord._election._is_standby = True

        with patch.object(coord, 'manual_register') as mock_register:
            coord.apply_state_snapshot({
                "nodes": {
                    "node-0": {"host": "10.0.0.1", "port": 50051, "start_layer": 0, "end_layer": 15},
                    "node-1": {"host": "10.0.0.2", "port": 50052, "start_layer": 16, "end_layer": 31},
                },
                "node_order": ["node-0", "node-1"],
            })
            assert mock_register.call_count == 2

    def test_set_replication_peers(self):
        """set_replication_peers should store peer URLs."""
        from distllm.core.coordinator import Coordinator
        from distllm.core.coordinator_config import CoordinatorConfig

        coord = Coordinator(config=CoordinatorConfig(model_name="test"))
        peers = ["http://10.0.0.2:8000", "http://10.0.0.3:8000"]
        coord.set_replication_peers(peers)
        assert coord._election._replication_peers == peers

    # ── B6 regression: standby replication was a permanent no-op ──

    def _make_standby_election(self, is_leader=False):
        """Build a CoordinatorElection with a fake coordinator + mock HA election."""
        import distllm.core.ha_coordinator as ha_mod
        from distllm.core.coordinator_election import CoordinatorElection

        class _FakeCoordinator:
            """Minimal coordinator surface used by CoordinatorElection."""

            model_name = "test-model"
            port = 50050
            nodes: dict = {}
            node_order: list = []

            def manual_register(self, node_id, host, port, start_layer, end_layer):
                self.nodes[node_id] = {
                    "host": host,
                    "port": port,
                    "start_layer": start_layer,
                    "end_layer": end_layer,
                }

        fake = _FakeCoordinator()
        election = CoordinatorElection(fake)
        mock_ft = patch.object(ha_mod, "RayFaultTolerance")
        patched = mock_ft.start()
        patched.return_value.is_leader.return_value = is_leader
        # Plain pytest class (no unittest addCleanup): the autouse fixture
        # below stops registered patchers when the test finishes.
        self._active_patchers.append(mock_ft)
        return fake, election

    def test_standby_flag_set_when_ha_enabled_not_leader(self):
        """B6: enable_ha marks a non-leader node as standby."""
        fake, election = self._make_standby_election(is_leader=False)
        election.enable_ha(
            coordinator_id="standby-1",
            peer_coordinators=[("leader-0", "10.0.0.1", 50050)],
        )
        # Previously _is_standby stayed False forever → apply was a no-op.
        assert election._is_standby is True

    def test_standby_applies_leader_state_snapshot(self):
        """B6: standby node applies a state snapshot pushed by leader."""
        fake, election = self._make_standby_election(is_leader=False)
        election.enable_ha(coordinator_id="standby-1")

        election.apply_state_snapshot({
            "nodes": {
                "node-0": {"host": "10.0.0.1", "port": 50051, "start_layer": 0, "end_layer": 15},
                "node-1": {"host": "10.0.0.2", "port": 50052, "start_layer": 16, "end_layer": 31},
            },
            "node_order": ["node-0", "node-1"],
        })
        # Previously apply_state_snapshot returned early and never registered.
        assert "node-0" in fake.nodes
        assert "node-1" in fake.nodes

    def test_leader_pushed_state_applied_via_election_callback(self):
        """B6: state pushed by the leader over the election is applied."""
        fake, election = self._make_standby_election(is_leader=False)
        election.enable_ha(coordinator_id="standby-1")

        # Capture the callback the election invokes on leader state.
        on_state_change = election._ha_election.on_state_change
        callback = on_state_change.call_args[0][0]

        callback({
            "nodes": {
                "node-0": {"host": "10.0.0.1", "port": 50051, "start_layer": 0, "end_layer": 15},
            },
        })
        assert "node-0" in fake.nodes

    def test_standby_flag_reset_on_promotion(self):
        """B6: a promoted leader stops applying leader snapshots."""
        fake, election = self._make_standby_election(is_leader=False)
        election.enable_ha(coordinator_id="standby-1")
        assert election._is_standby is True

        # Simulate promotion: this node now leads.
        election._ha_election.is_leader.return_value = True
        election.apply_state_snapshot({
            "nodes": {
                "node-0": {"host": "10.0.0.1", "port": 50051, "start_layer": 0, "end_layer": 15},
            },
        })
        assert election._is_standby is False
        assert "node-0" not in fake.nodes
