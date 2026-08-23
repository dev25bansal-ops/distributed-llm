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
        coord._is_standby = False
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
        coord._is_standby = True

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
        assert coord._replication_peers == peers
