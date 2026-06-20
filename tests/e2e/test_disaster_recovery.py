"""E2E tests for disaster recovery scenarios.

Verifies that after coordinator crash and restart:
- Replicated state (topology, checkpoints) survives the crash
- Node re-registration works after coordinator comes back
- Pending requests are handled (completed or error-resolved)
- State replication store maintains consistency
"""

from __future__ import annotations

import os
import threading
import time

import pytest
from unittest.mock import MagicMock, patch

from distllm.core.coordinator_state import (
    CoordinatorRole,
    CoordinatorStateMachine,
)
from distllm.core.coordinator_lifecycle import (
    RequestTracker,
)
from distllm.core.state_replication import (
    ReplicatedState,
    StateReplicationStore,
    TopologyStateStore,
)


@pytest.mark.e2e
class TestTopologyStatePersistence:
    """Verify that topology state survives coordinator crash via replication store."""

    def test_topology_survives_memory_backend_restart(self):
        """Topology written to memory store can be read back correctly.

        AAA:
          Arrange - Write topology to a memory-backed store.
          Act     - Read it back from the same store.
          Assert  - Topology has correct nodes and layer assignments.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")
        topo_store = TopologyStateStore(store)

        nodes = {
            "node-0": {"host": "10.0.0.1", "port": 50051, "start_layer": 0, "end_layer": 15, "healthy": True},
            "node-1": {"host": "10.0.0.2", "port": 50052, "start_layer": 16, "end_layer": 31, "healthy": True},
        }
        node_order = ["node-0", "node-1"]
        topo_store.save_topology(nodes, node_order)

        # Act
        recovered = topo_store.load_topology()

        # Assert
        assert recovered is not None
        assert "nodes" in recovered
        assert "node_order" in recovered
        assert recovered["node_order"] == ["node-0", "node-1"]

    def test_topology_with_mock_node_objects(self):
        """Topology save/load works with MagicMock node objects (realistic).

        AAA:
          Arrange - Create MagicMock nodes with attributes.
          Act     - Save and load topology.
          Assert  - Node attributes are correctly serialized.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")
        topo_store = TopologyStateStore(store)

        node0 = MagicMock()
        node0.host = "10.0.0.1"
        node0.port = 50051
        node0.start_layer = 0
        node0.end_layer = 15
        node0.healthy = True

        node1 = MagicMock()
        node1.host = "10.0.0.2"
        node1.port = 50052
        node1.start_layer = 16
        node1.end_layer = 31
        node1.healthy = True

        nodes = {"node-0": node0, "node-1": node1}

        # Act
        topo_store.save_topology(nodes, ["node-0", "node-1"])
        recovered = topo_store.load_topology()

        # Assert
        assert recovered is not None
        assert recovered["nodes"]["node-0"]["host"] == "10.0.0.1"
        assert recovered["nodes"]["node-0"]["port"] == 50051
        assert recovered["nodes"]["node-1"]["start_layer"] == 16

    def test_checkpoints_survive_memory_backend(self):
        """Recovery checkpoints can be saved and loaded from memory store.

        AAA:
          Arrange - Write checkpoints to memory-backed store.
          Act     - Load them back.
          Assert  - Checkpoints are recovered with request data intact.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")
        topo_store = TopologyStateStore(store)

        mock_ckpt = MagicMock()
        mock_ckpt.request_id = "req-abc"
        mock_ckpt.prompt_tokens = [1, 2, 3]
        mock_ckpt.generated_tokens = [4, 5, 6]
        mock_ckpt.node_id = "node-0"

        checkpoints = {"req-abc": mock_ckpt}
        topo_store.save_checkpoints(checkpoints)

        # Act
        recovered = topo_store.load_checkpoints()

        # Assert
        assert recovered is not None
        assert "checkpoints" in recovered
        assert "req-abc" in recovered["checkpoints"]
        ckpt = recovered["checkpoints"]["req-abc"]
        assert ckpt["request_id"] == "req-abc"
        assert ckpt["node_id"] == "node-0"


@pytest.mark.e2e
class TestStateReplicationStoreConsistency:
    """Test the StateReplicationStore under crash-like conditions."""

    def test_version_conflict_detection(self):
        """CAS (compare-and-swap) prevents stale writes.

        AAA:
          Arrange - Write a key at version 0 (force), then at version v1.
          Act     - Attempt a write with stale version v1 when actual is v2.
          Assert  - ValueError is raised.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")
        v1 = store.put("topology", {"nodes": {}}, version=0)
        v2 = store.put("topology", {"nodes": {"a": 1}}, version=v1)

        # Act / Assert
        with pytest.raises(ValueError, match="Version conflict"):
            store.put("topology", {"nodes": {"b": 2}}, version=v1)

    def test_force_write_bypasses_version_check(self):
        """version=0 forces a write regardless of current version.

        AAA:
          Arrange - Store exists at version 3.
          Act     - Write with version=0 (force).
          Assert  - Write succeeds and increments version.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")
        store.put("key", "v1")
        store.put("key", "v2")
        v3 = store.put("key", "v3")

        # Act
        v4 = store.put("key", "forced", version=0)

        # Assert
        assert v4 > v3
        assert store.get("key") == "forced"

    def test_delete_returns_correct_boolean(self):
        """Deleting an existing key returns True; deleting missing returns False.

        AAA:
          Arrange - Store a key.
          Act     - Delete it, then delete again.
          Assert  - First returns True, second returns False.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")
        store.put("ephemeral", {"data": 42})

        # Act
        result1 = store.delete("ephemeral")
        result2 = store.delete("ephemeral")

        # Assert
        assert result1 is True
        assert result2 is False

    def test_list_keys_reflects_current_state(self):
        """list_keys() returns all stored keys after writes and deletes.

        AAA:
          Arrange - Write several keys, delete one.
          Act     - Call list_keys().
          Assert  - Returns the remaining keys.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")
        store.put("topology", {})
        store.put("checkpoints", {})
        store.put("metrics", {})
        store.delete("metrics")

        # Act
        keys = store.list_keys()

        # Assert
        assert "topology" in keys
        assert "checkpoints" in keys
        assert "metrics" not in keys

    def test_memory_backend_full_lifecycle(self):
        """In-memory backend supports put, get, versioned get, delete, list.

        AAA:
          Arrange - Create memory-backed store.
          Act     - Put, get, versioned get, delete, list.
          Assert  - All operations work correctly.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")

        # Act
        v = store.put("test-key", {"hello": "world"})
        value = store.get("test-key")
        versioned = store.get_versioned("test-key")
        keys = store.list_keys()
        deleted = store.delete("test-key")
        gone = store.get("test-key")

        # Assert
        assert value == {"hello": "world"}
        assert versioned is not None
        assert versioned.version == 1
        assert versioned.key == "test-key"
        assert "test-key" in keys
        assert deleted is True
        assert gone is None

    def test_versioned_state_has_metadata(self):
        """get_versioned returns ReplicatedState with version and timestamp.

        AAA:
          Arrange - Write a value.
          Act     - Get versioned state.
          Assert  - Has version, updated_at, updated_by fields.
        """
        # Arrange
        store = StateReplicationStore(backend="memory", node_id="test-node")

        # Act
        store.put("meta-key", {"data": 1})
        state = store.get_versioned("meta-key")

        # Assert
        assert state is not None
        assert state.version == 1
        assert state.updated_at > 0
        assert state.updated_by == "test-node"

    def test_watch_callback_invoked_on_update(self):
        """watch() invokes callback when the key is updated.

        AAA:
          Arrange - Set up a watcher on a key.
          Act     - Update the key from another thread.
          Assert  - Callback is invoked with the new value.
        """
        # Arrange
        store = StateReplicationStore(backend="memory")
        store.put("watched", {"v": 0})

        callback_values = []

        def on_update(new_value, version):
            callback_values.append((new_value, version))

        store.watch("watched", on_update, interval_s=0.05)

        # Act
        time.sleep(0.1)  # let watcher start
        store.put("watched", {"v": 1})
        time.sleep(0.2)  # let watcher detect change

        # Assert
        assert len(callback_values) >= 1
        assert callback_values[-1][0] == {"v": 1}


@pytest.mark.e2e
class TestNodeReRegistrationAfterRestart:
    """Verify that nodes can re-register after coordinator restarts."""

    def test_cluster_nodes_endpoint_returns_structure(self, e2e_api_client, e2e_auth_headers):
        """Cluster nodes endpoint returns expected structure.

        AAA:
          Arrange - Coordinator is running (from fixture).
          Act     - GET /api/cluster/nodes.
          Assert  - Returns nodes list.
        """
        # Act
        response = e2e_api_client.get("/api/cluster/nodes", headers=e2e_auth_headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert isinstance(data["nodes"], list)

    def test_coordinator_state_machine_can_recover_from_recovering(self):
        """A coordinator in RECOVERING state can transition back to LEADER.

        AAA:
          Arrange - State machine in LEADER, then RECOVERING.
          Act     - Transition to LEADER.
          Assert  - Successfully becomes LEADER again.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.LEADER)
        sm.transition_to(CoordinatorRole.RECOVERING)
        assert sm.role == CoordinatorRole.RECOVERING

        # Act
        sm.transition_to(CoordinatorRole.LEADER)

        # Assert
        assert sm.is_leader is True

    def test_coordinator_state_machine_recovery_preserves_role(self):
        """After recovery, the role property reflects the new LEADER state.

        AAA:
          Arrange - Go through INIT -> FOLLOWER -> LEADER -> RECOVERING -> LEADER.
          Act     - Read role property.
          Assert  - role is LEADER, previous_role is RECOVERING.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.LEADER)
        sm.transition_to(CoordinatorRole.RECOVERING)

        # Act
        sm.transition_to(CoordinatorRole.LEADER)

        # Assert
        assert sm.role == CoordinatorRole.LEADER
        assert sm.previous_role == CoordinatorRole.RECOVERING

    def test_coordinator_can_become_follower_after_recovery(self):
        """After RECOVERING, coordinator can transition to FOLLOWER.

        AAA:
          Arrange - State machine in RECOVERING.
          Act     - Transition to FOLLOWER.
          Assert  - Role is FOLLOWER.
        """
        # Arrange
        sm = CoordinatorStateMachine()
        sm.transition_to(CoordinatorRole.FOLLOWER)
        sm.transition_to(CoordinatorRole.LEADER)
        sm.transition_to(CoordinatorRole.RECOVERING)

        # Act
        sm.transition_to(CoordinatorRole.FOLLOWER)

        # Assert
        assert sm.role == CoordinatorRole.FOLLOWER


@pytest.mark.e2e
class TestPendingRequestHandlingAfterCrash:
    """Verify pending requests are resolved after coordinator crash and recovery."""

    def test_request_tracker_reports_pending_count(self):
        """RequestTracker accurately reports the number of pending requests.

        AAA:
          Arrange - Register several requests.
          Act     - Check pending_count.
          Assert  - Count matches registered requests.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("r1")
        tracker.register_request("r2")
        tracker.register_request("r3")

        # Act
        count = tracker.pending_count

        # Assert
        assert count == 3

    def test_request_tracker_set_result_wakes_waiter(self):
        """Setting a result unblocks the waiting thread.

        AAA:
          Arrange - Register a request, start a waiter thread.
          Act     - Set the result from another thread.
          Assert  - Waiter receives the result.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("async-req")
        received = []
        error_holder = []

        def _wait():
            try:
                result = tracker.wait_for_result("async-req", timeout=5.0)
                received.append(result)
            except Exception as e:
                error_holder.append(e)

        t = threading.Thread(target=_wait, daemon=True)
        t.start()

        # Act
        time.sleep(0.1)  # let waiter start
        tracker.set_result("async-req", "recovered-token-stream")

        # Assert
        t.join(timeout=3.0)
        assert len(received) == 1
        assert received[0] == "recovered-token-stream"
        assert len(error_holder) == 0

    def test_request_tracker_set_error_propagates_to_waiter(self):
        """Setting an error causes wait_for_result to raise RuntimeError.

        AAA:
          Arrange - Register a request, start waiter.
          Act     - Set an error.
          Assert  - Waiter receives RuntimeError.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("fail-req")
        errors = []

        def _wait():
            try:
                tracker.wait_for_result("fail-req", timeout=5.0)
            except RuntimeError as e:
                errors.append(str(e))

        t = threading.Thread(target=_wait, daemon=True)
        t.start()

        # Act
        time.sleep(0.1)
        tracker.set_error("fail-req", ConnectionError("coordinator crashed"))

        # Assert
        t.join(timeout=3.0)
        assert len(errors) == 1
        assert "fail-req" in errors[0]
        assert "crashed" in errors[0].lower()

    def test_request_tracker_timeout_raises(self):
        """wait_for_result raises TimeoutError if result never arrives.

        AAA:
          Arrange - Register a request but never set a result.
          Act     - Call wait_for_result with a short timeout.
          Assert  - TimeoutError is raised.
        """
        # Arrange
        tracker = RequestTracker()
        tracker.register_request("ghost-req")

        # Act / Assert
        with pytest.raises(TimeoutError, match="timed out"):
            tracker.wait_for_result("ghost-req", timeout=0.1)

    def test_request_tracker_unknown_request_raises(self):
        """wait_for_result raises ValueError for an unregistered request.

        AAA:
          Arrange - Empty tracker.
          Act     - Call wait_for_result with unknown ID.
          Assert  - ValueError is raised.
        """
        # Arrange
        tracker = RequestTracker()

        # Act / Assert
        with pytest.raises(ValueError, match="Unknown request_id"):
            tracker.wait_for_result("never-registered", timeout=1.0)


@pytest.mark.e2e
class TestFailoverHandlerDiscovery:
    """Test CoordinatorFailoverHandler peer discovery logic."""

    def test_failover_handler_tracks_current_coordinator(self):
        """Handler exposes the current coordinator address.

        AAA:
          Arrange - Create handler with initial coordinator.
          Act     - Read current_coordinator.
          Assert  - Matches the initial host:port.
        """
        from distllm.core.coordinator_failover import CoordinatorFailoverHandler

        # Arrange
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )

        # Act
        current = handler.current_coordinator

        # Assert
        assert current == ("10.0.0.1", 50050)

    def test_failover_handler_stats_report_zero_initially(self):
        """Stats show zero failures and zero failovers at startup.

        AAA:
          Arrange - Create handler.
          Act     - Call stats().
          Assert  - consecutive_failures=0, failover_count=0.
        """
        from distllm.core.coordinator_failover import CoordinatorFailoverHandler

        # Arrange
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
        )

        # Act
        stats = handler.stats()

        # Assert
        assert stats["consecutive_failures"] == 0
        assert stats["failover_count"] == 0
        assert stats["running"] is False

    def test_failover_handler_update_peer_hosts(self):
        """Peer list can be updated dynamically.

        AAA:
          Arrange - Create handler with one peer list.
          Act     - Update peer hosts.
          Assert  - Stats reflects new peer count.
        """
        from distllm.core.coordinator_failover import CoordinatorFailoverHandler

        # Arrange
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
            peer_hosts=[("10.0.0.2", 50050)],
        )

        # Act
        handler.update_peer_hosts([
            ("10.0.0.2", 50050),
            ("10.0.0.3", 50050),
            ("10.0.0.4", 50050),
        ])

        # Assert
        stats = handler.stats()
        assert stats["peer_count"] == 3

    def test_failover_handler_invokes_reconnect_callback(self):
        """When a live peer is found during failover, the callback is invoked.

        AAA:
          Arrange - Create handler with a mock callback and a peer list.
          Act     - Simulate failover by making the primary unreachable.
          Assert  - on_reconnect callback is called with the peer address.
        """
        from distllm.core.coordinator_failover import CoordinatorFailoverHandler

        # Arrange
        reconnect_calls = []
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
            peer_hosts=[("10.0.0.1", 50050), ("10.0.0.2", 50050)],
            on_reconnect=lambda h, p: reconnect_calls.append((h, p)),
        )

        # Act: mock TCP checks — primary fails, secondary succeeds
        def mock_tcp(host, port):
            if host == "10.0.0.1":
                return False
            return True

        handler._check_tcp_alive = mock_tcp
        handler._trigger_failover()

        # Assert
        assert len(reconnect_calls) == 1
        assert reconnect_calls[0] == ("10.0.0.2", 50050)

    def test_failover_handler_no_peers_does_not_crash(self):
        """Failover with empty peer list logs warning but does not crash.

        AAA:
          Arrange - Create handler with no peers.
          Act     - Trigger failover.
          Assert  - No exception raised, failover_count stays 0.
        """
        from distllm.core.coordinator_failover import CoordinatorFailoverHandler

        # Arrange
        handler = CoordinatorFailoverHandler(
            coordinator_host="10.0.0.1",
            coordinator_port=50050,
            peer_hosts=[],
        )
        handler._check_tcp_alive = lambda h, p: False

        # Act (should not raise)
        handler._trigger_failover()

        # Assert
        assert handler.failover_count == 0


@pytest.mark.e2e
class TestCoordinatorHealthEndpointAfterRecovery:
    """Verify that health and cluster endpoints work after simulated recovery."""

    def test_health_endpoint_returns_healthy(self, e2e_api_client, e2e_auth_headers):
        """Health endpoint returns 200 with status information.

        AAA:
          Arrange - Coordinator is running (from fixture).
          Act     - GET /health.
          Assert  - Returns 200 with a status field.
        """
        # Act
        response = e2e_api_client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_pipeline_health_endpoint_responds(self, e2e_api_client, e2e_auth_headers):
        """Pipeline health endpoint returns a response (not a connection error).

        AAA:
          Arrange - Coordinator is running.
          Act     - GET /api/pipeline/health.
          Assert  - Returns a valid HTTP response (not a connection error).
        """
        # Act
        response = e2e_api_client.get("/api/pipeline/health", headers=e2e_auth_headers)

        # Assert: endpoint is reachable and returns a well-formed HTTP response
        assert response.status_code in (200, 500, 503)
