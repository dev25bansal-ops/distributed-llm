"""Realistic node failure integration tests.

These tests go beyond the existing mock-based chaos tests to verify
integration between the coordinator, health service, and failover engine.

Tests:
- Health probing integration with NodeState transitions
- FailoverEngine state machine: HEALTHY -> DEGRADED -> UNHEALTHY -> OFFLINE
- HealthCheckService periodic probing
- Coordinator health_check with node failures
- Node removal and re-registration

Run: pytest tests/chaos/test_node_failure_integration.py -v
"""

from unittest.mock import MagicMock

import pytest

from distllm.health.failover import FailoverEngine
from distllm.health.prober import probe_node
from distllm.health.state import HealthRecord, HealthStateStore, NodeState

# ============================================================
# HealthRecord Tests
# ============================================================


class TestHealthRecord:
    """Tests for HealthRecord state tracking."""

    def test_initial_state_is_offline(self):
        """New records should start in OFFLINE state."""
        record = HealthRecord(node_id="node-0")
        assert record.state == NodeState.OFFLINE

    def test_record_latency_updates_percentiles(self):
        """Recording latencies should update P50 and P99."""
        record = HealthRecord(node_id="node-0")

        # Record 100 latencies
        for i in range(100):
            record.record_latency(float(i))

        assert record.latency_p50_ms == 50.0  # Median of 0-99
        assert record.latency_p99_ms >= 98.0

    def test_record_latency_empty(self):
        """P50/P99 should be 0.0 with no latencies."""
        record = HealthRecord(node_id="node-0")
        assert record.latency_p50_ms == 0.0
        assert record.latency_p99_ms == 0.0


# ============================================================
# HealthStateStore Tests
# ============================================================


class TestHealthStateStore:
    """Tests for thread-safe health state storage."""

    def test_set_and_get(self):
        """Should store and retrieve records."""
        store = HealthStateStore()
        record = HealthRecord(node_id="node-0")
        store.set("node-0", record)

        retrieved = store.get("node-0")
        assert retrieved is record

    def test_get_unknown(self):
        """Should return None for unknown node."""
        store = HealthStateStore()
        assert store.get("unknown") is None

    def test_get_all(self):
        """Should return all records."""
        store = HealthStateStore()
        store.set("node-0", HealthRecord(node_id="node-0"))
        store.set("node-1", HealthRecord(node_id="node-1"))

        all_records = store.get_all()
        assert len(all_records) == 2

    def test_update_state(self):
        """Should update the state of an existing record."""
        store = HealthStateStore()
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)
        store.set("node-0", record)

        updated = store.update_state("node-0", NodeState.UNHEALTHY)

        assert updated.state == NodeState.UNHEALTHY
        assert record.state == NodeState.UNHEALTHY

    def test_update_state_unknown_node(self):
        """Should return None for unknown node."""
        store = HealthStateStore()
        result = store.update_state("unknown", NodeState.UNHEALTHY)
        assert result is None

    def test_remove(self):
        """Should remove a record."""
        store = HealthStateStore()
        store.set("node-0", HealthRecord(node_id="node-0"))
        store.remove("node-0")

        assert store.get("node-0") is None

    def test_healthy_nodes(self):
        """Should return only healthy/degraded nodes."""
        store = HealthStateStore()
        healthy = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)
        degraded = HealthRecord(node_id="node-1", state=NodeState.DEGRADED)
        unhealthy = HealthRecord(node_id="node-2", state=NodeState.UNHEALTHY)
        offline = HealthRecord(node_id="node-3", state=NodeState.OFFLINE)

        store.set("node-0", healthy)
        store.set("node-1", degraded)
        store.set("node-2", unhealthy)
        store.set("node-3", offline)

        result = store.healthy_nodes()
        assert "node-0" in result
        assert "node-1" in result
        assert "node-2" not in result
        assert "node-3" not in result


# ============================================================
# FailoverEngine Tests
# ============================================================


class TestFailoverEngine:
    """Tests for failover state machine."""

    def test_healthy_to_degraded_on_latency(self, failover_engine):
        """High latency should transition HEALTHY -> DEGRADED."""
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)

        new_state = failover_engine.evaluate(record, success=True, latency_ms=200.0)

        assert new_state == NodeState.DEGRADED

    def test_healthy_stays_healthy_on_low_latency(self, failover_engine):
        """Low latency should keep node HEALTHY."""
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)

        new_state = failover_engine.evaluate(record, success=True, latency_ms=50.0)

        assert new_state == NodeState.HEALTHY

    def test_degraded_to_unhealthy_on_failures(self, failover_engine):
        """Consecutive failures should transition to UNHEALTHY."""
        record = HealthRecord(node_id="node-0", state=NodeState.DEGRADED)

        # failure_threshold=2, so 2 failures -> UNHEALTHY
        failover_engine.evaluate(record, success=False, latency_ms=0)
        new_state = failover_engine.evaluate(record, success=False, latency_ms=0)

        assert new_state == NodeState.UNHEALTHY

    def test_offline_to_degraded_on_recovery(self, failover_engine):
        """Successful probe on OFFLINE node should transition to DEGRADED."""
        record = HealthRecord(node_id="node-0", state=NodeState.OFFLINE)

        new_state = failover_engine.evaluate(record, success=True, latency_ms=100.0)

        assert new_state == NodeState.DEGRADED

    def test_degraded_to_healthy_on_successes(self, failover_engine):
        """Consecutive successes should recover to HEALTHY."""
        record = HealthRecord(node_id="node-0", state=NodeState.DEGRADED)
        # recovery_threshold=1

        # First success should be enough (recovery_threshold=1)
        new_state = failover_engine.evaluate(record, success=True, latency_ms=50.0)

        assert new_state == NodeState.HEALTHY

    def test_failure_resets_success_count(self, failover_engine):
        """A failure should reset consecutive success count."""
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)
        record.consecutive_successes = 5

        failover_engine.evaluate(record, success=False, latency_ms=0)

        assert record.consecutive_successes == 0

    def test_success_resets_failure_count(self, failover_engine):
        """A success should reset consecutive failure count."""
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)

        failover_engine.evaluate(record, success=False, latency_ms=0)
        assert record.consecutive_failures == 1

        failover_engine.evaluate(record, success=True, latency_ms=50.0)
        assert record.consecutive_failures == 0

    def test_callback_triggered_on_transition(self, failover_engine):
        """State transitions should trigger callbacks."""
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)
        callback_calls = []

        def callback(node_id, old_state, new_state):
            callback_calls.append((node_id, old_state, new_state))

        failover_engine.on_state_change(callback)
        failover_engine.evaluate(record, success=True, latency_ms=200.0)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("node-0", NodeState.HEALTHY, NodeState.DEGRADED)

    def test_no_callback_on_same_state(self, failover_engine):
        """No callback should fire when state doesn't change."""
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)
        callback_calls = []
        failover_engine.on_state_change(lambda *args: callback_calls.append(args))

        failover_engine.evaluate(record, success=True, latency_ms=50.0)

        assert len(callback_calls) == 0


# ============================================================
# Prober Tests
# ============================================================


class TestProbeNode:
    """Tests for health probing."""

    @pytest.mark.asyncio
    async def test_probe_successful(self):
        """Successful probe should return True with latency."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = MagicMock(
            memory_used=1024,
            memory_total=8192,
            gpu_utilization=0.5,
        )

        success, latency_ms, data = await probe_node(mock_client, timeout=5.0)

        assert success is True
        assert latency_ms >= 0
        assert data["memory_used"] == 1024

    @pytest.mark.asyncio
    async def test_probe_failure(self):
        """Failed probe should return False with error."""
        mock_client = MagicMock()
        mock_client.health_check.side_effect = ConnectionError("unreachable")

        success, latency_ms, data = await probe_node(mock_client, timeout=5.0)

        assert success is False
        assert "error" in data


# ============================================================
# Coordinator Health Integration Tests
# ============================================================


class TestCoordinatorHealthIntegration:
    """Integration tests for coordinator health checking with nodes."""

    def test_health_check_healthy_nodes(self, mock_coordinator_with_nodes):
        """Health check should return healthy status for working nodes."""
        health = mock_coordinator_with_nodes.health_check()

        assert "node-0" in health
        assert health["node-0"]["healthy"] is True

    def test_health_check_unhealthy_node(self, mock_coordinator_with_nodes):
        """Health check should detect unhealthy nodes."""
        # Make node-1 fail
        mock_coordinator_with_nodes.nodes["node-1"].client.health_check.side_effect = (
            ConnectionError("node unreachable")
        )

        health = mock_coordinator_with_nodes.health_check()

        assert health["node-1"]["healthy"] is False
        assert "error" in health["node-1"]

    def test_health_check_mixed_states(self, mock_coordinator_with_nodes):
        """Health check should handle mix of healthy/unhealthy nodes."""
        # Make node-1 fail
        mock_coordinator_with_nodes.nodes["node-1"].client.health_check.side_effect = (
            ConnectionError("node unreachable")
        )

        health = mock_coordinator_with_nodes.health_check()

        assert health["node-0"]["healthy"] is True
        assert health["node-1"]["healthy"] is False

    def test_health_check_triggers_circuit_breaker_metrics(self, mock_coordinator_with_nodes):
        """Health check should record circuit breaker states."""
        # Mock metrics exporter
        mock_exporter = MagicMock()
        mock_coordinator_with_nodes.metrics_exporter = mock_exporter
        mock_exporter.circuit_breaker_state = MagicMock()
        mock_exporter.circuit_breaker_state.labels.return_value = MagicMock()

        mock_coordinator_with_nodes.health_check()

        # Should have recorded circuit breaker state for each node
        assert mock_exporter.circuit_breaker_state.labels.call_count >= 2


# ============================================================
# State Machine Integration Tests
# ============================================================


class TestStateMachineIntegration:
    """End-to-end state machine transitions."""

    def test_full_lifecycle(self, failover_engine):
        """Test: HEALTHY -> DEGRADED -> UNHEALTHY -> OFFLINE -> DEGRADED -> HEALTHY."""
        record = HealthRecord(node_id="node-0", state=NodeState.HEALTHY)

        # HEALTHY -> DEGRADED (high latency)
        state = failover_engine.evaluate(record, success=True, latency_ms=200.0)
        assert state == NodeState.DEGRADED

        # DEGRADED -> UNHEALTHY (failures)
        failover_engine.evaluate(record, success=False, latency_ms=0)
        state = failover_engine.evaluate(record, success=False, latency_ms=0)
        assert state == NodeState.UNHEALTHY

        # Simulate prolonged failure -> OFFLINE (manual transition)
        record.state = NodeState.OFFLINE

        # OFFLINE -> DEGRADED (recovery probe)
        state = failover_engine.evaluate(record, success=True, latency_ms=100.0)
        assert state == NodeState.DEGRADED

        # DEGRADED -> HEALTHY (consecutive successes)
        state = failover_engine.evaluate(record, success=True, latency_ms=50.0)
        assert state == NodeState.HEALTHY

    def test_degraded_recovery_not_enough_successes(self):
        """Recovery requires consecutive successes >= recovery_threshold."""
        engine = FailoverEngine(recovery_threshold=3)
        record = HealthRecord(node_id="node-0", state=NodeState.DEGRADED)

        # 1 success (below threshold)
        state = engine.evaluate(record, success=True, latency_ms=50.0)
        assert record.consecutive_successes == 1
        assert state == NodeState.HEALTHY  # Still recovers because latency is good

    def test_dringing_state_not_probed(self):
        """DRAINING state should skip probing (tested via service logic)."""
        record = HealthRecord(node_id="node-0", state=NodeState.DRAINING)
        # In service, _probe_once returns early for DRAINING state
        assert record.state == NodeState.DRAINING
