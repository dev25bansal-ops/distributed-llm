"""Tests for ResourceManager: node lifecycle, health checks, circuit breaking.

Tests: CircuitBreakerConfig, NodeRegistration, circuit breaker core logic,
exponential backoff, health checks (sync/async), connection management,
metrics, chaos engineering, and edge cases.

Run: pytest tests/core/test_resource_manager.py -v
"""

import time
from unittest.mock import MagicMock, AsyncMock

import pytest

from distllm.core.resource_manager import (
    ResourceManager,
    NodeRegistration,
    CircuitBreakerConfig,
)
from distllm.config.loader import NodeRole
from distllm.errors.types import NodeUnreachableError, GRPCTimeoutError


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        config = CircuitBreakerConfig()
        assert config.threshold == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0

    def test_custom_values(self):
        config = CircuitBreakerConfig(threshold=5, base_delay=2.0, max_delay=120.0)
        assert config.threshold == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0


class TestNodeRegistration:
    """Tests for NodeRegistration creation."""

    def test_basic_registration(self):
        reg = NodeRegistration(
            node_id="node-0", host="localhost", port=50051,
            start_layer=0, end_layer=5,
        )
        assert reg.node_id == "node-0"
        assert reg.host == "localhost"
        assert reg.port == 50051
        assert reg.start_layer == 0
        assert reg.end_layer == 5
        assert reg.healthy is True
        assert reg.role == NodeRole.AUTO
        assert reg.expert_ids == []
        assert reg.cluster_id == "default"
        assert reg.version == "stable"

    def test_clients_created(self):
        reg = NodeRegistration(
            node_id="node-0", host="localhost", port=50051,
            start_layer=0, end_layer=5,
        )
        assert reg.client is not None
        assert reg.async_client is not None

    def test_custom_role_and_experts(self):
        reg = NodeRegistration(
            node_id="node-0", host="localhost", port=50051,
            start_layer=0, end_layer=5,
            role=NodeRole.PREFILL, expert_ids=[1, 3, 5],
            cluster_id="us-east",
        )
        assert reg.role == NodeRole.PREFILL
        assert reg.expert_ids == [1, 3, 5]
        assert reg.cluster_id == "us-east"

    def test_spot_instance_properties(self):
        reg = NodeRegistration(
            node_id="spot-1", host="10.0.0.1", port=50051,
            start_layer=0, end_layer=5,
            instance_type="g5.xlarge", cost_per_hour=1.5, is_spot=True,
        )
        assert reg.instance_type == "g5.xlarge"
        assert reg.cost_per_hour == 1.5
        assert reg.is_spot is True


class TestResourceManagerInit:
    """Tests for ResourceManager initialization."""

    def test_default_config(self):
        rm = ResourceManager()
        assert rm.cb_config.threshold == 3
        assert rm.cb_config.base_delay == 1.0
        assert rm.cb_config.max_delay == 60.0
        assert rm._node_failure_counts == {}

    def test_custom_config(self):
        config = CircuitBreakerConfig(threshold=10, base_delay=5.0, max_delay=300.0)
        rm = ResourceManager(cb_config=config)
        assert rm.cb_config.threshold == 10
        assert rm.cb_config.base_delay == 5.0
        assert rm.cb_config.max_delay == 300.0

    def test_initial_metrics(self):
        rm = ResourceManager()
        assert rm._metrics["node_failures"] == 0
        assert rm._metrics["errors"] == 0

    def test_thread_lock(self):
        rm = ResourceManager()
        assert rm._lock is not None


class TestCircuitBreakerCore:
    """Tests for core circuit breaker logic."""

    def test_check_no_failures(self):
        rm = ResourceManager()
        assert rm.check_circuit_breaker("node-0") is False

    def test_check_unknown_node(self):
        rm = ResourceManager()
        assert rm.check_circuit_breaker("nonexistent") is False

    def test_record_success_clears_failures(self):
        rm = ResourceManager()
        rm._node_failure_counts["node-0"] = 2
        rm._node_recovery_time["node-0"] = time.time() + 100
        rm.record_success("node-0")
        assert rm._node_failure_counts["node-0"] == 0
        assert "node-0" not in rm._node_recovery_time

    def test_success_on_unknown_node(self):
        rm = ResourceManager()
        rm.record_success("new-node")
        assert rm._node_failure_counts.get("new-node") == 0

    def test_record_failure_increments(self):
        rm = ResourceManager()
        rm.record_failure("node-0")
        assert rm._node_failure_counts["node-0"] == 1

    def test_record_failure_accumulates(self):
        rm = ResourceManager()
        for _ in range(5):
            rm.record_failure("node-0")
        assert rm._node_failure_counts["node-0"] == 5

    def test_failure_increments_metrics(self):
        rm = ResourceManager()
        rm.record_failure("node-0")
        assert rm._metrics["node_failures"] == 1
        assert rm._metrics["errors"] == 1


class TestCircuitBreakerThreshold:
    """Tests for failure threshold behavior."""

    def test_below_threshold_not_open(self):
        rm = ResourceManager()
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is False

    def test_at_threshold_opens(self):
        rm = ResourceManager()
        for _ in range(3):
            rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True

    def test_above_threshold_stays_open(self):
        rm = ResourceManager()
        for _ in range(5):
            rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True

    def test_gradual_opening(self):
        rm = ResourceManager()
        for i in range(5):
            rm.record_failure("node-0")
            if i < 2:
                assert rm.check_circuit_breaker("node-0") is False
            else:
                assert rm.check_circuit_breaker("node-0") is True

    def test_custom_threshold(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=5))
        for _ in range(4):
            rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is False
        rm.record_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True


class TestCircuitBreakerBackoff:
    """Tests for exponential backoff."""

    def test_first_backoff_is_base_delay(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=3, base_delay=2.0))
        before = time.time()
        for _ in range(3):
            rm.record_failure("node-0")
        after = time.time()
        rec = rm._node_recovery_time["node-0"]
        assert before + 2.0 <= rec <= after + 2.0 + 0.1

    def test_backoff_doubles_each_failure(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=3, base_delay=1.0))
        for _ in range(4):
            rm.record_failure("node-0")
        rec = rm._node_recovery_time["node-0"]
        assert time.time() + 2.0 - 0.5 <= rec <= time.time() + 2.0 + 0.5

    def test_backoff_capped_at_max(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(max_delay=30.0))
        for _ in range(10):
            rm.record_failure("node-0")
        assert rm._node_recovery_time["node-0"] <= time.time() + 30.0 + 1.0

    def test_cooldown_elapsed_allows_retry(self):
        rm = ResourceManager()
        for _ in range(3):
            rm.record_failure("node-0")
        rm._node_recovery_time["node-0"] = time.time() - 10
        assert rm.check_circuit_breaker("node-0") is False

    def test_multiple_nodes_independent_backoff(self):
        rm = ResourceManager()
        for _ in range(3):
            rm.record_failure("node-a")
        rm.record_failure("node-b")
        assert rm.check_circuit_breaker("node-a") is True
        assert rm.check_circuit_breaker("node-b") is False

class TestHealthCheckSync:
    """Tests for synchronous health checks."""

    def test_all_healthy(self):
        rm = ResourceManager()
        mock_nodes = {}
        for i in range(2):
            node = MagicMock()
            node.client.health_check.return_value = MagicMock(
                healthy=True, memory_used=2048, memory_total=8192
            )
            mock_nodes[f"healthy-{i}"] = node
        results = rm.health_check_all(mock_nodes)
        assert len(results) == 2
        for result in results.values():
            assert result["healthy"] is True

    def test_unhealthy_node(self):
        rm = ResourceManager()
        node = MagicMock()
        node.client.health_check.side_effect = NodeUnreachableError(
            node_id="bad-node", host="localhost", port=50051
        )
        results = rm.health_check_all({"bad-node": node})
        assert results["bad-node"]["healthy"] is False
        assert "error" in results["bad-node"]

    def test_grpc_timeout(self):
        rm = ResourceManager()
        node = MagicMock()
        node.client.health_check.side_effect = GRPCTimeoutError(
            node_id="timeout-node", host="localhost", port=50051, timeout=5.0
        )
        results = rm.health_check_all({"timeout-node": node})
        assert results["timeout-node"]["healthy"] is False

    def test_connection_error(self):
        rm = ResourceManager()
        node = MagicMock()
        node.client.health_check.side_effect = ConnectionError("refused")
        results = rm.health_check_all({"bad-node": node})
        assert results["bad-node"]["healthy"] is False

    def test_circuit_breaker_integration(self):
        rm = ResourceManager()
        node = MagicMock()
        for _ in range(3):
            rm.record_failure("cb-node")
        results = rm.health_check_all({"cb-node": node})
        assert results["cb-node"]["healthy"] is False
        assert "Circuit breaker" in results["cb-node"]["error"]
        node.client.health_check.assert_not_called()

    def test_empty_nodes(self):
        rm = ResourceManager()
        assert rm.health_check_all({}) == {}


class TestHealthCheckAsync:
    """Tests for asynchronous health checks."""

    @pytest.mark.asyncio
    async def test_all_healthy_async(self):
        rm = ResourceManager()
        mock_nodes = {}
        for i in range(2):
            node = MagicMock()
            node.async_client.health_check = AsyncMock(return_value=MagicMock(
                healthy=True, memory_used=2048, memory_total=8192
            ))
            mock_nodes[f"healthy-{i}"] = node
        results = await rm.health_check_all_async(mock_nodes)
        assert len(results) == 2
        for result in results.values():
            assert result["healthy"] is True

    @pytest.mark.asyncio
    async def test_unhealthy_node_async(self):
        rm = ResourceManager()
        node = MagicMock()
        node.async_client.health_check = AsyncMock(
            side_effect=NodeUnreachableError(
                node_id="bad-node", host="localhost", port=50051
            )
        )
        results = await rm.health_check_all_async({"bad-node": node})
        assert results["bad-node"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_async(self):
        rm = ResourceManager()
        node = MagicMock()
        for _ in range(3):
            rm.record_failure("cb-node")
        results = await rm.health_check_all_async({"cb-node": node})
        assert results["cb-node"]["healthy"] is False
        assert "Circuit breaker" in results["cb-node"]["error"]

    @pytest.mark.asyncio
    async def test_exception_handling_async(self):
        rm = ResourceManager()
        node = MagicMock()
        node.async_client.health_check = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        results = await rm.health_check_all_async({"bad-node": node})
        assert results["bad-node"]["healthy"] is False



class TestConnectionManagement:
    """Tests for node connection lifecycle."""

    def test_close_all(self):
        rm = ResourceManager()
        mock_nodes = {f"node-{i}": MagicMock() for i in range(3)}
        rm.close_all(mock_nodes)
        for node in mock_nodes.values():
            node.close.assert_called_once()

    def test_close_all_handles_errors(self):
        rm = ResourceManager()
        node = MagicMock()
        node.close.side_effect = Exception("Close error")
        rm.close_all({"bad-node": node})
        node.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all_async(self):
        rm = ResourceManager()
        mock_nodes = {f"node-{i}": MagicMock() for i in range(3)}
        for node in mock_nodes.values():
            node.async_client.close = AsyncMock()
        await rm.close_all_async(mock_nodes)
        for node in mock_nodes.values():
            node.async_client.close.assert_called_once()
            node.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all_async_handles_errors(self):
        rm = ResourceManager()
        node = MagicMock()
        node.async_client.close = AsyncMock(side_effect=Exception("err"))
        await rm.close_all_async({"bad-node": node})
        node.async_client.close.assert_called_once()
        node.close.assert_called_once()


class TestResourceManagerMetrics:
    """Tests for ResourceManager metrics tracking."""

    def test_get_metrics(self):
        rm = ResourceManager()
        rm._metrics["node_failures"] = 5
        metrics = rm.get_metrics()
        assert metrics["node_failures"] == 5

    def test_increment_metric_existing(self):
        rm = ResourceManager()
        rm.increment_metric("node_failures", 3)
        assert rm._metrics["node_failures"] == 3
        rm.increment_metric("node_failures", 2)
        assert rm._metrics["node_failures"] == 5

    def test_increment_metric_new(self):
        rm = ResourceManager()
        rm.increment_metric("custom", 10)
        assert rm._metrics["custom"] == 10

    def test_increment_default_value(self):
        rm = ResourceManager()
        rm.increment_metric("node_failures")
        assert rm._metrics["node_failures"] == 1

class TestChaosEngineering:
    """Tests for chaos engineering features."""

    def test_simulate_node_failure(self):
        rm = ResourceManager()
        before = time.time()
        rm.simulate_node_failure("node-0")
        assert rm._node_failure_counts["node-0"] == rm.cb_config.threshold
        assert rm.check_circuit_breaker("node-0") is True
        rec = rm._node_recovery_time["node-0"]
        assert before + 3599 < rec < before + 3601

    def test_simulate_node_failure_multiple_nodes(self):
        rm = ResourceManager()
        rm.simulate_node_failure("node-a")
        rm.simulate_node_failure("node-b")
        assert rm.check_circuit_breaker("node-a") is True
        assert rm.check_circuit_breaker("node-b") is True

    def test_simulate_then_success_resets(self):
        rm = ResourceManager()
        rm.simulate_node_failure("node-0")
        assert rm.check_circuit_breaker("node-0") is True
        rm.record_success("node-0")
        assert rm.check_circuit_breaker("node-0") is False
        assert rm._node_failure_counts["node-0"] == 0


class TestResourceManagerEdgeCases:
    """Tests for edge cases."""

    def test_rapid_failures_overflow(self):
        rm = ResourceManager()
        for _ in range(100):
            rm.record_failure("node-0")
        assert rm._node_failure_counts["node-0"] == 100
        assert rm.check_circuit_breaker("node-0") is True

    def test_failure_then_success_cycle(self):
        rm = ResourceManager()
        for _ in range(3):
            for _ in range(2):
                rm.record_failure("node-0")
            assert rm.check_circuit_breaker("node-0") is False
            rm.record_success("node-0")
            assert rm._node_failure_counts["node-0"] == 0

    def test_failure_count_persists_after_cooldown(self):
        rm = ResourceManager()
        for _ in range(5):
            rm.record_failure("node-0")
        assert rm._node_failure_counts["node-0"] == 5
        assert rm.check_circuit_breaker("node-0") is True
        rm._node_recovery_time["node-0"] = time.time() - 10
        assert rm.check_circuit_breaker("node-0") is False

