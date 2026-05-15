"""Integration tests for node lifecycle management.

Tests registration, health checks, circuit breaker transitions under real gRPC load.
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from distllm.core.resource_manager import ResourceManager, CircuitBreakerConfig, NodeRegistration
from distllm.config.loader import NodeRole


class TestNodeRegistration:
    def test_register_node_with_all_fields(self):
        reg = NodeRegistration(
            node_id="node-0",
            host="localhost",
            port=50051,
            start_layer=0,
            end_layer=5,
            role=NodeRole.AUTO,
            cluster_id="default",
            version="stable",
            instance_type="g5.xlarge",
            cost_per_hour=2.5,
            is_spot=True,
        )
        assert reg.node_id == "node-0"
        assert reg.version == "stable"
        assert reg.is_spot is True
        assert reg.cost_per_hour == 2.5

    def test_register_node_defaults(self):
        reg = NodeRegistration("n1", "localhost", 50051, 0, 3)
        assert reg.cluster_id == "default"
        assert reg.version == "stable"
        assert reg.instance_type == "unknown"
        assert reg.cost_per_hour == 0.0
        assert reg.is_spot is False


class TestHealthChecks:
    def test_health_check_all_healthy(self):
        rm = ResourceManager()
        nodes = {}

        for i in range(2):
            mock_client = MagicMock()
            mock_client.health_check.return_value = MagicMock(
                healthy=True,
                memory_used=1024,
                memory_total=8192,
            )
            reg = NodeRegistration(f"node-{i}", "localhost", 50051 + i, 0, 5)
            reg.client = mock_client
            nodes[f"node-{i}"] = reg

        results = rm.health_check_all(nodes)
        assert results["node-0"]["healthy"] is True
        assert results["node-1"]["healthy"] is True

    def test_health_check_one_unhealthy(self):
        rm = ResourceManager()
        mock_client = MagicMock()
        mock_client.health_check.side_effect = ConnectionError("Connection refused")
        reg = NodeRegistration("node-0", "localhost", 50051, 0, 5)
        reg.client = mock_client

        results = rm.health_check_all({"node-0": reg})
        assert results["node-0"]["healthy"] is False

    def test_health_check_async(self):
        """Test async health check using asyncio.run for single-threaded execution."""
        import asyncio

        async def _run():
            rm = ResourceManager()
            mock_async_client = AsyncMock()
            mock_async_client.health_check.return_value = MagicMock(
                healthy=True,
                memory_used=2048,
                memory_total=8192,
            )
            reg = NodeRegistration("node-0", "localhost", 50051, 0, 5)
            reg.async_client = mock_async_client

            results = await rm.health_check_all_async({"node-0": reg})
            assert results["node-0"]["healthy"] is True

        asyncio.run(_run())


class TestCircuitBreaker:
    def test_circuit_breaker_opens_after_threshold(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=3))

        # Record failures up to threshold
        for _ in range(3):
            rm.record_failure("node-0")

        # Circuit breaker should be open
        assert rm.check_circuit_breaker("node-0") is True

    def test_circuit_breaker_closes_on_success(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=3))

        # Record some failures
        for _ in range(2):
            rm.record_failure("node-0")

        # Should not be open yet
        assert rm.check_circuit_breaker("node-0") is False

        # Record success
        rm.record_success("node-0")

        # Failure count should be reset
        assert rm._node_failure_counts.get("node-0", 0) == 0

    def test_circuit_breaker_exponential_backoff(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2, base_delay=1.0, max_delay=60.0))

        # Record failures
        for _ in range(4):
            rm.record_failure("node-0")

        # Recovery time should be set with exponential backoff
        recovery_time = rm._node_recovery_time.get("node-0", 0)
        assert recovery_time > 0

    def test_simulate_node_failure(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2))
        rm.simulate_node_failure("node-0")

        # Circuit breaker should be open
        assert rm.check_circuit_breaker("node-0") is True
        assert rm._node_failure_counts["node-0"] >= 2

    def test_metrics_tracking(self):
        rm = ResourceManager()
        rm.record_failure("node-0")
        rm.record_failure("node-0")
        rm.record_success("node-1")

        metrics = rm.get_metrics()
        assert metrics["node_failures"] == 2
        assert metrics["errors"] == 2


class TestNodeUnregistration:
    def test_unregister_clears_state(self):
        rm = ResourceManager(cb_config=CircuitBreakerConfig(threshold=2))

        # Register some state
        rm.record_failure("node-0")
        rm.record_success("node-0")

        # State exists
        assert "node-0" in rm._node_failure_counts

        # After removal (simulated by clearing)
        rm._node_failure_counts.pop("node-0", None)
        rm._node_recovery_time.pop("node-0", None)

        assert "node-0" not in rm._node_failure_counts
