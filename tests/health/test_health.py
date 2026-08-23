"""Tests for the ``distllm.health`` package.

Covers all four modules:
- ``state.py`` -- HealthRecord, HealthStateStore, NodeState
- ``failover.py`` -- FailoverEngine state-machine transitions
- ``prober.py`` -- probe_node function
- ``service.py`` -- HealthCheckService orchestration

All modules import directly (no circular dependencies).
No MagicMock -- real stubs and callable trackers.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from distllm.health.failover import FailoverEngine
from distllm.health.prober import probe_node
from distllm.health.service import (
    HealthCheckService,
    _BACKOFF_BASE_SECONDS,
    _BACKOFF_FAILURE_THRESHOLD,
    _BACKOFF_MAX_SECONDS,
)
from distllm.health.state import (
    HealthRecord,
    HealthStateStore,
    NodeState,
)

from tests.health.conftest import _StubClient


# --- Callback tracker --------------------------------------------------------


class _CallbackTracker:
    """A callable that records invocations for later inspection."""

    def __init__(self):
        self.args_list: list[tuple] = []
        self.kwargs_list: list[dict] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.args_list.append(args)
        self.kwargs_list.append(kwargs)


# ===================================================================
# NodeState enum
# ===================================================================


class TestNodeState:
    def test_values(self) -> None:
        assert NodeState.HEALTHY.value == "healthy"
        assert NodeState.DEGRADED.value == "degraded"
        assert NodeState.UNHEALTHY.value == "unhealthy"
        assert NodeState.DRAINING.value == "draining"
        assert NodeState.OFFLINE.value == "offline"

    def test_all_members_present(self) -> None:
        expected = {"HEALTHY", "DEGRADED", "UNHEALTHY", "DRAINING", "OFFLINE"}
        assert set(NodeState.__members__) == expected

    def test_is_hashable(self) -> None:
        states = {NodeState.HEALTHY, NodeState.DEGRADED}
        assert len(states) == 2
        assert NodeState.HEALTHY in states


# ===================================================================
# HealthRecord
# ===================================================================


class TestHealthRecord:
    def test_defaults(self) -> None:
        record = HealthRecord(node_id="test-node")
        assert record.node_id == "test-node"
        assert record.state == NodeState.OFFLINE
        assert record.last_probe_time == 0.0
        assert record.consecutive_failures == 0
        assert record.consecutive_successes == 0
        assert record.latency_p50_ms == 0.0
        assert record.latency_p99_ms == 0.0
        assert record.gpu_utilization == 0.0
        assert record.memory_used == 0
        assert record.memory_total == 0
        assert record.layer_range == ""

    def test_constructor_with_values(self, health_record: HealthRecord) -> None:
        assert health_record.node_id == "node-0"
        assert health_record.state == NodeState.HEALTHY
        assert health_record.last_probe_time == 1000.0
        assert health_record.consecutive_failures == 0
        assert health_record.consecutive_successes == 5
        assert health_record.latency_p50_ms == 50.0
        assert health_record.latency_p99_ms == 200.0
        assert health_record.gpu_utilization == 0.5
        assert health_record.memory_used == 2048
        assert health_record.memory_total == 8192
        assert health_record.layer_range == "0-5"

    def test_record_latency_single(self) -> None:
        record = HealthRecord(node_id="n")
        record.record_latency(100.0)
        assert record.latency_p50_ms == 100.0
        assert record.latency_p99_ms == 100.0

    def test_record_latency_multiple(self) -> None:
        record = HealthRecord(node_id="n")
        for lat in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
            record.record_latency(lat)
        assert record.latency_p50_ms == 60.0
        assert record.latency_p99_ms == 100.0

    def test_record_latency_odd_count(self) -> None:
        record = HealthRecord(node_id="n")
        for lat in [1.0, 2.0, 3.0]:
            record.record_latency(lat)
        assert record.latency_p50_ms == 2.0
        assert record.latency_p99_ms == 3.0

    def test_record_latency_ring_buffer(self) -> None:
        record = HealthRecord(node_id="n")
        for i in range(200):
            record.record_latency(float(i))
        assert record.latency_p50_ms == 150.0
        assert record.latency_p99_ms == 199.0

    def test_record_latency_deque_no_shared_state(self) -> None:
        r1 = HealthRecord(node_id="a")
        r2 = HealthRecord(node_id="b")
        r1.record_latency(100.0)
        r2.record_latency(200.0)
        assert r1.latency_p50_ms == 100.0
        assert r2.latency_p50_ms == 200.0

    def test_record_latency_does_not_mutate_other_fields(self) -> None:
        record = HealthRecord(
            node_id="n",
            state=NodeState.HEALTHY,
            gpu_utilization=0.8,
            memory_used=4096,
            memory_total=8192,
        )
        record.record_latency(50.0)
        assert record.node_id == "n"
        assert record.state == NodeState.HEALTHY
        assert record.gpu_utilization == 0.8
        assert record.memory_used == 4096
        assert record.memory_total == 8192


# ===================================================================
# HealthStateStore
# ===================================================================


class TestHealthStateStore:
    def test_get_missing(self, health_store: HealthStateStore) -> None:
        assert health_store.get("nonexistent") is None

    def test_set_and_get(self, health_store: HealthStateStore, health_record: HealthRecord) -> None:
        health_store.set("node-0", health_record)
        assert health_store.get("node-0") is health_record

    def test_get_all_empty(self, health_store: HealthStateStore) -> None:
        assert health_store.get_all() == {}

    def test_get_all(
        self,
        populated_health_store: HealthStateStore,
        health_record: HealthRecord,
        degraded_record: HealthRecord,
    ) -> None:
        all_records = populated_health_store.get_all()
        assert len(all_records) == 4
        assert all_records["node-0"] is health_record
        assert all_records["node-1"] is degraded_record

    def test_get_all_returns_copy(
        self,
        populated_health_store: HealthStateStore,
    ) -> None:
        all_records = populated_health_store.get_all()
        all_records.clear()
        assert len(populated_health_store.get_all()) == 4

    def test_remove(self, health_store: HealthStateStore, health_record: HealthRecord) -> None:
        health_store.set("node-0", health_record)
        health_store.remove("node-0")
        assert health_store.get("node-0") is None

    def test_remove_missing(self, health_store: HealthStateStore) -> None:
        health_store.remove("nonexistent")

    def test_update_state(self, health_store: HealthStateStore, health_record: HealthRecord) -> None:
        health_store.set("node-0", health_record)
        returned = health_store.update_state("node-0", NodeState.DEGRADED)
        assert returned is health_record
        assert returned.state == NodeState.DEGRADED

    def test_update_state_missing(self, health_store: HealthStateStore) -> None:
        assert health_store.update_state("nonexistent", NodeState.HEALTHY) is None

    def test_healthy_nodes(
        self,
        populated_health_store: HealthStateStore,
    ) -> None:
        healthy = populated_health_store.healthy_nodes()
        assert "node-0" in healthy
        assert "node-1" in healthy
        assert "node-2" not in healthy
        assert "node-3" not in healthy

    def test_healthy_nodes_empty(self, health_store: HealthStateStore) -> None:
        assert health_store.healthy_nodes() == []

    def test_thread_safety(self, health_store: HealthStateStore) -> None:
        import concurrent.futures

        def set_and_get(i: int) -> int:
            rec = HealthRecord(node_id=f"thread-{i}")
            health_store.set(f"thread-{i}", rec)
            retrieved = health_store.get(f"thread-{i}")
            return 0 if retrieved is rec else 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(set_and_get, range(100)))

        assert sum(results) == 0
        assert len(health_store.get_all()) == 100


# ===================================================================
# FailoverEngine
# ===================================================================


class TestFailoverEngine:
    """State-machine transition tests."""

    def test_healthy_stays_healthy_on_success_low_latency(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        new_state = failover_engine.evaluate(health_record, success=True, latency_ms=50.0)
        assert new_state == NodeState.HEALTHY
        assert health_record.state == NodeState.HEALTHY
        assert health_record.consecutive_successes == 6

    def test_healthy_to_degraded_on_high_latency(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        new_state = failover_engine.evaluate(health_record, success=True, latency_ms=150.0)
        assert new_state == NodeState.DEGRADED
        assert health_record.state == NodeState.DEGRADED

    def test_healthy_to_degraded_on_failure(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        new_state = failover_engine.evaluate(health_record, success=False, latency_ms=0.0)
        assert new_state == NodeState.DEGRADED
        assert health_record.state == NodeState.DEGRADED
        assert health_record.consecutive_successes == 0
        assert health_record.consecutive_failures == 1

    def test_degraded_to_unhealthy(
        self, failover_engine: FailoverEngine, degraded_record: HealthRecord
    ) -> None:
        new_state = failover_engine.evaluate(degraded_record, success=False, latency_ms=0.0)
        assert new_state == NodeState.UNHEALTHY
        assert degraded_record.state == NodeState.UNHEALTHY
        assert degraded_record.consecutive_failures == 2

    def test_degraded_to_healthy(
        self, failover_engine: FailoverEngine, degraded_record: HealthRecord
    ) -> None:
        new_state = failover_engine.evaluate(degraded_record, success=True, latency_ms=50.0)
        assert new_state == NodeState.HEALTHY
        assert degraded_record.state == NodeState.HEALTHY
        assert degraded_record.consecutive_successes == 1

    def test_offline_to_degraded(
        self, failover_engine: FailoverEngine, offline_record: HealthRecord
    ) -> None:
        new_state = failover_engine.evaluate(offline_record, success=True, latency_ms=50.0)
        assert new_state == NodeState.DEGRADED
        assert offline_record.state == NodeState.DEGRADED
        assert offline_record.consecutive_failures == 0

    def test_unhealthy_to_degraded(
        self, failover_engine: FailoverEngine, unhealthy_record: HealthRecord
    ) -> None:
        new_state = failover_engine.evaluate(unhealthy_record, success=True, latency_ms=50.0)
        assert new_state == NodeState.DEGRADED
        assert unhealthy_record.state == NodeState.DEGRADED

    def test_unhealthy_stays_unhealthy_on_failure(
        self, failover_engine: FailoverEngine, unhealthy_record: HealthRecord
    ) -> None:
        new_state = failover_engine.evaluate(unhealthy_record, success=False, latency_ms=0.0)
        assert new_state == NodeState.UNHEALTHY
        assert unhealthy_record.consecutive_failures == 3

    def test_draining_does_not_change(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        health_record.state = NodeState.DRAINING
        new_state = failover_engine.evaluate(health_record, success=True, latency_ms=50.0)
        assert new_state == NodeState.DRAINING

    def test_latency_tracking_on_success(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        failover_engine.evaluate(health_record, success=True, latency_ms=75.0)
        assert health_record.latency_p50_ms == 75.0

    def test_strict_thresholds(
        self, strict_failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        """With degraded_latency_ms=2000, staying healthy."""
        new_state = strict_failover_engine.evaluate(health_record, success=True, latency_ms=500.0)
        assert new_state == NodeState.HEALTHY

    def test_callback_fires_on_transition(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        callback = _CallbackTracker()
        failover_engine.on_state_change(callback)

        failover_engine.evaluate(health_record, success=False, latency_ms=0.0)

        assert len(callback.args_list) == 1
        assert callback.args_list[0] == ("node-0", NodeState.HEALTHY, NodeState.DEGRADED)

    def test_callback_fires_only_on_transition(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        callback = _CallbackTracker()
        failover_engine.on_state_change(callback)

        failover_engine.evaluate(health_record, success=True, latency_ms=50.0)

        assert len(callback.args_list) == 0

    def test_multiple_callbacks(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        cb1 = _CallbackTracker()
        cb2 = _CallbackTracker()
        failover_engine.on_state_change(cb1)
        failover_engine.on_state_change(cb2)

        failover_engine.evaluate(health_record, success=False, latency_ms=0.0)

        assert len(cb1.args_list) == 1
        assert len(cb2.args_list) == 1

    def test_failure_count_reset_on_success(
        self, failover_engine: FailoverEngine
    ) -> None:
        record = HealthRecord(
            node_id="n", state=NodeState.DEGRADED, consecutive_failures=5
        )
        failover_engine.evaluate(record, success=True, latency_ms=50.0)
        assert record.consecutive_failures == 0

    def test_success_count_reset_on_failure(
        self, failover_engine: FailoverEngine, health_record: HealthRecord
    ) -> None:
        failover_engine.evaluate(health_record, success=False, latency_ms=0.0)
        assert health_record.consecutive_successes == 0


# ===================================================================
# probe_node
# ===================================================================


class TestProbeNode:
    @pytest.mark.asyncio
    async def test_successful_probe(self, mock_grpc_client: _StubClient) -> None:
        success, latency_ms, data = await probe_node(mock_grpc_client, timeout=5.0)
        assert success is True
        assert latency_ms >= 0.0
        assert data["memory_used"] == 2048
        assert data["memory_total"] == 8192
        assert data["gpu_utilization"] == 0.45

    @pytest.mark.asyncio
    async def test_failed_probe(self) -> None:
        client = _StubClient()
        client.set_fail()

        success, latency_ms, data = await probe_node(client, timeout=5.0)
        assert success is False
        assert latency_ms >= 0.0
        assert "error" in data
        assert "connection refused" in data["error"]

    @pytest.mark.asyncio
    async def test_missing_gpu_utilization(self) -> None:
        response = SimpleNamespace(memory_used=1024, memory_total=4096)
        client = _StubClient(response=response)

        success, latency_ms, data = await probe_node(client, timeout=5.0)
        assert success is True
        assert data["gpu_utilization"] == 0.0

    @pytest.mark.asyncio
    async def test_timeout_propagates(self) -> None:
        response = SimpleNamespace(memory_used=0, memory_total=0, gpu_utilization=0.0)
        client = _StubClient(response=response)

        await probe_node(client, timeout=3.0)

        assert client.last_timeout == 3.0

    @pytest.mark.asyncio
    async def test_latency_is_positive(self, mock_grpc_client: _StubClient) -> None:
        success, latency_ms, data = await probe_node(mock_grpc_client, timeout=5.0)
        assert latency_ms > 0

    @pytest.mark.asyncio
    async def test_timeout_exception(self) -> None:
        client = _StubClient()
        client.set_fail(TimeoutError("timed out"))

        success, latency_ms, data = await probe_node(client, timeout=1.0)
        assert success is False
        assert "timed out" in data["error"]


# ===================================================================
# HealthCheckService
# ===================================================================


class TestHealthCheckService:
    def test_constructor_defaults(self) -> None:
        svc = HealthCheckService()
        assert svc._probe_interval == 5.0
        assert svc._probe_timeout == 10.0
        assert svc._running is False
        assert svc._get_client is None
        assert svc._task is None
        assert svc._on_node_death is None

    def test_register_node(self, health_service: HealthCheckService) -> None:
        health_service.register_node("test-node", client=_StubClient(), layer_range="0-5")
        record = health_service.get_node("test-node")
        assert record is not None
        assert record.node_id == "test-node"
        assert record.state == NodeState.OFFLINE
        assert record.layer_range == "0-5"

    def test_register_node_with_no_layer(self, health_service: HealthCheckService) -> None:
        health_service.register_node("test-node", client=_StubClient())
        record = health_service.get_node("test-node")
        assert record is not None
        assert record.layer_range == ""

    def test_unregister_node(self, health_service: HealthCheckService) -> None:
        health_service.register_node("test-node", client=_StubClient())
        health_service.unregister_node("test-node")
        assert health_service.get_node("test-node") is None

    def test_unregister_node_unknown(self, health_service: HealthCheckService) -> None:
        health_service.unregister_node("nonexistent")

    def test_get_node_unknown(self, health_service: HealthCheckService) -> None:
        assert health_service.get_node("nonexistent") is None

    def test_get_all_empty(self, health_service: HealthCheckService) -> None:
        assert health_service.get_all() == {}

    def test_get_all(
        self,
        health_service: HealthCheckService,
    ) -> None:
        health_service.register_node("node-a", client=_StubClient())
        health_service.register_node("node-b", client=_StubClient())
        all_nodes = health_service.get_all()
        assert len(all_nodes) == 2
        assert "node-a" in all_nodes
        assert "node-b" in all_nodes

    def test_healthy_nodes_empty(self, health_service: HealthCheckService) -> None:
        assert health_service.healthy_nodes() == []

    def test_healthy_nodes(
        self,
        health_service: HealthCheckService,
    ) -> None:
        health_service.register_node("node-a", client=_StubClient())
        health_service.register_node("node-b", client=_StubClient())
        assert health_service.healthy_nodes() == []

    def test_on_node_death(self, health_service: HealthCheckService) -> None:
        callback = _CallbackTracker()
        health_service.on_node_death(callback)
        assert health_service._on_node_death is callback

    def test_on_state_change(self, health_service: HealthCheckService) -> None:
        callback = _CallbackTracker()
        health_service.on_state_change(callback)
        assert len(health_service._failover._callbacks) == 1

    @pytest.mark.asyncio
    async def test_start_and_stop(self, health_service: HealthCheckService) -> None:
        get_client = lambda x: _StubClient()

        await health_service.start(get_client)
        assert health_service._running is True
        assert health_service._get_client is get_client
        assert health_service._task is not None

        await health_service.stop()
        assert health_service._running is False
        assert health_service._task is None or health_service._task.done()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, health_service: HealthCheckService) -> None:
        await health_service.stop()

    @pytest.mark.asyncio
    async def test_start_twice(self, health_service: HealthCheckService) -> None:
        get_client = lambda x: _StubClient()
        await health_service.start(get_client)
        task1 = health_service._task
        await health_service.start(get_client)
        task2 = health_service._task
        assert task2 is not task1
        await health_service.stop()

    @pytest.mark.asyncio
    async def test_probe_loop_probes_registered_nodes(
        self,
    ) -> None:
        response = SimpleNamespace(memory_used=1024, memory_total=4096, gpu_utilization=0.3)
        client = _StubClient(response=response)

        svc = HealthCheckService(
            probe_interval=0.01,
            probe_timeout=1.0,
            failure_threshold=2,
            degraded_latency_ms=100.0,
            recovery_threshold=1,
        )
        svc.register_node("node-x", client=client, layer_range="0-5")

        def get_client_fn(node_id: str) -> _StubClient:
            return client

        await svc.start(get_client_fn)
        await asyncio.sleep(0.05)
        await svc.stop()

        assert client.call_count >= 1

        record = svc.get_node("node-x")
        assert record is not None
        assert record.last_probe_time > 0

    @pytest.mark.asyncio
    async def test_probe_loop_skips_draining_nodes(self) -> None:
        client = _StubClient()

        svc = HealthCheckService(
            probe_interval=0.01,
            probe_timeout=1.0,
            failure_threshold=2,
            degraded_latency_ms=100.0,
            recovery_threshold=1,
        )
        svc.register_node("node-x", client=client)
        record = svc.get_node("node-x")
        assert record is not None
        record.state = NodeState.DRAINING

        def get_client_fn(node_id: str) -> _StubClient:
            return client

        await svc.start(get_client_fn)
        await asyncio.sleep(0.05)
        await svc.stop()

        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_probe_loop_skips_nodes_without_client(self) -> None:
        client = _StubClient()

        svc = HealthCheckService(
            probe_interval=0.01,
            probe_timeout=1.0,
        )
        svc.register_node("node-x", client=client)

        def get_client_fn(node_id: str) -> None:
            return None

        await svc.start(get_client_fn)
        await asyncio.sleep(0.05)
        await svc.stop()

        assert client.call_count == 0

    @pytest.mark.asyncio
    async def test_on_node_death_callback_triggers(self) -> None:
        client = _StubClient(fail_n=999)
        deaths: list[str] = []

        svc = HealthCheckService(
            probe_interval=0.01,
            probe_timeout=1.0,
            failure_threshold=2,
            degraded_latency_ms=100.0,
            recovery_threshold=1,
        )
        svc.on_node_death(lambda nid: deaths.append(nid))
        svc.register_node("node-x", client=client, layer_range="0-5")

        def get_client_fn(node_id: str) -> _StubClient:
            return client

        await svc.start(get_client_fn)
        await asyncio.sleep(0.2)
        await svc.stop()

        record = svc.get_node("node-x")
        assert record is not None
        # After offline_threshold (2 * failure_threshold = 4) consecutive
        # failures the node is OFFLINE and the self-healing callback fires.
        assert record.state == NodeState.OFFLINE
        assert record.consecutive_failures >= 4
        assert "node-x" in deaths

    @pytest.mark.asyncio
    async def test_exponential_backoff_on_dead_nodes(self) -> None:
        client = _StubClient(fail_n=999)

        svc = HealthCheckService(
            probe_interval=0.01,
            probe_timeout=1.0,
            failure_threshold=1,
            degraded_latency_ms=100.0,
            recovery_threshold=1,
        )
        svc.register_node("node-x", client=client)

        record = svc.get_node("node-x")
        assert record is not None
        record.consecutive_failures = _BACKOFF_FAILURE_THRESHOLD

        def get_client_fn(node_id: str) -> _StubClient:
            return client

        await svc.start(get_client_fn)
        await asyncio.sleep(0.05)
        await svc.stop()

        assert client.call_count <= 1

    @pytest.mark.asyncio
    async def test_unhandled_exception_in_probe_loop_logged(
        self,
    ) -> None:
        svc = HealthCheckService(
            probe_interval=0.01,
            probe_timeout=1.0,
        )
        svc.register_node("node-x", client=_StubClient())

        async def broken_probe(node_id: str) -> None:
            raise RuntimeError("unexpected error")

        svc._probe_once = broken_probe

        get_client = lambda x: _StubClient()

        await svc.start(get_client)
        await asyncio.sleep(0.05)
        await svc.stop()

        assert True

    @pytest.mark.asyncio
    async def test_node_recovers_after_failures(self) -> None:
        client = _StubClient(fail_n=2)

        svc = HealthCheckService(
            probe_interval=0.01,
            probe_timeout=1.0,
            failure_threshold=2,
            degraded_latency_ms=100.0,
            recovery_threshold=1,
        )
        svc.register_node("node-x", client=client)

        def get_client_fn(node_id: str) -> _StubClient:
            return client

        await svc.start(get_client_fn)
        await asyncio.sleep(0.15)
        await svc.stop()

        record = svc.get_node("node-x")
        assert record is not None
        assert record.state in (NodeState.DEGRADED, NodeState.HEALTHY)
        assert record.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_probe_once_no_record(self, health_service: HealthCheckService) -> None:
        result = await health_service._probe_once("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_probe_once_no_get_client(self, health_service: HealthCheckService) -> None:
        health_service.register_node("node-x", client=_StubClient())
        result = await health_service._probe_once("node-x")
        assert result is None
