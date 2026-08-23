"""Pytest fixtures for the health test suite.

All four health modules (state, failover, prober, service) import directly
without circular dependency issues, so no ``load_module`` / bootstrap is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from distllm.health.failover import FailoverEngine
from distllm.health.prober import probe_node
from distllm.health.service import HealthCheckService
from distllm.health.state import HealthRecord, HealthStateStore, NodeState


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------


class _StubClient:
    """Minimal stub for a gRPC NodeClient used in health probing tests.

    Provides a ``health_check(timeout=...)`` method that returns a
    ``SimpleNamespace`` with ``memory_used``, ``memory_total``, and
    ``gpu_utilization`` attributes (or a custom response passed at
    construction).

    Usage:
        client = _StubClient()                             # always succeeds
        client = _StubClient(fail_n=2)                     # fails first 2 calls
        client.set_fail(ConnectionError("custom"))         # permanent fail
    """

    def __init__(self, response=None, fail_n=0, fail_exc=None):
        self.response = response if response is not None else SimpleNamespace(
            memory_used=2048,
            memory_total=8192,
            gpu_utilization=0.45,
        )
        self._perm_fail = None
        self._fail_n = fail_n
        self._fail_exc = fail_exc if fail_exc is not None else ConnectionError("connection refused")
        self.call_count = 0
        self.last_timeout = None

    def set_fail(self, exc=None):
        """Make all subsequent ``health_check`` calls raise *exc*."""
        self._perm_fail = exc if exc is not None else ConnectionError("connection refused")
        self._fail_n = 0

    def health_check(self, timeout=10.0):
        self.call_count += 1
        self.last_timeout = timeout
        if self._perm_fail:
            raise self._perm_fail
        if self._fail_n > 0:
            self._fail_n -= 1
            raise self._fail_exc
        return self.response


# ---------------------------------------------------------------------------
# HealthRecord fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def health_record() -> HealthRecord:
    """Create a healthy node record."""
    return HealthRecord(
        node_id="node-0",
        state=NodeState.HEALTHY,
        last_probe_time=1000.0,
        consecutive_failures=0,
        consecutive_successes=5,
        latency_p50_ms=50.0,
        latency_p99_ms=200.0,
        gpu_utilization=0.5,
        memory_used=2048,
        memory_total=8192,
        layer_range="0-5",
    )


@pytest.fixture
def degraded_record() -> HealthRecord:
    """Create a degraded node record."""
    return HealthRecord(
        node_id="node-1",
        state=NodeState.DEGRADED,
        last_probe_time=1000.0,
        consecutive_failures=1,
        consecutive_successes=0,
        latency_p50_ms=3000.0,
        latency_p99_ms=5000.0,
        gpu_utilization=0.9,
        memory_used=7168,
        memory_total=8192,
        layer_range="6-11",
    )


@pytest.fixture
def unhealthy_record() -> HealthRecord:
    """Create an unhealthy node record.

    ``consecutive_failures`` is set to the UNHEALTHY threshold (matching the
    ``failover_engine`` fixture's ``failure_threshold=2``), so a single further
    failure stays UNHEALTHY and the node only reaches OFFLINE after the full
    ``offline_threshold`` (2x) of consecutive failures.
    """
    return HealthRecord(
        node_id="node-2",
        state=NodeState.UNHEALTHY,
        last_probe_time=1000.0,
        consecutive_failures=2,
        consecutive_successes=0,
        latency_p50_ms=0.0,
        latency_p99_ms=0.0,
        gpu_utilization=0.0,
        memory_used=0,
        memory_total=8192,
        layer_range="12-17",
    )


@pytest.fixture
def offline_record() -> HealthRecord:
    """Create an offline node record."""
    return HealthRecord(
        node_id="node-3",
        state=NodeState.OFFLINE,
        last_probe_time=1000.0,
        consecutive_failures=5,
        consecutive_successes=0,
        latency_p50_ms=0.0,
        latency_p99_ms=0.0,
        gpu_utilization=0.0,
        memory_used=0,
        memory_total=8192,
        layer_range="18-23",
    )


# ---------------------------------------------------------------------------
# HealthStateStore fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def health_store() -> HealthStateStore:
    """Create an empty HealthStateStore."""
    return HealthStateStore()


@pytest.fixture
def populated_health_store(
    health_record: HealthRecord,
    degraded_record: HealthRecord,
    unhealthy_record: HealthRecord,
    offline_record: HealthRecord,
) -> HealthStateStore:
    """Create a HealthStateStore with one record of each state."""
    store = HealthStateStore()
    store.set(health_record.node_id, health_record)
    store.set(degraded_record.node_id, degraded_record)
    store.set(unhealthy_record.node_id, unhealthy_record)
    store.set(offline_record.node_id, offline_record)
    return store


# ---------------------------------------------------------------------------
# FailoverEngine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def failover_engine() -> FailoverEngine:
    """FailoverEngine with low thresholds for fast testing."""
    return FailoverEngine(
        failure_threshold=2,
        degraded_latency_ms=100.0,
        recovery_threshold=1,
    )


@pytest.fixture
def strict_failover_engine() -> FailoverEngine:
    """FailoverEngine with higher thresholds (production-like)."""
    return FailoverEngine(
        failure_threshold=3,
        degraded_latency_ms=2000.0,
        recovery_threshold=2,
    )


# ---------------------------------------------------------------------------
# Stub client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_grpc_client() -> _StubClient:
    """Create a stub gRPC NodeClient for health probing."""
    return _StubClient()


# ---------------------------------------------------------------------------
# HealthCheckService fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def health_service() -> HealthCheckService:
    """HealthCheckService with low thresholds for fast testing."""
    return HealthCheckService(
        probe_interval=0.01,
        probe_timeout=1.0,
        failure_threshold=2,
        degraded_latency_ms=100.0,
        recovery_threshold=1,
    )


@pytest.fixture
def health_service_strict() -> HealthCheckService:
    """HealthCheckService with production-like thresholds."""
    return HealthCheckService(
        probe_interval=5.0,
        probe_timeout=10.0,
        failure_threshold=3,
        degraded_latency_ms=2000.0,
        recovery_threshold=2,
    )


@pytest.fixture
def registered_service(health_service: HealthCheckService) -> HealthCheckService:
    """HealthCheckService with a node registered."""
    health_service.register_node(
        node_id="node-0",
        client=_StubClient(),
        layer_range="0-5",
    )
    return health_service
