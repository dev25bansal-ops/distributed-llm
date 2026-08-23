"""Regression tests for CRITICAL fix C3:

Self-healing was dead code. ``FailoverEngine.evaluate()`` mutated
``record.state`` in place and returned it, while ``HealthCheckService``
compared ``record.state != new_state`` *after* the mutation -> the comparison
was always False, so the ``OFFLINE`` transition branch and the
``on_node_death`` self-healing callback were unreachable. Dead nodes never
recovered.

Fix: (1) ``evaluate`` now produces an ``OFFLINE`` state after
``offline_threshold`` consecutive failures; (2) the service captures
``old_state`` *before* calling ``evaluate`` so the transition is detected and
``on_node_death`` fires.

These tests drive the REAL ``HealthCheckService._probe_once`` path with a
failing client and assert the node reaches OFFLINE and the self-healing
callback fires.
"""

from __future__ import annotations

import asyncio

import pytest

from distllm.health.failover import FailoverEngine
from distllm.health.service import HealthCheckService
from distllm.health.state import HealthRecord, NodeState


class _FailingClient:
    """A node client whose health check always fails."""

    def health_check(self, timeout=10.0):  # pragma: no cover - trivial
        raise ConnectionError("connection refused")


def _make_service() -> HealthCheckService:
    # failure_threshold=2, recovery_threshold=1 -> offline_threshold=4 (2*2)
    return HealthCheckService(
        probe_interval=0.01,
        probe_timeout=1.0,
        failure_threshold=2,
        degraded_latency_ms=100.0,
        recovery_threshold=1,
    )


@pytest.mark.asyncio
async def test_node_reaches_offline_and_self_heals():
    """Real probe path: sustained failures -> OFFLINE -> on_node_death fires."""
    svc = _make_service()
    deaths: list[str] = []
    svc.on_node_death(lambda node_id: deaths.append(node_id))

    svc.register_node(node_id="n1", client=_FailingClient(), layer_range="0-5")
    svc._get_client = lambda node_id: _FailingClient()

    for _ in range(5):
        await svc._probe_once("n1")

    record = svc.get_node("n1")
    assert record is not None
    assert record.state == NodeState.OFFLINE
    assert deaths == ["n1"], "on_node_death must fire when a node goes OFFLINE"


def test_failover_engine_reaches_offline():
    """Unit-level: evaluate() now yields OFFLINE past offline_threshold."""
    eng = FailoverEngine(failure_threshold=3, recovery_threshold=1)
    rec = HealthRecord(node_id="x", state=NodeState.HEALTHY)
    for _ in range(3):
        eng.evaluate(rec, success=False, latency_ms=0.0)
    assert rec.state == NodeState.UNHEALTHY
    for _ in range(3):
        eng.evaluate(rec, success=False, latency_ms=0.0)
    assert rec.state == NodeState.OFFLINE
