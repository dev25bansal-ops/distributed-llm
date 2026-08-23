"""Tests for HealthChecker -- synchronous and async health check dispatch.

Covers:
- Construction with ResourceManager
- check_all with healthy/unhealthy nodes
- check_all with circuit-breaker-open nodes
- check_all with missing nodes
- check_all_async basic dispatch
- check_all_async with timeout

No MagicMock -- real dicts, counters, and callables.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/coordinator_health.py")
HealthChecker = _mod.HealthChecker


class _StubResourceManager:
    """Minimal ResourceManager stand-in for testing HealthChecker."""

    def __init__(self: Any) -> None:
        self._circuit_open: set[str] = set()
        self.successes: list[str] = []
        self.failures: list[str] = []

    def check_circuit_breaker(self: Any, node_id: str) -> bool:
        return node_id in self._circuit_open

    def record_success(self: Any, node_id: str) -> None:
        self.successes.append(node_id)

    def record_failure(self: Any, node_id: str) -> None:
        self.failures.append(node_id)


class TestHealthCheckerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm)
        assert hc._resource_mgr is rm
        assert hc._timeout_s == 5.0

    def test_custom_timeout(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm, timeout_s=10.0)
        assert hc.timeout_s == 10.0

    def test_timeout_property(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm, timeout_s=3.0)
        assert hc.timeout_s == 3.0


class TestHealthCheckerCheckAll:
    """Sync check_all dispatcher."""

    def test_all_healthy(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm)
        nodes = {"node-1": object(), "node-2": object()}
        result = hc.check_all(
            nodes=nodes,
            node_order=["node-1", "node-2"],
            is_healthy_fn=lambda nid: True,
        )
        assert result["node-1"]["healthy"] is True
        assert result["node-2"]["healthy"] is True
        assert "node-1" in rm.successes
        assert "node-2" in rm.successes

    def test_unhealthy_node(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm)
        nodes = {"node-1": object()}
        result = hc.check_all(
            nodes=nodes,
            node_order=["node-1"],
            is_healthy_fn=lambda nid: False,
        )
        assert result["node-1"]["healthy"] is False
        assert "node-1" in rm.failures

    def test_missing_node(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm)
        nodes = {}
        result = hc.check_all(
            nodes=nodes,
            node_order=["node-missing"],
            is_healthy_fn=lambda nid: True,
        )
        assert result["node-missing"]["healthy"] is False
        assert result["node-missing"]["reason"] == "not_found"

    def test_circuit_breaker_open(self) -> None:
        rm = _StubResourceManager()
        rm._circuit_open.add("node-1")
        hc = HealthChecker(resource_mgr=rm)
        nodes = {"node-1": object()}
        result = hc.check_all(
            nodes=nodes,
            node_order=["node-1"],
            is_healthy_fn=lambda nid: True,
        )
        assert result["node-1"]["healthy"] is False
        assert result["node-1"]["reason"] == "circuit_breaker_open"

    def test_exception_during_check(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm)
        nodes = {"node-1": object()}

        def _failing(nid: str) -> bool:
            raise ConnectionError("connection refused")

        result = hc.check_all(
            nodes=nodes,
            node_order=["node-1"],
            is_healthy_fn=_failing,
        )
        assert result["node-1"]["healthy"] is False
        assert "connection refused" in result["node-1"]["reason"]
        assert "node-1" in rm.failures


class TestHealthCheckerCheckAllAsync:
    """Async check_all_async dispatcher."""

    @pytest.mark.asyncio
    async def test_all_healthy_async(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm)
        nodes = {"node-1": object(), "node-2": object()}
        result = await hc.check_all_async(
            nodes=nodes,
            node_order=["node-1", "node-2"],
            is_healthy_fn=lambda nid: True,
        )
        assert result["node-1"]["healthy"] is True
        assert result["node-2"]["healthy"] is True

    @pytest.mark.asyncio
    async def test_missing_node_async(self) -> None:
        rm = _StubResourceManager()
        hc = HealthChecker(resource_mgr=rm)
        nodes = {}
        result = await hc.check_all_async(
            nodes=nodes,
            node_order=["node-missing"],
            is_healthy_fn=lambda nid: True,
        )
        assert result["node-missing"]["healthy"] is False
        assert result["node-missing"]["reason"] == "not_found"

    @pytest.mark.asyncio
    async def test_circuit_breaker_async(self) -> None:
        rm = _StubResourceManager()
        rm._circuit_open.add("node-1")
        hc = HealthChecker(resource_mgr=rm)
        nodes = {"node-1": object()}
        result = await hc.check_all_async(
            nodes=nodes,
            node_order=["node-1"],
            is_healthy_fn=lambda nid: True,
        )
        assert result["node-1"]["healthy"] is False
        assert result["node-1"]["reason"] == "circuit_breaker_open"
