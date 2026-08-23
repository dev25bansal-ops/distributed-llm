"""Tests for MetricsCollector -- aggregates metrics from latency, straggler, recovery.

Covers:
- Construction with no trackers
- collect returns empty dict when no trackers
- collect includes latency metrics
- collect includes straggler stats
- collect includes recovery metrics

No MagicMock -- real dict-based stubs for sub-trackers.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/metrics_collector.py")
MetricsCollector = _mod.MetricsCollector


class _StubLatencyTracker:
    """Minimal latency tracker."""

    def get_all_avg(self: Any) -> dict[str, float]:
        return {"avg_latency_ms": 42.5, "p99_latency_ms": 100.0}


class _StubStragglerDetector:
    """Minimal straggler detector."""

    def stats(self: Any) -> dict[str, Any]:
        return {"stragglers_detected": 2, "slow_nodes": ["node-3"]}


class _StubRecoveryManager:
    """Minimal recovery manager."""

    def get_metrics(self: Any) -> dict[str, Any]:
        return {"recoveries": 1, "last_recovery": "2025-01-01T00:00:00"}


class TestMetricsCollectorConstruction:
    """Construction and initial state."""

    def test_no_trackers(self) -> None:
        mc = MetricsCollector()
        assert mc._latency_tracker is None
        assert mc._straggler_detector is None
        assert mc._recovery_manager is None

    def test_with_trackers(self) -> None:
        lt = _StubLatencyTracker()
        sd = _StubStragglerDetector()
        rm = _StubRecoveryManager()
        mc = MetricsCollector(latency_tracker=lt, straggler_detector=sd, recovery_manager=rm)
        assert mc._latency_tracker is lt
        assert mc._straggler_detector is sd
        assert mc._recovery_manager is rm


class TestMetricsCollectorCollect:
    """Collecting metrics."""

    def test_collect_empty(self) -> None:
        mc = MetricsCollector()
        result = mc.collect()
        assert result == {}

    def test_collect_with_latency(self) -> None:
        mc = MetricsCollector(latency_tracker=_StubLatencyTracker())
        result = mc.collect()
        assert "latency" in result
        assert result["latency"]["avg_latency_ms"] == 42.5

    def test_collect_with_straggler(self) -> None:
        mc = MetricsCollector(straggler_detector=_StubStragglerDetector())
        result = mc.collect()
        assert "straggler" in result
        assert result["straggler"]["stragglers_detected"] == 2

    def test_collect_with_recovery(self) -> None:
        mc = MetricsCollector(recovery_manager=_StubRecoveryManager())
        result = mc.collect()
        assert "recovery" in result
        assert result["recovery"]["recoveries"] == 1

    def test_collect_all(self) -> None:
        mc = MetricsCollector(
            latency_tracker=_StubLatencyTracker(),
            straggler_detector=_StubStragglerDetector(),
            recovery_manager=_StubRecoveryManager(),
        )
        result = mc.collect()
        assert "latency" in result
        assert "straggler" in result
        assert "recovery" in result
