"""Tests for MetricsManager -- thread-safe metrics store with counters and gauges.

Covers:
- Construction and initial state
- record (gauge) and increment (counter)
- get merges counters and gauges
- get_prometheus returns typed format
- reset restores initial state

No MagicMock -- real dicts and threading.Lock.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/coordinator_metrics.py")
MetricsManager = _mod.MetricsManager


class TestMetricsManagerConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        mm = MetricsManager()
        assert mm._metrics == {}
        assert mm._counters["total_requests"] == 0
        assert mm._counters["errors"] == 0

    def test_initial_counters_preinitialized(self) -> None:
        mm = MetricsManager()
        assert mm._counters.get("total_requests") == 0
        assert mm._counters.get("errors") == 0


class TestMetricsManagerRecord:
    """Gauge recording."""

    def test_record_sets_gauge(self) -> None:
        mm = MetricsManager()
        mm.record("latency_ms", 42.5)
        assert mm._metrics["latency_ms"] == 42.5

    def test_record_overwrites_previous(self) -> None:
        mm = MetricsManager()
        mm.record("latency_ms", 10.0)
        mm.record("latency_ms", 20.0)
        assert mm._metrics["latency_ms"] == 20.0

    def test_record_multiple_gauges(self) -> None:
        mm = MetricsManager()
        mm.record("latency_ms", 1.0)
        mm.record("memory_mb", 256.0)
        assert len(mm._metrics) == 2


class TestMetricsManagerIncrement:
    """Counter increment."""

    def test_increment_default_amount(self) -> None:
        mm = MetricsManager()
        mm.increment("errors")
        assert mm._counters["errors"] == 1

    def test_increment_custom_amount(self) -> None:
        mm = MetricsManager()
        mm.increment("errors", 5)
        assert mm._counters["errors"] == 5

    def test_increment_new_counter(self) -> None:
        mm = MetricsManager()
        mm.increment("custom_metric")
        assert mm._counters["custom_metric"] == 1

    def test_increment_accumulates(self) -> None:
        mm = MetricsManager()
        mm.increment("errors", 2)
        mm.increment("errors", 3)
        assert mm._counters["errors"] == 5


class TestMetricsManagerGet:
    """Combined get()."""

    def test_get_returns_merged(self) -> None:
        mm = MetricsManager()
        mm.increment("errors", 1)
        mm.record("latency_ms", 30.0)
        result = mm.get()
        assert result["errors"] == 1
        assert result["latency_ms"] == 30.0
        assert result["total_requests"] == 0

    def test_get_returns_copy(self) -> None:
        mm = MetricsManager()
        mm.record("x", 1.0)
        result = mm.get()
        result["x"] = 999.0
        assert mm._metrics["x"] == 1.0


class TestMetricsManagerPrometheus:
    """Prometheus export."""

    def test_get_prometheus_format(self) -> None:
        mm = MetricsManager()
        mm.increment("errors", 3)
        mm.record("latency_ms", 50.0)
        prom = mm.get_prometheus()
        assert prom["errors"]["value"] == 3.0
        assert prom["errors"]["type"] == "counter"
        assert prom["latency_ms"]["value"] == 50.0
        assert prom["latency_ms"]["type"] == "gauge"

    def test_get_prometheus_has_total_requests(self) -> None:
        mm = MetricsManager()
        prom = mm.get_prometheus()
        assert "total_requests" in prom
        assert prom["total_requests"]["type"] == "counter"


class TestMetricsManagerReset:
    """Reset restores initial state."""

    def test_reset_clears_gauges(self) -> None:
        mm = MetricsManager()
        mm.record("latency_ms", 100.0)
        mm.increment("errors", 10)
        mm.reset()
        assert mm._metrics == {}
        assert mm._counters["total_requests"] == 0
        assert mm._counters["errors"] == 0

    def test_reset_removes_custom_counters(self) -> None:
        mm = MetricsManager()
        mm.increment("custom_metric", 5)
        mm.reset()
        assert "custom_metric" not in mm._counters
