"""Metrics collection and aggregation for the coordinator.

Provides a simple in-memory metrics store with counters, gauges, and
histograms (percentile + Prometheus-bucket export).
"""

from __future__ import annotations

import threading
from typing import Any


class Histogram:
    """Bounded sample histogram with percentile + Prometheus bucket export.

    Records floats and exposes count/mean and P50/P95/P99 (nearest-rank) plus a
    Prometheus-compatible ``le`` bucket series.
    """

    _CURRENT_BUCKETS = (10.0, 50.0, 100.0, 250.0, 500.0, 1000.0)

    def __init__(self, name: str, max_samples: int = 5000):
        self.name = name
        self._max = max_samples
        self._samples: list[float] = []

    def record(self, value: float) -> None:
        if len(self._samples) < self._max:
            self._samples.append(float(value))

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def mean(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def _percentile(self, pct: float) -> float:
        n = len(self._samples)
        if n == 0:
            return 0.0
        ordered = sorted(self._samples)
        idx = int(pct / 100.0 * n)
        idx = min(max(idx - 1, 0), n - 1)
        return ordered[idx]

    @property
    def p50(self) -> float:
        return self._percentile(50.0)

    @property
    def p95(self) -> float:
        return self._percentile(95.0)

    @property
    def p99(self) -> float:
        return self._percentile(99.0)

    def to_prometheus(self) -> dict[str, Any]:
        buckets: dict[str, float] = {}
        for b in self._CURRENT_BUCKETS:
            buckets[f"le_{b}"] = float(sum(v <= b for v in self._samples))
        buckets["le_+Inf"] = float(len(self._samples))
        return {
            "type": "histogram",
            "sample_count": self.count,
            "buckets": buckets,
        }


class MetricsManager:
    """Thread-safe metrics store with counter, gauge, and histogram support.

    Usage::

        mm = MetricsManager()
        mm.record("latency_ms", 42.5)
        mm.increment("errors")
        mm.observe("request_latency_ms", 12.0)  # histogram sample
        print(mm.get())         # flat dict incl. histogram p95
        print(mm.get_prometheus())  # Prometheus-compatible dict
    """

    def __init__(self) -> None:
        self._metrics: dict[str, float] = {}
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()
        # Pre-initialize standard counters
        self._counters["total_requests"] = 0
        self._counters["errors"] = 0

    def record(self, name: str, value: float) -> None:
        """Record a gauge metric (overwrites previous value)."""
        with self._lock:
            self._metrics[name] = value

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, value: float) -> None:
        """Record a histogram sample under *name*."""
        with self._lock:
            hist = self._histograms.get(name)
            if hist is None:
                hist = Histogram(name)
                self._histograms[name] = hist
            hist.record(value)

    def histogram(self, name: str) -> Histogram | None:
        with self._lock:
            return self._histograms.get(name)

    def get(self) -> dict[str, Any]:
        """Return all metrics as a flat dict.

        Counters and gauges are merged; histograms contribute ``<name>_p95``.
        """
        with self._lock:
            result: dict[str, Any] = {}
            result.update(self._counters)
            result.update(self._metrics)
            for name, hist in self._histograms.items():
                result[f"{name}_p95"] = hist.p95
            return result

    def get_prometheus(self) -> dict[str, Any]:
        """Return metrics in a Prometheus-compatible format.

        Returns a dict with metric names as keys and dicts of
        ``{"value": float, "type": "counter"|"gauge"}`` or histogram objects.
        """
        with self._lock:
            result: dict[str, Any] = {}
            for name, value in self._counters.items():
                result[name] = {"value": float(value), "type": "counter"}
            for name, value in self._metrics.items():
                result[name] = {"value": value, "type": "gauge"}
            for name, hist in self._histograms.items():
                result[name] = hist.to_prometheus()
            return result

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._histograms.clear()
            self._counters["total_requests"] = 0
            self._counters["errors"] = 0
