"""Metrics collection and aggregation for the coordinator.

Provides a simple in-memory metrics store with counters and gauges,
plus Prometheus-compatible export.
"""

from __future__ import annotations

import threading
from typing import Any


class MetricsManager:
    """Thread-safe metrics store with counter and gauge support.

    Usage::

        mm = MetricsManager()
        mm.record("latency_ms", 42.5)
        mm.increment("errors")
        print(mm.get())         # {"latency_ms": 42.5, "errors": 1, "total_requests": 0}
        print(mm.get_prometheus())  # Prometheus-compatible dict
    """

    def __init__(self) -> None:
        self._metrics: dict[str, float] = {}
        self._counters: dict[str, int] = {}
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

    def get(self) -> dict[str, Any]:
        """Return all metrics as a flat dict.

        Counters and gauges are merged. Counter values are ints,
        gauge values are floats.
        """
        with self._lock:
            result: dict[str, Any] = {}
            result.update(self._counters)
            result.update(self._metrics)
            return result

    def get_prometheus(self) -> dict[str, Any]:
        """Return metrics in a Prometheus-compatible format.

        Returns a dict with metric names as keys and dicts of
        ``{"value": float, "type": "counter"|"gauge"}`` as values.
        """
        with self._lock:
            result: dict[str, Any] = {}
            for name, value in self._counters.items():
                result[name] = {"value": float(value), "type": "counter"}
            for name, value in self._metrics.items():
                result[name] = {"value": value, "type": "gauge"}
            return result

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._counters["total_requests"] = 0
            self._counters["errors"] = 0
