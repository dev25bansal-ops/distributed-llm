"""Unified metrics/observability system for distllm.

Provides a singleton ``MetricsRegistry`` that manages ``Counter``,
``Histogram``, and ``Gauge`` metric objects and can render them in
Prometheus text exposition format.  A :func:`timed` context manager is
included for conveniently recording operation durations as histograms.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Sequence, Tuple

__all__ = [
    "MetricsRegistry",
    "Counter",
    "Histogram",
    "Gauge",
    "PipelineLatencyTracker",
    "StageLatencies",
    "WindowedLatencyTracker",
    "timed",
    "get_metrics_registry",
]

# ---------------------------------------------------------------------------
# Helper -- label rendering
# ---------------------------------------------------------------------------

_PROMETHEUS_LABEL_ESCAPE = str.maketrans({"\\": "\\\\", '"': '\\"', "\n": "\\n"})


def _quote(s: str) -> str:
    return '"' + s.translate(_PROMETHEUS_LABEL_ESCAPE) + '"'


def _sanitise(name: str) -> str:
    """Replace any character that isn't valid in a Prometheus metric name."""
    return "".join(c if c.isalnum() else "_" for c in name)


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------


@dataclass
class Counter:
    """A monotonically-increasing counter metric."""

    name: str
    description: str
    labels: Tuple[str, ...] = ()
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0) -> None:
        """Increment the counter by *amount* (default 1)."""
        if amount < 0:
            raise ValueError(f"Counter {self.name}: amount must be non-negative, got {amount}")
        with self._lock:
            self._value += amount


@dataclass
class Histogram:
    """A histogram metric that records observations in configurable buckets."""

    name: str
    description: str
    labels: Tuple[str, ...] = ()
    buckets: Tuple[float, ...] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        float("inf"),
    )
    _counts: List[int] = field(default_factory=list)
    _sum: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not self._counts:
            self._counts = [0] * len(self.buckets)

    def observe(self, amount: float) -> None:
        """Record a single observation."""
        with self._lock:
            self._sum += amount
            for i, boundary in enumerate(self.buckets):
                if amount <= boundary:
                    self._counts[i] += 1


@dataclass
class Gauge:
    """A gauge metric that can be set to an arbitrary value."""

    name: str
    description: str
    labels: Tuple[str, ...] = ()
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, val: float) -> None:
        """Set the gauge to *val*."""
        with self._lock:
            self._value = val

    def inc(self, amount: float = 1.0) -> None:
        """Increase the gauge by *amount* (default 1)."""
        with self._lock:
            self._value += amount


# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------


class MetricsRegistry:
    """Thread-safe singleton registry for all application metrics.

    Usage::

        reg = get_metrics_registry()
        req_counter = reg.counter("http_requests_total", "Total HTTP requests")
        dur_hist = reg.histogram("http_duration_seconds", "Request duration", buckets=(0.1, 0.5, 1.0))
        pool_gauge = reg.gauge("connection_pool_size", "Active DB connections")

        req_counter.inc()
        dur_hist.observe(0.234)
        pool_gauge.set(42)

        print(reg.export_prometheus())
    """

    _instance: Optional["MetricsRegistry"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._counters: Dict[str, Counter] = OrderedDict()
        self._histograms: Dict[str, Histogram] = OrderedDict()
        self._gauges: Dict[str, Gauge] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()

    # -- Metric constructors ------------------------------------------------

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Sequence[str]] = None,
    ) -> Counter:
        """Return (or create) a :class:`Counter` metric.

        If a counter with the same *name* already exists it is returned
        as-is (the registry is idempotent).
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(
                    name=name,
                    description=description,
                    labels=tuple(labels) if labels else (),
                )
            return self._counters[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Sequence[float]] = None,
        labels: Optional[Sequence[str]] = None,
    ) -> Histogram:
        """Return (or create) a :class:`Histogram` metric.

        If a histogram with the same *name* already exists it is returned
        as-is.
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name=name,
                    description=description,
                    buckets=tuple(buckets) if buckets else Histogram.buckets,
                )
            return self._histograms[name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[Sequence[str]] = None,
    ) -> Gauge:
        """Return (or create) a :class:`Gauge` metric.

        If a gauge with the same *name* already exists it is returned
        as-is.
        """
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(
                    name=name,
                    description=description,
                    labels=tuple(labels) if labels else (),
                )
            return self._gauges[name]

    # -- Prometheus exposition format ---------------------------------------

    def export_prometheus(self) -> str:
        """Render all registered metrics in `Prometheus text format`_.

        .. _Prometheus text format: https://prometheus.io/docs/instrumenting/exposition-formats/
        """
        lines: List[str] = []

        with self._lock:
            # counters
            for c in self._counters.values():
                if c.description:
                    lines.append(f"# HELP {_sanitise(c.name)} {_quote(c.description)}")
                    lines.append(f"# TYPE {_sanitise(c.name)} counter")
                with c._lock:
                    labels_str = _label_str(c.labels, None)
                    lines.append(f"{_sanitise(c.name)}{labels_str} {c._value}")
                lines.append("")

            # histograms
            for h in self._histograms.values():
                if h.description:
                    lines.append(f"# HELP {_sanitise(h.name)} {_quote(h.description)}")
                    lines.append(f"# TYPE {_sanitise(h.name)} histogram")
                with h._lock:
                    for i, boundary in enumerate(h.buckets):
                        bucket_label = _label_str(
                            h.labels, (("le", str(boundary)),)
                        )
                        lines.append(
                            f"{_sanitise(h.name)}_bucket{bucket_label} {h._counts[i]}"
                        )
                    lines.append(
                        f"{_sanitise(h.name)}_sum{_label_str(h.labels, None)} {h._sum}"
                    )
                    lines.append(
                        f"{_sanitise(h.name)}_count{_label_str(h.labels, None)} "
                        f"{h._counts[-1] if h._counts else 0}"
                    )
                lines.append("")

            # gauges
            for g in self._gauges.values():
                if g.description:
                    lines.append(f"# HELP {_sanitise(g.name)} {_quote(g.description)}")
                    lines.append(f"# TYPE {_sanitise(g.name)} gauge")
                with g._lock:
                    labels_str = _label_str(g.labels, None)
                    lines.append(f"{_sanitise(g.name)}{labels_str} {g._value}")
                lines.append("")

        return "\n".join(lines)

    # -- Reset ---------------------------------------------------------------

    def reset(self) -> None:
        """Clear all registered metrics (useful in tests)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


# ---------------------------------------------------------------------------
# Label-rendering helper
# ---------------------------------------------------------------------------


def _label_str(
    labels: Tuple[str, ...],
    extra: Optional[Tuple[Tuple[str, str], ...]],
) -> str:
    """Render label key-value pairs as ``{key="val",key="val"}``.

    If there are no labels and no extras, returns an empty string.
    """
    parts: List[str] = []
    for label in labels:
        parts.append(f'{label}="{label}"')
    if extra:
        for k, v in extra:
            parts.append(f'{k}="{v}"')
    if not parts:
        return ""
    return "{" + ",".join(parts) + "}"


# ---------------------------------------------------------------------------
# Latency breakdown tracker (P50/P95/P99 per pipeline stage)
# ---------------------------------------------------------------------------


@dataclass
class StageLatencies:
    """Latency statistics for a single pipeline stage."""
    stage_name: str
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    p999_ms: float = 0.0
    samples: int = 0


@dataclass
class WindowedLatencyTracker:
    """Moving-window latency tracker for a pipeline stage.

    Usage::

        tracker = WindowedLatencyTracker(window_size=1000)
        tracker.record(42.5)  # latency in ms
        stats = tracker.stats()  # -> StageLatencies
    """

    window_size: int = 1000
    _samples: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)
        if len(self._samples) > self.window_size:
            self._samples.pop(0)

    def stats(self, stage_name: str = "") -> StageLatencies:
        if not self._samples:
            return StageLatencies(stage_name=stage_name)
        sorted_samples = sorted(self._samples)
        n = len(sorted_samples)
        return StageLatencies(
            stage_name=stage_name,
            p50_ms=sorted_samples[n // 2],
            p95_ms=sorted_samples[int(n * 0.95)],
            p99_ms=sorted_samples[int(n * 0.99)],
            p999_ms=sorted_samples[int(n * 0.999)],
            samples=n,
        )


class PipelineLatencyTracker:
    """Aggregates per-stage latency across all pipeline stages.

    Usage::

        tracker = PipelineLatencyTracker()
        tracker.record_stage("node-0", 45.0)
        tracker.record_stage("node-1", 52.3)
        breakdown = tracker.breakdown()  # -> list[StageLatencies]
    """

    def __init__(self, window_size: int = 1000) -> None:
        self._stages: dict[str, WindowedLatencyTracker] = {}
        self._window_size = window_size

    def record_stage(self, stage_name: str, latency_ms: float) -> None:
        if stage_name not in self._stages:
            self._stages[stage_name] = WindowedLatencyTracker(
                window_size=self._window_size,
            )
        self._stages[stage_name].record(latency_ms)

    def breakdown(self) -> list[StageLatencies]:
        return [
            t.stats(name) for name, t in self._stages.items()
        ]

    def stats(self) -> dict:
        bd = self.breakdown()
        return {
            "stages": [
                {
                    "name": s.stage_name,
                    "p50_ms": s.p50_ms,
                    "p95_ms": s.p95_ms,
                    "p99_ms": s.p99_ms,
                    "samples": s.samples,
                }
                for s in bd
            ],
            "total_stages": len(bd),
        }


# ---------------------------------------------------------------------------
# Timed context manager
# ---------------------------------------------------------------------------


@contextmanager
def timed(
    registry: MetricsRegistry,
    name: str,
    description: str = "",
    buckets: Optional[Sequence[float]] = None,
) -> Generator[None, None, None]:
    """Context manager that records the duration of a block as a histogram.

    Usage::

        reg = get_metrics_registry()
        with timed(reg, "db_query_duration_seconds", buckets=(0.01, 0.1, 1.0)):
            result = run_query()

    The histogram is created via *registry* on first use.
    """
    h = registry.histogram(name, description=description, buckets=buckets)
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - start
        h.observe(elapsed)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_registry_instance: Optional[MetricsRegistry] = None
_registry_lock: threading.Lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """Return the application-wide singleton :class:`MetricsRegistry`.

    Creates the instance on first call.  This function is thread-safe.
    """
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = MetricsRegistry()
    return _registry_instance
