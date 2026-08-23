"""Metrics completeness module for distributed-llm.

Provides connection pool metrics, queue latency histogram, tenant cost metrics,
and a combined MetricsCompleteness class with periodic background refresh.

prometheus_client is imported optionally; when unavailable all metric
operations become silent no-ops so the application never crashes.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from distllm.core.connection_pool import ConnectionPool

# ---------------------------------------------------------------------------
# Optional prometheus_client — no-op stubs when not installed
# ---------------------------------------------------------------------------

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False

    class _NoopMetric:
        """Silent no-op stand-in for any prometheus_client metric type.

        Every method (``labels``, ``set``, ``inc``, ``observe``, …) returns
        ``self`` so callers can chain without raising ``AttributeError``.
        """

        def __getattr__(self, name: str) -> _NoopMetric:
            return self

        def __call__(self, *args: Any, **kwargs: Any) -> None:
            return None

        def labels(self, *args: Any, **kwargs: Any) -> _NoopMetric:
            return self

        def set(self, value: float) -> None:
            return None

        def inc(self, amount: float = 1) -> None:
            return None

        def observe(self, amount: float) -> None:
            return None

    # Module-level singletons so every ``from metrics_completeness import Gauge``
    # works without raising ``ImportError``.
    CollectorRegistry = _NoopMetric  # type: ignore[assignment,misc]
    Counter = _NoopMetric  # type: ignore[assignment,misc]
    Gauge = _NoopMetric  # type: ignore[assignment,misc]
    Histogram = _NoopMetric  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# ConnectionPoolMetrics
# ---------------------------------------------------------------------------

class ConnectionPoolMetrics:
    """Connection pool Prometheus metrics.

    Reads a :class:`~distllm.core.connection_pool.ConnectionPool` and exports
    the following Gauge metrics, each carrying a ``"pool"`` label:

    * ``distllm_connection_pool_active`` — connections currently checked out.
    * ``distllm_connection_pool_idle`` — connections sitting idle in the pool.
    * ``distllm_connection_pool_wait_queue`` — callers waiting for a connection
      (exposed as ``0`` because the current pool has no wait queue).
    * ``distllm_connection_pool_total_created`` — cumulative connections created.
    * ``distllm_connection_pool_total_closed`` — cumulative connections
      evicted or explicitly closed.

    Parameters
    ----------
    pool:
        The :class:`~distllm.core.connection_pool.ConnectionPool` to monitor.
    registry:
        Optional Prometheus :class:`CollectorRegistry`. A new one is created
        when ``None``.
    pool_name:
        Value for the ``"pool"`` label (default ``"default"``).
    """

    def __init__(
        self,
        pool: ConnectionPool,
        registry: CollectorRegistry | None = None,
        pool_name: str = "default",
    ) -> None:
        if PROMETHEUS_AVAILABLE and registry is None:
            registry = CollectorRegistry()

        self._pool = pool
        self._pool_name = pool_name
        self._registry = registry  # type: CollectorRegistry | _NoopMetric

        self.active: Gauge = Gauge(
            "distllm_connection_pool_active",
            "Currently active connections in use",
            ["pool"],
            registry=self._registry,
        )
        self.idle: Gauge = Gauge(
            "distllm_connection_pool_idle",
            "Currently idle connections in the pool",
            ["pool"],
            registry=self._registry,
        )
        self.wait_queue: Gauge = Gauge(
            "distllm_connection_pool_wait_queue",
            "Connections waiting in queue",
            ["pool"],
            registry=self._registry,
        )
        self.total_created: Gauge = Gauge(
            "distllm_connection_pool_total_created",
            "Total connections created since pool start",
            ["pool"],
            registry=self._registry,
        )
        self.total_closed: Gauge = Gauge(
            "distllm_connection_pool_total_closed",
            "Total connections closed / evicted since pool start",
            ["pool"],
            registry=self._registry,
        )

    def refresh(self) -> None:
        """Pull fresh stats from the monitored pool and update every gauge."""
        labels = [self._pool_name]
        pool_stats = self._pool.stats()

        idle_val = self._pool.total_pooled
        total_created_val = pool_stats.creates
        total_closed_val = pool_stats.evictions

        # Connections that were created, are still alive, and are *not* idle
        # are considered active.
        alive = total_created_val - total_closed_val
        active_val = alive - idle_val if alive > idle_val else 0

        self.active.labels(*labels).set(float(active_val))
        self.idle.labels(*labels).set(float(idle_val))
        self.wait_queue.labels(*labels).set(0.0)
        self.total_created.labels(*labels).set(float(total_created_val))
        self.total_closed.labels(*labels).set(float(total_closed_val))


# ---------------------------------------------------------------------------
# QueueLatencyHistogram
# ---------------------------------------------------------------------------

class QueueLatencyHistogram:
    """Queue latency histogram metric.

    Exports a Prometheus Histogram::

        distllm_request_queue_wait_seconds

    with the following bucket boundaries (seconds)::

        [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

    Parameters
    ----------
    registry:
        Optional Prometheus :class:`CollectorRegistry`. A new one is created
        when ``None``.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        if PROMETHEUS_AVAILABLE and registry is None:
            registry = CollectorRegistry()

        self._registry = registry  # type: CollectorRegistry | _NoopMetric

        self.histogram: Histogram = Histogram(
            "distllm_request_queue_wait_seconds",
            "Time requests spent waiting in the queue before processing",
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            registry=self._registry,
        )

    def observe(self, seconds: float) -> None:
        """Record a single queue-wait observation.

        Parameters
        ----------
        seconds:
            Wall-clock time the request waited in the queue.
        """
        self.histogram.observe(seconds)


# ---------------------------------------------------------------------------
# TenantCostMetrics
# ---------------------------------------------------------------------------

class TenantCostMetrics:
    """Tenant-level cost and token tracking.

    Exports two Prometheus Counters::

        distllm_tenant_cost_total    (labels: tenant_id, model)
        distllm_tenant_tokens_total  (labels: tenant_id, direction)

    Parameters
    ----------
    registry:
        Optional Prometheus :class:`CollectorRegistry`. A new one is created
        when ``None``.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        if PROMETHEUS_AVAILABLE and registry is None:
            registry = CollectorRegistry()

        self._registry = registry  # type: CollectorRegistry | _NoopMetric

        self.cost_total: Counter = Counter(
            "distllm_tenant_cost_total",
            "Total cost per tenant per model",
            ["tenant_id", "model"],
            registry=self._registry,
        )
        self.tokens_total: Counter = Counter(
            "distllm_tenant_tokens_total",
            "Total tokens per tenant by direction (input / output)",
            ["tenant_id", "direction"],
            registry=self._registry,
        )

    def add_cost(self, tenant_id: str, model: str, amount: float) -> None:
        """Record a cost increment.

        Parameters
        ----------
        tenant_id:
            The tenant identifier.
        model:
            The model name (e.g. ``"llama-3-70b"``).
        amount:
            Monetary amount to add.
        """
        self.cost_total.labels(tenant_id=tenant_id, model=model).inc(amount)

    def add_tokens(
        self, tenant_id: str, direction: str, count: int | float
    ) -> None:
        """Record a token-count increment.

        Parameters
        ----------
        tenant_id:
            The tenant identifier.
        direction:
            Token direction — ``"input"`` or ``"output"``.
        count:
            Number of tokens to add.
        """
        self.tokens_total.labels(
            tenant_id=tenant_id, direction=direction
        ).inc(count)


# ---------------------------------------------------------------------------
# MetricsCompleteness  (combined, with periodic refresh)
# ---------------------------------------------------------------------------

class MetricsCompleteness:
    """Combined metrics completeness monitor.

    Aggregates :class:`ConnectionPoolMetrics`, :class:`QueueLatencyHistogram`,
    and :class:`TenantCostMetrics` into a single interface with an optional
    periodic background refresh loop.

    The class may be used as a context manager::

        with MetricsCompleteness(pool) as mc:
            mc.queue_latency.observe(0.3)
            mc.tenant_costs.add_cost("tenant-1", "llama-3-70b", 0.005)
            # … metrics are refreshed every 15 s in the background …

    Parameters
    ----------
    pool:
        The :class:`~distllm.core.connection_pool.ConnectionPool` to monitor.
    pool_name:
        Label value for the ``"pool"`` label (default ``"default"``).
    registry:
        Optional Prometheus :class:`CollectorRegistry`. A new one is created
        when ``None``.
    refresh_interval:
        Seconds between periodic metric refreshes (default ``15``).
    """

    def __init__(
        self,
        pool: ConnectionPool,
        pool_name: str = "default",
        registry: CollectorRegistry | None = None,
        refresh_interval: float = 15.0,
    ) -> None:
        if PROMETHEUS_AVAILABLE and registry is None:
            registry = CollectorRegistry()

        self._registry = registry  # type: CollectorRegistry | _NoopMetric
        self._refresh_interval = refresh_interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self.pool_metrics = ConnectionPoolMetrics(
            pool=pool,
            registry=self._registry,
            pool_name=pool_name,
        )
        self.queue_latency = QueueLatencyHistogram(registry=self._registry)
        self.tenant_costs = TenantCostMetrics(registry=self._registry)

    @property
    def registry(self) -> CollectorRegistry:
        """Return the :class:`CollectorRegistry` used by all sub-metrics."""
        return self._registry  # type: ignore[return-value]

    # ── Life cycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the periodic background refresh loop.

        The loop runs once every ``refresh_interval`` seconds and calls
        :meth:`refresh`. Safe to call multiple times — subsequent calls
        are no-ops while the loop is already running.
        """
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop,
            daemon=True,
            name="metrics-completeness-refresh",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the periodic background refresh loop.

        Waits up to 5 seconds for the thread to exit. Safe to call
        multiple times.
        """
        self._running = False
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            self._thread = None

    def refresh(self) -> None:
        """Perform a one-shot refresh of all periodic sub-metrics.

        Currently refreshes :attr:`pool_metrics`; other sub-metrics are
        event-driven (their ``observe`` / ``add_*`` methods are called
        directly by application code).
        """
        self.pool_metrics.refresh()

    # ── Context-manager support ───────────────────────────────────────────

    def __enter__(self) -> MetricsCompleteness:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    # ── Internal ──────────────────────────────────────────────────────────

    def _refresh_loop(self) -> None:
        """Background loop body."""
        while not self._stop_event.wait(self._refresh_interval):
            try:
                self.refresh()
            except Exception:
                # Swallow so the loop doesn't die on a transient failure.
                pass
