"""Prometheus metrics exporter for distributed-llm.

Provides a CollectorRegistry with all standard distllm metrics
and a FastAPI-compatible /metrics response via generate_latest.
"""

import contextlib

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # [observability] extra not installed — degrade to no-ops
    _PROMETHEUS_AVAILABLE = False

    class _CollectorRegistry:
        def __init__(self):
            pass

    class _NoopMetric:
        """Drop-in stub: every method is a no-op, every label returns self."""

        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    CollectorRegistry = _CollectorRegistry
    Counter = _NoopMetric
    Histogram = _NoopMetric
    Gauge = _NoopMetric


class DistLLMPrometheusExporter:
    """Prometheus metrics collector for all distributed-llm components.

    When ``prometheus_client`` is missing (the ``[observability]`` extra is
    not installed), every metric becomes a no-op stub so the rest of the
    server still imports and runs; ``generate_metrics()`` then returns an
    empty exposition and the ``/metrics`` endpoint serves nothing rather
    than crashing the process.
    """

    def __init__(self):
        self.registry = CollectorRegistry()
        # Watermarks for cumulative counters bridged from subsystems that
        # report totals (recovery manager, spot ensemble). populate_gauges
        # runs on every /metrics scrape; counters are only incremented by the
        # growth since the previous scrape so repeated scrapes don't inflate
        # them and a subsystem restart (value goes backwards) re-bases.
        self._bridge_watermarks: dict[str, float] = {}

        # --- Request metrics (with model/tenant labels) ---
        self.requests_total = Counter(
            "distllm_requests_total",
            "Total requests processed",
            ["method", "status", "model", "tenant"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "distllm_request_latency_seconds",
            "End-to-end request latency",
            ["method", "model", "tenant"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry,
        )

        # --- Token generation metrics ---
        self.tokens_generated = Counter(
            "distllm_tokens_generated_total",
            "Total tokens generated",
            ["model", "tenant"],
            registry=self.registry,
        )
        self.token_latency = Histogram(
            "distllm_token_generation_latency_seconds",
            "Time to generate a single token",
            ["model"],
            buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )
        self.tokens_per_second = Gauge(
            "distllm_tokens_per_second",
            "Current token generation rate",
            ["model"],
            registry=self.registry,
        )

        # --- Node metrics ---
        self.node_health = Gauge(
            "distllm_node_health",
            "Node health status (1=healthy, 0=unhealthy)",
            ["node_id", "layer_range"],
            registry=self.registry,
        )
        self.node_gpu_utilization = Gauge(
            "distllm_node_gpu_utilization_percent",
            "GPU utilization percentage",
            ["node_id"],
            registry=self.registry,
        )
        self.node_gpu_memory_bytes = Gauge(
            "distllm_node_gpu_memory_bytes",
            "GPU memory used in bytes",
            ["node_id"],
            registry=self.registry,
        )
        self.node_gpu_memory_total_bytes = Gauge(
            "distllm_node_gpu_memory_total_bytes",
            "GPU memory total in bytes",
            ["node_id"],
            registry=self.registry,
        )
        self.node_latency_p50 = Gauge(
            "distllm_node_latency_p50_ms",
            "p50 inference latency per node in milliseconds",
            ["node_id"],
            registry=self.registry,
        )
        self.node_latency_p99 = Gauge(
            "distllm_node_latency_p99_ms",
            "p99 inference latency per node in milliseconds",
            ["node_id"],
            registry=self.registry,
        )

        # --- Coordinator metrics ---
        self.coordinator_queue_depth = Gauge(
            "distllm_coordinator_queue_depth",
            "Pending requests in the batch scheduler queue",
            registry=self.registry,
        )
        self.coordinator_active_requests = Gauge(
            "distllm_coordinator_active_requests",
            "Currently processing requests",
            registry=self.registry,
        )
        self.circuit_breaker_state = Gauge(
            "distllm_circuit_breaker_state",
            "Circuit breaker state per target node (0=closed, 1=open)",
            ["target_node"],
            registry=self.registry,
        )

        # --- Error metrics (with model/tenant labels) ---
        self.errors_total = Counter(
            "distllm_errors_total",
            "Total errors encountered",
            ["type", "model", "tenant"],
            registry=self.registry,
        )

        # --- Alerting-critical metrics ---
        self.kv_cache_usage_ratio = Gauge(
            "distllm_kv_cache_usage_ratio",
            "KV cache memory usage ratio (0.0-1.0)",
            ["node_id"],
            registry=self.registry,
        )
        self.request_duration_seconds = Histogram(
            "distllm_request_duration_seconds",
            "Request duration in seconds",
            ["method", "model", "tenant"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )
        self.active_nodes = Gauge(
            "distllm_active_nodes",
            "Number of active healthy nodes",
            registry=self.registry,
        )

        # --- Anomaly detection ---
        self.anomaly_detected_total = Counter(
            "distllm_anomaly_detected_total",
            "Total anomaly detection events",
            ["metric", "type"],
            registry=self.registry,
        )

        # --- Self-healing recovery metrics ---
        self.recovery_total = Counter(
            "distllm_recovery_total",
            "Total node recovery events triggered",
            registry=self.registry,
        )
        self.recovery_sequences_recovered = Counter(
            "distllm_recovery_sequences_recovered_total",
            "Total in-flight sequences recovered from failed nodes",
            registry=self.registry,
        )
        self.recovery_sequences_lost = Counter(
            "distllm_recovery_sequences_lost_total",
            "Total in-flight sequences lost due to node failure",
            registry=self.registry,
        )
        self.recovery_duration_ms = Histogram(
            "distllm_recovery_duration_ms",
            "Duration of node recovery in milliseconds",
            buckets=[10, 50, 100, 500, 1000, 5000, 10000, 30000],
            registry=self.registry,
        )
        self.draining_nodes = Gauge(
            "distllm_draining_nodes",
            "Number of nodes currently in draining state",
            registry=self.registry,
        )
        self.dead_nodes = Gauge(
            "distllm_dead_nodes",
            "Number of nodes marked as dead (awaiting replacement)",
            registry=self.registry,
        )

        # --- Cost tracking metrics ---
        self.cost_per_hour_total = Gauge(
            "distllm_cost_per_hour_total",
            "Total cost per hour for all active nodes",
            registry=self.registry,
        )
        self.budget_remaining = Gauge(
            "distllm_budget_remaining",
            "Remaining budget per hour",
            registry=self.registry,
        )
        self.spot_interruptions_total = Counter(
            "distllm_spot_interruptions_total",
            "Total spot instance interruptions",
            registry=self.registry,
        )
        self.request_cost_total = Counter(
            "distllm_request_cost_total",
            "Estimated $ cost per request",
            ["model", "tenant"],
            registry=self.registry,
        )
        self.request_gpu_hours = Counter(
            "distllm_request_gpu_hours",
            "GPU-hours consumed per request",
            ["model", "tenant"],
            registry=self.registry,
        )

    def record_anomaly(self, metric: str, type_: str = "threshold") -> None:
        """Record an anomaly-detection event (``distllm_anomaly_detected_total``).

        The intended producer is the API's anomaly hookup: every time the
        anomaly detector classifies a sample as anomalous for a metric, it
        reports the event here so the anomaly counter moves. Without a call
        site the family stays registered-but-empty.
        """
        # A duplicate label set or a torn-down registry must not break the
        # request path — the anomaly counter is advisory telemetry.
        with contextlib.suppress(Exception):
            self.anomaly_detected_total.labels(metric=metric, type=type_).inc()

    def _sync_counter(self, counter, key: str, value: float) -> None:
        """Bridge a subsystem's cumulative total into a Prometheus counter.

        Subsystems (recovery manager, spot ensemble) expose lifetime totals
        rather than counters; /metrics scrapes call this on every scrape, so
        only the growth since the previous scrape is added. A total that
        moved backwards (source restarted, its counter re-based) is treated
        as a fresh baseline and its current value added — never subtracted.
        """
        try:
            value = float(value or 0)
        except (TypeError, ValueError):
            return
        last = self._bridge_watermarks.get(key)
        if last is None or value < last:
            if value:
                counter.inc(value)
        elif value > last:
            counter.inc(value - last)
        self._bridge_watermarks[key] = value

    def populate_gauges(self, coordinator=None) -> None:
        """Populate gauge and bridged-counter metrics from coordinator state.

        Called on every /metrics scrape. Every read is defensive: coordinator
        stand-ins (tests, degraded deployments) may expose partial state, and
        a scrape must never fail because one subsystem is absent.
        """
        if coordinator is None:
            return
        scheduler = getattr(coordinator, "scheduler", None)
        if scheduler is not None:
            self.coordinator_queue_depth.set(getattr(scheduler, "pending_count", 0) or 0)
            self.coordinator_active_requests.set(getattr(scheduler, "active_count", 0) or 0)

        nodes = getattr(coordinator, "nodes", None)
        if not isinstance(nodes, dict):
            nodes = {}
        self.active_nodes.set(len(nodes))

        resource_mgr = getattr(coordinator, "_resource_mgr", None)
        tracker = getattr(coordinator, "_latency_tracker", None)
        cost_per_hour_total = 0.0

        for node_id, reg in nodes.items():
            cb_open = False
            if resource_mgr is not None and hasattr(resource_mgr, "check_circuit_breaker"):
                try:
                    cb_open = bool(resource_mgr.check_circuit_breaker(node_id))
                except Exception:
                    cb_open = False
            self.circuit_breaker_state.labels(target_node=str(node_id)).set(1 if cb_open else 0)

            if reg is None:
                continue

            # Health: NodeRegistration exposes `.healthy`, PipelineNode
            # exposes `.is_healthy`. The old guard checked a `health_status`
            # attribute that neither class defines, so distllm_node_health
            # was never written even after the wiring was made reachable.
            healthy = getattr(reg, "healthy", None)
            if healthy is None:
                healthy = getattr(reg, "is_healthy", None)
            if healthy is None:
                healthy = getattr(reg, "health_status", None)
            if healthy is not None:
                start = getattr(reg, "start_layer", None)
                end = getattr(reg, "end_layer", None)
                if start is not None and end is not None:
                    layer_range = f"{start}-{end}"
                else:
                    layer_range = str(getattr(reg, "layer_range", ""))
                self.node_health.labels(node_id=str(node_id), layer_range=layer_range).set(1 if healthy else 0)

            # GPU memory: used = total - free (Profile RPC / HealthCheck
            # populate both fields on the node classes).
            mem_total = getattr(reg, "gpu_memory_total", 0) or 0
            mem_free = getattr(reg, "gpu_memory_free", 0) or 0
            if mem_total > 0:
                used = max(mem_total - mem_free, 0)
                self.node_gpu_memory_bytes.labels(node_id=str(node_id)).set(used)
                self.node_gpu_memory_total_bytes.labels(node_id=str(node_id)).set(mem_total)
                # Prefer the node's own compute-utilization reading when it
                # carries one (the HealthCheck RPC's gpu_utilization field);
                # neither node class stores it today, so derive the
                # memory-pressure percentage as the working default.
                util = getattr(reg, "gpu_utilization", None)
                if util is None:
                    util = used / mem_total * 100.0
                self.node_gpu_utilization.labels(node_id=str(node_id)).set(float(util))

            # Per-node latency percentiles. Prefer the sliding-window
            # LatencyTracker; fall back to the pipeline node's last measured
            # hop latency (PipelineNode.latency_ms) when the tracker has no
            # samples for that node yet.
            values = None
            if tracker is not None and hasattr(tracker, "get_measurements"):
                try:
                    values = tracker.get_measurements(node_id)
                except Exception:
                    values = None
            if not values:
                last_ms = getattr(reg, "latency_ms", None)
                if last_ms:
                    values = [float(last_ms)]
            if values:
                ordered = sorted(values)
                p50 = ordered[max(0, int(len(ordered) * 0.50) - 1)]
                p99 = ordered[max(0, int(len(ordered) * 0.99) - 1)]
                self.node_latency_p50.labels(node_id=str(node_id)).set(p50)
                self.node_latency_p99.labels(node_id=str(node_id)).set(p99)

            cost_per_hour_total += float(getattr(reg, "cost_per_hour", 0.0) or 0.0)

        # Cost of the whole active fleet (NodeRegistration carries its
        # hourly price; PipelineNode does not, so this stays 0 for it).
        self.cost_per_hour_total.set(cost_per_hour_total)

        # KV cache pressure from the local paged-attention pools.
        get_backends = getattr(coordinator, "_get_paged_backends", None)
        if callable(get_backends):
            try:
                backends = get_backends() or []
            except Exception:
                backends = []
            for idx, mgr in enumerate(backends):
                try:
                    used_blocks = mgr.num_used_blocks
                    total_blocks = mgr._num_blocks
                except Exception:
                    continue
                if total_blocks:
                    label = getattr(mgr, "node_id", None) or ("local" if idx == 0 else f"local-{idx}")
                    self.kv_cache_usage_ratio.labels(node_id=str(label)).set(used_blocks / total_blocks)

        # Recovery metrics from the recovery manager. The previous code read
        # `coordinator._recovery`, which no Coordinator defines — the real
        # attribute is `_recovery_manager` — so every recovery family was
        # dead even after the health-status fix.
        recovery = getattr(coordinator, "_recovery_manager", None) or getattr(coordinator, "_recovery", None)
        if recovery is not None and hasattr(recovery, "get_metrics"):
            try:
                rec_metrics = recovery.get_metrics()
            except Exception:
                rec_metrics = None
            if isinstance(rec_metrics, dict):
                self.draining_nodes.set(rec_metrics.get("draining_nodes", 0))
                self.dead_nodes.set(rec_metrics.get("dead_nodes", 0))
                self._sync_counter(self.recovery_total, "recovery:total", rec_metrics.get("recoveries", 0))
                self._sync_counter(
                    self.recovery_sequences_recovered,
                    "recovery:sequences_recovered",
                    rec_metrics.get("sequences_recovered", 0),
                )
                self._sync_counter(
                    self.recovery_sequences_lost,
                    "recovery:sequences_lost",
                    rec_metrics.get("sequences_lost", 0),
                )
            if hasattr(recovery, "get_recovery_history"):
                try:
                    history = recovery.get_recovery_history()
                except Exception:
                    history = []
                if isinstance(history, list):
                    seen = int(self._bridge_watermarks.get("recovery:history_len", 0))
                    for entry in history[seen:]:
                        duration = entry.get("duration_ms") if isinstance(entry, dict) else None
                        if duration is not None:
                            self.recovery_duration_ms.observe(float(duration))
                    self._bridge_watermarks["recovery:history_len"] = len(history)

        # Spot interruptions from the spot ensemble manager.
        spot = getattr(coordinator, "_spot_ensemble", None)
        if spot is not None and hasattr(spot, "stats"):
            try:
                spot_stats = spot.stats
            except Exception:
                spot_stats = None
            if isinstance(spot_stats, dict):
                self._sync_counter(
                    self.spot_interruptions_total,
                    "spot:interruptions",
                    spot_stats.get("interruptions", 0),
                )

        # Remaining hourly budget across configured tenant budgets (only set
        # when at least one budget exists; no budgets means no value rather
        # than a misleading zero).
        try:
            from distllm.core.cost_tracker import get_cost_tracker

            tracker_obj = get_cost_tracker()
            budgets = getattr(tracker_obj, "_budgets", None)
            if isinstance(budgets, dict) and budgets:
                remaining = 0.0
                any_budget = False
                for tenant_id in list(budgets):
                    summary = tracker_obj.get_cost_summary(tenant_id)
                    budget = summary.get("budget", {}) if isinstance(summary, dict) else {}
                    max_hour = float(budget.get("max_cost_per_hour", 0.0) or 0.0)
                    if max_hour > 0:
                        any_budget = True
                        current = float(budget.get("current_hour_cost", 0.0) or 0.0)
                        remaining += max(0.0, max_hour - current)
                if any_budget:
                    self.budget_remaining.set(remaining)
        except Exception:
            pass

    def generate_metrics(self) -> bytes:
        """Generate Prometheus text exposition format."""
        if not _PROMETHEUS_AVAILABLE:
            return b""

        from prometheus_client import generate_latest

        return generate_latest(self.registry)
