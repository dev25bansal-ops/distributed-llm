"""Prometheus metrics exporter for distributed-llm.

Provides a CollectorRegistry with all standard distllm metrics
and a FastAPI-compatible /metrics response via generate_latest.
"""

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge


class DistLLMPrometheusExporter:
    """Prometheus metrics collector for all distributed-llm components."""

    def __init__(self):
        self.registry = CollectorRegistry()

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

    def populate_gauges(self, coordinator=None) -> None:
        """Populate gauge metrics from current coordinator state."""
        if coordinator is None:
            return
        scheduler = getattr(coordinator, "scheduler", None)
        if scheduler is not None:
            self.coordinator_queue_depth.set(getattr(scheduler, "pending_count", 0))
            self.coordinator_active_requests.set(getattr(scheduler, "active_count", 0))
        nodes = getattr(coordinator, "nodes", {}) or {}
        self.active_nodes.set(len(nodes))
        for node_id in nodes:
            cb = coordinator._check_circuit_breaker(node_id) if hasattr(coordinator, "_check_circuit_breaker") else False
            self.circuit_breaker_state.labels(target_node=node_id).set(1 if cb else 0)
            reg = nodes.get(node_id)
            if reg and hasattr(reg, "health_status"):
                self.node_health.labels(node_id=node_id, layer_range=str(getattr(reg, "layer_range", ""))).set(1 if reg.health_status else 0)

        # Populate recovery metrics from the recovery manager
        recovery = getattr(coordinator, "_recovery", None)
        if recovery is not None:
            rec_metrics = recovery.get_metrics()
            self.draining_nodes.set(rec_metrics.get("draining_nodes", 0))
            self.dead_nodes.set(rec_metrics.get("dead_nodes", 0))

    def generate_metrics(self) -> bytes:
        """Generate Prometheus text exposition format."""
        from prometheus_client import generate_latest

        return generate_latest(self.registry)
