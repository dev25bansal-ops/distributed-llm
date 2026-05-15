"""Prometheus metrics exporter for distributed-llm.

Provides a CollectorRegistry with all standard distllm metrics
and a FastAPI-compatible /metrics response via generate_latest.
"""

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge


class DistLLMPrometheusExporter:
    """Prometheus metrics collector for all distributed-llm components."""

    def __init__(self):
        self.registry = CollectorRegistry()

        # --- Request metrics ---
        self.requests_total = Counter(
            "distllm_requests_total",
            "Total requests processed",
            ["method", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "distllm_request_latency_seconds",
            "End-to-end request latency",
            ["method"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry,
        )

        # --- Token generation metrics ---
        self.tokens_generated = Counter(
            "distllm_tokens_generated_total",
            "Total tokens generated",
            registry=self.registry,
        )
        self.token_latency = Histogram(
            "distllm_token_generation_latency_seconds",
            "Time to generate a single token",
            buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry,
        )
        self.tokens_per_second = Gauge(
            "distllm_tokens_per_second",
            "Current token generation rate",
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

        # --- Error metrics ---
        self.errors_total = Counter(
            "distllm_errors_total",
            "Total errors encountered",
            ["type"],
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
            ["method"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )
        self.active_nodes = Gauge(
            "distllm_active_nodes",
            "Number of active healthy nodes",
            registry=self.registry,
        )

        # --- Cost metrics (Feature 18) ---
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

    def generate_metrics(self) -> bytes:
        """Generate Prometheus text exposition format."""
        from prometheus_client import generate_latest

        return generate_latest(self.registry)
