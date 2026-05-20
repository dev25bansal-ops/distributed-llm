"""Metrics manager for the Coordinator facade.

Handles metric recording, retrieval, and Prometheus-format export.
Extracted from the Coordinator class.
"""

import threading


class MetricsManager:
    """Thread-safe metrics recording and reporting.

    Attributes:
        _metrics: Dict of metric name -> value.
        _metrics_lock: Threading lock for concurrent access.
    """

    def __init__(self):
        self._metrics: dict[str, float] = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_generation_time": 0.0,
            "errors": 0,
            "node_failures": 0,
        }
        self._metrics_lock = threading.Lock()

    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value (thread-safe).

        Args:
            metric_name: Name of the metric.
            value: Value to add (accumulated for numeric metrics).
        """
        with self._metrics_lock:
            if metric_name in self._metrics:
                if isinstance(self._metrics[metric_name], (int, float)):
                    self._metrics[metric_name] += value
                else:
                    self._metrics[metric_name] = value
            else:
                self._metrics[metric_name] = value

    def get(self) -> dict[str, float]:
        """Get a copy of current metrics (thread-safe).

        Returns:
            Dict of metric name -> value.
        """
        with self._metrics_lock:
            return dict(self._metrics)

    def get_prometheus(self) -> dict[str, float]:
        """Get current metrics in Prometheus-compatible format (thread-safe).

        Returns:
            Dict of Prometheus metric name -> value.
        """
        with self._metrics_lock:
            avg_tokens_per_sec = 0.0
            gen_time = self._metrics.get("total_generation_time", 0)
            tokens = self._metrics.get("total_tokens_generated", 0)
            if gen_time > 0 and tokens > 0:
                avg_tokens_per_sec = tokens / gen_time

            return {
                "distllm_requests_total": self._metrics["total_requests"],
                "distllm_tokens_generated_total": self._metrics["total_tokens_generated"],
                "distllm_generation_time_seconds_total": round(self._metrics["total_generation_time"], 3),
                "distllm_errors_total": self._metrics["errors"],
                "distllm_node_failures_total": self._metrics["node_failures"],
                "distllm_avg_tokens_per_second": round(avg_tokens_per_sec, 2),
            }

    def increment(self, metric_name: str, value: float = 1.0) -> None:
        """Increment a metric counter (thread-safe).

        Args:
            metric_name: Name of the metric.
            value: Amount to increment by (default 1.0).
        """
        self.record(metric_name, value)
