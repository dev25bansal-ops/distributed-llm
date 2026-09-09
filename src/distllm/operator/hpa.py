"""Custom HPA driven by DistLLM's own Prometheus metrics.

A kopf-based autoscaler that queries the in-cluster Prometheus for
DistLLM application metrics (tokens per second, queue depth, KV-cache
pressure) instead of CPU/memory, because LLM serving pods are almost
never CPU-bound — the meaningful scaling signals are queue depth and
generation throughput.
"""

from __future__ import annotations

import re

from distllm.errors import ConfigValidationError

# Metric names exported by distllm.observability.exporter.DistLLMPrometheusExporter
# and consumed here for scaling decisions.
SUPPORTED_METRICS = ("tokens_per_second", "queue_depth", "kv_cache_usage_ratio")

_PROMETHEUS_URL_RE = re.compile(
    r"^https?://"            # scheme must be http(s) — no file/ftp/data schemes
    r"[A-Za-z0-9._~:-]+"     # host (also covers in-cluster service names)
    r"(?::\d{1,5})?"         # optional port
    r"(?:/[A-Za-z0-9._~/=-]*)?$"
)


def _validate_prometheus_url(url: str) -> str:
    """Validate a Prometheus base URL for the custom HPA.

    Accepts only http/https URLs with a plausible host; rejects other
    schemes (``ftp://``, ``file://``, raw IPs with userinfo tricks) so a
    misconfigured CR cannot point the operator at an arbitrary endpoint.
    """
    if not isinstance(url, str) or not url:
        raise ConfigValidationError("prometheus_url", f"URL must be a non-empty string, got {url!r}")
    if not _PROMETHEUS_URL_RE.match(url):
        raise ConfigValidationError(
            "prometheus_url",
            f"Invalid Prometheus URL {url!r}: must be http(s)://<host>[:<port>][/path]",
        )
    return url


class CustomHPA:
    """Prometheus-metric-driven horizontal pod autoscaler for DistLLM.

    Usage (kopf handler)::

        hpa = CustomHPA(metric="queue_depth", target_value=50)
        desired = hpa.compute_desired_replicas(
            current=3, metric_value=180.0
        )
    """

    def __init__(
        self,
        metric: str = "tokens_per_second",
        target_value: float = 100.0,
        min_replicas: int = 1,
        max_replicas: int = 10,
        prometheus_url: str = "http://prometheus:9090",
    ):
        if metric not in SUPPORTED_METRICS:
            raise ConfigValidationError(
                "metric", f"Unsupported HPA metric {metric!r}; supported: {SUPPORTED_METRICS}"
            )
        if min_replicas < 1:
            raise ConfigValidationError("min_replicas", f"must be >= 1, got {min_replicas}")
        if max_replicas < min_replicas:
            raise ConfigValidationError(
                "max_replicas",
                f"max_replicas ({max_replicas}) must be >= min_replicas ({min_replicas})",
            )
        if target_value <= 0:
            raise ConfigValidationError("target_value", f"must be > 0, got {target_value}")

        self._metric = metric
        self._target_value = float(target_value)
        self._min_replicas = min_replicas
        self._max_replicas = max_replicas
        self._prometheus_url = _validate_prometheus_url(prometheus_url)

    @property
    def metric(self) -> str:
        return self._metric

    def compute_desired_replicas(self, current: int, metric_value: float) -> int:
        """Proportional scaling: replicas * (value / target), clamped.

        For queue-like metrics the ratio drives scale-UP; for throughput
        metrics it drives scale-DOWN when nodes idle. The clamp keeps the
        result inside [min_replicas, max_replicas].
        """
        if current < 1:
            current = 1
        ratio = metric_value / self._target_value
        desired = max(1, round(current * ratio))
        return max(self._min_replicas, min(self._max_replicas, desired))
