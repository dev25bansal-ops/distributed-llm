"""OpenTelemetry metrics for distributed LLM.

The OTel SDK lives in the ``observability`` extra. Importing it unconditionally
made `import distllm.api.server` fail with

    ModuleNotFoundError: No module named 'opentelemetry.sdk'

for anyone who installed ``distributed-llm[cli,self-hosted]``, because
``server.py`` imports this package at module scope -- so the console script
`distllm-api` was installed but could not start. The SDK is optional here:
the module now imports cleanly and raises a message naming the extra only if
`setup_metrics` is actually called without it.
"""

from __future__ import annotations

try:  # opentelemetry-api may be present without the SDK
    from opentelemetry import metrics
except ImportError:  # pragma: no cover - no OTel at all
    metrics = None  # type: ignore[assignment]

try:
    from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
    from opentelemetry.sdk.metrics.export import (
        MetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource

    _HAS_SDK = True
except ImportError:  # pragma: no cover - the common case without the extra
    _HAS_SDK = False
    Counter = Histogram = MeterProvider = object  # type: ignore[assignment,misc]
    MetricExporter = object  # type: ignore[assignment,misc]
    PeriodicExportingMetricReader = object  # type: ignore[assignment,misc]
    Resource = None  # type: ignore[assignment]


_MISSING_SDK_MESSAGE = (
    "OpenTelemetry metrics require the SDK: "
    'pip install "distributed-llm[observability]"'
)


def setup_metrics(
    service_name: str = "distllm",
    endpoint: str | None = None,
    exporter: MetricExporter = None,
) -> "metrics.MeterProvider":
    """Initialize OpenTelemetry metrics.

    Args:
        service_name: Name of the service for metric resources.
        endpoint: Optional OTLP exporter endpoint.
        exporter: Optional custom metric exporter.

    Returns:
        Configured MeterProvider.

    Raises:
        RuntimeError: when the observability extra is not installed.
    """
    if not _HAS_SDK:
        raise RuntimeError(_MISSING_SDK_MESSAGE)

    resource = Resource.create({"service.name": service_name})

    readers = []
    if endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        otlp_exporter = OTLPMetricExporter(endpoint=endpoint)
        readers.append(PeriodicExportingMetricReader(otlp_exporter))

    if exporter:
        readers.append(PeriodicExportingMetricReader(exporter))

    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    return provider


def get_meter():
    """Get the default meter for distllm."""
    return metrics.get_meter("distllm")


class DistLLMMetrics:
    """Pre-defined OpenTelemetry metrics for distributed LLM."""

    def __init__(self):
        meter = get_meter()

        self.node_latency: Histogram = meter.create_histogram(
            name="distllm_node_latency_seconds",
            description="Latency of gRPC calls to worker nodes",
            unit="s",
        )

        self.generation_duration: Histogram = meter.create_histogram(
            name="distllm_generation_duration_seconds",
            description="End-to-end text generation duration",
            unit="s",
        )

        self.tokens_generated: Counter = meter.create_counter(
            name="distllm_tokens_generated_total",
            description="Total number of tokens generated",
            unit="1",
        )

        # Draft model metrics (speculative decoding)
        self.draft_calls: Counter = meter.create_counter(
            name="distllm_draft_calls_total",
            description="Total remote draft model calls",
            unit="1",
        )

        self.draft_latency: Histogram = meter.create_histogram(
            name="distllm_draft_latency_seconds",
            description="Latency of remote draft model calls",
            unit="s",
        )

        self.draft_acceptance: Histogram = meter.create_histogram(
            name="distllm_draft_acceptance_rate",
            description="Draft token acceptance rate",
            unit="1",
        )

    def record_node_latency(self, node_id: str, duration: float):
        """Record latency for a node gRPC call."""
        self.node_latency.record(duration, {"node_id": node_id})

    def record_generation(self, duration: float, num_tokens: int):
        """Record a complete generation event."""
        self.generation_duration.record(duration)
        self.tokens_generated.add(num_tokens)

    def record_draft_call(self, duration: float, accepted: int, drafted: int, error: bool = False):
        """Record a remote draft model call.

        Args:
            duration: Round-trip latency in seconds.
            accepted: Number of tokens accepted by target.
            drafted: Number of tokens drafted.
            error: Whether the call failed.
        """
        result = "error" if error else "success"
        self.draft_calls.add(1, {"result": result})
        self.draft_latency.record(duration)
        if drafted > 0:
            self.draft_acceptance.record(accepted / drafted)
