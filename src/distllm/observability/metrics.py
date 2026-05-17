"""OpenTelemetry metrics for distributed LLM."""

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, MetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import Histogram, Counter


def setup_metrics(
    service_name: str = "distllm",
    endpoint: str | None = None,
    exporter: MetricExporter = None,
) -> metrics.MeterProvider:
    """Initialize OpenTelemetry metrics.

    Args:
        service_name: Name of the service for metric resources.
        endpoint: Optional OTLP exporter endpoint.
        exporter: Optional custom metric exporter.

    Returns:
        Configured MeterProvider.
    """
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

    def record_node_latency(self, node_id: str, duration: float):
        """Record latency for a node gRPC call."""
        self.node_latency.record(duration, {"node_id": node_id})

    def record_generation(self, duration: float, num_tokens: int):
        """Record a complete generation event."""
        self.generation_duration.record(duration)
        self.tokens_generated.add(num_tokens)
