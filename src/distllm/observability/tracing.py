"""OpenTelemetry tracing setup for distributed LLM."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ALWAYS_ON,
    ALWAYS_OFF,
    ParentBased,
)
from opentelemetry.instrumentation.grpc import GRPCInstrumentor


def _get_sampler(
    sampling_strategy: str = "head",
    sampling_ratio: float = 1.0,
):
    """Select an OpenTelemetry sampler based on strategy.

    Args:
        sampling_strategy: "head", "tail", "always", or "none".
        sampling_ratio: Probability for head/tail sampling (0.0-1.0).

    Returns:
        An OpenTelemetry Sampler instance.
    """
    if sampling_strategy == "always":
        return ALWAYS_ON
    if sampling_strategy == "none":
        return ALWAYS_OFF
    if sampling_strategy == "head":
        return ParentBased(TraceIdRatioBased(sampling_ratio))
    if sampling_strategy == "tail":
        # Tail-based: always sample errors, sample others at ratio
        return ParentBased(
            root=TraceIdRatioBased(sampling_ratio),
            remote_sampled=ALWAYS_ON,
            remote_not_sampled=TraceIdRatioBased(sampling_ratio),
        )
    # Default: sample everything
    return ALWAYS_ON


def setup_tracing(
    service_name: str = "distllm",
    endpoint: str | None = None,
    exporters: list[SpanExporter] | None = None,
    sampling_strategy: str = "head",
    sampling_ratio: float = 1.0,
) -> trace.TracerProvider:
    """Initialize OpenTelemetry tracing with gRPC instrumentation.

    Args:
        service_name: Name of the service for trace resources.
        endpoint: Optional OTLP exporter endpoint (e.g. "http://localhost:4317").
        exporters: Optional list of custom span exporters.
        sampling_strategy: Sampling strategy — "head", "tail", "always", "none".
        sampling_ratio: Sampling probability (0.0-1.0) for head/tail strategies.

    Returns:
        Configured TracerProvider.
    """
    resource = Resource.create({"service.name": service_name})
    sampler = _get_sampler(sampling_strategy, sampling_ratio)
    provider = TracerProvider(resource=resource, sampler=sampler)
    trace.set_tracer_provider(provider)

    if endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if exporters:
        for exporter in exporters:
            provider.add_span_processor(BatchSpanProcessor(exporter))

    GRPCInstrumentor().instrument()

    return provider


def inject_request_id(metadata: list, request_id: str) -> list:
    """Add x-request-id to gRPC metadata for trace correlation.

    Args:
        metadata: Existing gRPC metadata list of (key, value) tuples.
        request_id: Unique request identifier.

    Returns:
        Updated metadata list with x-request-id appended.
    """
    if metadata is None:
        metadata = []
    metadata.append(("x-request-id", request_id))
    return metadata


def extract_request_id(metadata: list) -> str | None:
    """Extract x-request-id from gRPC metadata.

    Args:
        metadata: gRPC metadata list of (key, value) tuples.

    Returns:
        Request ID if present, else None.
    """
    for key, value in metadata:
        if key == "x-request-id":
            return value
    return None
