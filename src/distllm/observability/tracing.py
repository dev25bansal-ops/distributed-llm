"""OpenTelemetry tracing setup for distributed LLM."""

from typing import Optional, List, Tuple

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.grpc import GRPCInstrumentor


def setup_tracing(
    service_name: str = "distllm",
    endpoint: Optional[str] = None,
    exporters: Optional[List[SpanExporter]] = None,
) -> trace.TracerProvider:
    """Initialize OpenTelemetry tracing with gRPC instrumentation.

    Args:
        service_name: Name of the service for trace resources.
        endpoint: Optional OTLP exporter endpoint (e.g. "http://localhost:4317").
        exporters: Optional list of custom span exporters.

    Returns:
        Configured TracerProvider.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
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


def extract_request_id(metadata: list) -> Optional[str]:
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
