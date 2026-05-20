"""OpenTelemetry tracing setup for distributed LLM.

Provides trace provider setup, W3C trace context propagation across
distributed nodes, and request-to-trace correlation helpers.
"""

from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ALWAYS_ON,
    ALWAYS_OFF,
    ParentBased,
)
try:
    from opentelemetry.instrumentation.grpc import GRPCInstrumentor
except ImportError:
    GRPCInstrumentor = None  # type: ignore[misc,assignment]
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
try:
    from opentelemetry.propagators.textmap import DictGetter
except ImportError:
    DictGetter = None  # type: ignore[misc,assignment]


# W3C TraceContext propagator for cross-node trace correlation
_tracecontext_propagator = TraceContextTextMapPropagator()


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

    if GRPCInstrumentor is not None:
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


def inject_trace_context(metadata: dict | None = None) -> dict:
    """Inject W3C trace context into outgoing gRPC metadata.

    Propagates the current span's traceparent/tracestate headers
    so that downstream nodes join the same trace.

    Args:
        metadata: Existing metadata dict to inject into.

    Returns:
        Updated metadata dict with traceparent and tracestate.
    """
    if metadata is None:
        metadata = {}
    carrier = {}
    _tracecontext_propagator.inject(carrier)
    for key, value in carrier.items():
        metadata[key] = value
    return metadata


def extract_trace_context(metadata: dict) -> None:
    """Extract W3C trace context from incoming gRPC metadata.

    Sets the current OpenTelemetry context to the parent trace
    so that new spans are linked to the distributed trace.

    Args:
        metadata: Incoming metadata dict with traceparent/tracestate.
    """
    if DictGetter is None:
        return
    ctx = _tracecontext_propagator.extract(DictGetter(metadata))
    if ctx is not None:
        context.attach(ctx)


def get_current_trace_id() -> str | None:
    """Get the current trace ID as a hex string.

    Returns:
        Trace ID (32 hex chars) or None if no active span.
    """
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_current_span_id() -> str | None:
    """Get the current span ID as a hex string.

    Returns:
        Span ID (16 hex chars) or None if no active span.
    """
    span = trace.get_current_span()
    if span is not None and span.is_recording():
        return format(span.get_span_context().span_id, "016x")
    return None
