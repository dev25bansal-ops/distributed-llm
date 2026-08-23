"""OpenTelemetry instrumentation for the DistLLM SDK.

Provides optional distributed tracing for all API calls.
Enable by installing ``distllm-sdk[otel]`` and configuring
an OTel exporter::

    from distllm_sdk import DistLLMClient
    from distllm_sdk.tracing import setup_tracing

    setup_tracing(service_name="my-app")
    client = DistLLMClient(...)

Spans are created for every ``_request`` / ``_request_raw`` call
with attributes for method, path, status code, and latency.
"""

from __future__ import annotations

from typing import Any

_OTEL_AVAILABLE = False
_tracer_provider: Any = None
_tracer: Any = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource

    _OTEL_AVAILABLE = True
except ImportError:
    trace = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment,misc]
    BatchSpanProcessor = None  # type: ignore[assignment,misc]
    OTLPSpanExporter = None  # type: ignore[assignment,misc]
    Resource = None  # type: ignore[assignment,misc]


def setup_tracing(
    service_name: str = "distllm-sdk",
    otlp_endpoint: str | None = None,
    resource_attributes: dict[str, str] | None = None,
) -> bool:
    """Configure OpenTelemetry tracing for the SDK.

    Args:
        service_name: Service name for the traced application.
        otlp_endpoint: OTLP HTTP exporter endpoint (e.g. ``http://localhost:4318``).
                        Falls back to ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.
        resource_attributes: Additional resource attributes.

    Returns:
        True if tracing was enabled, False if opentelemetry is not installed.
    """
    global _tracer_provider, _tracer

    if not _OTEL_AVAILABLE:
        return False

    import os as _os

    attrs = {"service.name": service_name, **(resource_attributes or {})}
    resource = Resource.create(attrs)

    provider = TracerProvider(resource=resource)
    endpoint = otlp_endpoint or _os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    _tracer_provider = provider
    _tracer = trace.get_tracer(service_name)
    return True


def shutdown_tracing() -> None:
    """Flush and shut down the OTel tracer provider."""
    if _tracer_provider is not None:
        _tracer_provider.shutdown()


def get_tracer() -> Any:
    """Return the current tracer, or a no-op tracer if OTel is not configured."""
    if _tracer is not None:
        return _tracer
    if _OTEL_AVAILABLE and trace is not None:
        return trace.get_tracer("distllm-sdk")
    return None
