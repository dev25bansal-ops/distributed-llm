"""OpenTelemetry instrumentation for the distributed inference pipeline.

Provides context-propagation-aware :func:`trace` wrappers for the key
subsystems (pipeline, federation, recovery) so that a single end-user
request generates a coherent span tree across all cluster hops.

Usage::

    from distllm.dist.tracing import Tracer

    tracer = Tracer(service_name="distllm-coordinator")
    tracer.instrument_pipeline()
    tracer.instrument_federation()
    tracer.instrument_recovery()

    with tracer.start_span("inference_request") as span:
        span.set_attribute("model", "llama-70b")
        result = pipeline.run(...)

Design
------
- Lazy initialisation: no import-time side effects.
- Graceful degradation: when ``opentelemetry`` is not installed all
  methods become no-ops (zero overhead).
- Propagates trace context via W3C ``traceparent`` header so federated
  clusters see the same trace across cluster boundaries.
"""

from __future__ import annotations

import asyncio
import functools
import os
from typing import Any, Callable

from loguru import logger

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import SpanKind
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

    # Stub types so the module compiles without opentelemetry installed.
    class SpanKind:  # type: ignore[no-redef]
        INTERNAL = 0
        SERVER = 1
        CLIENT = 2
        PRODUCER = 3
        CONSUMER = 4

    class _FakeSpan:
        def set_attribute(self, key: str, value: Any) -> None: ...
        def add_event(self, name: str, attributes: dict | None = None) -> None: ...
        def set_status(self, status: Any) -> None: ...
        def end(self) -> None: ...

        def __enter__(self): return self
        def __exit__(self, *args): ...

    class _FakeTracer:
        def start_span(self, name: str, **kwargs) -> _FakeSpan:
            return _FakeSpan()
        def start_as_current_span(self, name: str, **kwargs) -> _FakeSpan:
            return _FakeSpan()

    class _FakeContextPropagator:
        def inject(self, carrier: dict, context: Any | None = None) -> None: ...
        def extract(self, carrier: dict, context: Any | None = None) -> Any: ...

    TraceContextTextMapPropagator = _FakeContextPropagator  # type: ignore[assignment, misc]

_DEFAULT_OTLP_ENDPOINT = os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "http://localhost:4318/v1/traces",
)


class Tracer:
    """OpenTelemetry tracer for distributed inference subsystems.

    Attributes:
        provider: The configured :class:`TracerProvider`.
        tracer: The named :class:`trace.Tracer` instance.
        propagator: W3C TraceContext propagator for cross-cluster context.
    """

    def __init__(
        self,
        service_name: str = "distllm-node",
        otlp_endpoint: str = _DEFAULT_OTLP_ENDPOINT,
        console_export: bool = False,
        resource_attributes: dict[str, Any] | None = None,
    ):
        self._service_name = service_name
        self._instrumented: set[str] = set()
        self.propagator = TraceContextTextMapPropagator()

        if not _OTEL_AVAILABLE:
            self.tracer = _FakeTracer()  # type: ignore[assignment]
            self.provider = None
            logger.debug(
                "OpenTelemetry not installed — tracing is a no-op. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-http"
            )
            return

        resource = Resource.create({
            "service.name": service_name,
            **(resource_attributes or {}),
        })
        self.provider = TracerProvider(resource=resource)

        # OTLP exporter (default: local OTel collector).
        self.provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )

        # Optional console output for local debugging.
        if console_export:
            self.provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )

        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(service_name)

    # ── Span helpers ──────────────────────────────────────────────────

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Create and return a new span (manual lifecycle)."""
        span = self.tracer.start_span(name, kind=kind, attributes=attributes)
        return span

    @property
    def current_span(self) -> Any:
        """Return the currently active span (or a no-op span)."""
        if _OTEL_AVAILABLE:
            return trace.get_current_span()
        return _FakeSpan()

    # ── Decorator ─────────────────────────────────────────────────────

    def trace(
        self,
        span_name: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator that wraps a function in an OpenTelemetry span.

        Usage::

            tracer = Tracer()

            @tracer.trace("federated_forward")
            async def forward_to_peer(request):
                ...

        When ``span_name`` is ``None`` the decorated function's
        ``__qualname__`` is used.
        """
        def decorator(func: Callable) -> Callable:
            name = span_name or func.__qualname__

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(name, kind=kind, attributes=attributes):
                    return func(*args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(name, kind=kind, attributes=attributes):
                    return await func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        return decorator

    # ── Subsystem instrumentations ─────────────────────────────────────

    def instrument_pipeline(self) -> None:
        """Patch :class:`AsyncPipelineEngine` methods with tracing spans.

        Idempotent — safe to call multiple times.
        """
        key = "pipeline"
        if key in self._instrumented:
            return

        try:
            from distllm.dist.async_pipeline import AsyncPipelineEngine

            self._patch_method(AsyncPipelineEngine, "forward", "pipeline.forward",
                               SpanKind.CLIENT)
            self._patch_method(AsyncPipelineEngine, "forward_stage", "pipeline.stage",
                               SpanKind.INTERNAL)

            self._instrumented.add(key)
            logger.debug("Tracing instrumented: pipeline")
        except ImportError as e:
            logger.warning(f"Failed to instrument pipeline: {e}")

    def instrument_federation(self) -> None:
        """Patch :class:`FederationCoordinator` methods with tracing spans.

        Propagates trace context via W3C ``traceparent`` header so that
        downstream peered clusters see the same trace across cluster
        boundaries.
        """
        key = "federation"
        if key in self._instrumented:
            return

        try:
            from distllm.dist.federation import FederationCoordinator

            self._patch_method(FederationCoordinator, "forward_request",
                               "federation.forward", SpanKind.CLIENT)
            self._patch_method(FederationCoordinator, "forward_request_streaming",
                               "federation.forward_streaming", SpanKind.CLIENT)
            self._patch_method(FederationCoordinator, "forward_with_cache_affinity",
                               "federation.forward_cached", SpanKind.CLIENT)
            self._patch_method(FederationCoordinator, "check_peer_health",
                               "federation.health_check", SpanKind.CLIENT)

            self._instrumented.add(key)
            logger.debug("Tracing instrumented: federation")
        except ImportError as e:
            logger.warning(f"Failed to instrument federation: {e}")

    def instrument_recovery(self) -> None:
        """Patch :class:`NodeRecoveryManager` methods with tracing spans."""
        key = "recovery"
        if key in self._instrumented:
            return

        try:
            from distllm.dist.recovery import NodeRecoveryManager

            self._patch_method(NodeRecoveryManager, "on_node_failure",
                               "recovery.on_failure", SpanKind.INTERNAL)
            self._patch_method(NodeRecoveryManager, "save_checkpoint",
                               "recovery.save_checkpoint", SpanKind.INTERNAL)
            self._patch_method(NodeRecoveryManager, "save_to_disk",
                               "recovery.persist_checkpoints", SpanKind.INTERNAL)
            self._patch_method(NodeRecoveryManager, "load_from_disk",
                               "recovery.load_checkpoints", SpanKind.INTERNAL)

            self._instrumented.add(key)
            logger.debug("Tracing instrumented: recovery")
        except ImportError as e:
            logger.warning(f"Failed to instrument recovery: {e}")

    def instrument_nccl(self) -> None:
        """Patch :class:`NcclTransport` methods with tracing spans."""
        key = "nccl"
        if key in self._instrumented:
            return

        try:
            from distllm.dist.nccl import NcclTransport

            self._patch_method(NcclTransport, "send", "nccl.send", SpanKind.CLIENT)
            self._patch_method(NcclTransport, "recv", "nccl.recv", SpanKind.CLIENT)
            self._patch_method(NcclTransport, "all_reduce", "nccl.all_reduce",
                               SpanKind.CLIENT)
            self._patch_method(NcclTransport, "broadcast", "nccl.broadcast",
                               SpanKind.CLIENT)
            self._patch_method(NcclTransport, "all_gather", "nccl.all_gather",
                               SpanKind.CLIENT)

            self._instrumented.add(key)
            logger.debug("Tracing instrumented: nccl")
        except ImportError as e:
            logger.warning(f"Failed to instrument nccl: {e}")

    def instrument_all(self) -> None:
        """Convenience: instrument all known subsystems."""
        self.instrument_pipeline()
        self.instrument_federation()
        self.instrument_recovery()
        self.instrument_nccl()

    # ── Internal ──────────────────────────────────────────────────────

    def _patch_method(self, cls: type, method_name: str,
                      span_name: str, kind: SpanKind) -> None:
        """Monkey-patch *method_name* on *cls* with a tracing wrapper."""
        original = getattr(cls, method_name, None)
        if original is None:
            logger.warning(f"Method {cls.__name__}.{method_name} not found, skipping")
            return

        tracer = self.tracer
        is_async = asyncio.iscoroutinefunction(original)

        if is_async:

            async def traced(self, *args, **kwargs):  # type: ignore[no-redef]
                with tracer.start_as_current_span(span_name, kind=kind):
                    return await original(self, *args, **kwargs)
        else:

            def traced(self, *args, **kwargs):  # type: ignore[no-redef]
                with tracer.start_as_current_span(span_name, kind=kind):
                    return original(self, *args, **kwargs)

        traced.__name__ = method_name
        traced.__qualname__ = f"{cls.__name__}.{method_name}"
        setattr(cls, method_name, traced)
