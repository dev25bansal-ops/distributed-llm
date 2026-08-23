"""W3C Trace-Context compliant tracing middleware for FastAPI/Starlette.

Parses the incoming ``traceparent`` header to join an existing distributed
trace, or generates a new trace ID for root spans.  Creates a root span
for every request and emits sub-span markers for the auth, rate-limiting,
and inference phases.

Integration with OpenTelemetry
------------------------------
The middleware first attempts to use the project's existing OpenTelemetry
setup (``distllm.observability.tracing``).  If OTel is not installed or not
configured (e.g. in lightweight test scenarios), it falls back to a pure
Python implementation that produces the same trace context format and
response headers.

State interface (``request.state``)
-----------------------------------
After dispatch, the following attributes are available to downstream
middleware and route handlers::

    request.state.trace_id       # 32-char hex string
    request.state.span_id        # 16-char hex string (root span ID)
    request.state.trace_flags    # 2-char hex string (01 = sampled)
    request.state.traceparent    # full traceparent header value
    request.state.tracer         # SpanFactory for creating child spans
    request.state.root_span      # root span object (OTel Span or _Span)

W3C ``traceparent`` format
---------------------------
``{version:02x}-{trace_id:032x}-{parent_span_id:016x}-{trace_flags:02x}``

- **version**: ``00`` (current specification)
- **trace_id**: 16 random bytes, hex-encoded (128-bit)
- **parent_span_id**: 8 random bytes, hex-encoded (64-bit)
- **trace_flags**: ``01`` (sampled) or ``00`` (not sampled)

Response header
---------------
Every response includes the ``traceparent`` header so downstream services
can continue the trace.

Registration
------------
Register this middleware **last** (after every other ``add_middleware`` call)
so it wraps the entire pipeline and captures the full request lifecycle::

    app.add_middleware(...)                # all existing middleware
    app.add_middleware(BodyCacheMiddleware)
    app.add_middleware(TracingMiddleware)  # outermost — full trace coverage
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


# ---------------------------------------------------------------------------
# Pure-Python fallback types (used when OpenTelemetry is unavailable)
# ---------------------------------------------------------------------------


@dataclass
class _SpanEvent:
    """A named event attached to a span at a point in time."""

    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Span:
    """Minimal span implementation for the non-OTel fallback path.

    Tracks timing, attributes, and child events.  The interface deliberately
    mirrors a subset of ``opentelemetry.trace.Span`` so that the middleware
    can switch backends transparently.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_time: float  # monotonic clock
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[_SpanEvent] = field(default_factory=list)

    # ── OTel-compatible helpers ──────────────────────────────────────

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a key-value attribute on this span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record a named event with optional attributes."""
        self.events.append(
            _SpanEvent(name=name, timestamp=time.monotonic(), attributes=attributes or {})
        )

    def end(self) -> None:
        """End the span, recording the wall-clock end time."""
        self.end_time = time.monotonic()

    def is_recording(self) -> bool:
        """Return ``True`` while the span has not been ended."""
        return self.end_time is None

    # ── Introspection helpers ────────────────────────────────────────

    def duration_ms(self) -> float:
        """Return the span's elapsed wall-clock time in milliseconds."""
        end = self.end_time if self.end_time is not None else time.monotonic()
        return (end - self.start_time) * 1000.0


class _SpanFactory(Protocol):
    """Factory protocol matching ``opentelemetry.trace.Tracer.start_span``."""

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = ...,
        parent_span_id: str | None = ...,
    ) -> _Span: ...


class _SimpleTracer:
    """Pure-Python tracer that creates ``_Span`` instances.

    This is a near-zero-overhead factory that emits structured trace data
    to the log at span end.  It is designed for environments where the
    full OpenTelemetry SDK is not available (unit tests, lightweight
    deployments, etc.).
    """

    def __init__(self, service_name: str = "distllm-api") -> None:
        self._service_name = service_name

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
    ) -> _Span:
        """Create and return a new ``_Span``.

        Args:
            name: Span name (e.g. ``"auth"``, ``"inference"``).
            attributes: Optional key-value pairs to attach immediately.
            parent_span_id: 16-char hex parent ID, or ``None`` for root.

        Returns:
            A started ``_Span`` instance.
        """
        span = _Span(
            name=name,
            trace_id="",  # will be set by the caller or attached via parent
            span_id=secrets.token_hex(8),
            parent_span_id=parent_span_id,
            start_time=time.monotonic(),
            attributes=dict(attributes or {}),
        )
        return span


# ---------------------------------------------------------------------------
# OpenTelemetry integration (lazy import — graceful degradation)
# ---------------------------------------------------------------------------

_OTEL_AVAILABLE: bool = False
_otel_trace: Any = None
_otel_propagator: Any = None

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[no-redef]
    from opentelemetry.trace.propagation.tracecontext import (  # type: ignore[no-redef]
        TraceContextTextMapPropagator,
    )

    _otel_propagator = TraceContextTextMapPropagator()
    _OTEL_AVAILABLE = True
except ImportError:
    _otel_trace = None
    _otel_propagator = None


# W3C Trace-Context constants
_TRACEPARENT_VERSION: str = "00"
_TRACE_FLAGS_SAMPLED: str = "01"
_TRACE_FLAGS_NOT_SAMPLED: str = "00"


# ---------------------------------------------------------------------------
# Helper: parse / generate trace context
# ---------------------------------------------------------------------------


def _parse_traceparent(header: str) -> dict[str, str] | None:
    """Parse a W3C ``traceparent`` header into its component fields.

    Expected format::

        ``00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01``

    Returns a dict with keys ``version``, ``trace_id``, ``parent_span_id``,
    and ``trace_flags``, or *None* if the header is malformed.
    """
    parts = header.strip().split("-")
    if len(parts) != 4:
        return None
    if len(parts[0]) != 2:
        return None
    if len(parts[1]) != 32:
        return None
    if len(parts[2]) != 16:
        return None
    if len(parts[3]) != 2:
        return None
    # Basic hex validation (digits are already checked by len)
    try:
        int(parts[1], 16)
        int(parts[2], 16)
        int(parts[3], 16)
    except ValueError:
        return None
    return {
        "version": parts[0],
        "trace_id": parts[1],
        "parent_span_id": parts[2],
        "trace_flags": parts[3],
    }


def _generate_trace_context(sampled: bool = True) -> dict[str, str]:
    """Generate a fresh W3C trace context for a root span.

    Args:
        sampled: Whether the trace should be marked as sampled.

    Returns:
        Dict with ``version``, ``trace_id``, ``parent_span_id``,
        ``trace_flags`` keys.
    """
    trace_id = secrets.token_hex(16)  # 128-bit
    span_id = secrets.token_hex(8)  # 64-bit
    flags = _TRACE_FLAGS_SAMPLED if sampled else _TRACE_FLAGS_NOT_SAMPLED
    return {
        "version": _TRACEPARENT_VERSION,
        "trace_id": trace_id,
        "parent_span_id": span_id,
        "trace_flags": flags,
    }


def _format_traceparent(ctx: dict[str, str]) -> str:
    """Reassemble a ``traceparent`` header from its component dict."""
    return (
        f"{ctx['version']}-{ctx['trace_id']}-{ctx['parent_span_id']}"
        f"-{ctx['trace_flags']}"
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class TracingMiddleware(BaseHTTPMiddleware):
    """W3C Trace-Context compliant tracing middleware.

    On every request:

    * Parses the incoming ``traceparent`` header.  If valid, the request joins
      the existing distributed trace.  Otherwise a new trace ID is generated.
    * Creates a **root span** named ``"<METHOD> <path>`` (e.g. ``"POST
      /v1/chat/completions"``) and stores it on ``request.state.root_span``.
    * After the response is produced, adds sub-span *events* for the
      **auth**, **rate-limiting**, and **inference** phases by inspecting
      ``request.state`` markers placed by downstream middleware.
    * Emits a ``traceparent`` response header so callers and downstream
      services can continue the distributed trace.
    * Logs the root-span duration at ``INFO`` level.

    The middleware transparently uses OpenTelemetry when available and falls
    back to a pure-Python implementation otherwise.
    """

    # Paths exempt from tracing (health probes, metrics scraping, docs).
    SKIP_PATHS: frozenset[str] = frozenset({
        "/health",
        "/ready",
        "/live",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
    })

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Apply tracing: parse/generate context, create root span, emit events."""
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # ---- 1. Parse or generate trace context ----
        raw_header = request.headers.get("traceparent", "")
        if raw_header:
            parsed = _parse_traceparent(raw_header)
            if parsed is not None:
                trace_ctx = parsed
                is_root = False
            else:
                # Malformed header — silently regenerate
                trace_ctx = _generate_trace_context()
                is_root = True
        else:
            trace_ctx = _generate_trace_context()
            is_root = True

        # Expose on request.state for downstream consumers
        request.state.trace_id = trace_ctx["trace_id"]
        request.state.span_id = trace_ctx["parent_span_id"]  # our root span ID
        request.state.trace_flags = trace_ctx["trace_flags"]
        request.state.traceparent = _format_traceparent(trace_ctx)

        # ---- 2. Create root span ----
        span_name = f"{request.method} {request.url.path}"
        span_kind = "server" if not is_root else "root"

        if _OTEL_AVAILABLE:
            root_span, tracer = self._create_otel_root_span(
                request, trace_ctx, span_name, raw_header, span_kind,
            )
        else:
            root_span, tracer = self._create_fallback_root_span(
                trace_ctx, span_name, raw_header, span_kind,
            )

        request.state.root_span = root_span
        request.state.tracer = tracer
        request.state._trace_attrs: dict[str, Any] = {}

        # ---- 3. Process the request ----
        start_wall = time.monotonic()
        try:
            response = await call_next(request)
        except BaseException as exc:
            # Record the exception on the span before re-raising
            self._record_exception(root_span, exc)
            root_span.end()
            raise

        elapsed_ms = (time.monotonic() - start_wall) * 1000.0

        # ---- 4. Emit sub-span markers ----
        self._emit_auth_event(root_span, request)
        self._emit_rate_limit_event(root_span, request, response)
        self._emit_inference_event(root_span, request, response)
        self._emit_response_attributes(root_span, response, elapsed_ms)

        # ---- 5. End root span and add response header ----
        root_span.end()

        # Use the root span's own span_id from OTel (may differ from the
        # parent_span_id we generated if OTel generated a new span ID).
        span_id = self._get_span_id(root_span) or trace_ctx["parent_span_id"]
        response.headers["traceparent"] = _format_traceparent({
            "version": _TRACEPARENT_VERSION,
            "trace_id": trace_ctx["trace_id"],
            "parent_span_id": span_id,
            "trace_flags": trace_ctx["trace_flags"],
        })

        # ---- 6. Log duration ----
        logger.info(
            "Trace | {} {} | trace_id={} span_id={} status={} duration_ms={:.1f}",
            request.method,
            request.url.path,
            trace_ctx["trace_id"],
            span_id,
            response.status_code,
            elapsed_ms,
        )

        return response

    # ------------------------------------------------------------------
    # OTel span creation
    # ------------------------------------------------------------------

    def _create_otel_root_span(
        self,
        request: Request,
        trace_ctx: dict[str, str],
        span_name: str,
        raw_header: str,
        span_kind: str,
    ) -> tuple[Any, Any]:
        """Create a root span using the OpenTelemetry SDK."""
        tracer = _otel_trace.get_tracer("distllm.api")

        # If we have an incoming trace context, extract it via the propagator
        # so the new span correctly inherits the remote parent.
        if raw_header and _otel_propagator is not None:
            carrier = {"traceparent": raw_header}
            ctx = _otel_propagator.extract(carrier=carrier)
            span = tracer.start_span(
                name=span_name,
                context=ctx,
                kind=_otel_trace.SpanKind.SERVER,
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.target": request.url.path,
                    "http.host": request.url.hostname or "",
                    "trace.type": span_kind,
                    "service.name": "distllm-api",
                },
            )
        else:
            # Root span (no parent)
            span = tracer.start_span(
                name=span_name,
                kind=_otel_trace.SpanKind.SERVER,
                attributes={
                    "http.method": request.method,
                    "http.url": str(request.url),
                    "http.target": request.url.path,
                    "http.host": request.url.hostname or "",
                    "trace.type": span_kind,
                    "service.name": "distllm-api",
                },
            )
        return span, tracer

    # ------------------------------------------------------------------
    # Fallback span creation
    # ------------------------------------------------------------------

    def _create_fallback_root_span(
        self,
        trace_ctx: dict[str, str],
        span_name: str,
        raw_header: str,
        span_kind: str,
    ) -> tuple[_Span, _SimpleTracer]:
        """Create a root span using the pure-Python fallback."""
        tracer = _SimpleTracer()
        span = _Span(
            name=span_name,
            trace_id=trace_ctx["trace_id"],
            span_id=trace_ctx["parent_span_id"],
            parent_span_id=None,  # root span
            start_time=time.monotonic(),
            attributes={
                "trace.type": span_kind,
            },
        )
        return span, tracer

    # ------------------------------------------------------------------
    # Sub-span event helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_auth_event(span: Any, request: Request) -> None:
        """Add an event to the root span for the auth phase.

        If ``AuthMiddleware`` set ``request.state.api_key_role`` then auth
        completed successfully; otherwise the request may have been rejected
        or bypassed auth for an exempt path.
        """
        role = getattr(request.state, "api_key_role", None)
        key_id = getattr(request.state, "api_key_id", None)
        if role is not None:
            span.add_event(
                "auth.completed",
                attributes={
                    "auth.role": role,
                    "auth.key_id": key_id or "",
                },
            )
        else:
            span.add_event(
                "auth.skipped",
                attributes={"auth.reason": "no api_key_role in request.state"},
            )

    @staticmethod
    def _emit_rate_limit_event(span: Any, request: Request, response: Response) -> None:
        """Add an event for rate-limiting (applied or passed through)."""
        if response.status_code == 429:
            span.add_event(
                "rate_limit.throttled",
                attributes={
                    "http.status_code": 429,
                    "rate_limit.retry_after": response.headers.get("Retry-After", ""),
                },
            )
        else:
            span.add_event("rate_limit.passed")

    @staticmethod
    def _emit_inference_event(span: Any, request: Request, response: Response) -> None:
        """Add an event for the inference phase."""
        status = response.status_code
        if status == 200:
            span.add_event(
                "inference.completed",
                attributes={"http.status_code": status},
            )
        elif status in (504, 503, 502):
            span.add_event(
                "inference.failed",
                attributes={
                    "http.status_code": status,
                    "inference.reason": response.headers.get(
                        "X-Error-Type", "unknown"
                    ),
                },
            )
        else:
            # Non-200, non-error: probably a redirect or informational
            span.add_event(
                "inference.skipped",
                attributes={
                    "http.status_code": status,
                    "inference.reason": "request did not reach route handler",
                },
            )

    @staticmethod
    def _emit_response_attributes(
        span: Any,
        response: Response,
        elapsed_ms: float,
    ) -> None:
        """Set standard response attributes on the root span."""
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("http.duration_ms", elapsed_ms)
        content_length = response.headers.get("content-length")
        if content_length:
            span.set_attribute("http.response_content_length", int(content_length))

    @staticmethod
    def _record_exception(span: Any, exc: BaseException) -> None:
        """Record an exception on the span before re-raising."""
        try:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc))
            span.add_event("exception", attributes={"exception.message": str(exc)})
        except Exception:
            pass  # best-effort

    @staticmethod
    def _get_span_id(span: Any) -> str | None:
        """Extract the 16-char hex span ID from a span object.

        Works with both OTel spans and ``_Span`` instances.
        """
        if hasattr(span, "span_id"):
            return span.span_id
        if hasattr(span, "get_span_context"):
            ctx = span.get_span_context()
            if ctx is not None:
                return format(ctx.span_id, "016x")
        return None
