"""Tracing configuration with tail-based sampling, Loki export, and exemplars.

Provides:
    - TailBasedSampler — head-based 10 % sampling + always-sample-errors
      + tail latency detection via a companion span processor.
    - TraceLokiExporter — exports trace IDs as Loki labels for log-trace
      correlation.
    - ExemplarExporter — attaches exemplars (trace_id, span_id) to
      Prometheus histogram observations for latency, queue_wait, ttft.
    - TracingConfigurator — combines all three into a complete tracing
      pipeline and wires it into a FastAPI application.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

# ── Optional OpenTelemetry imports ─────────────────────────────────────

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SpanExportResult,
        SpanExporter,
    )
    from opentelemetry.sdk.trace.sampling import (
        ParentBased,
        Sampler,
        SamplingResult,
        TraceIdRatioBased,
    )
    # OTel SDK >= 1.27 renamed Decision; support older versions too.
    try:
        from opentelemetry.sdk.trace.sampling import Decision as SamplingDecision
    except ImportError:  # pragma: no cover
        from opentelemetry.sdk.trace.sampling import (  # type: ignore[no-redef]
            SamplingDecision,
        )
    from opentelemetry.trace import StatusCode
    from opentelemetry.sdk.resources import Resource

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False

    # Stubs so the module can be imported without OTel installed.
    class Sampler:  # type: ignore[no-redef]
        """Stand-in when OpenTelemetry is not available."""

    class SpanExporter:  # type: ignore[no-redef]
        """Stand-in when OpenTelemetry is not available."""

    class SamplingDecision:
        RECORD_AND_SAMPLE = 1

    class SamplingResult:  # type: ignore[no-redef]
        def __init__(
            self,
            decision: Any = None,
            attributes: dict[str, Any] | None = None,
            trace_state: Any = None,
        ) -> None:
            self.decision = decision
            self.attributes = attributes
            self.trace_state = trace_state

    class StatusCode:  # type: ignore[no-redef]
        ERROR = 1

    class ReadableSpan:  # type: ignore[no-redef]
        ...

    class SpanExportResult:  # type: ignore[no-redef]
        SUCCESS = 0
        FAILURE = 1

    class TracerProvider:  # type: ignore[no-redef]
        def add_span_processor(self, processor: Any) -> None: ...


try:
    from prometheus_client import Histogram
except ImportError:  # pragma: no cover
    Histogram = None  # type: ignore[misc,assignment]


# ── TailBasedSampler ───────────────────────────────────────────────────


class TailBasedSampler(Sampler):
    """Samples traces based on tail latency (p99 > threshold).

    Strategy:

        *   **Head-based** — ``head_ratio`` (default 10 %) of all traces are
            probabilistically sampled at span-creation time.
        *   **Always sample errors** — any span that carries an ``error``
            attribute or whose status is ``ERROR`` is sampled regardless of the
            head-based decision.
        *   **Tail-based latency detection** — after a span completes, its
            duration is compared to ``threshold_ms``.  If the duration exceeds
            the threshold, the trace ID is recorded so that *all future* spans
            in that trace are sampled at 100 %.

    The companion :class:`TailBasedSpanProcessor` must be added to the
    ``TracerProvider`` so that the sampler receives span-end data::

        sampler = TailBasedSampler(threshold_ms=3000.0)
        provider = TracerProvider(sampler=sampler)
        provider.add_span_processor(sampler.create_span_processor())
    """

    def __init__(
        self,
        threshold_ms: float = 5000.0,
        head_ratio: float = 0.1,
    ) -> None:
        self._threshold_ns = int(threshold_ms * 1_000_000)
        self._head_ratio = head_ratio
        # Lazily initialised so the class can be imported without OTel.
        self._head_sampler: Sampler | None = None
        # Trace IDs that must be fully sampled (slow or errored).
        self._force_sampled: set[int] = set()
        self._lock = threading.Lock()

    def _get_head_sampler(self) -> Sampler:
        """Create the head-based sampler on first access."""
        if self._head_sampler is None:
            self._head_sampler = ParentBased(
                TraceIdRatioBased(self._head_ratio),
            )
        return self._head_sampler

    # ── Sampler interface ──────────────────────────────────────────────

    def should_sample(  # type: ignore[override]
        self,
        parent_context: Any | None = None,
        trace_id: int = 0,
        name: str = "",
        kind: Any | None = None,
        attributes: dict[str, Any] | None = None,
        links: list[Any] | None = None,
        trace_state: Any | None = None,
    ) -> SamplingResult:
        """Return a sampling decision for the given span parameters."""
        # 1. Already force-sampled trace (slow / errored)?
        with self._lock:
            if trace_id in self._force_sampled:
                return SamplingResult(
                    SamplingDecision.RECORD_AND_SAMPLE,
                    attributes=(
                        {"tail.sampled": True}
                        if attributes is not None
                        else None
                    ),
                    trace_state=trace_state,
                )

        # 2. Delegate to head-based sampler.
        result = self._get_head_sampler().should_sample(
            parent_context,
            trace_id,
            name,
            kind,
            attributes,
            links,
            trace_state,
        )

        # 3. Already sampled at head — done.
        if result.decision == SamplingDecision.RECORD_AND_SAMPLE:
            return result

        # 4. Error detected at span-creation time.
        if attributes is not None:
            error_attr = (
                attributes.get("error")
                or attributes.get("status.description", "")
                == "error"
            )
            if error_attr:
                with self._lock:
                    self._force_sampled.add(trace_id)
                return SamplingResult(
                    SamplingDecision.RECORD_AND_SAMPLE,
                    attributes=(
                        {"tail.sampled_by_error": True}
                        if attributes is not None
                        else None
                    ),
                    trace_state=trace_state,
                )

        # 5. Not sampled.
        return result

    def get_description(self) -> str:
        """Human-readable description of this sampler."""
        return (
            f"TailBasedSampler{{"
            f"threshold_ms={self._threshold_ns / 1e6:.0f}, "
            f"head_ratio={self._head_ratio}}}"
        )

    # ── Tail-latency feedback ──────────────────────────────────────────

    def record_latency(self, trace_id: int, duration_ns: int) -> None:
        """Record a completed span's duration.

        Called by the companion span processor after each span ends.
        If ``duration_ns`` exceeds the threshold, the trace is force-sampled
        from that point onward.
        """
        if duration_ns > self._threshold_ns:
            with self._lock:
                self._force_sampled.add(trace_id)

    def record_error(self, trace_id: int) -> None:
        """Record an error observed at span end.

        Called by the companion span processor when a span ends with
        ``StatusCode.ERROR``.
        """
        with self._lock:
            self._force_sampled.add(trace_id)

    def create_span_processor(self) -> TailBasedSpanProcessor:
        """Return a companion span processor that feeds back span-end data."""
        return TailBasedSpanProcessor(self)


class TailBasedSpanProcessor:
    """Companion :class:`SpanProcessor` for :class:`TailBasedSampler`.

    Sits in the span-processor chain and evaluates each completed span's
    duration and status against the sampler's threshold.  Slow or errored
    spans cause the sampler to force-sample the entire trace going forward.
    """

    def __init__(self, sampler: TailBasedSampler) -> None:
        self._sampler = sampler

    def on_start(
        self,
        span: Any,
        parent_context: Any | None = None,
    ) -> None:
        """No-op at span start."""
        return None

    def on_end(self, span: Any) -> None:
        """Inspect the completed span and notify the sampler."""
        ctx = span.get_span_context()
        duration_ns = (span.end_time or 0) - (span.start_time or 0)
        if duration_ns > 0:
            self._sampler.record_latency(ctx.trace_id, duration_ns)

        if span.status and span.status.status_code == StatusCode.ERROR:
            self._sampler.record_error(ctx.trace_id)

    def shutdown(self) -> None:
        """No-op shutdown."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op force-flush."""
        return True


# ── TraceLokiExporter ──────────────────────────────────────────────────


class TraceLokiExporter(SpanExporter):
    """Export span trace_id / span_id as Loki labels for log-trace correlation.

    Pushes span metadata to the Grafana Loki push API.  Each span is
    encoded as a Loki log entry whose labels include ``trace_id``,
    ``span_id``, ``span_name``, and ``service`` so that logs and traces
    can be correlated in a Grafana Explore view.
    """

    def __init__(
        self,
        loki_url: str = "http://localhost:3100/loki/api/v1/push",
        service_name: str = "distllm",
        labels: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._loki_url = loki_url.rstrip("/")
        self._service_name = service_name
        self._extra_labels = labels or {}
        self._timeout = timeout
        self._http_client: Any = None  # lazy init

    # ── SpanExporter interface ─────────────────────────────────────────

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        """Export a batch of spans to the Loki push API."""
        if not _OTEL_AVAILABLE or not spans:
            return SpanExportResult.SUCCESS

        streams: dict[str, dict[str, Any]] = {}

        for span in spans:
            ctx = span.get_span_context()
            base_labels = {
                "service": self._service_name,
                "trace_id": f"{ctx.trace_id:032x}",
                "span_id": f"{ctx.span_id:016x}",
                "span_name": span.name,
                **self._extra_labels,
            }
            if span.status and span.status.status_code == StatusCode.ERROR:
                base_labels["error"] = "true"

            label_key = json.dumps(base_labels, sort_keys=True)
            if label_key not in streams:
                streams[label_key] = {"stream": base_labels, "values": []}

            ts_ns = span.end_time or int(time.time() * 1e9)
            entry = json.dumps(
                {
                    "name": span.name,
                    "kind": str(span.kind),
                    "attributes": dict(span.attributes or {}),
                    "status": (
                        str(span.status.status_code)
                        if span.status
                        else None
                    ),
                    "duration_ns": (span.end_time or 0)
                    - (span.start_time or 0),
                },
                default=str,
            )
            streams[label_key]["values"].append([str(ts_ns), entry])

        payload = {"streams": list(streams.values())}

        try:
            client = self._http_client
            if client is None:
                import httpx

                self._http_client = httpx.Client(timeout=self._timeout)
                client = self._http_client

            resp = client.post(
                self._loki_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code not in (200, 204):
                # Export failures are non-fatal — avoid raising.
                pass
        except Exception:
            pass

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force-flush is a no-op for this exporter."""
        return True


# ── ExemplarExporter ───────────────────────────────────────────────────


class ExemplarExporter:
    """Attach exemplars (trace_id, span_id) to Prometheus histogram observations.

    Wraps Prometheus histograms so that ``observe()`` calls include the
    current OpenTelemetry trace context as an exemplar, enabling trace-metric
    correlation in Prometheus + Grafana.

    Target histograms (by convention): ``latency``, ``queue_wait``, ``ttft``.

    Usage::

        ex = ExemplarExporter()
        latency_hist = ex.register_histogram("latency", my_latency_histogram)

        # Later, in a request handler:
        ex.observe("latency", 0.35, labels={"method": "generate"})
        # The observation includes {trace_id, span_id} as an exemplar.
    """

    def __init__(self) -> None:
        self._histograms: dict[str, Histogram] = {}

    def register_histogram(
        self,
        name: str,
        histogram: Histogram,
    ) -> Histogram:
        """Register a Prometheus histogram to receive exemplars.

        Args:
            name: Logical name used with :meth:`observe` (e.g. ``"latency"``).
            histogram: A ``prometheus_client.Histogram`` instance.

        Returns:
            The same ``histogram`` (callers may ignore the return value).
        """
        self._histograms[name] = histogram
        return histogram

    def create_histogram(
        self,
        name: str,
        metric_name: str,
        documentation: str,
        label_names: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Create a new Prometheus Histogram and register it for exemplars.

        Args:
            name: Logical name used with :meth:`observe`.
            metric_name: Prometheus metric name (e.g. ``"my_latency_seconds"``).
            documentation: Metric help text.
            label_names: Optional list of label keys.
            buckets: Optional bucket boundaries.

        Returns:
            The newly created ``Histogram``.
        """
        if Histogram is None:
            raise ImportError("prometheus_client is not installed")

        hist = Histogram(
            metric_name,
            documentation,
            labelnames=label_names or [],
            buckets=buckets or Histogram.DEFAULT_BUCKETS,
        )
        return self.register_histogram(name, hist)

    def observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Observe a value on the named histogram with exemplars attached.

        Args:
            name: Logical histogram name (registered via
                :meth:`register_histogram` or :meth:`create_histogram`).
            value: Observed value.
            labels: Optional label values to scope the observation.
        """
        histogram = self._histograms.get(name)
        if histogram is None:
            return

        trace_id, span_id = self._current_trace_context()
        exemplar: dict[str, str] | None = None
        if trace_id is not None:
            exemplar = {"trace_id": trace_id, "span_id": span_id or ""}

        if labels:
            histogram.labels(**labels).observe(value, exemplar=exemplar)
        else:
            histogram.observe(value, exemplar=exemplar)

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _current_trace_context() -> tuple[str | None, str | None]:
        """Extract trace_id and span_id from the current OTel span."""
        if not _OTEL_AVAILABLE:
            return None, None
        try:
            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.is_valid:
                return (
                    f"{ctx.trace_id:032x}",
                    f"{ctx.span_id:016x}",
                )
        except Exception:
            pass
        return None, None


# ── TracingConfigurator ────────────────────────────────────────────────


class TracingConfigurator:
    """Combine tail-based sampling, Loki log-trace export, and exemplars.

    Usage::

        configurator = TracingConfigurator(
            service_name="distllm",
            threshold_ms=3000.0,
            head_ratio=0.1,
            loki_url="http://loki:3100/loki/api/v1/push",
        )
        provider = configurator.configure(app)

    The configurator installs:

        *   ``TracerProvider`` configured with a :class:`TailBasedSampler`
        *   ``BatchSpanProcessor`` with a :class:`TraceLokiExporter`
        *   An :class:`ExemplarExporter` instance (accessible via
            :attr:`exemplar_exporter`)
        *   Optional ASGI middleware for trace-context propagation
    """

    def __init__(
        self,
        service_name: str = "distllm",
        threshold_ms: float = 5000.0,
        head_ratio: float = 0.1,
        loki_url: str | None = None,
        sampler: TailBasedSampler | None = None,
    ) -> None:
        self._service_name = service_name
        self._threshold_ms = threshold_ms
        self._head_ratio = head_ratio
        self._loki_url = loki_url
        self._sampler = sampler or TailBasedSampler(
            threshold_ms=threshold_ms,
            head_ratio=head_ratio,
        )
        self._exemplar_exporter = ExemplarExporter()

    # ── properties ─────────────────────────────────────────────────────

    @property
    def sampler(self) -> TailBasedSampler:
        """The configured :class:`TailBasedSampler` instance."""
        return self._sampler

    @property
    def exemplar_exporter(self) -> ExemplarExporter:
        """The configured :class:`ExemplarExporter` instance."""
        return self._exemplar_exporter

    # ── public API ─────────────────────────────────────────────────────

    def configure(
        self,
        app: Any = None,
    ) -> TracerProvider | None:
        """Set up the full tracing pipeline.

        Args:
            app: Optional FastAPI (or ASGI) application.  When provided,
                the configurator installs ASGI middleware for automatic
                trace-context propagation from incoming HTTP headers and
                response header injection (``X-Trace-Id``, ``X-Span-Id``).

        Returns:
            The configured ``TracerProvider``, or ``None`` if OpenTelemetry
            is not installed.
        """
        if not _OTEL_AVAILABLE:
            return None

        # 1. Build resource.
        resource = Resource.create({"service.name": self._service_name})

        # 2. Create provider with tail-based sampler.
        provider = TracerProvider(
            resource=resource,
            sampler=self._sampler,
        )

        # 3. Add companion span processor for tail-latency feedback.
        provider.add_span_processor(self._sampler.create_span_processor())

        # 4. Add Loki exporter if a URL was provided.
        if self._loki_url:
            loki_exporter = TraceLokiExporter(
                loki_url=self._loki_url,
                service_name=self._service_name,
            )
            provider.add_span_processor(
                BatchSpanProcessor(loki_exporter),
            )

        # 5. Set global tracer provider.
        trace.set_tracer_provider(provider)

        # 6. Install ASGI middleware for automatic context propagation.
        if app is not None:
            self._install_middleware(app)

        return provider

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _install_middleware(app: Any) -> None:
        """Install trace-context propagation middleware on a FastAPI app."""
        # -- W3C trace-context extraction from incoming headers --
        try:
            from starlette.middleware.base import BaseHTTPMiddleware

            async def tracing_middleware(
                request: Any,
                call_next: Any,
            ) -> Any:
                # Extract W3C traceparent from incoming request headers.
                traceparent = request.headers.get("traceparent")
                tracestate = request.headers.get("tracestate")
                if traceparent and _OTEL_AVAILABLE:
                    try:
                        from opentelemetry import (
                            context as otel_context,
                        )
                        from opentelemetry.trace.propagation.tracecontext import (  # noqa: E501
                            TraceContextTextMapPropagator,
                        )

                        carrier: dict[str, str] = {
                            "traceparent": traceparent,
                        }
                        if tracestate:
                            carrier["tracestate"] = tracestate
                        propagator = TraceContextTextMapPropagator()
                        ctx = propagator.extract(carrier)
                        if ctx is not None:
                            otel_context.attach(ctx)
                    except Exception:
                        pass

                response = await call_next(request)

                # Inject trace context into response headers.
                if _OTEL_AVAILABLE:
                    try:
                        span = trace.get_current_span()
                        if span and span.is_recording():
                            ctx = span.get_span_context()
                            response.headers["X-Trace-Id"] = (
                                f"{ctx.trace_id:032x}"
                            )
                            response.headers["X-Span-Id"] = (
                                f"{ctx.span_id:016x}"
                            )
                    except Exception:
                        pass

                return response

            app.add_middleware(BaseHTTPMiddleware, dispatch=tracing_middleware)
        except Exception:
            pass

        # -- OpenTelemetry ASGI middleware for automatic span creation --
        try:
            from opentelemetry.instrumentation.asgi import (
                OpenTelemetryMiddleware,
            )

            app.add_middleware(OpenTelemetryMiddleware)
        except (ImportError, Exception):
            pass
