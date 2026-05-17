"""FastAPI middleware for OpenTelemetry tracing and RED metrics.

Wires per-request OTel spans, RED metrics (Rate, Errors, Duration) with
model/tenant labels, cost tracking, and anomaly detection into the
FastAPI request lifecycle.
"""

import time

from fastapi import Request
from loguru import logger
from opentelemetry import trace
from opentelemetry.trace import StatusCode
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Per-request OpenTelemetry span, RED metrics, cost tracking, anomaly detection."""

    def __init__(
        self,
        app,
        metrics_exporter=None,
        cost_tracker=None,
        anomaly_detector=None,
    ):
        super().__init__(app)
        self._metrics_exporter = metrics_exporter
        self._cost_tracker = cost_tracker
        self._anomaly_detector = anomaly_detector
        self._tracer = trace.get_tracer("distllm.api")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        request_id = getattr(request.state, "request_id", "")
        method = request.method
        path = request.url.path
        model = getattr(request.state, "model", "distributed-llm")
        tenant = getattr(request.state, "tenant", "default")

        start_time = time.monotonic()

        with self._tracer.start_as_current_span(
            f"HTTP {method} {path}",
            attributes={
                "http.method": method,
                "http.target": path,
                "http.request_id": request_id,
                "llm.model": model,
                "llm.tenant": tenant,
            },
        ) as span:
            # Inject trace context into loguru
            span_ctx = span.get_span_context()
            ctx_logger = logger.bind(
                trace_id=f"{span_ctx.trace_id:032x}",
                span_id=f"{span_ctx.span_id:016x}",
                request_id=request_id,
            )

            try:
                response = await call_next(request)
                duration = time.monotonic() - start_time

                self._record_red_metrics(
                    method, path, model, tenant, response.status_code, duration
                )

                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.duration_s", duration)

                # Anomaly detection
                if self._anomaly_detector:
                    self._anomaly_detector.record("http_request_duration", duration)

                # Cost tracking completion
                if self._cost_tracker and request_id:
                    request_cost = getattr(request.state, "_request_cost", None)
                    if request_cost is not None:
                        self._record_cost_metrics(model, tenant, request_cost)

                return response

            except Exception as exc:
                duration = time.monotonic() - start_time

                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)

                self._record_red_metrics(method, path, model, tenant, 500, duration, is_error=True)

                if self._anomaly_detector:
                    self._anomaly_detector.record("http_error_rate", 1.0)

                raise

    def _record_red_metrics(self, method, path, model, tenant, status_code, duration, is_error=False):
        """Record Rate, Errors, Duration metrics with model/tenant labels."""
        exporter = self._metrics_exporter
        if not exporter:
            return

        status = "error" if is_error or status_code >= 500 else "success"

        exporter.requests_total.labels(
            method=method, status=status, model=model, tenant=tenant,
        ).inc()

        exporter.request_latency.labels(
            method=method, model=model, tenant=tenant,
        ).observe(duration)

        exporter.request_duration_seconds.labels(
            method=method, model=model, tenant=tenant,
        ).observe(duration)

        if is_error or status_code >= 400:
            exporter.errors_total.labels(
                type=f"http_{status_code}", model=model, tenant=tenant,
            ).inc()

    def _record_cost_metrics(self, model, tenant, cost_data):
        """Record cost and GPU-hour metrics."""
        exporter = self._metrics_exporter
        if not exporter or not cost_data:
            return

        cost = cost_data.get("cost", 0.0)
        gpu_hours = cost_data.get("gpu_hours", 0.0)

        if cost > 0:
            exporter.request_cost_total.labels(model=model, tenant=tenant).inc(cost)
        if gpu_hours > 0:
            exporter.request_gpu_hours.labels(model=model, tenant=tenant).inc(gpu_hours)
