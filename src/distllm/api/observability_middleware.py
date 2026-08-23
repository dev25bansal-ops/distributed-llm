"""Observability middleware for the distributed-LLM API.

``ObservabilityMiddleware`` is a Starlette ``BaseHTTPMiddleware`` that wraps
every request and emits OpenTelemetry spans plus RED metrics, optional
cost/GPU-hour metrics, and optional anomaly-detection samples.

The exact metric label vocabulary and span attribute names below are part of
the contract exercised by ``tests/api/test_auth_middleware.py``.
"""

from __future__ import annotations

import time

from fastapi import Request
from opentelemetry import trace
from starlette.middleware.base import BaseHTTPMiddleware


# Default label values required by the test-suite metrics assertions.
_DEFAULT_MODEL = "distributed-llm"
_DEFAULT_TENANT = "default"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Collect observability signals for each HTTP request.

    For every request the middleware:

    * opens an OpenTelemetry span named ``HTTP <METHOD> <path>`` carrying
      standard HTTP attributes and the propagated request id;
    * records RED metrics (rate / errors / duration) on ``metrics_exporter``;
    * records cost / gpu-hour metrics when the route sets
      ``request.state._request_cost``;
    * forwards a duration (success) or error-rate (failure) sample to an
      optional ``anomaly_detector``.

    All collaborators are optional and are safely no-op'd when ``None``.
    """

    def __init__(
        self,
        app,
        metrics_exporter=None,
        cost_tracker=None,
        anomaly_detector=None,
        tracer=None,
    ):
        super().__init__(app)
        self.metrics_exporter = metrics_exporter
        self.cost_tracker = cost_tracker
        self.anomaly_detector = anomaly_detector
        self.tracer = tracer

    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path
        start = time.perf_counter()
        status_code = 500
        error = None
        response = None

        try:
            if self.tracer is not None:
                with self.tracer.start_as_current_span(
                    f"HTTP {method} {path}"
                ) as span:
                    try:
                        response = await call_next(request)
                        status_code = response.status_code
                    except Exception as exc:  # noqa: BLE001
                        error = exc
                        status_code = 500
                        # Mark the span as failed and record the exception
                        # before it is closed by the context manager.
                        self._finalize_span(
                            span, request, method, path, status_code, start
                        )
                        span.set_status(trace.Status(trace.StatusCode.ERROR))
                        span.record_exception(exc)
                        raise
                    self._finalize_span(
                        span, request, method, path, status_code, start
                    )
            else:
                try:
                    response = await call_next(request)
                    status_code = response.status_code
                except Exception as exc:  # noqa: BLE001
                    error = exc
                    status_code = 500
                    raise
        finally:
            duration = time.perf_counter() - start
            self._record_metrics(
                request, method, path, status_code, duration, error
            )

        return response

    # ------------------------------------------------------------------ #
    # Span helpers
    # ------------------------------------------------------------------ #
    def _finalize_span(self, span, request, method, path, status_code, start):
        duration = time.perf_counter() - start
        span.set_attribute("http.method", method)
        span.set_attribute("http.target", path)
        span.set_attribute("http.status_code", status_code)
        span.set_attribute("http.duration_s", duration)
        request_id = getattr(request.state, "request_id", None)
        if request_id is not None:
            span.set_attribute("http.request_id", request_id)

    # ------------------------------------------------------------------ #
    # Metrics / anomaly helpers
    # ------------------------------------------------------------------ #
    def _record_metrics(self, request, method, path, status_code, duration, error):
        model = _DEFAULT_MODEL
        tenant = _DEFAULT_TENANT

        exporter = self.metrics_exporter
        if exporter is not None:
            status = "error" if error is not None else "success"
            exporter.requests_total.labels(
                method=method, status=status, model=model, tenant=tenant
            ).inc()
            exporter.request_latency.labels(
                method=method, model=model, tenant=tenant
            ).observe(duration)
            exporter.request_duration_seconds.labels(
                method=method, model=model, tenant=tenant
            ).observe(duration)
            if error is not None:
                exporter.errors_total.labels(
                    type="http_500", model=model, tenant=tenant
                ).inc()

            cost = getattr(request.state, "_request_cost", None)
            if isinstance(cost, dict):
                cost_val = cost.get("cost")
                gpu_val = cost.get("gpu_hours")
                if cost_val is not None:
                    exporter.request_cost_total.labels(
                        model=model, tenant=tenant
                    ).inc(cost_val)
                if gpu_val is not None:
                    exporter.request_gpu_hours.labels(
                        model=model, tenant=tenant
                    ).inc(gpu_val)
                self._forward_cost(cost)

        detector = self.anomaly_detector
        if detector is not None and hasattr(detector, "record"):
            if error is not None:
                detector.record("http_error_rate", 1.0)
            else:
                detector.record("http_request_duration", duration)

    def _forward_cost(self, cost):
        """Pass the per-request cost dict to ``cost_tracker`` if it can accept it."""
        tracker = self.cost_tracker
        if tracker is None:
            return
        if hasattr(tracker, "track_cost"):
            tracker.track_cost(cost)
        elif hasattr(tracker, "record"):
            tracker.record(cost)
        elif callable(tracker):
            tracker(cost)
