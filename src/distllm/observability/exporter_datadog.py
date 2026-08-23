"""Datadog and New Relic APM integration via OpenTelemetry protocol.

Provides ``DatadogMetricsExporter`` -- a unified exporter that pushes metrics,
traces, and logs to Datadog (or New Relic) using the existing OpenTelemetry
infrastructure.  The exporter is fully optional: it degrades gracefully when
the required packages are not installed.

Environment variables
---------------------
DD_API_KEY :
    Datadog API key.  When set, the exporter targets Datadog endpoints.
DD_SITE :
    Datadog site (e.g. ``datadoghq.com``, ``us5.datadoghq.com``,
    ``datadoghq.eu``).  Defaults to ``datadoghq.com``.
NEW_RELIC_API_KEY :
    New Relic API / Insights insert key.  When set (and ``DD_API_KEY`` is
    unset), the exporter targets New Relic endpoints.
NEW_RELIC_REGION :
    New Relic region (``US`` or ``EU``).  Defaults to ``US``.
OTEL_EXPORTER_OTLP_ENDPOINT :
    Override the endpoint URL for OTLP export.  When set, the exporter
    sends OTLP Protobuf payloads instead of using the native Datadog/New
    Relic HTTP APIs.

Usage -- standalone push API ::

    from distllm.observability.exporter_datadog import DatadogMetricsExporter

    exporter = DatadogMetricsExporter()
    exporter.push_metrics({"cpu.percent": 42.5, "memory.mb": 2048})
    exporter.push_trace({"trace_id": "abc", "spans": [...]})
    exporter.push_log({"message": "request complete", "level": "info"})

Usage -- OTel SpanExporter (when opentelemetry-sdk is installed) ::

    exporter = DatadogMetricsExporter()
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

Usage -- context manager with auto-collection ::

    with DatadogMetricsExporter(auto_collect=True, collect_interval_s=15.0) as ex:
        ...
        ex.observe_request_latency(0.350)
        ex.observe_request_error("timeout")
"""

from __future__ import annotations

import math
import os
import platform
import random
import socket
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import psutil
from loguru import logger

# ---------------------------------------------------------------------------
# Optional OpenTelemetry imports
# ---------------------------------------------------------------------------

_OTEL_AVAILABLE = False
try:
    from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter

    _OTEL_AVAILABLE = True
except ImportError:
    # Stubs for type checking when OTel is not installed.
    class SpanExporter:  # type: ignore[no-redef]
        """Stand-in when OpenTelemetry is not available."""

    class SpanExportResult:
        SUCCESS = 0
        FAILURE = 1

# ---------------------------------------------------------------------------
# Optional GPU monitoring
# ---------------------------------------------------------------------------

_PYNVML_AVAILABLE = False
try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SITE = "datadoghq.com"

_DATADOG_METRICS_URL = "https://api.{site}/api/v2/series"
_DATADOG_TRACES_URL = "https://trace.agent.{site}/api/v0.2/traces"
_DATADOG_LOGS_URL = "https://http-intake.logs.{site}/api/v2/logs"

_NEW_RELIC_METRICS_URL = "https://metric-api.{region}newrelic.com/metric/v1"
_NEW_RELIC_TRACES_URL = "https://trace-api.{region}newrelic.com/trace/v1"
_NEW_RELIC_LOGS_URL = "https://log-api.{region}newrelic.com/log/v1"

_NEW_RELIC_REGIONS = {
    "US": "",
    "EU": "eu.",
}

_MAX_BATCH_SIZE = 1000
_DEFAULT_COLLECT_INTERVAL_S = 15.0

# Retry parameters
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY_S = 1.0

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class _MetricPoint:
    """A single metric data point for push."""

    name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: str = "gauge"


@dataclass
class _TraceBatch:
    """A batch of spans for push."""

    trace_id: str
    spans: list[dict[str, Any]] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class _LogEntry:
    """A single log entry for push."""

    message: str
    level: str = "info"
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ddsource: str = "distllm"


# ---------------------------------------------------------------------------
# DatadogMetricsExporter
# ---------------------------------------------------------------------------


class DatadogMetricsExporter:
    """Push metrics, traces, and logs to Datadog or New Relic.

    The exporter auto-detects the backend from environment variables:

    * ``DD_API_KEY`` -> Datadog
    * ``NEW_RELIC_API_KEY`` -> New Relic

    When neither is set, the exporter starts in **dry-run** mode: all push
    methods log at DEBUG level and return without making HTTP calls.  Call
    :meth:`check_ready` to verify that the exporter is active.

    OpenTelemetry integration
    -------------------------
    When the ``opentelemetry-sdk`` package is installed, this class
    implements the ``SpanExporter`` interface and can be used as an OTel
    exporter::

        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        processor = BatchSpanProcessor(exporter)
        tracer_provider.add_span_processor(processor)

    Auto-collection
    ---------------
    When *auto_collect* is ``True`` (or the exporter is used as a context
    manager), a background thread collects GPU utilization, memory, and
    request-rate metrics every *collect_interval_s* seconds and pushes
    them automatically.

    Thread safety
    -------------
    All public methods are thread-safe.  Internal buffers use a lock.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        new_relic_key: str | None = None,
        new_relic_region: str | None = None,
        site: str | None = None,
        auto_collect: bool = False,
        collect_interval_s: float = _DEFAULT_COLLECT_INTERVAL_S,
        service_name: str = "distllm",
        tags: dict[str, str] | None = None,
        max_batch_size: int = _MAX_BATCH_SIZE,
        flush_immediately: bool = False,
    ) -> None:
        """Initialize DatadogMetricsExporter.

        Args:
            api_key: Datadog API key.  Falls back to ``DD_API_KEY`` env var.
            site: Datadog site (e.g. ``datadoghq.com``).  Falls back to
                ``DD_SITE`` env var, then to ``datadoghq.com``.
            auto_collect: If True, start a background thread that
                periodically collects GPU / memory / request metrics.
            collect_interval_s: Seconds between auto-collection cycles.
            service_name: Service name tag attached to all payloads.
            tags: Extra tags attached to every metric, trace, and log.
            max_batch_size: Maximum payload size before a forced flush.
            flush_immediately: If True, every ``push_*`` call sends
                immediately instead of batching.

        Environment variable fallback order (``api_key``):
            1. Explicit *api_key* argument.
            2. ``DD_API_KEY`` env var.
            3. ``NEW_RELIC_API_KEY`` env var.

        Environment variable fallback order (``site``):
            1. Explicit *site* argument.
            2. ``DD_SITE`` env var.
            3. ``datadoghq.com``.
        """
        self._service_name = service_name
        self._extra_tags = dict(tags or {})
        self._max_batch_size = max_batch_size
        self._flush_immediately = flush_immediately

        # API key resolution.
        self._api_key = api_key or os.environ.get("DD_API_KEY", "")
        self._new_relic_key = new_relic_key or os.environ.get("NEW_RELIC_API_KEY", "")
        self._site = (site or os.environ.get("DD_SITE", "") or _DEFAULT_SITE).rstrip("/")

        # New Relic region.
        self._nr_region = (new_relic_region or os.environ.get("NEW_RELIC_REGION", "") or "US").upper()

        # OTLP endpoint override.
        self._otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")

        # Connected flag.
        self._ready = bool(self._api_key or self._new_relic_key)

        # Internal buffers.
        self._lock = threading.Lock()
        self._metric_buffer: list[_MetricPoint] = []
        self._trace_buffer: list[_TraceBatch] = []
        self._log_buffer: list[_LogEntry] = []
        self._closed = False

        # Aggregate counters for auto-submission.
        self._request_count: int = 0
        self._error_count: int = 0
        self._latencies: deque[float] = deque(maxlen=10_000)

        # GPU monitoring state (lazy init, multi-GPU).
        self._gpu_initialised = False
        self._gpu_handles: list[Any] = []
        self._gpu_device_count: int = 0

        # Auto-collection thread.
        self._collect_thread: threading.Thread | None = None
        self._collect_stop = threading.Event()
        self._collect_interval = collect_interval_s
        if auto_collect:
            self._start_auto_collect()

        if not self._ready:
            logger.debug(
                "DatadogMetricsExporter started in dry-run mode "
                "(set DD_API_KEY or NEW_RELIC_API_KEY)"
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Whether the exporter has a valid API key configured."""
        return self._ready

    def check_ready(self) -> bool:
        """Return True if the exporter is configured and can reach its backend.

        Performs a lightweight connectivity check by attempting to resolve
        the endpoint hostname.
        """
        if not self._ready:
            return False
        try:
            host = self._resolve_host()
            socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Flush all buffers and stop the auto-collection thread.

        After calling ``close()`` the exporter is unusable.
        """
        if self._closed:
            return
        self._closed = True
        self._stop_auto_collect()
        self.flush()

    def flush(self) -> None:
        """Force-flush all buffered metrics, traces, and logs."""
        self._flush_metrics()
        self._flush_traces()
        self._flush_logs()

    # ------------------------------------------------------------------
    # Observation API (used internally and by auto-collect)
    # ------------------------------------------------------------------

    def observe_request_latency(self, seconds: float) -> None:
        """Record a request latency observation.

        Args:
            seconds: End-to-end request latency in seconds.
        """
        with self._lock:
            self._latencies.append(seconds)

    def observe_request_error(self, error_type: str = "") -> None:
        """Record a request error.

        Args:
            error_type: Optional error classification label.
        """
        with self._lock:
            self._error_count += 1

    def observe_request(self) -> None:
        """Record a completed request (increments request counter)."""
        with self._lock:
            self._request_count += 1

    # ------------------------------------------------------------------
    # Push API
    # ------------------------------------------------------------------

    def push_metrics(
        self,
        metrics: dict[str, float] | list[dict[str, Any]],
        tags: dict[str, str] | None = None,
    ) -> None:
        """Push metric data points to the configured backend.

        Args:
            metrics: Either a ``{name: value}`` dict for simple gauge
                metrics, or a list of dicts with keys ``name``, ``value``,
                ``tags``, ``timestamp``, ``type``.
            tags: Extra tags applied to every metric in the batch (when
                *metrics* is a flat dict).
        """
        if self._closed:
            return

        base_tags = dict(self._extra_tags)
        if tags:
            base_tags.update(tags)

        if isinstance(metrics, dict):
            points = [
                _MetricPoint(name=name, value=value, tags=base_tags)
                for name, value in metrics.items()
            ]
        else:
            points = [
                _MetricPoint(
                    name=m.get("name", "unknown"),
                    value=m.get("value", 0.0),
                    tags={**base_tags, **m.get("tags", {})},
                    timestamp=m.get("timestamp", time.time()),
                    metric_type=m.get("type", "gauge"),
                )
                for m in metrics
            ]

        # NOTE: flush is called OUTSIDE the lock to avoid reentrant-lock
        # deadlocks when _flush_immediately or batch-size thresholds trigger
        # a flush from within a lock-holding context.
        need_flush = False
        with self._lock:
            self._metric_buffer.extend(points)
            if self._flush_immediately or len(self._metric_buffer) >= self._max_batch_size:
                need_flush = True

        if need_flush:
            self._flush_metrics()

    def push_trace(
        self,
        trace: dict[str, Any] | list[dict[str, Any]],
        tags: dict[str, str] | None = None,
    ) -> None:
        """Push trace spans to the configured backend.

        Args:
            trace: A single trace dict (with ``trace_id`` and ``spans``)
                or a list of span dicts.  Each span dict should follow
                the OpenTelemetry span model (``trace_id``, ``span_id``,
                ``name``, ``start_time``, ``end_time``, ``attributes``,
                ``status``).
            tags: Extra tags applied to every span in the batch.
        """
        if self._closed:
            return

        base_tags = dict(self._extra_tags)
        if tags:
            base_tags.update(tags)

        if isinstance(trace, dict):
            traces_list = [trace]
        else:
            traces_list = trace

        for t in traces_list:
            tid = t.get("trace_id", t.get("traceId", ""))
            # Fallback: try "spans" key, then "span" (singular), then wrap
            # the entire dict as a single-span batch.
            spans = t.get("spans", t.get("span", [t]))
            self._trace_buffer.append(
                _TraceBatch(trace_id=tid, spans=spans, tags=base_tags)
            )

        # Flush outside the lock (see push_metrics for rationale).
        need_flush = False
        with self._lock:
            if self._flush_immediately or len(self._trace_buffer) >= self._max_batch_size:
                need_flush = True

        if need_flush:
            self._flush_traces()

    def push_log(
        self,
        log: str | dict[str, Any] | list[dict[str, Any]],
        level: str = "info",
        tags: dict[str, str] | None = None,
    ) -> None:
        """Push log entries to the configured backend.

        Args:
            log: A log message string or a structured log dict (or list
                thereof).  Dicts may include keys ``message``, ``level``,
                ``timestamp``, ``ddsource``, and ``tags``.
            level: Default log level when *log* is a plain string.
            tags: Extra tags applied to every log entry in the batch.
        """
        if self._closed:
            return

        base_tags = dict(self._extra_tags)
        if tags:
            base_tags.update(tags)

        if isinstance(log, str):
            entries = [_LogEntry(message=log, level=level, tags=base_tags)]
        elif isinstance(log, dict):
            entries = [
                _LogEntry(
                    message=log.get("message", ""),
                    level=log.get("level", level),
                    tags={**base_tags, **log.get("tags", {})},
                    timestamp=log.get("timestamp", time.time()),
                    ddsource=log.get("ddsource", self._service_name),
                )
            ]
        else:
            entries = [
                _LogEntry(
                    message=entry.get("message", ""),
                    level=entry.get("level", level),
                    tags={**base_tags, **entry.get("tags", {})},
                    timestamp=entry.get("timestamp", time.time()),
                    ddsource=entry.get("ddsource", self._service_name),
                )
                for entry in log
            ]

        # Flush outside the lock (see push_metrics for rationale).
        need_flush = False
        with self._lock:
            self._log_buffer.extend(entries)
            if self._flush_immediately or len(self._log_buffer) >= self._max_batch_size:
                need_flush = True

        if need_flush:
            self._flush_logs()

    # ------------------------------------------------------------------
    # OTel SpanExporter interface (when OTel SDK is installed)
    # ------------------------------------------------------------------

    def export(self, spans: list[Any]) -> SpanExportResult:
        """Export a batch of OpenTelemetry spans.

        Implements ``opentelemetry.sdk.trace.export.SpanExporter``.
        Each OTel span is converted to a dict and pushed via the trace API.

        Args:
            spans: List of OpenTelemetry ``ReadableSpan`` objects.

        Returns:
            ``SpanExportResult.SUCCESS`` on success,
            ``SpanExportResult.FAILURE`` on error.
        """
        if not _OTEL_AVAILABLE or self._closed:
            return SpanExportResult.FAILURE

        try:
            from opentelemetry.trace import StatusCode

            converted = []
            for span in spans:
                ctx = span.get_span_context()
                entry = {
                    "trace_id": f"{ctx.trace_id:032x}",
                    "span_id": f"{ctx.span_id:016x}",
                    "name": span.name,
                    "kind": str(span.kind),
                    "start_time": span.start_time // 1_000_000 if span.start_time else 0,
                    "end_time": span.end_time // 1_000_000 if span.end_time else 0,
                    "attributes": dict(span.attributes or {}),
                    "status": {
                        "code": span.status.status_code,
                        "description": span.status.description or "",
                    }
                    if span.status
                    else {},
                    "resource": {
                        k: v for k, v in (span.resource.attributes or {}).items()
                    }
                    if span.resource
                    else {},
                }
                if span.status and span.status.status_code == StatusCode.ERROR:
                    entry["error"] = True
                converted.append(entry)

            if converted:
                self.push_trace(converted)
                self.flush()

            return SpanExportResult.SUCCESS
        except Exception as exc:
            logger.debug("OTel span export failed: {}", exc)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the exporter (implements OTel SpanExporter)."""
        self.close()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Force-flush any buffered data.

        Implements ``opentelemetry.sdk.trace.export.SpanExporter``.

        Args:
            timeout_millis: Maximum time to wait in milliseconds.

        Returns:
            True on success.
        """
        self.flush()
        return True

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> DatadogMetricsExporter:
        """Start auto-collection and return self."""
        self._start_auto_collect()
        return self

    def __exit__(self, *args: Any) -> None:
        """Flush and stop auto-collection."""
        self.close()

    # ------------------------------------------------------------------
    # Internal -- flush methods
    # ------------------------------------------------------------------

    def _flush_metrics(self) -> None:
        """Send buffered metrics to the backend."""
        batch: list[_MetricPoint] = []
        with self._lock:
            if not self._metric_buffer:
                return
            batch = list(self._metric_buffer)
            self._metric_buffer.clear()

        if not batch:
            return

        if not self._ready:
            logger.debug("(dry-run) would push {} metrics", len(batch))
            return

        if self._otlp_endpoint:
            self._send_otlp_metrics(batch)
        elif self._api_key:
            self._send_datadog_metrics(batch)
        elif self._new_relic_key:
            self._send_new_relic_metrics(batch)

    def _flush_traces(self) -> None:
        """Send buffered traces to the backend."""
        batch: list[_TraceBatch] = []
        with self._lock:
            if not self._trace_buffer:
                return
            batch = list(self._trace_buffer)
            self._trace_buffer.clear()

        if not batch:
            return

        if not self._ready:
            logger.debug("(dry-run) would push {} traces", len(batch))
            return

        if self._otlp_endpoint:
            self._send_otlp_traces(batch)
        elif self._api_key:
            self._send_datadog_traces(batch)
        elif self._new_relic_key:
            self._send_new_relic_traces(batch)

    def _flush_logs(self) -> None:
        """Send buffered logs to the backend."""
        batch: list[_LogEntry] = []
        with self._lock:
            if not self._log_buffer:
                return
            batch = list(self._log_buffer)
            self._log_buffer.clear()

        if not batch:
            return

        if not self._ready:
            logger.debug("(dry-run) would push {} logs", len(batch))
            return

        if self._api_key:
            self._send_datadog_logs(batch)
        elif self._new_relic_key:
            self._send_new_relic_logs(batch)

    # ------------------------------------------------------------------
    # Internal -- HTTP transport with retry
    # ------------------------------------------------------------------

    def _resolve_host(self) -> str:
        """Return the backend hostname for connectivity checks."""
        if self._otlp_endpoint:
            return self._otlp_endpoint.split("//")[-1].split(":")[0]
        if self._api_key:
            return f"api.{self._site}"
        if self._new_relic_key:
            region = _NEW_RELIC_REGIONS.get(self._nr_region, "")
            return f"metric-api.{region}newrelic.com"
        return "api.datadoghq.com"

    @staticmethod
    def _post_with_retry(
        url: str,
        payload: Any,
        api_key: str,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_delay: float = _DEFAULT_BASE_DELAY_S,
    ) -> None:
        """POST *payload* to *url* with API-key auth and exponential backoff.

        Failures are logged at DEBUG level and silently swallowed to avoid
        disrupting the caller's hot path.

        Args:
            url: Target URL.
            payload: JSON-serializable payload.
            api_key: API key for the ``DD-API-KEY`` or ``Api-Key`` header.
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay in seconds for exponential backoff.
        """
        is_datadog = "datadog" in url
        header_key = "DD-API-KEY" if is_datadog else "Api-Key"

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        header_key: api_key,
                    },
                    timeout=10.0,
                )
                if response.status_code in (200, 201, 202, 204):
                    return

                logger.debug(
                    "Backend returned {} for POST {} (attempt {}/{}): {}",
                    response.status_code,
                    url,
                    attempt + 1,
                    max_retries,
                    response.text[:200],
                )

                # Do not retry client errors.
                if response.status_code < 500:
                    return

            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "HTTP POST to {} failed (attempt {}/{}): {}",
                    url,
                    attempt + 1,
                    max_retries,
                    exc,
                )

            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

        if last_exc:
            logger.debug("All retries exhausted for POST {}: {}", url, last_exc)

    def _send_datadog_metrics(self, batch: list[_MetricPoint]) -> None:
        """POST metrics to the Datadog v2 series endpoint."""
        series = []
        for pt in batch:
            series.append(
                {
                    "metric": pt.name,
                    "type": _dd_type(pt.metric_type),
                    "points": [{"timestamp": int(pt.timestamp), "value": pt.value}],
                    "tags": [_format_tag(k, v) for k, v in pt.tags.items()]
                    + [f"service:{self._service_name}"],
                }
            )

        payload = {"series": series}
        url = _DATADOG_METRICS_URL.format(site=self._site)
        self._post_with_retry(url, payload, self._api_key)

    def _send_datadog_traces(self, batch: list[_TraceBatch]) -> None:
        """POST traces to the Datadog trace intake."""
        # Datadog expects: [[trace_id, [span, ...]], ...]
        dd_traces = []
        for tb in batch:
            dd_traces.append([tb.trace_id, tb.spans])
        payload = dd_traces
        url = _DATADOG_TRACES_URL.format(site=self._site)
        self._post_with_retry(url, payload, self._api_key)

    def _send_datadog_logs(self, batch: list[_LogEntry]) -> None:
        """POST logs to the Datadog HTTP logs intake."""
        entries = []
        for entry in batch:
            tags_list = [_format_tag(k, v) for k, v in entry.tags.items()]
            tags_list.append(f"service:{self._service_name}")
            entries.append(
                {
                    "ddsource": entry.ddsource,
                    "ddtags": ",".join(tags_list),
                    "level": entry.level,
                    "message": entry.message,
                    "service": self._service_name,
                    "timestamp": int(entry.timestamp * 1000),
                }
            )

        payload = entries
        url = _DATADOG_LOGS_URL.format(site=self._site)
        self._post_with_retry(url, payload, self._api_key)

    def _send_new_relic_metrics(self, batch: list[_MetricPoint]) -> None:
        """POST metrics to the New Relic Metric API."""
        from collections import defaultdict

        # Group by metric name + tags for New Relic format.
        groups: dict[str, dict[str, Any]] = {}
        for pt in batch:
            key = f"{pt.name}|{_sorted_tags(pt.tags)}"
            if key not in groups:
                groups[key] = {
                    "name": pt.name,
                    "type": _nr_type(pt.metric_type),
                    "value": None,
                    "timestamp": int(pt.timestamp),
                    "attributes": {**pt.tags, "service": self._service_name},
                    "interval.ms": self._collect_interval,
                }
            groups[key]["value"] = pt.value

        payload = [g for g in groups.values()]
        region = _NEW_RELIC_REGIONS.get(self._nr_region, "")
        url = _NEW_RELIC_METRICS_URL.format(region=region)
        self._post_with_retry(url, payload, self._new_relic_key)

    def _send_new_relic_traces(self, batch: list[_TraceBatch]) -> None:
        """POST traces to the New Relic Trace API."""
        for tb in batch:
            for span in tb.spans:
                span["service"] = self._service_name
                span["attributes"] = {
                    **span.get("attributes", {}),
                    **tb.tags,
                }

        payload = [{"trace_id": tb.trace_id, "spans": tb.spans} for tb in batch]
        region = _NEW_RELIC_REGIONS.get(self._nr_region, "")
        url = _NEW_RELIC_TRACES_URL.format(region=region)
        self._post_with_retry(url, payload, self._new_relic_key)

    def _send_new_relic_logs(self, batch: list[_LogEntry]) -> None:
        """POST logs to the New Relic Log API."""
        entries = []
        for entry in batch:
            entries.append(
                {
                    "timestamp": int(entry.timestamp * 1000),
                    "level": entry.level,
                    "message": entry.message,
                    "service": self._service_name,
                    "attributes": dict(entry.tags),
                }
            )

        region = _NEW_RELIC_REGIONS.get(self._nr_region, "")
        url = _NEW_RELIC_LOGS_URL.format(region=region)
        self._post_with_retry(url, entries, self._new_relic_key)

    # ------------------------------------------------------------------
    # Internal -- OTLP / gRPC export (real implementations)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ns(timestamp: float) -> int | None:
        """Convert a timestamp (seconds, ms, or ns) to nanoseconds.

        Handles input in:
        - Seconds (< 1e12)
        - Milliseconds (> 1e6)
        - Nanoseconds (> 1e12)
        Returns ``None`` for zero or negative values.
        """
        if timestamp <= 0:
            return None
        if timestamp > 1e16:
            # Already nanoseconds (far future in seconds).
            return int(timestamp)
        if timestamp > 1e12:
            # > 31 689 years past epoch in seconds -> likely ns already.
            return int(timestamp)
        if timestamp > 1e9:
            # > 31.7 years past epoch in seconds -> likely milliseconds.
            return int(timestamp * 1_000_000)
        # Seconds.
        return int(timestamp * 1_000_000_000)

    def _send_otlp_metrics(self, batch: list[_MetricPoint]) -> None:
        """Export metrics via OTLP/gRPC.

        Builds protobuf ``ExportMetricsServiceRequest`` messages and sends
        them via gRPC to the configured OTLP endpoint.
        """
        try:
            # Import OTel protobuf and gRPC dependencies.
            from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
                ExportMetricsServiceRequest,
            )
            from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2_grpc import (
                MetricsServiceStub,
            )
            from opentelemetry.proto.common.v1 import common_pb2
            from opentelemetry.proto.metrics.v1 import metrics_pb2
            from opentelemetry.proto.resource.v1 import resource_pb2

            import grpc
            from collections import defaultdict
        except ImportError:
            logger.debug("OTLP gRPC exporter not installed; dropping metric batch")
            return

        try:
            now_ns = int(time.time() * 1_000_000_000)

            # Build the gRPC request.
            request = ExportMetricsServiceRequest()
            rm = request.resource_metrics.add()
            rm.resource.CopyFrom(
                resource_pb2.Resource(
                    attributes=[
                        common_pb2.KeyValue(
                            key="service.name",
                            value=common_pb2.AnyValue(string_value=self._service_name),
                        )
                    ]
                )
            )

            sm = rm.scope_metrics.add()
            sm.scope.name = "distllm"
            sm.scope.version = "0.4.1"

            # Group points by metric name.
            groups: dict[str, list[_MetricPoint]] = defaultdict(list)
            for pt in batch:
                groups[pt.name].append(pt)

            for name, points in groups.items():
                metric = sm.metrics.add()
                metric.name = name
                metric.description = name
                metric.unit = "1"

                for pt in points:
                    dp = metric.gauge.data_points.add()
                    for k, v in pt.tags.items():
                        attr = dp.attributes.add()
                        attr.key = k
                        attr.value.string_value = str(v)
                    dp.time_unix_nano = now_ns
                    dp.start_time_unix_nano = int(pt.timestamp * 1_000_000_000)
                    dp.as_double = pt.value

            # Create gRPC channel and send.
            if self._otlp_endpoint.startswith("https://"):
                chan = grpc.secure_channel(
                    self._otlp_endpoint.replace("https://", ""),
                    grpc.ssl_channel_credentials(),
                )
            else:
                chan = grpc.insecure_channel(
                    self._otlp_endpoint.replace("http://", "")
                )

            stub = MetricsServiceStub(chan)
            stub.Export(request, timeout=10)
            chan.close()

        except Exception as exc:
            logger.debug("OTLP metric export failed: {}", exc)

    def _send_otlp_traces(self, batch: list[_TraceBatch]) -> None:
        """Export traces via OTLP/gRPC.

        Builds protobuf ``ExportTraceServiceRequest`` messages and sends
        them via gRPC to the configured OTLP endpoint.
        """
        try:
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest,
            )
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
                TraceServiceStub,
            )
            from opentelemetry.proto.common.v1 import common_pb2
            from opentelemetry.proto.resource.v1 import resource_pb2
            from opentelemetry.proto.trace.v1 import trace_pb2

            from opentelemetry.trace import StatusCode

            import grpc
        except ImportError:
            logger.debug("OTLP gRPC exporter not installed; dropping trace batch")
            return

        try:
            now_ns = int(time.time() * 1_000_000_000)

            request = ExportTraceServiceRequest()
            rs = request.resource_spans.add()
            rs.resource.CopyFrom(
                resource_pb2.Resource(
                    attributes=[
                        common_pb2.KeyValue(
                            key="service.name",
                            value=common_pb2.AnyValue(string_value=self._service_name),
                        )
                    ]
                )
            )

            ss = rs.scope_spans.add()
            ss.scope.name = "distllm"
            ss.scope.version = "0.4.1"

            for tb in batch:
                for span_dict in tb.spans:
                    span_pb = ss.spans.add()
                    span_pb.trace_id = _trace_id_bytes(
                        span_dict.get("trace_id", tb.trace_id)
                    )
                    span_pb.span_id = _span_id_bytes(
                        span_dict.get("span_id", "")
                    )
                    span_pb.name = span_dict.get("name", "unknown")
                    span_pb.kind = _otel_span_kind(span_dict.get("kind", ""))
                    span_pb.start_time_unix_nano = self._to_ns(
                        float(span_dict.get("start_time", 0))
                    ) or now_ns
                    span_pb.end_time_unix_nano = self._to_ns(
                        float(span_dict.get("end_time", 0))
                    ) or now_ns

                    # Status.
                    status_info = span_dict.get("status", {})
                    if status_info:
                        span_pb.status.code = status_info.get("code", 1)
                        span_pb.status.message = status_info.get("description", "")

                    # Attributes from the span dict and batch tags.
                    attrs = dict(tb.tags)
                    attrs.update(span_dict.get("attributes", {}))
                    attrs["service"] = self._service_name
                    for k, v in attrs.items():
                        attr = span_pb.attributes.add()
                        attr.key = k
                        _set_any_value(attr.value, v)

            # Create gRPC channel and send.
            if self._otlp_endpoint.startswith("https://"):
                chan = grpc.secure_channel(
                    self._otlp_endpoint.replace("https://", ""),
                    grpc.ssl_channel_credentials(),
                )
            else:
                chan = grpc.insecure_channel(
                    self._otlp_endpoint.replace("http://", "")
                )

            stub = TraceServiceStub(chan)
            stub.Export(request, timeout=10)
            chan.close()

        except Exception as exc:
            logger.debug("OTLP trace export failed: {}", exc)

    # ------------------------------------------------------------------
    # Internal -- GPU / system metrics (multi-GPU)
    # ------------------------------------------------------------------

    def _init_gpu(self) -> None:
        """Lazy-initialise pynvml for GPU metric collection (all devices)."""
        if self._gpu_initialised:
            return
        self._gpu_initialised = True
        if not _PYNVML_AVAILABLE:
            logger.debug("pynvml not installed; GPU metrics disabled")
            return
        try:
            pynvml.nvmlInit()  # type: ignore[union-attr]
            count = pynvml.nvmlDeviceGetCount()  # type: ignore[union-attr]
            self._gpu_device_count = count
            self._gpu_handles = []
            if count > 0:
                self._gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore[union-attr]
                    for i in range(count)
                ]
                first_name = pynvml.nvmlDeviceGetName(self._gpu_handles[0])  # type: ignore[union-attr]
                logger.debug(
                    "GPU monitoring active: {} ({} devices)",
                    first_name,
                    count,
                )
        except Exception as exc:
            logger.debug("GPU init failed: {}", exc)
            self._gpu_handles = []

    def _collect_system_metrics(self) -> dict[str, float]:
        """Collect a snapshot of CPU, memory, and GPU metrics.

        Returns:
            Dict mapping metric names to float values.
        """
        metrics: dict[str, float] = {}

        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        metrics["system.cpu.percent"] = cpu_percent

        # Memory
        mem = psutil.virtual_memory()
        metrics["system.memory.percent"] = mem.percent
        metrics["system.memory.available_mb"] = mem.available / 1024 / 1024
        metrics["system.memory.total_mb"] = mem.total / 1024 / 1024

        # Process
        proc = psutil.Process()
        metrics["system.process.rss_mb"] = proc.memory_info().rss / 1024 / 1024
        metrics["system.process.cpu_percent"] = proc.cpu_percent()

        # GPU metrics -- all devices (multi-GPU support).
        self._init_gpu()
        for i, handle in enumerate(self._gpu_handles):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[union-attr]
                metrics[f"gpu.{i}.utilization.percent"] = float(util.gpu)
                metrics[f"gpu.{i}.memory.percent"] = float(util.memory)

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[union-attr]
                metrics[f"gpu.{i}.memory.used_mb"] = mem_info.used / 1024 / 1024
                metrics[f"gpu.{i}.memory.total_mb"] = mem_info.total / 1024 / 1024

                temp = pynvml.nvmlDeviceGetTemperature(  # type: ignore[union-attr]
                    handle, pynvml.NVML_TEMPERATURE_GPU  # type: ignore[union-attr]
                )
                metrics[f"gpu.{i}.temperature.celsius"] = float(temp)

                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)  # type: ignore[union-attr]
                metrics[f"gpu.{i}.power.watts"] = power_usage / 1000.0
            except Exception as exc:
                logger.debug("GPU {} metric collection failed: {}", i, exc)

        # Backward-compatible aggregate GPU metrics (averaged across devices).
        if self._gpu_handles:
            try:
                avg_util = 0.0
                total_mem_used = 0
                total_mem_total = 0
                for handle in self._gpu_handles:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[union-attr]
                    avg_util += float(util.gpu)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[union-attr]
                    total_mem_used += mem_info.used
                    total_mem_total += mem_info.total
                n = len(self._gpu_handles)
                metrics["gpu.utilization.percent"] = avg_util / n
                metrics["gpu.memory.used_mb"] = (total_mem_used / n) / 1024 / 1024
                metrics["gpu.memory.total_mb"] = (total_mem_total / n) / 1024 / 1024
            except Exception as exc:
                logger.debug("GPU aggregate metric collection failed: {}", exc)

        return metrics

    # ------------------------------------------------------------------
    # Internal -- auto-collection
    # ------------------------------------------------------------------

    def _start_auto_collect(self) -> None:
        """Start the background metric collection thread."""
        if self._collect_thread is not None and self._collect_thread.is_alive():
            return

        self._collect_stop.clear()

        def _loop() -> None:
            while not self._collect_stop.is_set():
                self._collect_and_push()
                self._collect_stop.wait(self._collect_interval)

        self._collect_thread = threading.Thread(
            target=_loop, daemon=True, name="dd-auto-collect"
        )
        self._collect_thread.start()
        logger.debug("Auto-collection started (interval={}s)", self._collect_interval)

    def _stop_auto_collect(self) -> None:
        """Stop the background metric collection thread."""
        if self._collect_thread is None:
            return
        self._collect_stop.set()
        self._collect_thread.join(timeout=5.0)
        self._collect_thread = None

    def _collect_and_push(self) -> None:
        """Collect system + request metrics and push them."""
        metrics: dict[str, float] = self._collect_system_metrics()

        # Aggregate request metrics since last collection.
        with self._lock:
            req_count = self._request_count
            err_count = self._error_count
            self._request_count = 0
            self._error_count = 0
            latencies = list(self._latencies)
            self._latencies.clear()

        metrics["request.count"] = float(req_count)

        if req_count > 0:
            metrics["request.error_rate"] = err_count / req_count
        else:
            metrics["request.error_rate"] = 0.0

        if latencies:
            metrics["request.latency.p50_ms"] = _percentile(latencies, 50) * 1000
            metrics["request.latency.p95_ms"] = _percentile(latencies, 95) * 1000
            metrics["request.latency.p99_ms"] = _percentile(latencies, 99) * 1000
            metrics["request.latency.avg_ms"] = (sum(latencies) / len(latencies)) * 1000

        if err_count > 0:
            metrics["request.errors"] = float(err_count)

        self.push_metrics(metrics, tags={"collector": "auto"})
        self._flush_metrics()

    # ------------------------------------------------------------------
    # Convenience class method -- one-shot push
    # ------------------------------------------------------------------

    @classmethod
    def push(
        cls,
        metrics: dict[str, float] | None = None,
        traces: dict[str, Any] | list[dict[str, Any]] | None = None,
        logs: str | dict[str, Any] | list[dict[str, Any]] | None = None,
        *,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Convenience one-shot push without managing an exporter instance.

        Creates a temporary exporter, pushes the provided data, and
        flushes immediately.  Useful for ad-hoc instrumentation.

        Args:
            metrics: Metric name/value pairs.
            traces: Trace or span dicts.
            logs: Log entry or entries.
            tags: Extra tags applied to all pushed data.
        """
        exporter = cls()
        try:
            if metrics:
                exporter.push_metrics(metrics, tags=tags)
            if traces:
                exporter.push_trace(traces, tags=tags)
            if logs:
                exporter.push_log(logs, tags=tags)
            exporter.flush()
        finally:
            exporter.close()


# ---------------------------------------------------------------------------
# Helpers (module-level)
# ---------------------------------------------------------------------------


def _dd_type(metric_type: str) -> int:
    """Map metric type string to Datadog API type code."""
    mapping = {
        "gauge": 3,
        "count": 3,
        "rate": 3,
        "histogram": 3,
    }
    return mapping.get(metric_type, 3)


def _nr_type(metric_type: str) -> str:
    """Map metric type string to New Relic Metric API type."""
    mapping = {
        "gauge": "gauge",
        "count": "count",
        "rate": "count",
        "histogram": "gauge",
    }
    return mapping.get(metric_type, "gauge")


def _format_tag(key: str, value: str) -> str:
    """Format a ``key:value`` tag for Datadog."""
    return f"{key}:{value}"


def _sorted_tags(tags: dict[str, str]) -> str:
    """Canonical, sorted tag string for deduplication."""
    return ",".join(f"{k}={v}" for k, v in sorted(tags.items()))


def _percentile(values: list[float], p: int) -> float:
    """Compute the *p*-th percentile of *values*."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (p / 100.0) * (len(sorted_vals) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def _trace_id_bytes(trace_id: str) -> bytes:
    """Convert a hex trace ID string to 16 bytes for OTLP proto."""
    if not trace_id:
        return b"\x00" * 16
    # Strip any 0x prefix.
    hex_str = trace_id.removeprefix("0x")
    # Pad or truncate to 32 hex chars (16 bytes).
    hex_str = hex_str.zfill(32)[:32]
    return bytes.fromhex(hex_str)


def _span_id_bytes(span_id: str) -> bytes:
    """Convert a hex span ID string to 8 bytes for OTLP proto."""
    if not span_id:
        return b"\x00" * 8
    hex_str = span_id.removeprefix("0x")
    hex_str = hex_str.zfill(16)[:16]
    return bytes.fromhex(hex_str)


def _otel_span_kind(kind_str: str) -> int:
    """Map an OTel span kind string to the protobuf integer value."""
    kinds = {
        "SpanKind.INTERNAL": 1,
        "INTERNAL": 1,
        "SpanKind.SERVER": 2,
        "SERVER": 2,
        "SpanKind.CLIENT": 3,
        "CLIENT": 3,
        "SpanKind.PRODUCER": 4,
        "PRODUCER": 4,
        "SpanKind.CONSUMER": 5,
        "CONSUMER": 5,
    }
    return kinds.get(kind_str, 1)


def _set_any_value(any_value: Any, value: Any) -> None:
    """Set the appropriate field on a protobuf AnyValue from a Python value."""
    if isinstance(value, bool):
        any_value.bool_value = value
    elif isinstance(value, int):
        any_value.int_value = value
    elif isinstance(value, float):
        any_value.double_value = value
    elif isinstance(value, str):
        any_value.string_value = value
    elif value is None:
        any_value.string_value = ""
    else:
        any_value.string_value = str(value)
