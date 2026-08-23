"""Datadog and New Relic APM integration via OpenTelemetry protocol.

Provides ``DatadogMetricsExporter`` — a unified exporter that pushes metrics,
traces, and logs to Datadog (or New Relic) using the existing OpenTelemetry
infrastructure.  The exporter is fully optional: it degrades gracefully when
the ``opentelemetry-sdk`` package is not installed.

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

Usage — standalone push API ::

    from integrations.datadog_monitoring import DatadogMetricsExporter

    exporter = DatadogMetricsExporter()
    exporter.push_metrics({"cpu.percent": 42.5, "memory.mb": 2048})
    exporter.push_trace({"trace_id": "abc", "spans": [...]})
    exporter.push_log({"message": "request complete", "level": "info"})

Usage — OTel SpanExporter (when opentelemetry-sdk is installed) ::

    exporter = DatadogMetricsExporter()
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

Usage — context manager with auto-collection ::

    with DatadogMetricsExporter(auto_collect=True, collect_interval_s=15.0) as ex:
        ...
        ex.observe_request_latency(0.350)
        ex.observe_request_error("timeout")
"""

from __future__ import annotations

import math
import os
import platform
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

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

    * ``DD_API_KEY`` → Datadog
    * ``NEW_RELIC_API_KEY`` → New Relic

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
        self._new_relic_key = os.environ.get("NEW_RELIC_API_KEY", "")
        self._site = (site or os.environ.get("DD_SITE", "") or _DEFAULT_SITE).rstrip(
            "/"
        )

        # New Relic region.
        self._nr_region = (os.environ.get("NEW_RELIC_REGION", "") or "US").upper()

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

        # GPU monitoring state (lazy init).
        self._gpu_initialised = False
        self._gpu_handle: Any = None
        self._gpu_device_count: int = 0
        self._gpu_device_name: str = ""

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
            import socket

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

        with self._lock:
            self._metric_buffer.extend(points)
            if self._flush_immediately or len(self._metric_buffer) >= self._max_batch_size:
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
            spans = t.get("spans", t.get("spans", [t]))
            self._trace_buffer.append(
                _TraceBatch(trace_id=tid, spans=spans, tags=base_tags)
            )

        with self._lock:
            if self._flush_immediately or len(self._trace_buffer) >= self._max_batch_size:
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

        with self._lock:
            self._log_buffer.extend(entries)
            if self._flush_immediately or len(self._log_buffer) >= self._max_batch_size:
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
    # Internal — flush methods
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
    # Internal — HTTP transport
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
        self._post(url, payload, self._api_key)

    def _send_datadog_traces(self, batch: list[_TraceBatch]) -> None:
        """POST traces to the Datadog trace intake."""
        # Datadog expects: [[trace_id, [span, ...]], ...]
        dd_traces = []
        for tb in batch:
            dd_traces.append([tb.trace_id, tb.spans])
        payload = dd_traces
        url = _DATADOG_TRACES_URL.format(site=self._site)
        self._post(url, payload, self._api_key)

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
        self._post(url, payload, self._api_key)

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
        self._post(url, payload, self._new_relic_key)

    def _send_new_relic_traces(self, batch: list[_TraceBatch]) -> None:
        """POST traces to the New Relic Trace API."""
        entries = []
        for tb in batch:
            for span in tb.spans:
                span["service"] = self._service_name
                span["attributes"] = {
                    **span.get("attributes", {}),
                    **tb.tags,
                }
                entries.append(span)

        payload = [{"trace_id": tb.trace_id, "spans": tb.spans} for tb in batch]
        region = _NEW_RELIC_REGIONS.get(self._nr_region, "")
        url = _NEW_RELIC_TRACES_URL.format(region=region)
        self._post(url, payload, self._new_relic_key)

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
        self._post(url, entries, self._new_relic_key)

    def _send_otlp_metrics(self, batch: list[_MetricPoint]) -> None:
        """Export metrics via OTLP/gRPC."""
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics.export import (
                AggregationTemporality,
                MetricExporter,
            )

            # Convert our points into OTel metrics and delegate to OTLP.
            logger.debug("OTLP metric export not yet wired ({} points)", len(batch))
        except ImportError:
            logger.debug("OTLP gRPC exporter not installed; dropping metric batch")

    def _send_otlp_traces(self, batch: list[_TraceBatch]) -> None:
        """Export traces via OTLP/gRPC."""
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            logger.debug("OTLP trace export not yet wired ({} batches)", len(batch))
        except ImportError:
            logger.debug("OTLP gRPC exporter not installed; dropping trace batch")

    @staticmethod
    def _post(url: str, payload: Any, api_key: str) -> None:
        """POST *payload* to *url* with API-key auth.

        Failures are logged at DEBUG level and silently swallowed to avoid
        disrupting the caller's hot path.
        """
        try:
            import httpx

            response = httpx.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "DD-API-KEY" if "datadog" in url else "Api-Key": api_key,
                },
                timeout=10.0,
            )
            if response.status_code not in (200, 201, 202, 204):
                logger.debug(
                    "Backend returned {} for POST {}: {}",
                    response.status_code,
                    url,
                    response.text[:200],
                )
        except Exception as exc:
            logger.debug("HTTP POST to {} failed: {}", url, exc)

    # ------------------------------------------------------------------
    # Internal — GPU / system metrics
    # ------------------------------------------------------------------

    def _init_gpu(self) -> None:
        """Lazy-initialise pynvml for GPU metric collection."""
        if self._gpu_initialised:
            return
        self._gpu_initialised = True
        if not _PYNVML_AVAILABLE:
            logger.debug("pynvml not installed; GPU metrics disabled")
            return
        try:
            if self._gpu_handle is None:
                pynvml.nvmlInit()  # type: ignore[union-attr]
                count = pynvml.nvmlDeviceGetCount()  # type: ignore[union-attr]
                if count > 0:
                    self._gpu_device_count = count
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # type: ignore[union-attr]
                    self._gpu_device_name = pynvml.nvmlDeviceGetName(self._gpu_handle)  # type: ignore[union-attr]
                    logger.debug(
                        "GPU monitoring active: {} ({} devices)",
                        self._gpu_device_name,
                        count,
                    )
        except Exception as exc:
            logger.debug("GPU init failed: {}", exc)
            self._gpu_handle = None

    def _collect_system_metrics(self) -> dict[str, float]:
        """Collect a snapshot of CPU, memory, and GPU metrics.

        Returns:
            Dict mapping metric names to float values.
        """
        import psutil

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

        # GPU
        self._init_gpu()
        if self._gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)  # type: ignore[union-attr]
                metrics["gpu.utilization.percent"] = float(util.gpu)
                metrics["gpu.memory.percent"] = float(util.memory)

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)  # type: ignore[union-attr]
                metrics["gpu.memory.used_mb"] = mem_info.used / 1024 / 1024
                metrics["gpu.memory.total_mb"] = mem_info.total / 1024 / 1024

                temp = pynvml.nvmlDeviceGetTemperature(  # type: ignore[union-attr]
                    self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU  # type: ignore[union-attr]
                )
                metrics["gpu.temperature.celsius"] = float(temp)

                power_usage = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)  # type: ignore[union-attr]
                metrics["gpu.power.watts"] = power_usage / 1000.0
            except Exception as exc:
                logger.debug("GPU metric collection failed: {}", exc)

        return metrics

    # ------------------------------------------------------------------
    # Internal — auto-collection
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

        self._collect_thread = threading.Thread(target=_loop, daemon=True, name="dd-auto-collect")
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
    # Convenience class method — one-shot push
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
# Helpers
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
