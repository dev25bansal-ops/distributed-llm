"""Native OpenTelemetry LogRecord bridge for distributed-llm.

This module makes every stdlib :class:`logging.LogRecord` emitted inside an
active OpenTelemetry span carry ``trace_id`` / ``span_id`` / ``trace_flags`` and
(optionally) forward to an OTLP log collector (Tempo / Loki / OTel Collector)
via the SDK ``LoggingHandler``.

Two independent capabilities, each with graceful fallback:

1.  **Trace-context injection into LogRecords** (works whenever the OTel *API*
    is importable).  We install a ``logging.LogRecordFactory`` so that **every**
    record -- no matter which logger emits it -- gets ``record.trace_id``,
    ``record.span_id`` and ``record.trace_flags`` set from the current span
    context at creation time.  Text logs are thereby trace-correlated even when
    OTLP export is disabled/unavailable.  (A :class:`OTelTraceContextFilter`
    is also provided for handler-level use if desired.)

2.  **OTLP export of logs** (only when ``opentelemetry-sdk`` *and* the OTLP log
    exporter are importable).  When an ``otlp_endpoint`` is supplied we attach a
    :class:`opentelemetry.sdk._logs.LoggingHandler` backed by a
    ``LoggerProvider`` + ``BatchLogRecordProcessor`` + ``OTLPLogExporter``.

If ``opentelemetry`` is entirely missing, or the SDK / OTLP exporter cannot be
imported, nothing crashes: we record the intent (a ``logging`` debug record) and
return a ``status`` dict describing what was and was not wired.  This keeps the
function safe to call unconditionally from ``api/server.py`` startup.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Module-level state so callers (and tests) can introspect what was wired.
_OTEL_LOGGING_STATE: dict[str, Any] = {
    "trace_context_injected": False,
    "handler_attached": False,
    "otlp_endpoint": None,
    "sdk_available": False,
    "otlp_exporter_available": False,
    "status": "not_initialised",
}
_STATE_LOCK = threading.Lock()

# Guard so the LogRecordFactory is wrapped at most once per process.
_FACTORY_INSTALLED = False
_FACTORY_LOCK = threading.Lock()


# ── Span context extraction ────────────────────────────────────────────────
def _extract_span_context() -> tuple[str, str, int]:
    """Return ``(trace_id, span_id, trace_flags)`` for the current span.

    ``trace_id`` / ``span_id`` are empty strings and ``trace_flags`` is ``0``
    when there is no active valid span.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx is not None and ctx.trace_id and ctx.is_valid:
            return (
                f"{ctx.trace_id:032x}",
                f"{ctx.span_id:016x}",
                int(ctx.trace_flags),
            )
    except Exception:
        # OTel API absent, or span context extraction failed -- not fatal.
        pass
    return ("", "", 0)


# ── Injection mechanism: LogRecordFactory (covers every record) ─────────────
def _make_trace_context_factory(
    original: Callable[..., logging.LogRecord],
) -> Callable[..., logging.LogRecord]:
    """Wrap a LogRecord factory so produced records carry span context."""

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = original(*args, **kwargs)
        trace_id, span_id, trace_flags = _extract_span_context()
        record.trace_id = trace_id
        record.span_id = span_id
        record.trace_flags = trace_flags
        return record

    return factory


def _install_trace_context_factory() -> bool:
    """Install the trace-context ``LogRecordFactory`` (idempotent)."""
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return True
    with _FACTORY_LOCK:
        if _FACTORY_INSTALLED:
            return True
        try:
            original = logging.getLogRecordFactory()
            logging.setLogRecordFactory(_make_trace_context_factory(original))
            _FACTORY_INSTALLED = True
            return True
        except Exception:
            return False


class OTelTraceContextFilter(logging.Filter):
    """Inject the active span's ``trace_id`` / ``span_id`` into a record.

    Provided for handler-level use. The default bridge uses a
    :class:`logging.LogRecordFactory` instead (which is more robust because it
    covers records emitted by *any* logger, not just those routed through a
    particular handler).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        trace_id, span_id, trace_flags = _extract_span_context()
        record.trace_id = trace_id
        record.span_id = span_id
        record.trace_flags = trace_flags
        return True


# ── Testable import helpers (so tests can simulate SDK / exporter presence) ──
def _import_sdk_logs():
    """Import and return the OTel SDK logs primitives, or raise ImportError."""
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    return {
        "LoggerProvider": LoggerProvider,
        "LoggingHandler": LoggingHandler,
        "BatchLogRecordProcessor": BatchLogRecordProcessor,
        "Resource": Resource,
    }


def _import_otlp_log_exporter():
    """Import and return the OTLP log exporter class, or raise ImportError."""
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

    return OTLPLogExporter


# ── Public API ──────────────────────────────────────────────────────────────
def setup_otel_logging(
    service_name: str = "distllm",
    otlp_endpoint: str | None = None,
    enable: bool = True,
) -> dict[str, Any]:
    """Wire the native OTel logging bridge onto the root logger.

    Args:
        service_name: Service identifier used for the OTel ``Resource`` and as
            the fallback ``service.name`` attribute on exported logs.
        otlp_endpoint: If set, attach a :class:`LoggingHandler` that exports
            logs over OTLP to the given collector URL.  When ``None`` (default)
            only the trace-context injection factory is installed (text logs
            stay trace-correlated but nothing is exported).
        enable: Master switch.  When ``False`` no bridge is installed and a
            status of ``"disabled"`` is returned.

    Returns:
        A status dict describing what was wired, e.g.::

            {
                "status": "ok",
                "trace_context_injected": True,
                "handler_attached": True,
                "otlp_endpoint": "http://localhost:4318/v1/logs",
                "sdk_available": True,
                "otlp_exporter_available": True,
            }

        The ``status`` field is one of ``"ok"``, ``"disabled"``,
        ``"otel_api_missing"``, ``"sdk_missing"`` or ``"otlp_exporter_missing"``.
    """
    with _STATE_LOCK:
        if not enable:
            _OTEL_LOGGING_STATE.update(
                {
                    "trace_context_injected": False,
                    "handler_attached": False,
                    "otlp_endpoint": otlp_endpoint,
                    "status": "disabled",
                }
            )
            logger.debug("OTel logging bridge disabled by caller")
            return dict(_OTEL_LOGGING_STATE)

        root_logger = logging.getLogger()

        # ── Capability probe ───────────────────────────────────────────
        sdk_available = False
        otlp_exporter_available = False
        sdk = None
        try:
            sdk = _import_sdk_logs()
            sdk_available = True
        except Exception:
            sdk_available = False

        try:
            _import_otlp_log_exporter()
            otlp_exporter_available = True
        except Exception:
            otlp_exporter_available = False

        # ── Always: trace-context injection into LogRecords ────────────
        # Uses a LogRecordFactory so EVERY stdlib record (any logger) carries
        # trace_id/span_id. Only needs the OTel *API*; if even that is missing
        # we gracefully skip and report.
        trace_context_injected = False
        try:
            from opentelemetry import trace  # noqa: F401  (API probe)

            trace_context_injected = _install_trace_context_factory()
        except Exception:
            trace_context_injected = False

        # ── Optionally: OTLP export handler ───────────────────────────
        handler_attached = False
        status = "ok"
        if otlp_endpoint:
            if not sdk_available:
                status = "sdk_missing"
                logger.debug(
                    "OTel SDK not available -- skipping OTLP log handler. "
                    "Install opentelemetry-sdk to enable log export."
                )
            elif not otlp_exporter_available:
                status = "otlp_exporter_missing"
                logger.debug(
                    "OTLP log exporter not available -- skipping OTLP log "
                    "handler. Install opentelemetry-exporter-otlp-proto-http."
                )
            else:
                try:
                    OTLPLogExporter = _import_otlp_log_exporter()
                    LoggerProvider = sdk["LoggerProvider"]
                    LoggingHandler = sdk["LoggingHandler"]
                    BatchLogRecordProcessor = sdk["BatchLogRecordProcessor"]
                    Resource = sdk["Resource"]

                    provider = LoggerProvider(
                        resource=Resource.create({"service.name": service_name})
                    )
                    provider.add_log_record_processor(
                        BatchLogRecordProcessor(OTLPLogExporter(endpoint=otlp_endpoint))
                    )
                    handler = LoggingHandler(
                        level=logging.NOTSET, logger_provider=provider
                    )
                    root_logger.addHandler(handler)
                    handler_attached = True
                except Exception as e:  # never crash startup on exporter issues
                    status = "otlp_exporter_missing"
                    logger.debug(f"Failed to attach OTLP log handler: {e}")

        if not trace_context_injected and not sdk_available:
            status = "otel_api_missing"

        _OTEL_LOGGING_STATE.update(
            {
                "trace_context_injected": trace_context_injected,
                "handler_attached": handler_attached,
                "otlp_endpoint": otlp_endpoint,
                "sdk_available": sdk_available,
                "otlp_exporter_available": otlp_exporter_available,
                "status": status,
            }
        )
        logger.debug(
            "OTel logging bridge initialised: status=%s injected=%s handler=%s",
            status,
            trace_context_injected,
            handler_attached,
        )
        return dict(_OTEL_LOGGING_STATE)


def get_otel_logging_state() -> dict[str, Any]:
    """Return the current bridge state (useful for diagnostics / tests)."""
    with _STATE_LOCK:
        return dict(_OTEL_LOGGING_STATE)
