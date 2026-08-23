"""N2 — Regression test for the native OpenTelemetry LogRecord bridge.

Proves (model-free, no live Tempo/Grafana collector):

  (1) A stdlib log record emitted *inside* an active OTel span carries that
      span's ``trace_id`` + ``span_id`` (and ``trace_flags``) via the
      trace-context ``LogRecordFactory`` installed by
      :func:`setup_otel_logging`.
  (2) Outside a span, those fields are absent / zero — and crucially the call
      does **not** crash (graceful behaviour).
  (3) When the OTel SDK is present, :func:`setup_otel_logging` reports a state
      where the bridge was actually wired (``trace_context_injected`` True and,
      for an ``otlp_endpoint`` with an exporter available, ``handler_attached``
      True).  On a real exporter we additionally confirm a live
      :class:`LoggingHandler` is attached to the root logger.
  (4) Graceful no-op when the OTLP exporter is unavailable: with the SDK present
      but no ``opentelemetry-exporter-otlp-proto-http`` package, requesting an
      ``otlp_endpoint`` must NOT raise and must report
      ``status == "otlp_exporter_missing"`` (or skip the handler cleanly).

Honest scope: this verifies the *trace-context injection into LogRecords* end
to end, plus the wiring/return-contract of the OTLP handler path.  Actual OTLP
export to a collector is wired but not exercised against a live Tempo/Loki here.
"""

from __future__ import annotations

import logging

import pytest

from distllm.observability import otel_logging


# ── Test helpers ────────────────────────────────────────────────────────────
def _make_capturing_logger() -> tuple[logging.Logger, list[logging.LogRecord]]:
    """Return a fresh stdlib logger + list that captures every emitted record.

    Uses its own handler (not the root logger) so the assertions are isolated
    from global logging configuration / loguru.  The trace-context factory is
    installed on the *module-global* LogRecordFactory, so records created by
    this logger inherit the injected trace_id/span_id regardless of handler.
    """
    captured: list[logging.LogRecord] = []
    test_logger = logging.getLogger(f"n2_test.{id(captured)}")
    test_logger.setLevel(logging.DEBUG)
    test_logger.propagate = False
    # Remove any handlers from previous runs (id is unique, so usually none).
    for h in list(test_logger.handlers):
        test_logger.removeHandler(h)

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _Capture()
    test_logger.addHandler(handler)
    return test_logger, captured


def _active_span_tuple():
    """Create a real, valid OTel span via the SDK and return (cm, ctx)."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({"service.name": "n2-test"}))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("n2-test")
    span_cm = tracer.start_as_current_span("n2-span")
    span = span_cm.__enter__()
    ctx = span.get_span_context()
    return span_cm, ctx


# ── Tests ───────────────────────────────────────────────────────────────────
def test_01_log_record_inside_span_carries_trace_and_span_id():
    """A record emitted within an active span gets the span's ids."""
    # Ensure the factory is installed (idempotent).  In CI the SDK is present.
    state = otel_logging.setup_otel_logging(
        service_name="n2-test", otlp_endpoint=None, enable=True
    )
    # The environment under test has the OTel SDK installed.
    assert state["sdk_available"] is True, (
        "N2 requires the opentelemetry SDK in .venv311 to prove injection"
    )
    assert state["trace_context_injected"] is True

    test_logger, captured = _make_capturing_logger()
    span_cm, ctx = _active_span_tuple()
    try:
        test_logger.info("hello from inside a span")
    finally:
        span_cm.__exit__(None, None, None)

    assert captured, "expected at least one captured record"
    rec = captured[0]
    assert rec.trace_id == f"{ctx.trace_id:032x}", (
        f"trace_id mismatch: rec={rec.trace_id!r} span={ctx.trace_id:032x}"
    )
    assert rec.span_id == f"{ctx.span_id:016x}", (
        f"span_id mismatch: rec={rec.span_id!r} span={ctx.span_id:016x}"
    )
    # trace_flags is the raw W3C flags byte (int), present and non-negative.
    assert isinstance(rec.trace_flags, int)
    assert rec.trace_flags == int(ctx.trace_flags)


def test_02_log_record_outside_span_has_no_crash_and_empty_ids():
    """Outside a span the ids are empty/zero and nothing raises."""
    otel_logging.setup_otel_logging(
        service_name="n2-test", otlp_endpoint=None, enable=True
    )

    test_logger, captured = _make_capturing_logger()
    # Deliberately NOT inside any span.
    test_logger.warning("no active span here")

    assert captured, "expected a captured record"
    rec = captured[0]
    assert rec.trace_id == "", (
        f"expected empty trace_id outside a span, got {rec.trace_id!r}"
    )
    assert rec.span_id == "", (
        f"expected empty span_id outside a span, got {rec.span_id!r}"
    )
    assert rec.trace_flags == 0


def test_03_otel_logging_handler_attaches_when_sdk_present():
    """With an OTLP endpoint + exporter available, a LoggingHandler attaches.

    The real ``opentelemetry-exporter-otlp-proto-http`` package is often not
    installed, so we inject a *fake* OTLP log exporter via monkeypatch to prove
    the handler-attach wiring executes and a genuine SDK ``LoggingHandler`` is
    placed on the root logger.  This exercises the same code path a live
    collector would use.
    """
    # The SDK itself must be present to reach the handler-attach branch.
    try:
        otel_logging._import_sdk_logs()
        sdk_present = True
    except Exception:
        sdk_present = False
    if not sdk_present:
        pytest.skip("opentelemetry-sdk not installed; cannot exercise handler path")

    from opentelemetry.sdk._logs import LoggingHandler as SDKLoggingHandler

    constructed: dict[str, Any] = {}

    class _FakeOTLPLogExporter:
        def __init__(self, *, endpoint=None, **kwargs):
            constructed["endpoint"] = endpoint
            constructed["instance"] = self

        def export(self, *a, **k):
            return None

        def shutdown(self, *a, **k):
            return None

    # Inject the fake exporter (and keep the real SDK import path).
    real_import_exporter = otel_logging._import_otlp_log_exporter
    otel_logging._import_otlp_log_exporter = lambda: _FakeOTLPLogExporter
    try:
        state = otel_logging.setup_otel_logging(
            service_name="n2-test",
            otlp_endpoint="http://localhost:4318/v1/logs",
            enable=True,
        )
    finally:
        otel_logging._import_otlp_log_exporter = real_import_exporter

    assert state["sdk_available"] is True
    assert state["otlp_exporter_available"] is True, (
        "fake exporter should make exporter_available True"
    )
    assert state["handler_attached"] is True, (
        f"expected LoggingHandler attached, got state={state}"
    )
    # Confirm a real SDK LoggingHandler is now on the root logger and that the
    # fake exporter was actually wired with the requested endpoint.
    root = logging.getLogger()
    assert any(isinstance(h, SDKLoggingHandler) for h in root.handlers), (
        "no SDK LoggingHandler found on the root logger"
    )
    assert constructed.get("endpoint") == "http://localhost:4318/v1/logs"


def test_04_graceful_noop_when_otlp_exporter_unavailable():
    """Requesting OTLP export without the exporter must not crash."""
    # If the exporter happens to be installed, this path is covered by test_03;
    # we still verify the *contract* that a missing exporter is handled.
    try:
        otel_logging._import_otlp_log_exporter()
        exporter_available = True
    except Exception:
        exporter_available = False

    if exporter_available:
        pytest.skip(
            "OTLP log exporter is installed; graceful-missing path not "
            "exerciseable here (covered by test_03's success)."
        )

    # SDK is present but exporter package is not -> must be graceful.
    state = otel_logging.setup_otel_logging(
        service_name="n2-test",
        otlp_endpoint="http://localhost:4318/v1/logs",
        enable=True,
    )
    # Must NOT raise; status reports the missing exporter and no handler is
    # attached (the trace-context factory still works on the SDK side).
    assert state["sdk_available"] is True
    assert state["otlp_exporter_available"] is False
    assert state["handler_attached"] is False
    assert state["status"] in ("otlp_exporter_missing", "sdk_missing")
    # Trace-context injection should still be wired (SDK present => API present).
    assert state["trace_context_injected"] is True


def test_05_disable_switch_is_a_clean_noop():
    """enable=False installs nothing and reports 'disabled'."""
    state = otel_logging.setup_otel_logging(
        service_name="n2-test", otlp_endpoint=None, enable=False
    )
    assert state["status"] == "disabled"
    assert state["trace_context_injected"] is False
    assert state["handler_attached"] is False
