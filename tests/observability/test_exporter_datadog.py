"""Tests for observability/exporter_datadog.py -- Datadog/New Relic APM exporter.

Covers initialisation, push API, flush methods, race-condition fix,
push_trace fallback fix, OTLP stubs, multi-GPU support, retry logic,
auto-collection, context manager, OTel SpanExporter interface, and
module-level helpers.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

# Module under test (pre-imported for clarity in patch paths)
MODULE = "distllm.observability.exporter_datadog"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_env():
    """Clear monitoring-related env vars before each test."""
    keys = [
        "DD_API_KEY",
        "DD_SITE",
        "NEW_RELIC_API_KEY",
        "NEW_RELIC_REGION",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.fixture
def dry_exporter():
    """Create a dry-run exporter (no API keys)."""
    from distllm.observability.exporter_datadog import DatadogMetricsExporter

    return DatadogMetricsExporter()


@pytest.fixture
def ready_exporter():
    """Create an exporter with a Datadog API key."""
    from distllm.observability.exporter_datadog import DatadogMetricsExporter

    return DatadogMetricsExporter(api_key="test-key-12345", site="datadoghq.com")


# ---------------------------------------------------------------------------
# Imports and module structure
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Verify basic module and class structure."""

    def test_module_import(self):
        from distllm.observability import exporter_datadog

        assert exporter_datadog is not None

    def test_class_exported(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        assert DatadogMetricsExporter is not None

    def test_class_in_init(self):
        from distllm.observability import DatadogMetricsExporter

        assert DatadogMetricsExporter is not None

    def test_helpers_exist(self):
        from distllm.observability.exporter_datadog import (
            _dd_type,
            _format_tag,
            _nr_type,
            _percentile,
            _sorted_tags,
            _trace_id_bytes,
            _span_id_bytes,
            _otel_span_kind,
            _set_any_value,
        )

        assert callable(_dd_type)
        assert callable(_format_tag)
        assert callable(_nr_type)
        assert callable(_percentile)
        assert callable(_sorted_tags)
        assert callable(_trace_id_bytes)
        assert callable(_span_id_bytes)
        assert callable(_otel_span_kind)
        assert callable(_set_any_value)

    def test_data_containers(self):
        from distllm.observability.exporter_datadog import (
            _LogEntry,
            _MetricPoint,
            _TraceBatch,
        )

        mp = _MetricPoint(name="cpu", value=50.0)
        assert mp.name == "cpu"
        assert mp.value == 50.0
        assert mp.metric_type == "gauge"

        tb = _TraceBatch(trace_id="abc", spans=[{"name": "test"}])
        assert tb.trace_id == "abc"
        assert tb.spans == [{"name": "test"}]

        le = _LogEntry(message="hello")
        assert le.message == "hello"
        assert le.level == "info"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    """Exporter construction and dry-run detection."""

    def test_default_dry_run(self, dry_exporter):
        assert not dry_exporter.is_ready
        assert not dry_exporter._ready

    def test_with_api_key(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(api_key="abc123")
        assert ex.is_ready
        assert ex._api_key == "abc123"

    def test_with_env_var(self):
        os.environ["DD_API_KEY"] = "env-key"
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        assert ex.is_ready
        assert ex._api_key == "env-key"

    def test_with_new_relic_key(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(new_relic_key="nr-key")
        assert ex.is_ready
        assert ex._new_relic_key == "nr-key"

    def test_site_default(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        assert ex._site == "datadoghq.com"

    def test_site_from_env(self):
        os.environ["DD_SITE"] = "us5.datadoghq.com"
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        assert ex._site == "us5.datadoghq.com"

    def test_tags_stored(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        tags = {"env": "test", "team": "ml"}
        ex = DatadogMetricsExporter(tags=tags)
        assert ex._extra_tags == tags

    def test_service_name_default(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        assert ex._service_name == "distllm"

    def test_flush_immediately(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(flush_immediately=True)
        assert ex._flush_immediately

    def test_max_batch_size(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(max_batch_size=50)
        assert ex._max_batch_size == 50

    def test_new_relic_region_param(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(
            new_relic_key="nr-key", new_relic_region="EU"
        )
        assert ex._nr_region == "EU"

    def test_new_relic_region_env(self):
        os.environ["NEW_RELIC_API_KEY"] = "nr-key"
        os.environ["NEW_RELIC_REGION"] = "EU"
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        assert ex._nr_region == "EU"


# ---------------------------------------------------------------------------
# Push API
# ---------------------------------------------------------------------------


class TestPushMetrics:
    """push_metrics method."""

    def test_push_metrics_dict(self, dry_exporter):
        dry_exporter.push_metrics({"cpu": 50.0, "mem": 1024})
        assert len(dry_exporter._metric_buffer) == 2
        assert dry_exporter._metric_buffer[0].name == "cpu"
        assert dry_exporter._metric_buffer[0].value == 50.0

    def test_push_metrics_list(self, dry_exporter):
        data = [
            {"name": "cpu", "value": 60.0, "tags": {"host": "a"}},
            {"name": "mem", "value": 2048},
        ]
        dry_exporter.push_metrics(data)
        assert len(dry_exporter._metric_buffer) == 2
        assert dry_exporter._metric_buffer[0].tags.get("host") == "a"

    def test_push_metrics_with_tags(self, dry_exporter):
        dry_exporter.push_metrics({"cpu": 70.0}, tags={"env": "prod"})
        pt = dry_exporter._metric_buffer[0]
        assert pt.tags.get("env") == "prod"

    def test_push_metrics_closed(self, dry_exporter):
        dry_exporter.close()
        dry_exporter.push_metrics({"cpu": 1.0})
        assert len(dry_exporter._metric_buffer) == 0

    def test_push_metrics_flush_immediately(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(flush_immediately=True)
        ex.push_metrics({"cpu": 42.0})
        assert len(ex._metric_buffer) == 0

    def test_push_metrics_overflow_flush(self, dry_exporter):
        dry_exporter._max_batch_size = 3
        dry_exporter.push_metrics({"a": 1})
        dry_exporter.push_metrics({"b": 2})
        dry_exporter.push_metrics({"c": 3})
        assert len(dry_exporter._metric_buffer) == 0


class TestPushTrace:
    """push_trace method."""

    def test_push_trace_dict_with_spans(self, dry_exporter):
        trace = {"trace_id": "abc123", "spans": [{"name": "op1"}, {"name": "op2"}]}
        dry_exporter.push_trace(trace)
        assert len(dry_exporter._trace_buffer) == 1
        assert dry_exporter._trace_buffer[0].trace_id == "abc123"
        assert len(dry_exporter._trace_buffer[0].spans) == 2

    def test_push_trace_fallback_span_key(self, dry_exporter):
        """Verify the fallback from 'spans' to 'span' (bugfix)."""
        trace = {"trace_id": "xyz", "span": [{"name": "single"}]}
        dry_exporter.push_trace(trace)
        assert len(dry_exporter._trace_buffer) == 1
        assert dry_exporter._trace_buffer[0].trace_id == "xyz"
        assert len(dry_exporter._trace_buffer[0].spans) == 1
        assert dry_exporter._trace_buffer[0].spans[0]["name"] == "single"

    def test_push_trace_not_a_dict(self, dry_exporter):
        """When trace is a list of span dicts."""
        spans = [{"name": "span1"}, {"name": "span2"}]
        dry_exporter.push_trace(spans)
        assert len(dry_exporter._trace_buffer) == 2

    def test_push_trace_tags(self, dry_exporter):
        dry_exporter.push_trace(
            {"trace_id": "t1", "spans": [{"name": "op"}]}, tags={"env": "test"}
        )
        assert dry_exporter._trace_buffer[0].tags.get("env") == "test"

    def test_push_trace_with_traceId(self, dry_exporter):
        dry_exporter.push_trace({"traceId": "t1", "spans": [{"name": "op"}]})
        assert dry_exporter._trace_buffer[0].trace_id == "t1"

    def test_push_trace_closed(self, dry_exporter):
        dry_exporter.close()
        dry_exporter.push_trace({"trace_id": "t", "spans": []})
        assert len(dry_exporter._trace_buffer) == 0


class TestPushLog:
    """push_log method."""

    def test_push_log_string(self, dry_exporter):
        dry_exporter.push_log("hello world", level="info")
        assert len(dry_exporter._log_buffer) == 1
        assert dry_exporter._log_buffer[0].message == "hello world"
        assert dry_exporter._log_buffer[0].level == "info"

    def test_push_log_dict(self, dry_exporter):
        log = {"message": "system alert", "level": "warn", "tags": {"component": "gpu"}}
        dry_exporter.push_log(log)
        assert len(dry_exporter._log_buffer) == 1
        assert dry_exporter._log_buffer[0].level == "warn"
        assert dry_exporter._log_buffer[0].tags.get("component") == "gpu"

    def test_push_log_list(self, dry_exporter):
        logs = [
            {"message": "msg1", "level": "info"},
            {"message": "msg2", "level": "error"},
        ]
        dry_exporter.push_log(logs)
        assert len(dry_exporter._log_buffer) == 2

    def test_push_log_closed(self, dry_exporter):
        dry_exporter.close()
        dry_exporter.push_log("test")
        assert len(dry_exporter._log_buffer) == 0


# ---------------------------------------------------------------------------
# Observation API
# ---------------------------------------------------------------------------


class TestObservation:
    """Observation counters."""

    def test_observe_request(self, dry_exporter):
        dry_exporter.observe_request()
        assert dry_exporter._request_count == 1

    def test_observe_error(self, dry_exporter):
        dry_exporter.observe_request_error("timeout")
        assert dry_exporter._error_count == 1

    def test_observe_latency(self, dry_exporter):
        dry_exporter.observe_request_latency(0.5)
        assert list(dry_exporter._latencies) == [0.5]

    def test_latency_queue_maxlen(self, dry_exporter):
        for i in range(10_005):
            dry_exporter.observe_request_latency(float(i))
        assert len(dry_exporter._latencies) == 10_000


# ---------------------------------------------------------------------------
# Flush and race-condition fix
# ---------------------------------------------------------------------------


class TestFlush:
    """Flush methods and the race-condition fix (flush called outside lock)."""

    def test_flush_metrics_empty(self, dry_exporter):
        dry_exporter._flush_metrics()

    def test_flush_traces_empty(self, dry_exporter):
        dry_exporter._flush_traces()

    def test_flush_logs_empty(self, dry_exporter):
        dry_exporter._flush_logs()

    def test_flush_metrics_dry_run(self, dry_exporter):
        dry_exporter.push_metrics({"cpu": 10.0})
        dry_exporter._flush_metrics()
        assert len(dry_exporter._metric_buffer) == 0

    def test_flush_traces_dry_run(self, dry_exporter):
        dry_exporter.push_trace({"trace_id": "t", "spans": [{"name": "op"}]})
        assert len(dry_exporter._trace_buffer) == 1
        dry_exporter._flush_traces()
        assert len(dry_exporter._trace_buffer) == 0

    def test_flush_logs_dry_run(self, dry_exporter):
        dry_exporter.push_log("test")
        dry_exporter._flush_logs()
        assert len(dry_exporter._log_buffer) == 0

    def test_flush_all(self, dry_exporter):
        dry_exporter.push_metrics({"cpu": 1.0})
        dry_exporter.push_trace({"trace_id": "t", "spans": [{"name": "op"}]})
        dry_exporter.push_log("test")
        dry_exporter.flush()
        assert len(dry_exporter._metric_buffer) == 0
        assert len(dry_exporter._trace_buffer) == 0
        assert len(dry_exporter._log_buffer) == 0

    def test_flush_with_immediate_mode_no_deadlock(self):
        """Verify flush_immediately=True does NOT cause a deadlock."""
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(flush_immediately=True)
        ex.push_metrics({"cpu": 1.0})
        ex.push_trace({"trace_id": "t", "spans": [{"name": "op"}]})
        ex.push_log("test")

    def test_flush_concurrent_safety(self, dry_exporter):
        errors: list[Exception] = []

        def pusher():
            try:
                for i in range(100):
                    dry_exporter.push_metrics({f"m{i}": float(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pusher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert not errors


# ---------------------------------------------------------------------------
# HTTP POST with retry
# ---------------------------------------------------------------------------


class TestRetry:
    """Retry logic in _post_with_retry."""

    @patch(f"{MODULE}.time.sleep", return_value=None)
    @patch(f"{MODULE}.httpx")
    def test_retry_success_first_attempt(self, mock_httpx, mock_sleep, ready_exporter):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_httpx.post.return_value = mock_response
        ready_exporter._post_with_retry(
            "https://api.datadoghq.com/v1/series", {}, "key"
        )
        assert mock_httpx.post.call_count == 1

    @patch(f"{MODULE}.time.sleep", return_value=None)
    @patch(f"{MODULE}.httpx")
    def test_retry_on_server_error(self, mock_httpx, mock_sleep, ready_exporter):
        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500
        mock_resp_500.text = "error"
        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_httpx.post.side_effect = [mock_resp_500, mock_resp_200]
        ready_exporter._post_with_retry(
            "https://api.datadoghq.com/v1/series", {}, "key"
        )
        assert mock_httpx.post.call_count == 2

    @patch(f"{MODULE}.time.sleep", return_value=None)
    @patch(f"{MODULE}.httpx")
    def test_retry_not_on_client_error(self, mock_httpx, mock_sleep, ready_exporter):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "bad request"
        mock_httpx.post.return_value = mock_resp
        ready_exporter._post_with_retry(
            "https://api.datadoghq.com/v1/series", {}, "key"
        )
        assert mock_httpx.post.call_count == 1

    @patch(f"{MODULE}.time.sleep", return_value=None)
    @patch(f"{MODULE}.httpx")
    def test_retry_exhausted(self, mock_httpx, mock_sleep, ready_exporter):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "error"
        mock_httpx.post.return_value = mock_resp
        ready_exporter._post_with_retry(
            "https://api.datadoghq.com/v1/series", {}, "key", max_retries=3
        )
        assert mock_httpx.post.call_count == 3

    @patch(f"{MODULE}.time.sleep", return_value=None)
    @patch(f"{MODULE}.httpx")
    def test_retry_on_connection_error(self, mock_httpx, mock_sleep, ready_exporter):
        mock_httpx.post.side_effect = [
            ConnectionError("dns failed"),
            MagicMock(status_code=200),
        ]
        ready_exporter._post_with_retry(
            "https://api.datadoghq.com/v1/series", {}, "key", max_retries=3
        )
        assert mock_httpx.post.call_count == 2

    @patch(f"{MODULE}.time.sleep", return_value=None)
    @patch(f"{MODULE}.httpx")
    def test_retry_exponential_delay(self, mock_httpx, mock_sleep, ready_exporter):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "fail"
        mock_httpx.post.return_value = mock_resp
        ready_exporter._post_with_retry(
            "https://api.datadoghq.com/v1/series",
            {},
            "key",
            max_retries=4,
            base_delay=1.0,
        )
        assert mock_sleep.call_count == 3
        calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert 1.0 <= calls[0] <= 2.0
        assert 2.0 <= calls[1] <= 3.0
        assert 4.0 <= calls[2] <= 5.0


# ---------------------------------------------------------------------------
# Backend-specific sending (Datadog)
# ---------------------------------------------------------------------------


class TestSendDatadog:
    """Datadog HTTP API calls."""

    @patch(f"{MODULE}.DatadogMetricsExporter._post_with_retry")
    def test_send_datadog_metrics(self, mock_post, ready_exporter):
        from distllm.observability.exporter_datadog import _MetricPoint

        batch = [_MetricPoint(name="cpu", value=50.0, tags={"env": "prod"})]
        ready_exporter._send_datadog_metrics(batch)
        mock_post.assert_called_once()
        url, payload, api_key = mock_post.call_args[0]
        assert "api." in url
        assert "series" in payload
        assert payload["series"][0]["metric"] == "cpu"

    @patch(f"{MODULE}.DatadogMetricsExporter._post_with_retry")
    def test_send_datadog_traces(self, mock_post, ready_exporter):
        from distllm.observability.exporter_datadog import _TraceBatch

        batch = [
            _TraceBatch(
                trace_id="abc",
                spans=[{"name": "op1"}, {"name": "op2"}],
                tags={"env": "prod"},
            )
        ]
        ready_exporter._send_datadog_traces(batch)
        mock_post.assert_called_once()
        _, payload, _ = mock_post.call_args[0]
        assert len(payload) == 1
        assert payload[0][0] == "abc"
        assert len(payload[0][1]) == 2

    @patch(f"{MODULE}.DatadogMetricsExporter._post_with_retry")
    def test_send_datadog_logs(self, mock_post, ready_exporter):
        from distllm.observability.exporter_datadog import _LogEntry

        entry = _LogEntry(message="test log", level="info", tags={"env": "prod"})
        batch = [entry]
        ready_exporter._send_datadog_logs(batch)
        mock_post.assert_called_once()
        _, payload, _ = mock_post.call_args[0]
        assert payload[0]["message"] == "test log"
        assert payload[0]["level"] == "info"


# ---------------------------------------------------------------------------
# Backend-specific sending (New Relic)
# ---------------------------------------------------------------------------


class TestSendNewRelic:
    """New Relic HTTP API calls."""

    @patch(f"{MODULE}.DatadogMetricsExporter._post_with_retry")
    def test_send_new_relic_metrics(self, mock_post, ready_exporter):
        from distllm.observability.exporter_datadog import _MetricPoint

        ex = ready_exporter
        ex._api_key = ""
        ex._new_relic_key = "nr-key"
        batch = [_MetricPoint(name="cpu", value=50.0, tags={"env": "prod"})]
        ex._send_new_relic_metrics(batch)
        mock_post.assert_called_once()
        _, payload, _ = mock_post.call_args[0]
        assert payload[0]["name"] == "cpu"
        assert payload[0]["value"] == 50.0

    @patch(f"{MODULE}.DatadogMetricsExporter._post_with_retry")
    def test_send_new_relic_traces(self, mock_post, ready_exporter):
        from distllm.observability.exporter_datadog import _TraceBatch

        ex = ready_exporter
        ex._api_key = ""
        ex._new_relic_key = "nr-key"
        batch = [
            _TraceBatch(
                trace_id="abc",
                spans=[{"name": "op1"}],
                tags={"env": "prod"},
            )
        ]
        ex._send_new_relic_traces(batch)
        mock_post.assert_called_once()
        _, payload, _ = mock_post.call_args[0]
        assert payload[0]["trace_id"] == "abc"

    @patch(f"{MODULE}.DatadogMetricsExporter._post_with_retry")
    def test_send_new_relic_logs(self, mock_post, ready_exporter):
        from distllm.observability.exporter_datadog import _LogEntry

        ex = ready_exporter
        ex._api_key = ""
        ex._new_relic_key = "nr-key"
        entry = _LogEntry(message="test", level="warn")
        ex._send_new_relic_logs([entry])
        mock_post.assert_called_once()
        _, payload, _ = mock_post.call_args[0]
        assert payload[0]["message"] == "test"


# ---------------------------------------------------------------------------
# OTLP stubs
# ---------------------------------------------------------------------------


class TestOTLPStubs:
    """OTLP gRPC export stubs (real implementations)."""

    def test_otlp_metrics_import_error_caught(self, ready_exporter):
        """When OTLP exporter packages are missing, no exception raised."""
        ready_exporter._otlp_endpoint = "http://localhost:4317"
        from distllm.observability.exporter_datadog import _MetricPoint

        batch = [_MetricPoint(name="cpu", value=50.0)]
        ready_exporter._send_otlp_metrics(batch)

    def test_otlp_traces_import_error_caught(self, ready_exporter):
        from distllm.observability.exporter_datadog import _TraceBatch

        ready_exporter._otlp_endpoint = "http://localhost:4317"
        batch = [_TraceBatch(trace_id="abc", spans=[{"name": "op1"}])]
        ready_exporter._send_otlp_traces(batch)


# ---------------------------------------------------------------------------
# OTel SpanExporter interface
# ---------------------------------------------------------------------------


class TestOTelInterface:
    """export(), shutdown(), force_flush() methods."""

    def test_export_closed(self, dry_exporter):
        dry_exporter.close()
        from distllm.observability.exporter_datadog import SpanExportResult

        result = dry_exporter.export([MagicMock()])
        assert result == SpanExportResult.FAILURE

    def test_export_empty_list(self, dry_exporter):
        """Empty span list should succeed (nothing to fail on)."""
        result = dry_exporter.export([])
        from distllm.observability.exporter_datadog import SpanExportResult

        assert result == SpanExportResult.SUCCESS

    def test_export_exception(self, dry_exporter):
        with patch.object(dry_exporter, "push_trace", side_effect=ValueError("boom")):
            from distllm.observability.exporter_datadog import SpanExportResult

            result = dry_exporter.export([MagicMock()])
            assert result == SpanExportResult.FAILURE

    def test_shutdown(self, dry_exporter):
        with patch.object(dry_exporter, "close") as mock_close:
            dry_exporter.shutdown()
            mock_close.assert_called_once()

    def test_force_flush(self, dry_exporter):
        with patch.object(dry_exporter, "flush") as mock_flush:
            result = dry_exporter.force_flush()
            mock_flush.assert_called_once()
            assert result is True


# ---------------------------------------------------------------------------
# Auto-collection
# ---------------------------------------------------------------------------


class TestAutoCollect:
    """Background metric collection."""

    def test_auto_collect_starts_thread(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(auto_collect=True, collect_interval_s=60)
        assert ex._collect_thread is not None
        assert ex._collect_thread.is_alive()
        ex.close()

    def test_auto_collect_not_started_by_default(self, dry_exporter):
        assert dry_exporter._collect_thread is None

    def test_stop_auto_collect(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(auto_collect=True)
        assert ex._collect_thread is not None
        ex._stop_auto_collect()
        assert ex._collect_thread is None
        ex.close()

    def test_close_stops_auto_collect(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(auto_collect=True)
        ex.close()
        assert ex._collect_thread is None

    def test_collect_and_push_basic(self, dry_exporter):
        dry_exporter.observe_request()
        dry_exporter.observe_request_latency(0.5)
        dry_exporter._collect_and_push()
        assert len(dry_exporter._metric_buffer) == 0


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """__enter__ / __exit__ behavior."""

    def test_context_manager_starts_auto_collect(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        with DatadogMetricsExporter() as ex:
            assert ex._collect_thread is not None
            assert ex._collect_thread.is_alive()

    def test_context_manager_flushes_on_exit(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        with patch.object(DatadogMetricsExporter, "flush") as mock_flush:
            with DatadogMetricsExporter() as ex:
                ex.push_metrics({"cpu": 1.0})
            mock_flush.assert_called()

    def test_context_manager_calls_close(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        with patch.object(DatadogMetricsExporter, "close") as mock_close:
            with DatadogMetricsExporter():
                pass
            mock_close.assert_called()


# ---------------------------------------------------------------------------
# Convenience class method
# ---------------------------------------------------------------------------


class TestClassMethodPush:
    """DatadogMetricsExporter.push() class method."""

    @patch(f"{MODULE}.DatadogMetricsExporter.flush")
    @patch(f"{MODULE}.DatadogMetricsExporter.push_metrics")
    @patch(f"{MODULE}.DatadogMetricsExporter.close")
    def test_push_metrics_only(self, mock_close, mock_push_metrics, mock_flush):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        DatadogMetricsExporter.push(metrics={"cpu": 42.0})
        mock_push_metrics.assert_called_once_with({"cpu": 42.0}, tags=None)

    @patch(f"{MODULE}.DatadogMetricsExporter.flush")
    @patch(f"{MODULE}.DatadogMetricsExporter.push_trace")
    @patch(f"{MODULE}.DatadogMetricsExporter.close")
    def test_push_traces_only(self, mock_close, mock_push_trace, mock_flush):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        trace = {"trace_id": "t1", "spans": [{"name": "op"}]}
        DatadogMetricsExporter.push(traces=trace)
        mock_push_trace.assert_called_once_with(trace, tags=None)

    @patch(f"{MODULE}.DatadogMetricsExporter.flush")
    @patch(f"{MODULE}.DatadogMetricsExporter.push_log")
    @patch(f"{MODULE}.DatadogMetricsExporter.close")
    def test_push_logs_only(self, mock_close, mock_push_log, mock_flush):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        DatadogMetricsExporter.push(logs="hello")
        mock_push_log.assert_called_once_with("hello", tags=None)

    @patch(f"{MODULE}.DatadogMetricsExporter.flush")
    @patch(f"{MODULE}.DatadogMetricsExporter.close")
    def test_push_with_tags(self, mock_close, mock_flush):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        DatadogMetricsExporter.push(metrics={"cpu": 1.0}, tags={"env": "prod"})


# ---------------------------------------------------------------------------
# check_ready / _resolve_host
# ---------------------------------------------------------------------------


class TestConnectivity:
    """Host resolution and connectivity checks."""

    def test_resolve_host_datadog(self, ready_exporter):
        host = ready_exporter._resolve_host()
        assert "datadoghq.com" in host

    def test_resolve_host_new_relic(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        ex._new_relic_key = "nr-key"
        host = ex._resolve_host()
        assert "newrelic.com" in host

    def test_resolve_host_otlp(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(api_key="key")
        ex._otlp_endpoint = "http://my-otlp:4317"
        host = ex._resolve_host()
        assert host == "my-otlp"

    def test_resolve_host_fallback(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        host = ex._resolve_host()
        assert host == "api.datadoghq.com"

    def test_check_ready_not_ready(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        assert not ex.check_ready()

    @patch(f"{MODULE}.socket.getaddrinfo", return_value=[(None, None, None, None, None)])
    def test_check_ready_success(self, mock_getaddrinfo, ready_exporter):
        assert ready_exporter.check_ready()

    @patch(f"{MODULE}.socket.getaddrinfo", side_effect=OSError("no route"))
    def test_check_ready_failure(self, mock_getaddrinfo, ready_exporter):
        assert not ready_exporter.check_ready()


# ---------------------------------------------------------------------------
# Multi-GPU support
# ---------------------------------------------------------------------------


class TestMultiGPU:
    """Multi-GPU initialisation and metric collection."""

    @patch(f"{MODULE}._PYNVML_AVAILABLE", True)
    def test_init_gpu_multiple_devices(self):
        """Verify all GPU handles are initialised."""
        with patch(f"{MODULE}.pynvml") as mock_pynvml:
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetCount.return_value = 4
            mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
                MagicMock() for _ in range(4)
            ]
            from distllm.observability.exporter_datadog import DatadogMetricsExporter

            ex = DatadogMetricsExporter()
            ex._init_gpu()
            assert ex._gpu_device_count == 4
            assert len(ex._gpu_handles) == 4
            assert mock_pynvml.nvmlDeviceGetHandleByIndex.call_count == 4

    @patch(f"{MODULE}._PYNVML_AVAILABLE", False)
    def test_init_gpu_no_pynvml(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        ex._init_gpu()
        assert ex._gpu_device_count == 0
        assert len(ex._gpu_handles) == 0

    @patch(f"{MODULE}._PYNVML_AVAILABLE", True)
    def test_init_gpu_init_failure(self):
        with patch(f"{MODULE}.pynvml") as mock_pynvml:
            mock_pynvml.nvmlInit.side_effect = RuntimeError("NVML failure")
            from distllm.observability.exporter_datadog import DatadogMetricsExporter

            ex = DatadogMetricsExporter()
            ex._init_gpu()
            assert len(ex._gpu_handles) == 0

    @patch(f"{MODULE}._PYNVML_AVAILABLE", True)
    def test_collect_system_metrics_multi_gpu(self):
        """Verify per-GPU and aggregate GPU metrics are collected."""
        with (
            patch(f"{MODULE}.psutil") as mock_psutil,
            patch(f"{MODULE}.pynvml") as mock_pynvml,
        ):
            # Mock psutil
            mock_psutil.cpu_percent.return_value = 50.0
            mock_mem = MagicMock()
            mock_mem.percent = 60.0
            mock_mem.available = 8 * 1024**3
            mock_mem.total = 16 * 1024**3
            mock_psutil.virtual_memory.return_value = mock_mem
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value = MagicMock(rss=500 * 1024**2)
            mock_proc.cpu_percent.return_value = 25.0
            mock_psutil.Process.return_value = mock_proc

            # Mock pynvml with 2 GPUs
            mock_pynvml.nvmlInit.return_value = None
            mock_pynvml.nvmlDeviceGetCount.return_value = 2

            mock_handle0 = MagicMock()
            mock_handle1 = MagicMock()
            mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [
                mock_handle0,
                mock_handle1,
            ]

            # Per-GPU mocks -- set return_value so both the per-GPU and the
            # subsequent aggregate loops can consume the same values.
            util_gpu0 = MagicMock(gpu=80, memory=75)
            util_gpu1 = MagicMock(gpu=60, memory=50)
            mem_gpu0 = MagicMock(used=4 * 1024**3, total=8 * 1024**3)
            mem_gpu1 = MagicMock(used=2 * 1024**3, total=8 * 1024**3)

            _util_count: int = 0
            _mem_count: int = 0

            def _get_util(handle: Any) -> MagicMock:
                nonlocal _util_count
                _util_count += 1
                return util_gpu0 if _util_count % 2 == 1 else util_gpu1

            def _get_mem(handle: Any) -> MagicMock:
                nonlocal _mem_count
                _mem_count += 1
                return mem_gpu0 if _mem_count % 2 == 1 else mem_gpu1

            mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = _get_util
            mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = _get_mem
            mock_pynvml.nvmlDeviceGetTemperature.side_effect = None
            mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
            mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = None
            mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 135_000
            mock_pynvml.nvmlDeviceGetName.return_value = "Test GPU"
            mock_pynvml.NVML_TEMPERATURE_GPU = 0

            from distllm.observability.exporter_datadog import DatadogMetricsExporter

            ex = DatadogMetricsExporter()
            metrics = ex._collect_system_metrics()

            # Per-GPU metrics
            assert metrics["gpu.0.utilization.percent"] == 80.0
            assert metrics["gpu.1.utilization.percent"] == 60.0

            # Aggregate backward-compatible metrics
            assert metrics["gpu.utilization.percent"] == 70.0  # (80 + 60) / 2

    @patch(f"{MODULE}._PYNVML_AVAILABLE", False)
    def test_gpu_disabled_no_pynvml(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        ex._init_gpu()
        assert len(ex._gpu_handles) == 0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Module-level helper functions."""

    def test_dd_type(self):
        from distllm.observability.exporter_datadog import _dd_type

        assert _dd_type("gauge") == 3
        assert _dd_type("unknown") == 3

    def test_nr_type(self):
        from distllm.observability.exporter_datadog import _nr_type

        assert _nr_type("gauge") == "gauge"
        assert _nr_type("count") == "count"
        assert _nr_type("rate") == "count"

    def test_format_tag(self):
        from distllm.observability.exporter_datadog import _format_tag

        assert _format_tag("key", "val") == "key:val"

    def test_sorted_tags(self):
        from distllm.observability.exporter_datadog import _sorted_tags

        result = _sorted_tags({"b": "2", "a": "1"})
        assert result == "a=1,b=2"

    def test_percentile(self):
        from distllm.observability.exporter_datadog import _percentile

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 50) == 3.0
        assert _percentile(values, 100) == 5.0
        assert _percentile(values, 0) == 1.0

    def test_percentile_empty(self):
        from distllm.observability.exporter_datadog import _percentile

        assert _percentile([], 50) == 0.0

    def test_to_ns(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        # Seconds
        result = DatadogMetricsExporter._to_ns(1000.0)
        assert result == 1_000_000_000_000

        # Milliseconds (> 1e9 seconds ~ 31.7 years -> ms)
        result = DatadogMetricsExporter._to_ns(2_000_000_000.0)
        assert result == 2_000_000_000_000_000

        # Already ns (> 1e12)
        result = DatadogMetricsExporter._to_ns(1_500_000_000_000.0)
        assert result == 1_500_000_000_000

        # Zero or negative
        assert DatadogMetricsExporter._to_ns(0) is None
        assert DatadogMetricsExporter._to_ns(-1.0) is None

    def test_trace_id_bytes(self):
        from distllm.observability.exporter_datadog import _trace_id_bytes

        result = _trace_id_bytes("abcdef0123456789")
        assert len(result) == 16
        result = _trace_id_bytes("")
        assert result == b"\x00" * 16
        result = _trace_id_bytes("0xabc")
        assert len(result) == 16

    def test_span_id_bytes(self):
        from distllm.observability.exporter_datadog import _span_id_bytes

        result = _span_id_bytes("abcdef01")
        assert len(result) == 8
        result = _span_id_bytes("")
        assert result == b"\x00" * 8

    def test_otel_span_kind(self):
        from distllm.observability.exporter_datadog import _otel_span_kind

        assert _otel_span_kind("INTERNAL") == 1
        assert _otel_span_kind("SERVER") == 2
        assert _otel_span_kind("CLIENT") == 3
        assert _otel_span_kind("SpanKind.PRODUCER") == 4
        assert _otel_span_kind("CONSUMER") == 5
        assert _otel_span_kind("UNKNOWN") == 1

    def test_set_any_value(self):
        from distllm.observability.exporter_datadog import _set_any_value

        for py_val, attr, expected in [
            ("hello", "string_value", "hello"),
            (42, "int_value", 42),
            (3.14, "double_value", 3.14),
            (True, "bool_value", True),
            (None, "string_value", ""),
        ]:
            mock_any = MagicMock()
            _set_any_value(mock_any, py_val)
            assert getattr(mock_any, attr) == expected


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case scenarios."""

    def test_close_called_twice(self, dry_exporter):
        dry_exporter.close()
        dry_exporter.close()

    def test_flush_after_close(self, dry_exporter):
        dry_exporter.close()
        dry_exporter.flush()

    def test_large_batch(self, dry_exporter):
        """Pushing many metrics triggers batch-flush at max_batch_size."""
        dry_exporter._max_batch_size = 5
        for i in range(10):
            dry_exporter.push_metrics({"m": float(i)})
        assert len(dry_exporter._metric_buffer) < 5

    def test_otlp_endpoint_non_empty_buffer(self):
        """OTLP path is taken when endpoint is set (with data in buffer)."""
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        ex._otlp_endpoint = "http://localhost:4317"
        ex._ready = True
        ex._api_key = ""
        ex._new_relic_key = ""
        ex._otlp_endpoint = "http://localhost:4317"
        # Push a metric so the buffer is non-empty
        ex.push_metrics({"test": 1.0})
        with patch.object(ex, "_send_otlp_metrics") as mock_otlp:
            ex._flush_metrics()
            mock_otlp.assert_called_once()

    def test_context_manager_double_enter(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter()
        with ex:
            pass
        with ex:
            pass


# ---------------------------------------------------------------------------
# Integration-like: flush routing
# ---------------------------------------------------------------------------


class TestFlushRouting:
    """Verify correct backend routing based on config."""

    def test_routes_to_datadog_when_api_key(self, ready_exporter):
        with patch.object(ready_exporter, "_send_datadog_metrics") as mock_send:
            ready_exporter.push_metrics({"cpu": 1.0})
            ready_exporter._flush_metrics()
            mock_send.assert_called_once()

    def test_routes_to_new_relic_when_nr_key(self):
        from distllm.observability.exporter_datadog import DatadogMetricsExporter

        ex = DatadogMetricsExporter(api_key="")
        ex._new_relic_key = "nr-key"
        ex._ready = True
        with patch.object(ex, "_send_new_relic_metrics") as mock_send:
            ex.push_metrics({"cpu": 1.0})
            ex._flush_metrics()
            mock_send.assert_called_once()

    def test_routes_to_otlp_when_endpoint(self, ready_exporter):
        ready_exporter._otlp_endpoint = "http://localhost:4317"
        with patch.object(ready_exporter, "_send_otlp_metrics") as mock_send:
            ready_exporter.push_metrics({"cpu": 1.0})
            ready_exporter._flush_metrics()
            mock_send.assert_called_once()

    def test_dry_run_when_not_ready(self, dry_exporter):
        with patch.object(dry_exporter, "_send_datadog_metrics") as mock_send:
            dry_exporter.push_metrics({"cpu": 1.0})
            dry_exporter._flush_metrics()
            mock_send.assert_not_called()
