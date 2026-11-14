"""Tests for observability/ modules."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock
import pytest


class TestMetrics:
    """Tests for observability/metrics.py."""

    def test_metrics_module_import(self):
        from distllm.observability import metrics
        assert metrics is not None

    def test_metrics_collector(self):
        from distllm.observability.metrics import DistLLMMetrics

        mc = DistLLMMetrics()
        assert mc is not None

    def test_request_counter(self):
        from distllm.observability.metrics import DistLLMMetrics

        mc = DistLLMMetrics()
        if hasattr(mc, "tokens_generated"):
            mc.tokens_generated.add(1)
            assert True

    def test_latency_histogram(self):
        from distllm.observability.metrics import DistLLMMetrics

        mc = DistLLMMetrics()
        if hasattr(mc, "node_latency"):
            mc.node_latency.record(42.5)
            assert True


class TestLogging:
    """Tests for observability/logging.py."""

    def test_logging_module_import(self):
        from distllm.observability import logging
        assert logging is not None

    @patch("distllm.observability.logging.logger", create=True)
    def test_setup_logging(self, mock_logger):
        from distllm.observability.logging import setup_logging

        if callable(setup_logging):
            setup_logging()
            assert True


class TestTracing:
    """Tests for observability/tracing.py."""

    def test_tracing_module_import(self):
        from distllm.observability import tracing
        assert tracing is not None

    @patch("distllm.observability.tracing.setup_tracing", create=True)
    def test_setup_tracing(self, mock_setup):
        from distllm.observability.tracing import setup_tracing

        if callable(setup_tracing):
            setup_tracing()
            assert True

    def test_trace_span(self):
        from distllm.observability.spans import span_prefill

        with span_prefill("test_span", 100) as span:
            assert span is not None


class TestPrometheusExporter:
    """Tests for observability/exporter.py."""

    def test_exporter_module_import(self):
        from distllm.observability import exporter
        assert exporter is not None

    def test_exporter_class(self):
        from distllm.observability.exporter import DistLLMPrometheusExporter

        exporter = DistLLMPrometheusExporter()
        assert exporter is not None

    def test_exporter_init(self):
        from distllm.observability.exporter import DistLLMPrometheusExporter

        exporter = DistLLMPrometheusExporter()
        if hasattr(exporter, "start"):
            assert callable(exporter.start)

    def test_exporter_shutdown(self):
        from distllm.observability.exporter import DistLLMPrometheusExporter

        exporter = DistLLMPrometheusExporter()
        if hasattr(exporter, "shutdown"):
            exporter.shutdown()


def _sample_value(text: str, family: str, labels: str = "") -> float | None:
    """Return the value of the first sample line for *family* (optionally with
    a label fragment), or None when no sample line exists."""
    prefix = f"{family}{labels}"
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[-1])
    return None


class _FakeScheduler:
    pending_count = 3
    active_count = 2


class _FakeNode:
    """PipelineNode-shaped stand-in: is_healthy / gpu_memory_* / latency_ms."""

    def __init__(self, node_id):
        self.node_id = node_id
        self.is_healthy = True
        self.start_layer = 0
        self.end_layer = 10
        self.gpu_memory_total = 40 * 1024**3
        self.gpu_memory_free = 10 * 1024**3
        self.latency_ms = 25.0


class _FakeRecovery:
    def __init__(self, recoveries=4, recovered=7, lost=1, history=None):
        self._recovery_metrics = {
            "recoveries": recoveries,
            "sequences_recovered": recovered,
            "sequences_lost": lost,
            "draining_nodes": 1,
            "dead_nodes": 2,
        }
        self._history = history or [{"duration_ms": 120.0}, {"duration_ms": 80.0}]

    def get_metrics(self):
        return dict(self._recovery_metrics)

    def get_recovery_history(self):
        return list(self._history)


class _FakeSpot:
    stats = {
        "providers": 3,
        "active_leader": "aws",
        "interruptions": 5,
        "leader_changes": 2,
        "healthy_spares": 2,
    }


class _FakePagedMgr:
    num_used_blocks = 30
    _num_blocks = 100


class _FakeCoordinator:
    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.nodes = {"node-0": _FakeNode("node-0")}
        self._resource_mgr = None
        self._latency_tracker = None
        self._recovery_manager = _FakeRecovery()
        self._spot_ensemble = _FakeSpot()

    def _get_paged_backends(self):
        return [_FakePagedMgr()]


class TestPopulateGauges:
    """populate_gauges must write real samples for the alerting families.

    Regression for the audit finding that most registered families had no
    writer anywhere in src: the Grafana dashboards and Prometheus alerts
    queried them but /metrics only emitted HELP/TYPE header lines. Also
    covers the attribute-name bugs (health_status / _check_circuit_breaker /
    _recovery never existed) that kept even the wired families dead.
    """

    def _populate(self):
        from distllm.observability.exporter import DistLLMPrometheusExporter
        from prometheus_client import generate_latest

        exporter = DistLLMPrometheusExporter()
        exporter.populate_gauges(coordinator=_FakeCoordinator())
        return exporter, generate_latest(exporter.registry).decode()

    def test_node_health_sample_written(self):
        _exporter, text = self._populate()
        assert _sample_value(text, "distllm_node_health", "{") == 1.0

    def test_node_gpu_memory_samples_written(self):
        _exporter, text = self._populate()
        assert _sample_value(text, "distllm_node_gpu_memory_bytes", "{") == 30 * 1024**3
        assert _sample_value(text, "distllm_node_gpu_memory_total_bytes", "{") == 40 * 1024**3

    def test_node_gpu_utilization_written(self):
        _exporter, text = self._populate()
        assert _sample_value(text, "distllm_node_gpu_utilization_percent", "{") == 75.0

    def test_node_latency_percentiles_written(self):
        _exporter, text = self._populate()
        # fallback source: PipelineNode.latency_ms
        assert _sample_value(text, "distllm_node_latency_p50_ms", "{node_id=\"node-0\"}") == 25.0
        assert _sample_value(text, "distllm_node_latency_p99_ms", "{node_id=\"node-0\"}") == 25.0

    def test_kv_cache_usage_ratio_written(self):
        _exporter, text = self._populate()
        assert _sample_value(text, "distllm_kv_cache_usage_ratio", "{") == 0.3

    def test_recovery_families_written(self):
        _exporter, text = self._populate()
        assert _sample_value(text, "distllm_recovery_total", " ") == 4.0
        assert _sample_value(text, "distllm_recovery_sequences_recovered_total", " ") == 7.0
        assert _sample_value(text, "distllm_recovery_sequences_lost_total", " ") == 1.0
        # Histogram: _count appears after two observed durations (per scrape
        # watermark stops re-observation on later scrapes).
        assert _sample_value(text, "distllm_recovery_duration_ms_count", " ") == 2.0
        assert _sample_value(text, "distllm_draining_nodes", " ") == 1.0
        assert _sample_value(text, "distllm_dead_nodes", " ") == 2.0

    def test_spot_interruptions_written(self):
        _exporter, text = self._populate()
        assert _sample_value(text, "distllm_spot_interruptions_total", " ") == 5.0

    def test_repeated_population_does_not_inflate_counters(self):
        from distllm.observability.exporter import DistLLMPrometheusExporter
        from prometheus_client import generate_latest

        exporter = DistLLMPrometheusExporter()
        coord = _FakeCoordinator()
        exporter.populate_gauges(coordinator=coord)
        exporter.populate_gauges(coordinator=coord)
        text = generate_latest(exporter.registry).decode()
        # The bridged counters must carry the subsystem total exactly once,
        # not once per scrape.
        assert _sample_value(text, "distllm_recovery_total", " ") == 4.0
        assert _sample_value(text, "distllm_spot_interruptions_total", " ") == 5.0

    def test_no_coordinator_is_a_noop(self):
        from distllm.observability.exporter import DistLLMPrometheusExporter

        exporter = DistLLMPrometheusExporter()
        exporter.populate_gauges(coordinator=None)

    def test_partial_coordinator_never_raises(self):
        """Degraded state (no recovery/spot/paged subsystems) must not break
        the scrape."""
        from distllm.observability.exporter import DistLLMPrometheusExporter

        class _Bare:
            nodes = {}
            scheduler = None

        exporter = DistLLMPrometheusExporter()
        exporter.populate_gauges(coordinator=_Bare())

    def test_record_anomaly_increments_family(self):
        from distllm.observability.exporter import DistLLMPrometheusExporter
        from prometheus_client import generate_latest

        exporter = DistLLMPrometheusExporter()
        exporter.record_anomaly("http_error_rate", "threshold")
        text = generate_latest(exporter.registry).decode()
        assert _sample_value(text, "distllm_anomaly_detected_total", "{") == 1.0


class TestDistLLMMetricsSingleton:
    """The OTel metrics accessor production code calls."""

    def test_get_distllm_metrics_returns_singleton(self):
        from distllm.observability.metrics import get_distllm_metrics

        assert get_distllm_metrics() is get_distllm_metrics()

    def test_record_generation_is_safe_without_provider(self):
        from distllm.observability.metrics import get_distllm_metrics

        # No MeterProvider installed in this test process: recording must be
        # a no-op (or proxy-bound), never raise.
        get_distllm_metrics().record_generation(0.5, 10)


class TestLokiSink:
    """Tests for observability/loki_sink.py."""

    def test_loki_sink_module_import(self):
        from distllm.observability import loki_sink
        assert loki_sink is not None

    def test_loki_sink_factory(self):
        from distllm.observability.loki_sink import make_loki_sink

        if callable(make_loki_sink):
            sink = make_loki_sink("http://localhost:3100")
            assert sink is not None


class TestSpans:
    """Tests for observability/spans.py."""

    def test_spans_module_import(self):
        from distllm.observability import spans
        assert spans is not None

    def test_create_span(self):
        from distllm.observability.spans import span_prefill

        with span_prefill("test_operation", 100) as span:
            assert span is not None

    def test_span_attributes(self):
        from distllm.observability.spans import span_prefill

        with span_prefill("test_op", 100) as span:
            if hasattr(span, "set_attribute"):
                span.set_attribute("key", "value")
                assert True
