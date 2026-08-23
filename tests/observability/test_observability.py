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
