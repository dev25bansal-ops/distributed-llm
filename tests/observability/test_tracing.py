"""Tests for distributed tracing setup."""
from distllm.observability.tracing import setup_tracing, inject_trace_context


class TestTracing:
    def test_setup(self):
        tracer_provider = setup_tracing(service_name="test")
        assert tracer_provider is not None

    def test_trace_context(self):
        ctx = inject_trace_context({"existing": "val"})
        assert "existing" in ctx
