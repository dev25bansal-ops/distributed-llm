"""Real tests for tracing — Tracer instantiation without OpenTelemetry installed."""
from __future__ import annotations


class TestTracer:
    def test_tracer_init_no_op(self):
        """Without opentelemetry, tracer should fall back to no-op."""
        from distllm.dist.tracing import Tracer

        tracer = Tracer(service_name="test")
        assert tracer is not None

    def test_tracer_start_span_no_op(self):
        from distllm.dist.tracing import Tracer

        tracer = Tracer(service_name="test")
        span = tracer.start_span("test-span")
        span.set_attribute("key", "value")
        span.add_event("test-event")
        span.end()

    def test_tracer_decorator_sync(self):
        from distllm.dist.tracing import Tracer

        tracer = Tracer(service_name="test")

        @tracer.trace("test-func")
        def my_func(x: int) -> int:
            return x * 2

        assert my_func(21) == 42

    def test_instrument_all_no_crash(self):
        from distllm.dist.tracing import Tracer

        tracer = Tracer(service_name="test")
        tracer.instrument_all()  # Should not crash
