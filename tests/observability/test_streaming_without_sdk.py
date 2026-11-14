"""Streaming must work on an install without the [observability] extra.

A streamed chat completion returned HTTP 200 with a completely empty body on
every default install. The cause was an import placed inside the SSE
generator:

    File ".../distllm/api/streaming.py", line 422, in _stream_response
        from distllm.observability.spans import async_span_generation, record_ttft
    File ".../distllm/observability/spans.py", line 16, in <module>
        from opentelemetry import trace
    ModuleNotFoundError: No module named 'opentelemetry'

By the time a StreamingResponse's body iterator first runs, Starlette has
already sent the status line and headers. An exception there cannot become an
error response -- it aborts the stream, and the client keeps the 200 with
nothing in it. Found by streaming a completion against a real server on the
A10; the non-streaming path worked, which is why no unit test had caught it.

Two separate defects, both covered here:
  1. spans.py required the SDK to import at all.
  2. streaming.py imported it lazily, inside the iterator.
"""

from __future__ import annotations

import builtins
import importlib

import pytest


def _block_otel(monkeypatch):
    """Make `import opentelemetry...` fail the way a minimal install does."""
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            raise ImportError("simulated: no opentelemetry")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    return real_import


class TestSpansImportWithoutSDK:
    def test_spans_module_imports(self, monkeypatch):
        _block_otel(monkeypatch)
        import distllm.observability.spans as spans

        reloaded = importlib.reload(spans)
        try:
            assert reloaded.trace is None
            assert reloaded._tracer is not None
        finally:
            monkeypatch.undo()
            importlib.reload(spans)

    def test_span_helpers_still_yield_an_object(self, monkeypatch):
        """Callers use the yielded span, so a missing SDK must not yield None."""
        _block_otel(monkeypatch)
        import distllm.observability.spans as spans

        reloaded = importlib.reload(spans)
        try:
            with reloaded.span_prefill("req-1", 12) as span:
                span.set_attribute("k", "v")
                span.add_event("e", {"a": 1})
                reloaded.record_ttft(span, 0.5)
                reloaded.link_to_parent_node(span, "node-1", "req-1")
        finally:
            monkeypatch.undo()
            importlib.reload(spans)

    @pytest.mark.asyncio
    async def test_async_span_generation_yields_a_span(self, monkeypatch):
        _block_otel(monkeypatch)
        import distllm.observability.spans as spans

        reloaded = importlib.reload(spans)
        try:
            async with reloaded.async_span_generation("req", "m", 3, "t") as span:
                span.set_attribute("k", "v")
        finally:
            monkeypatch.undo()
            importlib.reload(spans)


class TestStreamingImportIsNotLazy:
    def test_observability_imported_at_module_scope(self):
        """The generator must not be the first thing to import it.

        If this symbol is resolved at call time, a broken dependency graph
        turns every streamed response into a silent empty 200 instead of a
        loud, diagnosable failure at import.
        """
        import distllm.api.streaming as streaming

        assert hasattr(streaming, "async_span_generation")
        assert hasattr(streaming, "record_ttft")

    def test_streaming_module_imports_without_sdk(self, monkeypatch):
        _block_otel(monkeypatch)
        import distllm.api.streaming as streaming

        reloaded = importlib.reload(streaming)
        try:
            assert reloaded._stream_response is not None
        finally:
            monkeypatch.undo()
            importlib.reload(streaming)


class TestStreamYieldsWhenCoordinatorMissing:
    """The generator must emit the role chunk and [DONE] even with no model.

    This is the shape a client sees; it also proves the generator starts
    producing before any optional dependency is touched.
    """

    @pytest.mark.asyncio
    async def test_empty_coordinator_still_produces_chunks(self):
        from types import SimpleNamespace

        import distllm.api.streaming as streaming

        request = SimpleNamespace(
            model="m",
            stream_options=None,
            user=None,
        )
        chunks = []
        async for piece in streaming._stream_response(
            "hello", request, "chat.completion.chunk", "chatcmpl-",
        ):
            chunks.append(piece)
        joined = "".join(chunks)
        assert "chat.completion.chunk" in joined
        assert "[DONE]" in joined