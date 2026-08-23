"""Benchmark: Streaming token throughput.

Measures tokens per second vs concurrent streaming connections.

Note: This benchmark exercises the SSE formatting and generator
machinery without a real model — it uses a mock generator.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from distllm.core.streaming_generator import StreamChunk, StreamingGenerator
from distllm.core.token_streaming_buffer import TokenStreamingBuffer


class TestStreamingThroughput:
    """Measure streaming chunk generation throughput."""

    def test_stream_chunk_sse_throughput(self, benchmark):
        """Throughput of StreamChunk.to_sse() serialization."""
        chunk = StreamChunk(
            id="test-id",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        )

        def _serialize():
            for _ in range(1000):
                chunk.to_sse()

        benchmark(_serialize)

    def test_token_buffer_throughput(self, benchmark):
        """Throughput of TokenStreamingBuffer append + flush cycle."""
        buffer = TokenStreamingBuffer()

        def _buffer_ops():
            for i in range(500):
                buffer.add_token({"text": f"token-{i}", "logprob": -0.5})
            buffer.finish()

        benchmark(_buffer_ops)

    def test_concurrent_sse_formatting(self, benchmark):
        """Multiple concurrent SSE formatters."""
        chunks = [
            StreamChunk(
                id=f"stream-{j}",
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": {"content": "x" * 100}, "finish_reason": None}],
            )
            for j in range(100)
        ]

        def _concurrent_sse():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda c: c.to_sse(), chunks))

        benchmark(_concurrent_sse)
