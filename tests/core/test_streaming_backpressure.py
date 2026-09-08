"""Regression tests for streaming backpressure infinite hang (C-05).

Bug C-05: with `max_buffer_size <= stream_chunk_size`, the backpressure
loop `while len(token_buffer) >= config.max_buffer_size:` slept forever,
because the flush (`if len(token_buffer) >= config.stream_chunk_size`)
was downstream of the loop and never became reachable while the buffer
sat at exactly max_buffer_size. The request hung permanently.

Fix: two guards -
  1. only wait while the buffer is over the limit AND a flush is not yet due
  2. bound total wait with `backpressure_timeout_s` (default 5 s)
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

torch = pytest.importorskip("torch")

sg = importlib.import_module("distllm.core.streaming_generator")
StreamingConfig = sg.StreamingConfig
StreamingGenerator = sg.StreamingGenerator


async def _drain(gen, timeout: float = 10.0):
    """Consume an async generator with a hard timeout."""
    out = []
    async for chunk in gen:
        out.append(chunk)
    return out


def _make_gen(n_tokens: int):
    """Simple async token producer."""
    async def gen_fn(prompt):
        for i in range(n_tokens):
            yield i, (i == n_tokens - 1)
    return gen_fn


def test_default_config_streams_all_tokens():
    """Sanity: default config (1 token per chunk) streams everything."""
    config = StreamingConfig()
    generator = StreamingGenerator(config=config)
    chunks = asyncio.run(
        _drain(generator.generate(prompt="hi", generate_fn=_make_gen(5)))
    )
    # 5 token chunks + the final [DONE] sentinel = 6 items total
    assert len(chunks) == 6


def test_pathological_config_does_not_hang():
    """C-05: max_buffer_size == stream_chunk_size must complete, not hang.

    Before the fix this hung forever. The test is wrapped in an explicit
    timeout so a regression fails loudly instead of hanging CI.
    """
    config = StreamingConfig(
        stream_chunk_size=8,
        max_buffer_size=8,
        backpressure_sleep_s=0.001,
        backpressure_timeout_s=0.5,
    )
    generator = StreamingGenerator(config=config)

    try:
        chunks = asyncio.run(
            asyncio.wait_for(
                _drain(generator.generate(prompt="hi", generate_fn=_make_gen(20))),
                timeout=10.0,
            )
        )
    except asyncio.TimeoutError:  # pragma: no cover - regression signal
        pytest.fail(
            "Streaming hung with max_buffer_size == stream_chunk_size. "
            "The C-05 backpressure fix has regressed."
        )

    # 20 tokens at 8 per chunk -> 2 full chunks + 1 final flush
    assert len(chunks) >= 1, "Expected at least one chunk to be emitted"


def test_max_buffer_smaller_than_chunk_completes():
    """max_buffer_size < stream_chunk_size also completes."""
    config = StreamingConfig(
        stream_chunk_size=16,
        max_buffer_size=4,
        backpressure_sleep_s=0.001,
        backpressure_timeout_s=0.5,
    )
    generator = StreamingGenerator(config=config)

    try:
        chunks = asyncio.run(
            asyncio.wait_for(
                _drain(generator.generate(prompt="hi", generate_fn=_make_gen(12))),
                timeout=10.0,
            )
        )
    except asyncio.TimeoutError:  # pragma: no cover
        pytest.fail("Streaming hung when max_buffer_size < stream_chunk_size")

    assert len(chunks) >= 1


def test_backpressure_timeout_field_default():
    """New config field exists with a sane default."""
    config = StreamingConfig()
    assert hasattr(config, "backpressure_timeout_s")
    assert config.backpressure_timeout_s > 0


def test_backpressure_timeout_is_bounded():
    """Even with a huge sleep interval, the bounded budget prevents a hang."""
    config = StreamingConfig(
        stream_chunk_size=8,
        max_buffer_size=8,
        backpressure_sleep_s=10.0,   # absurdly long sleep
        backpressure_timeout_s=0.05,  # tiny budget -> breaks immediately
    )
    generator = StreamingGenerator(config=config)

    try:
        asyncio.run(
            asyncio.wait_for(
                _drain(generator.generate(prompt="hi", generate_fn=_make_gen(8))),
                timeout=10.0,
            )
        )
    except asyncio.TimeoutError:  # pragma: no cover
        pytest.fail("Backpressure timeout budget did not bound the wait")
