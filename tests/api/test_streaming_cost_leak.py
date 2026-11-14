"""Regression: streamed chunks must not carry private ``_cost`` telemetry.

Defect: every streamed chunk got ``usage={"_cost": {...}}`` injected
unconditionally, even when the client never asked for usage and cost
tracking was off.  The OpenAI streaming wire format only permits a
``usage`` object on the final chunk, and only when the request sets
``stream_options.include_usage``.  An underscore-prefixed key is a
private extension that no OpenAI-compatible client expects and that a
strict SDK can reject.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import distllm.api.streaming as streaming
from distllm.api.api_state import g
from distllm.api.streaming import _stream_response


# ---------------------------------------------------------------------------
# Minimal stubs — no model, no tokenizer weights, no coordinator internals.
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        tokens = [1, 2, 3]
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    def decode(self, ids, **kwargs):
        return "hi"


class _FakeCoordinator:
    tokenizer = _FakeTokenizer()
    node_order = []
    local_partitioner = None
    model_name = "test-model"


class _FakeRequest:
    """Only the attributes ``_stream_response`` touches."""

    def __init__(self, include_usage: bool | None = None):
        self.model = "test-model"
        self.max_tokens = 3
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 0
        if include_usage is None:
            self.stream_options = None
        else:
            self.stream_options = {"include_usage": include_usage}


def _parse_chunks(raw_chunks: list[str]) -> list[dict]:
    out = []
    for chunk in raw_chunks:
        if not chunk.startswith("data: "):
            continue
        payload = chunk[len("data: "):].strip()
        if payload == "[DONE]":
            out.append({"__done__": True})
            continue
        out.append(json.loads(payload))
    return out


async def _fake_generate(prompt, request_id, max_tokens, temperature, top_p, top_k,
                         response_format=None):
    for i in range(3):
        yield "tok", None, (0.01 if i == 0 else None)


@pytest.fixture
def fake_stream(monkeypatch):
    """Wire a fake coordinator and token generator into the streaming module."""
    monkeypatch.setattr(streaming, "_generate_tokens", _fake_generate)
    original = g.coordinator
    g.coordinator = _FakeCoordinator()
    yield
    g.coordinator = original


async def _collect(request) -> list[dict]:
    chunks = []
    async for raw in _stream_response(
        "hello", request, "chat.completion.chunk", "chatcmpl-",
    ):
        chunks.append(raw)
    return _parse_chunks(chunks)


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_usage_on_intermediate_chunks_without_opt_in(fake_stream):
    """A plain streaming client must not receive ``usage`` mid-stream."""
    parsed = await _collect(_FakeRequest(include_usage=None))

    content_chunks = [c for c in parsed if c.get("choices")]
    assert content_chunks, "expected at least one content chunk"
    for chunk in content_chunks:
        assert "usage" not in chunk or chunk["usage"] is None, (
            "intermediate chunk carried `usage`: %r" % (chunk.get("usage"),)
        )


@pytest.mark.asyncio
async def test_no_private_cost_key_anywhere_without_opt_in(fake_stream):
    """No chunk may contain the private ``_cost`` extension without opt-in."""
    parsed = await _collect(_FakeRequest(include_usage=None))
    for chunk in parsed:
        assert "_cost" not in json.dumps(chunk), chunk


@pytest.mark.asyncio
async def test_no_usage_at_all_when_not_requested(fake_stream):
    """Without ``include_usage`` there must be no usage chunk at the end either."""
    parsed = await _collect(_FakeRequest(include_usage=None))
    usage_chunks = [c for c in parsed if c.get("usage")]
    assert usage_chunks == [], usage_chunks


# ---------------------------------------------------------------------------
# Opt-in behaviour must keep working
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opt_in_still_returns_final_usage_with_cost(fake_stream, monkeypatch):
    """``include_usage`` still yields one final usage chunk with cost data."""
    monkeypatch.setenv("DISTLLM_COST_TRACKING", "true")
    # api_settings is a module-level singleton; reset so the env var is read.
    import distllm.api.api_settings as api_settings_mod
    monkeypatch.setattr(api_settings_mod, "_settings", None)

    parsed = await _collect(_FakeRequest(include_usage=True))

    usage_chunks = [c for c in parsed if c.get("usage")]
    assert len(usage_chunks) == 1, usage_chunks
    usage = usage_chunks[0]["usage"]
    assert usage["completion_tokens"] == 3
    assert "prompt_tokens" in usage
    # Cost tracking is on, so the summary is attached under its own key.
    assert "cost" in usage

    # ...and no intermediate chunk carries a usage object.
    for chunk in parsed:
        if chunk.get("choices"):
            assert "usage" not in chunk or chunk["usage"] is None
