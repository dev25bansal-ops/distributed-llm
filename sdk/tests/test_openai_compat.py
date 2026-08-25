"""Regression tests: openai_compat adapter vs DistLLM's extended usage fields.

Wave-2 item 38. The adapter used strict ``Usage(**usage)`` unpacking, so any
DistLLM-native extension inside the ``usage`` object crashed parsing:

* ``/v1/embeddings`` always adds ``processing_time`` to usage
  (src/distllm/api/routes/embeddings.py) -> TypeError on every real response.
* Streaming usage chunks merge a ``cost`` summary into usage
  (src/distllm/api/streaming.py, core/streaming_cost.to_final_summary).
* ``/v1/chat/completions`` and ``/v1/completions`` add top-level
  ``generation_time`` (routes/chat.py, routes/completion.py).

Contract after the fix: unknown fields are ignored (ignore-extra semantics),
required structure is still validated with clear ValueErrors, and streaming
chunks surface the final usage chunk.

Both copies of the adapter are tested:
* packaged:  sdk/src/distllm_sdk/compat/openai_compat.py
* standalone: sdk/compat/openai_compat.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from distllm_sdk.compat import openai_compat as packaged


def _load_standalone():
    """Load sdk/compat/openai_compat.py under a private module name."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "compat", "openai_compat.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_openai_compat_standalone", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_openai_compat_standalone"] = mod  # dataclasses needs this
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=["packaged", "standalone"])
def compat(request):
    if request.param == "packaged":
        return packaged
    return _load_standalone()


# --------------------------------------------------------------------------- #
# Realistic server payloads (shapes lifted from the API route handlers)
# --------------------------------------------------------------------------- #

# routes/chat.py ChatCompletionResponse + V2 extras (system_fingerprint,
# api_version) and top-level generation_time.
CHAT_BODY_DISTLLM = {
    "id": "chatcmpl-1a2b3c4d5e6f",
    "object": "chat.completion.v2",
    "created": 1756000000,
    "model": "distributed-llm",
    "system_fingerprint": "fp_ab12cd34",
    "api_version": "2025-03-01",
    "generation_time": 0.142,
    "request_id": "req-777",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 4,
        "total_tokens": 13,
    },
}

# routes/embeddings.py: usage ALWAYS carries processing_time.
EMBEDDINGS_BODY_DISTLLM = {
    "object": "list",
    "model": "text-embedding-3-small",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
    ],
    "usage": {
        "prompt_tokens": 3,
        "total_tokens": 3,
        "processing_time": 0.031,
    },
}

# Defensive: /v1/completions gaining cost-annotated usage later.
COMPLETION_BODY_EXTENDED_USAGE = {
    "id": "cmpl-1a2b3c4d5e6f",
    "object": "text_completion",
    "created": 1756000000,
    "model": "distributed-llm",
    "generation_time": 0.098,
    "choices": [
        {"index": 0, "text": " world", "finish_reason": "stop"}
    ],
    "usage": {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
        "cost_usd": 0.000012,
        "processing_time": 0.098,
    },
}

# Streaming SSE body mirroring api/streaming.py: delta chunks, a final
# finish chunk, a usage chunk whose usage carries the merged cost summary
# (core/streaming_cost.to_final_summary), then [DONE].
SSE_LINES = [
    'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
    '"created":1756000000,"model":"distributed-llm","choices":'
    '[{"index":0,"delta":{"role":"assistant","content":"Hel"},'
    '"finish_reason":null}]}',
    'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
    '"created":1756000000,"model":"distributed-llm","choices":'
    '[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
    '"created":1756000000,"model":"distributed-llm","choices":'
    '[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk",'
    '"created":1756000000,"model":"distributed-llm","choices":[],'
    '"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11,'
    '"cost":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11,'
    '"cost_usd":0.000123,"ttft_ms":41.2}}} ',
    "data: [DONE]",
]


def _iter_chunks(compat, lines):
    """Parse an SSE line sequence exactly like ChatCompletions._stream_create."""
    chunks = []
    for line in lines:
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk = compat.ChatCompletionChunk.from_dict(json.loads(data))
            chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- #
# Extended fields accepted (the crash regression)
# --------------------------------------------------------------------------- #


class TestExtendedFieldsAccepted:
    def test_chat_completion_with_v2_and_generation_time(self, compat):
        resp = compat.ChatCompletion.from_dict(CHAT_BODY_DISTLLM)
        assert resp.id == "chatcmpl-1a2b3c4d5e6f"
        assert resp.model == "distributed-llm"
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content == "Hello!"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.prompt_tokens == 9
        assert resp.usage.completion_tokens == 4
        assert resp.usage.total_tokens == 13

    def test_tool_call_message_content_null(self, compat):
        # routes/chat.py sets content=None when tool_calls are present.
        body = {
            "id": "chatcmpl-tool",
            "model": "distributed-llm",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
        resp = compat.ChatCompletion.from_dict(body)
        assert resp.choices[0].message.content == ""
        assert resp.choices[0].finish_reason == "tool_calls"

    def test_embeddings_processing_time_no_crash(self, compat):
        """Every real /v1/embeddings response carries processing_time."""
        resp = compat.EmbeddingResponse.from_dict(EMBEDDINGS_BODY_DISTLLM)
        assert resp.model == "text-embedding-3-small"
        assert resp.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert resp.usage.prompt_tokens == 3
        assert resp.usage.total_tokens == 3

    def test_completion_extended_usage_no_crash(self, compat):
        resp = compat.Completion.from_dict(COMPLETION_BODY_EXTENDED_USAGE)
        assert resp.choices[0]["text"] == " world"
        assert resp.usage.total_tokens == 6
        assert resp.usage.prompt_tokens == 4

    def test_missing_usage_defaults_to_zero(self, compat):
        # routes/completion.py currently emits no usage at all.
        resp = compat.Completion.from_dict({
            "id": "cmpl-1",
            "choices": [{"index": 0, "text": "hi"}],
        })
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.total_tokens == 0

    def test_null_usage_defaults_to_zero(self, compat):
        resp = compat.ChatCompletion.from_dict({
            "id": "chatcmpl-1",
            "choices": [],
            "usage": None,
        })
        assert resp.usage.total_tokens == 0

    def test_noninteger_token_values_tolerated(self, compat):
        resp = compat.ChatCompletion.from_dict({
            "id": "chatcmpl-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 9.0,
                "completion_tokens": None,
                "total_tokens": "13",
                "cost_usd": 0.5,
            },
        })
        assert resp.usage.prompt_tokens == 9
        assert resp.usage.completion_tokens == 0
        assert resp.usage.total_tokens == 0  # non-numeric string -> 0


# --------------------------------------------------------------------------- #
# Malformed required structure still rejected
# --------------------------------------------------------------------------- #


class TestMalformedRequiredRejected:
    def test_payload_not_an_object(self, compat):
        with pytest.raises(ValueError):
            compat.ChatCompletion.from_dict(["not", "a", "dict"])

    def test_choices_not_a_list(self, compat):
        with pytest.raises(ValueError):
            compat.ChatCompletion.from_dict({"id": "x", "choices": "one"})

    def test_choice_not_an_object(self, compat):
        with pytest.raises(ValueError):
            compat.ChatCompletion.from_dict({"id": "x", "choices": ["msg"]})

    def test_message_not_an_object(self, compat):
        with pytest.raises(ValueError):
            compat.ChatCompletion.from_dict({
                "id": "x",
                "choices": [{"index": 0, "message": "hello there"}],
            })

    def test_content_not_a_string(self, compat):
        with pytest.raises(ValueError):
            compat.ChatCompletion.from_dict({
                "id": "x",
                "choices": [
                    {"message": {"role": "assistant", "content": 42}}
                ],
            })

    def test_role_not_a_string(self, compat):
        with pytest.raises(ValueError):
            compat.ChatCompletion.from_dict({
                "id": "x",
                "choices": [{"message": {"role": 7, "content": "hi"}}],
            })

    def test_completion_usage_not_an_object(self, compat):
        with pytest.raises(ValueError):
            compat.Completion.from_dict({"choices": [], "usage": "12 tokens"})

    def test_embedding_usage_not_an_object(self, compat):
        with pytest.raises(ValueError):
            compat.EmbeddingResponse.from_dict({
                "data": [],
                "usage": 42,
            })


# --------------------------------------------------------------------------- #
# Streaming round-trip (delta chunks + usage chunk + [DONE])
# --------------------------------------------------------------------------- #


class TestStreamingRoundTrip:
    def test_round_trip_collects_text_and_finish(self, compat):
        chunks = _iter_chunks(compat, SSE_LINES)
        assert len(chunks) == 4  # 2 delta + finish + usage
        text = ""
        for ch in chunks:
            for c in ch.choices:
                delta = c.get("delta", {})
                text += delta.get("content") or ""
        assert text == "Hello"
        assert chunks[-2].choices[0]["finish_reason"] == "stop"

    def test_usage_chunk_surfaces_token_counts(self, compat):
        """Final usage chunk (with embedded cost summary) is exposed."""
        lines = SSE_LINES[:4]  # exclude [DONE]; include the usage chunk
        chunks = _iter_chunks(compat, lines)
        usage_chunks = [c for c in chunks if not c.choices and c.usage]
        assert len(usage_chunks) == 1
        u = usage_chunks[0].usage
        assert u.prompt_tokens == 9
        assert u.completion_tokens == 2
        assert u.total_tokens == 11

    def test_plain_delta_chunk_has_no_usage(self, compat):
        chunks = _iter_chunks(compat, SSE_LINES[:1])
        assert chunks[0].usage is None
        assert chunks[0].id == "chatcmpl-abc"
        assert chunks[0].model == "distributed-llm"

    def test_malformed_chunk_rejected(self, compat):
        with pytest.raises(ValueError):
            compat.ChatCompletionChunk.from_dict({"choices": "delta"})
