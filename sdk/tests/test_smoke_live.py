"""Live-call smoke tests for distllm_sdk against an in-process mock server.

Each test spins a canned OpenAI-compatible HTTP surface using httpx.MockTransport
(no network, no threads) and asserts that both DistLLMClientSync and
DistLLMClient parse real wire responses into their typed dataclasses.

Run:  python -m pytest sdk/tests/test_smoke_live.py -v
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx

from distllm_sdk import (
    ChatCompletionResponse,
    CompletionResponse,
    EmbeddingResponse,
    ModelList,
    RetryConfig,
)
from distllm_sdk.client import DistLLMClient, DistLLMClientSync

# ── Canned OpenAI-compatible payloads ─────────────────────────────────────

MODELS_RESPONSE = {
    "object": "list",
    "data": [
        {"id": "distributed-llm", "object": "model", "created": 1700000000, "owned_by": "distllm"},
        {"id": "tiny-stories-1m", "object": "model", "created": 1700000001, "owned_by": "distllm"},
    ],
}

CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-smoke-001",
    "object": "chat.completion",
    "created": 1700000100,
    "model": "distributed-llm",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from the mock cluster!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 8,
        "total_tokens": 13,
    },
}

COMPLETIONS_RESPONSE = {
    "id": "cmpl-smoke-001",
    "object": "text_completion",
    "created": 1700000200,
    "model": "distributed-llm",
    "choices": [{"index": 0, "text": " Once upon a time.", "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
}

EMBEDDINGS_RESPONSE = {
    "object": "list",
    "model": "distributed-llm",
    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2},
}

HEALTH_RESPONSE = {"status": "ok", "nodes": 2, "uptime": 1234.5}

SSE_BODY = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"lo!"}}]}\n\n'
    "data: [DONE]\n\n"
)


# ── In-process mock OpenAI-compatible transport ───────────────────────────


def mock_openai_transport(captured: dict | None = None) -> httpx.MockTransport:
    """Build an httpx.MockTransport serving canned /v1/* + /health routes."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["auth"] = request.headers.get("authorization")
            try:
                captured["json"] = json.loads(request.content)
            except ValueError:
                captured["json"] = None

        path = request.url.path
        if request.method == "GET" and path == "/v1/models":
            return httpx.Response(200, json=MODELS_RESPONSE)
        if request.method == "POST" and path == "/v1/chat/completions":
            body = json.loads(request.content)
            if body.get("stream"):
                return httpx.Response(
                    200,
                    text=SSE_BODY,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=CHAT_COMPLETION_RESPONSE)
        if request.method == "POST" and path == "/v1/completions":
            return httpx.Response(200, json=COMPLETIONS_RESPONSE)
        if request.method == "POST" and path == "/v1/embeddings":
            return httpx.Response(200, json=EMBEDDINGS_RESPONSE)
        if request.method == "GET" and path == "/health":
            return httpx.Response(200, json=HEALTH_RESPONSE)
        return httpx.Response(404, json={"error": {"message": f"no route {path}", "type": "invalid_request_error"}})

    return httpx.MockTransport(handler)


def _sync_client(**kwargs) -> DistLLMClientSync:
    base_kwargs = dict(base_url="http://mock.local", api_key="sk-test-key", retry=RetryConfig(max_retries=0))
    base_kwargs.update(kwargs)
    return DistLLMClientSync(**base_kwargs)


def _async_client(**kwargs) -> DistLLMClient:
    base_kwargs = dict(base_url="http://mock.local", api_key="sk-test-key", retry=RetryConfig(max_retries=0))
    base_kwargs.update(kwargs)
    return DistLLMClient(**base_kwargs)


# ── Sync client smoke tests ───────────────────────────────────────────────


class TestSyncSmoke:
    def test_list_models_parses(self):
        client = _sync_client(transport=mock_openai_transport())
        with client:
            models = client.list_models()
        assert isinstance(models, ModelList)
        assert [m.id for m in models.data] == ["distributed-llm", "tiny-stories-1m"]
        assert models.data[0].owned_by == "distllm"

    def test_chat_completions_parses(self):
        client = _sync_client(transport=mock_openai_transport())
        with client:
            resp = client.chat_completions(messages=[{"role": "user", "content": "Hi"}])
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.id == "chatcmpl-smoke-001"
        assert resp.model == "distributed-llm"
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content == "Hello from the mock cluster!"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.prompt_tokens == 5
        assert resp.usage.total_tokens == 13
        # stats recorded for the call
        assert client.stats.total_calls == 1

    def test_completions_parses(self):
        client = _sync_client(transport=mock_openai_transport())
        with client:
            resp = client.completions(prompt="Tell me a story")
        assert isinstance(resp, CompletionResponse)
        assert resp.id == "cmpl-smoke-001"
        assert resp.choices[0].text == " Once upon a time."
        assert resp.usage.total_tokens == 7

    def test_embeddings_parse(self):
        client = _sync_client(transport=mock_openai_transport())
        with client:
            resp = client.embeddings(input="hello")
        assert isinstance(resp, EmbeddingResponse)
        assert resp.model == "distributed-llm"
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]

    def test_health_check(self):
        client = _sync_client(transport=mock_openai_transport())
        with client:
            health = client.health_check()
        assert health == HEALTH_RESPONSE

    def test_streaming_chat_yields_deltas(self):
        client = _sync_client(transport=mock_openai_transport())
        with client:
            deltas = list(client.chat_completions_stream(messages=[{"role": "user", "content": "Hi"}]))
        assert deltas == ["Hel", "lo!"]

    def test_auth_header_sent_and_payload_shape(self):
        captured: dict = {}
        client = _sync_client(transport=mock_openai_transport(captured))
        with client:
            client.chat_completions(messages=[{"role": "user", "content": "Hi"}], model="distributed-llm")
        assert captured["method"] == "POST"
        assert captured["path"] == "/v1/chat/completions"
        assert captured["auth"] == "Bearer sk-test-key"
        payload = captured["json"]
        assert payload["model"] == "distributed-llm"
        assert payload["messages"] == [{"role": "user", "content": "Hi"}]
        assert payload["stream"] is False


# ── Async client smoke tests ──────────────────────────────────────────────


class TestAsyncSmoke:
    def test_list_models_async(self):
        async def run():
            client = _async_client(transport=mock_openai_transport())
            async with client:
                return await client.list_models()

        models = asyncio.run(run())
        assert isinstance(models, ModelList)
        assert models.data[0].id == "distributed-llm"

    def test_chat_completions_async(self):
        async def run():
            client = _async_client(transport=mock_openai_transport())
            async with client:
                return await client.chat_completions(messages=[{"role": "user", "content": "Hi"}])

        resp = asyncio.run(run())
        assert isinstance(resp, ChatCompletionResponse)
        assert resp.choices[0].message.content == "Hello from the mock cluster!"
        assert resp.usage.completion_tokens == 8

    def test_streaming_chat_async(self):
        async def run():
            client = _async_client(transport=mock_openai_transport())
            async with client:
                out = []
                async for delta in client.chat_completions_stream(messages=[{"role": "user", "content": "Hi"}]):
                    out.append(delta)
                return out

        assert asyncio.run(run()) == ["Hel", "lo!"]
