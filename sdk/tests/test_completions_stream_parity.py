"""Parity tests: completions_stream must yield text STRINGS like
chat_completions_stream (W2-39, analysis A2-8).

Before the fix, both async and sync ``completions_stream`` yielded raw SSE
event dicts while ``chat_completions_stream`` yielded delta-content strings —
switching between the two changed the caller's contract (``"".join()``
raised TypeError on dicts, and framework adapters that expected strings
received dicts).

Covers both wire shapes the server can emit:
- real format: text at ``choices[0].text`` (api/streaming.py ``_build_chunk``
  non-chat branch),
- documented OpenAPI example format: text at ``choices[0].delta.text``.

Run:  python -m pytest sdk/tests/test_completions_stream_parity.py -v
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import httpx

from distllm_sdk.client import DistLLMClient, DistLLMClientSync

# Real DistLLM server wire format: text directly on the choice.
COMPLETIONS_SSE_BODY_TEXT = (
    'data: {"id":"cmpl-w239a","object":"text_completion.chunk","created":1700000300,'
    '"model":"distributed-llm","choices":[{"index":0,"text":"Hel","finish_reason":null}]}\n\n'
    'data: {"id":"cmpl-w239a","object":"text_completion.chunk","created":1700000300,'
    '"model":"distributed-llm","choices":[{"index":0,"text":"lo!","finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)

# Alternate documented shape (OpenAPI example): text nested in delta.
COMPLETIONS_SSE_BODY_DELTA = (
    'data: {"id":"cmpl-w239b","object":"text_completion.chunk","created":1700000301,'
    '"model":"distributed-llm","choices":[{"index":0,"delta":{"text":"Go"},"finish_reason":null}]}\n\n'
    'data: {"id":"cmpl-w239b","object":"text_completion.chunk","created":1700000301,'
    '"model":"distributed-llm","choices":[{"index":0,"delta":{"text":"odbye"},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)

CHAT_SSE_BODY = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"lo!"}}]}\n\n'
    "data: [DONE]\n\n"
)


def _stream_transport(sse_body: str) -> httpx.MockTransport:
    """Mock transport serving canned SSE bodies for the two streaming routes."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/v1/completions":
            return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})
        if request.method == "POST" and path == "/v1/chat/completions":
            return httpx.Response(200, text=CHAT_SSE_BODY, headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"error": {"message": f"no route {path}", "type": "invalid_request_error"}})

    return httpx.MockTransport(handler)


def _sync_client(transport: httpx.MockTransport) -> DistLLMClientSync:
    return DistLLMClientSync(base_url="http://mock.local", api_key="sk-test-key", transport=transport)


def _async_client(transport: httpx.MockTransport) -> DistLLMClient:
    return DistLLMClient(base_url="http://mock.local", api_key="sk-test-key", transport=transport)


class TestCompletionsStreamYieldsStrings:
    def test_sync_yields_strings(self):
        client = _sync_client(_stream_transport(COMPLETIONS_SSE_BODY_TEXT))
        with client:
            items = list(client.completions_stream(prompt="Tell me a story"))
        assert items == ["Hel", "lo!"], f"expected text strings, got {items!r}"
        assert all(isinstance(x, str) for x in items)

    def test_sync_joins_into_full_text(self):
        # Direct repro of the pre-fix bug: joining dicts raised TypeError.
        client = _sync_client(_stream_transport(COMPLETIONS_SSE_BODY_TEXT))
        with client:
            joined = "".join(client.completions_stream(prompt="hi"))
        assert joined == "Hello!"

    def test_sync_tolerates_delta_text_shape(self):
        client = _sync_client(_stream_transport(COMPLETIONS_SSE_BODY_DELTA))
        with client:
            items = list(client.completions_stream(prompt="hi"))
        assert items == ["Go", "odbye"]
        assert all(isinstance(x, str) for x in items)

    def test_async_yields_strings(self):
        async def run():
            client = _async_client(_stream_transport(COMPLETIONS_SSE_BODY_TEXT))
            async with client:
                out = []
                async for item in client.completions_stream(prompt="Tell me a story"):
                    out.append(item)
                return out

        items = asyncio.run(run())
        assert items == ["Hel", "lo!"], f"expected text strings, got {items!r}"
        assert all(isinstance(x, str) for x in items)

    def test_async_joins_into_full_text(self):
        async def run():
            client = _async_client(_stream_transport(COMPLETIONS_SSE_BODY_TEXT))
            async with client:
                return "".join([item async for item in client.completions_stream(prompt="hi")])

        assert asyncio.run(run()) == "Hello!"

    def test_async_tolerates_delta_text_shape(self):
        async def run():
            client = _async_client(_stream_transport(COMPLETIONS_SSE_BODY_DELTA))
            async with client:
                return [item async for item in client.completions_stream(prompt="hi")]

        items = asyncio.run(run())
        assert items == ["Go", "odbye"]
        assert all(isinstance(x, str) for x in items)


class TestCompletionsStreamParityWithChatStream:
    """Both stream APIs on one client must hand back the SAME item type."""

    def test_sync_item_type_matches_chat_stream(self):
        client = _sync_client(_stream_transport(COMPLETIONS_SSE_BODY_TEXT))
        with client:
            chat_items = list(client.chat_completions_stream(messages=[{"role": "user", "content": "Hi"}]))
            completion_items = list(client.completions_stream(prompt="hi"))
        assert all(isinstance(x, str) for x in chat_items)
        assert all(isinstance(x, str) for x in completion_items)
        assert {type(x) for x in chat_items} == {type(x) for x in completion_items}

    def test_async_item_type_matches_chat_stream(self):
        async def run():
            client = _async_client(_stream_transport(COMPLETIONS_SSE_BODY_TEXT))
            async with client:
                chat_items = [x async for x in client.chat_completions_stream(messages=[{"role": "user", "content": "Hi"}])]
                completion_items = [x async for x in client.completions_stream(prompt="hi")]
                return chat_items, completion_items

        chat_items, completion_items = asyncio.run(run())
        assert all(isinstance(x, str) for x in chat_items)
        assert all(isinstance(x, str) for x in completion_items)
        assert {type(x) for x in chat_items} == {type(x) for x in completion_items}
