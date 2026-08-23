"""Regression: sync and async chat_completions_stream must yield the SAME item
type (content strings).

F-036: the async stream yielded raw ``content`` strings while the sync stream
yielded full SSE event dicts — switching between them changed the caller's
contract.  Both now yield content strings.
"""

from __future__ import annotations

from unittest import mock

from distllm_sdk.client import DistLLMClientSync


def _mock_sync_stream_response(lines):
    """Return a mock client whose ``stream()`` yields SSE lines."""
    ctx = mock.Mock()
    ctx.iter_lines.return_value = iter(lines)
    ctx.raise_for_status.return_value = None
    stream_cm = mock.MagicMock()
    stream_cm.__enter__.return_value = ctx
    client = mock.MagicMock()
    client.stream.return_value = stream_cm
    return client, ctx


SSE_LINES = [
    'data: {"choices": [{"delta": {"content": "Hello"}}]}',
    'data: {"choices": [{"delta": {"content": " world"}}]}',
    'data: [DONE]',
]


class TestSyncAsyncStreamParity:
    def test_sync_chat_stream_yields_content_strings(self):
        client, _ = _mock_sync_stream_response(SSE_LINES)
        dllm = DistLLMClientSync(api_key="k", base_url="http://localhost:8000")
        dllm._client = client

        items = list(dllm.chat_completions_stream([{"role": "user", "content": "hi"}]))

        assert items == ["Hello", " world"], f"expected content strings, got {items!r}"
        assert all(isinstance(x, str) for x in items)

    def test_sync_async_yield_identical_types(self):
        import asyncio

        from distllm_sdk.streaming import parse_sse_stream_async

        # Async: feed the same SSE through the async parser + content extraction.
        async def mock_async_lines():
            for line in SSE_LINES:
                yield line

        mock_resp = mock.Mock()
        mock_resp.aiter_lines = mock_async_lines
        mock_resp.raise_for_status.return_value = None

        async def _collect():
            out = []
            async for event in parse_sse_stream_async(mock_resp):
                if "choices" in event and event["choices"]:
                    content = event["choices"][0].get("delta", {}).get("content")
                    if content:
                        out.append(content)
            return out

        async_items = asyncio.run(_collect())
        client, _ = _mock_sync_stream_response(SSE_LINES)
        dllm = DistLLMClientSync(api_key="k", base_url="http://localhost:8000")
        dllm._client = client
        sync_items = list(dllm.chat_completions_stream([{"role": "user", "content": "hi"}]))

        assert sync_items == async_items == ["Hello", " world"]
        assert all(type(a) is type(b) for a, b in zip(sync_items, async_items))