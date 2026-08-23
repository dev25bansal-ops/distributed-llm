"""Advanced unit tests for distllm-llamaindex.

Covers async code paths, batch embeddings, and streaming chat.
All DistLLM HTTP API calls are mocked — no live server required.
"""

import asyncio
import os
import sys
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make `integrations/` importable so `tools.py`'s `from _common.base_tool_provider`
# import resolves without a live server / installed editable metadata.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from llama_index.core.llms import ChatMessage, MessageRole  # noqa: E402


def _async_gen(items: List[Any]):
    """Return an async generator yielding the given items (for streaming mocks)."""

    async def gen():
        for item in items:
            yield item

    return gen()


def _make_chat_response(content: str):
    """Build a ChatCompletionResponse-like SDK object for async chat mocks."""
    from distllm.sdk.types import (
        ChatCompletionResponse,
        ChatChoice,
        ChatMessage as SDKChatMessage,
        UsageInfo,
    )

    mock_choice = MagicMock(spec=ChatChoice)
    mock_choice.index = 0
    mock_choice.message = SDKChatMessage(role="assistant", content=content)
    mock_choice.delta = None
    mock_choice.finish_reason = "stop"
    return ChatCompletionResponse(
        id="chat-id",
        model="test",
        choices=[mock_choice],
        created=123,
        object="chat.completion",
        usage=UsageInfo(prompt_tokens=5, completion_tokens=8, total_tokens=13),
        generation_time=0.3,
    )


def _make_completion_response(text: str):
    """Build a CompletionResponse-like SDK object for async completion mocks."""
    from distllm.sdk.types import (
        CompletionResponse,
        CompletionChoice,
        UsageInfo,
    )

    mock_choice = MagicMock(spec=CompletionChoice)
    mock_choice.text = text
    mock_choice.index = 0
    mock_choice.finish_reason = "stop"
    return CompletionResponse(
        id="comp-id",
        model="test",
        choices=[mock_choice],
        created=123,
        object="text_completion",
        usage=UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7),
        generation_time=0.2,
    )


def test_achat_mocked():
    """Verify achat awaits the async client's chat_completions."""
    from distllm_llamaindex.llms import DistLLM
    from llama_index.core.llms import ChatMessage, MessageRole

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    resp = _make_chat_response("async chat reply")

    with patch.object(llm._async_client, "chat_completions", new=AsyncMock(return_value=resp)):
        result = asyncio.run(llm.achat([ChatMessage(role=MessageRole.USER, content="Hi")]))
        assert result.message.content == "async chat reply"
        assert result.message.role == MessageRole.ASSISTANT


def test_achat_passes_payload_kwargs():
    """Verify achat forwards resolved temperature / max_tokens into the SDK call."""
    from distllm_llamaindex.llms import DistLLM
    from llama_index.core.llms import ChatMessage, MessageRole

    llm = DistLLM(model="test", base_url="http://localhost:8000", temperature=0.3)
    resp = _make_chat_response("ok")

    with patch.object(llm._async_client, "chat_completions", new=AsyncMock(return_value=resp)) as mock_call:
        asyncio.run(
            llm.achat(
                [ChatMessage(role=MessageRole.USER, content="Hi")],
                temperature=0.2,
                max_tokens=512,
            )
        )
        kwargs = mock_call.call_args.kwargs
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 512
        assert kwargs["model"] == "test"


def test_astream_chat_mocked():
    """Verify astream_chat yields content deltas from an async stream."""
    from distllm_llamaindex.llms import DistLLM
    from llama_index.core.llms import ChatMessage, MessageRole

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
    ]

    with patch.object(llm._async_client, "chat_completions_stream", return_value=_async_gen(chunks)):
        result = asyncio.run(
            _collect_astream_chat(llm, [ChatMessage(role=MessageRole.USER, content="Hi")])
        )
        # The wrapper skips the final empty-content chunk.
        assert [c.delta for c in result] == ["Hello", " world"]
        assert result[0].message.role == MessageRole.ASSISTANT


async def _collect_astream_chat(llm, messages):
    return [chunk async for chunk in llm.astream_chat(messages)]


def test_astream_chat_skips_empty_content():
    """Verify astream_chat skips chunks that carry no content delta."""
    from distllm_llamaindex.llms import DistLLM
    from llama_index.core.llms import ChatMessage, MessageRole

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    chunks = [
        {"choices": [{"delta": {}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "only"}, "finish_reason": "stop"}]},
    ]

    with patch.object(llm._async_client, "chat_completions_stream", return_value=_async_gen(chunks)):
        result = asyncio.run(
            _collect_astream_chat(llm, [ChatMessage(role=MessageRole.USER, content="Hi")])
        )
        # The empty delta chunk is skipped; only the "only" chunk is yielded.
        assert len(result) == 1
        assert result[0].delta == "only"


def test_astream_complete_mocked():
    """Verify astream_complete yields text deltas from an async stream."""
    from distllm_llamaindex.llms import DistLLM

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    chunks = [
        {"choices": [{"text": "distributed", "finish_reason": None}]},
        {"choices": [{"text": " inference", "finish_reason": "stop"}]},
    ]

    with patch.object(llm._async_client, "completions_stream", return_value=_async_gen(chunks), create=True):
        result = asyncio.run(_collect_astream_complete(llm, "explain"))
        assert [c.delta for c in result] == ["distributed", " inference"]
        assert result[0].text == "distributed"


async def _collect_astream_complete(llm, prompt):
    return [chunk async for chunk in llm.astream_complete(prompt)]


def test_aget_text_embedding_mocked():
    """Verify _aget_text_embedding awaits async embeddings for a single text."""
    from distllm_llamaindex.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    mock_resp = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    with patch.object(emb._async_client, "embeddings", new=AsyncMock(return_value=mock_resp)):
        result = asyncio.run(emb._aget_text_embedding("hello"))
        assert result == [0.1, 0.2, 0.3]


def test_aget_text_embedding_batch_mocked():
    """Verify _aget_text_embeddings returns one vector per input, in order."""
    from distllm_llamaindex.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    mock_resp = {
        "data": [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
            {"embedding": [0.5, 0.6]},
        ]
    }

    with patch.object(emb._async_client, "embeddings", new=AsyncMock(return_value=mock_resp)) as mock_call:
        result = asyncio.run(emb._aget_text_embeddings(["a", "b", "c"]))
        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        assert result[2] == [0.5, 0.6]
        # Async batch path forwards the full list to the SDK.
        assert mock_call.call_args.kwargs["input"] == ["a", "b", "c"]


def test_aget_query_embedding_mocked():
    """Verify _aget_query_embedding delegates to the async text embedding path."""
    from distllm_llamaindex.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    mock_resp = {"data": [{"embedding": [9.0, 8.0]}]}

    with patch.object(emb._async_client, "embeddings", new=AsyncMock(return_value=mock_resp)):
        result = asyncio.run(emb._aget_query_embedding("search query"))
        assert result == [9.0, 8.0]


def test_aget_text_embeddings_empty_mocked():
    """Verify async batch path returns an empty list when the SDK yields no data."""
    from distllm_llamaindex.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    mock_resp = {"data": []}

    with patch.object(emb._async_client, "embeddings", new=AsyncMock(return_value=mock_resp)):
        result = asyncio.run(emb._aget_text_embeddings([]))
        assert result == []


def test_achat_and_embeddings_use_separate_clients():
    """Verify async chat and async embeddings route through the async client only."""
    from distllm_llamaindex.llms import DistLLM
    from distllm_llamaindex.embeddings import DistLLMEmbeddings
    from llama_index.core.llms import ChatMessage, MessageRole

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    emb = DistLLMEmbeddings(base_url="http://localhost:8000")

    chat_resp = _make_chat_response("hi")
    emb_resp = {"data": [{"embedding": [1.0]}]}

    with patch.object(llm._async_client, "chat_completions", new=AsyncMock(return_value=chat_resp)) as llm_mock, \
         patch.object(emb._async_client, "embeddings", new=AsyncMock(return_value=emb_resp)) as emb_mock, \
         patch.object(llm._client, "chat_completions", new=MagicMock()) as llm_sync_mock, \
         patch.object(emb._client, "embeddings", new=MagicMock()) as emb_sync_mock:
        out = asyncio.run(_run_async_pair(llm, emb))
        assert out["chat"] == "hi"
        assert out["embed"] == [1.0]
        llm_mock.assert_called_once()
        emb_mock.assert_called_once()
        # Sync clients must not be touched in the async paths.
        llm_sync_mock.assert_not_called()
        emb_sync_mock.assert_not_called()


async def _run_async_pair(llm, emb):
    chat = await llm.achat([ChatMessage(role=MessageRole.USER, content="Hi")])
    vec = await emb._aget_query_embedding("q")
    return {"chat": chat.message.content, "embed": vec}  # noqa: F821


def test_stream_chat_and_async_chat_equivalent():
    """Verify sync streaming and async non-stream chat produce the same final content."""
    from distllm_llamaindex.llms import DistLLM
    from llama_index.core.llms import ChatMessage, MessageRole

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    stream_chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
    ]
    chat_resp = _make_chat_response("Hello world")

    with patch.object(llm._client, "chat_completions_stream", return_value=iter(stream_chunks)), \
         patch.object(llm._async_client, "chat_completions", new=AsyncMock(return_value=chat_resp)):
        sync_parts = [c.delta for c in llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Hi")])]
        async_content = asyncio.run(
            llm.achat([ChatMessage(role=MessageRole.USER, content="Hi")])
        ).message.content
        assert "".join(sync_parts) == async_content == "Hello world"
