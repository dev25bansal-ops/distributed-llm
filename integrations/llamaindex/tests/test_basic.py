"""Basic unit tests for distllm-llamaindex.

These tests use mocked SDK to avoid requiring a running server.
"""

from unittest.mock import MagicMock, patch, AsyncMock


def test_imports():
    """Verify all top-level imports work."""
    from distllm_llamaindex import DistLLM, DistLLMEmbeddings, DistLLMToolProvider
    assert DistLLM is not None
    assert DistLLMEmbeddings is not None
    assert DistLLMToolProvider is not None


def test_llm_init():
    """Verify DistLLM construction with minimal args."""
    from distllm_llamaindex.llms import DistLLM

    llm = DistLLM(model="test-model", base_url="http://localhost:8000")
    assert llm.model == "test-model"
    assert llm.base_url == "http://localhost:8000"
    assert llm.temperature == 0.7


def test_llm_metadata():
    """Verify DistLLM metadata property."""
    from distllm_llamaindex.llms import DistLLM

    llm = DistLLM(model="my-model", base_url="http://localhost:8000")
    meta = llm.metadata
    assert meta.model_name == "my-model"
    assert meta.is_chat_model is True
    assert meta.is_function_calling_model is True


def test_complete_mocked():
    """Verify complete calls the SDK correctly."""
    from distllm_llamaindex.llms import DistLLM
    from distllm.sdk.types import CompletionResponse, CompletionChoice, UsageInfo

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    mock_choice = MagicMock(spec=CompletionChoice)
    mock_choice.text = "distributed inference is cool"
    mock_choice.index = 0
    mock_choice.finish_reason = "stop"
    mock_resp = CompletionResponse(
        id="test-id",
        model="test",
        choices=[mock_choice],
        created=123,
        object="text_completion",
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        generation_time=0.5,
    )

    with patch.object(llm._client, "completions", return_value=mock_resp):
        result = llm.complete("What is distributed inference?")
        assert result.text == "distributed inference is cool"


def test_chat_mocked():
    """Verify chat calls the SDK correctly."""
    from distllm_llamaindex.llms import DistLLM
    from llama_index.core.llms import ChatMessage, MessageRole
    from distllm.sdk.types import ChatCompletionResponse, ChatChoice, ChatMessage as SDKChatMessage, UsageInfo

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    mock_choice = MagicMock(spec=ChatChoice)
    mock_choice.index = 0
    mock_choice.message = SDKChatMessage(role="assistant", content="Hello there!")
    mock_choice.delta = None
    mock_choice.finish_reason = "stop"
    mock_resp = ChatCompletionResponse(
        id="chat-id",
        model="test",
        choices=[mock_choice],
        created=123,
        object="chat.completion",
        usage=UsageInfo(prompt_tokens=5, completion_tokens=8, total_tokens=13),
        generation_time=0.3,
    )

    with patch.object(llm._client, "chat_completions", return_value=mock_resp):
        result = llm.chat([ChatMessage(role=MessageRole.USER, content="Hi")])
        assert result.message.content == "Hello there!"
        assert result.message.role == MessageRole.ASSISTANT


def test_stream_chat_mocked():
    """Verify stream_chat yields chunks correctly."""
    from distllm_llamaindex.llms import DistLLM
    from llama_index.core.llms import ChatMessage, MessageRole

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    mock_chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
    ]

    with patch.object(llm._client, "chat_completions_stream", return_value=iter(mock_chunks)):
        chunks = list(llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Hi")]))
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert chunks[2].delta == ""  # final chunk with finish_reason


def test_embeddings_init():
    """Verify DistLLMEmbeddings construction."""
    from distllm_llamaindex.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    assert emb.model == "distributed-llm"
    assert emb.base_url == "http://localhost:8000"


def test_embeddings_get_text_embedding_mocked():
    """Verify get_text_embedding calls the SDK correctly."""
    from distllm_llamaindex.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

    with patch.object(emb._client, "embeddings", return_value=mock_resp):
        result = emb.get_text_embedding("hello")
        assert result == [0.1, 0.2, 0.3]


def test_embeddings_get_text_embedding_batch_mocked():
    """Verify get_text_embedding_batch calls the SDK correctly."""
    from distllm_llamaindex.embeddings import DistLLMEmbeddings

    emb = DistLLMEmbeddings(base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]

    with patch.object(emb._client, "embeddings", return_value=mock_resp):
        result = emb.get_text_embedding_batch(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]


def test_tool_provider_defaults():
    """Verify DistLLMToolProvider returns default tools."""
    from distllm_llamaindex.tools import DistLLMToolProvider

    provider = DistLLMToolProvider(base_url="http://localhost:8000")
    tools = provider._default_tools()
    assert len(tools) == 3
    names = [t["name"] for t in tools]
    assert "distllm_chat" in names
    assert "distllm_complete" in names
    assert "distllm_embed" in names


def test_message_conversion():
    """Verify ChatMessage <-> DistLLM dict conversion."""
    from distllm_llamaindex.llms import _to_chat_message, _from_chat_response
    from llama_index.core.llms import ChatMessage, MessageRole

    # ChatMessage -> dict
    d = _to_chat_message(ChatMessage(role=MessageRole.USER, content="hello"))
    assert d == {"role": "user", "content": "hello"}

    d = _to_chat_message(ChatMessage(role=MessageRole.SYSTEM, content="be helpful"))
    assert d == {"role": "system", "content": "be helpful"}

    d = _to_chat_message(ChatMessage(role=MessageRole.ASSISTANT, content="world"))
    assert d == {"role": "assistant", "content": "world"}

    # dict -> ChatMessage
    msg = _from_chat_response({"role": "assistant", "content": "hi"})
    assert msg.content == "hi"
    assert msg.role == MessageRole.ASSISTANT


def test_async_complete_mocked():
    """Verify acomplete calls the SDK correctly."""
    from distllm_llamaindex.llms import DistLLM
    from distllm.sdk.types import CompletionResponse, CompletionChoice, UsageInfo
    import pytest

    llm = DistLLM(model="test", base_url="http://localhost:8000")
    mock_choice = MagicMock(spec=CompletionChoice)
    mock_choice.text = "async result"
    mock_choice.index = 0
    mock_choice.finish_reason = "stop"
    mock_resp = CompletionResponse(
        id="test-id",
        model="test",
        choices=[mock_choice],
        created=123,
        object="text_completion",
        usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        generation_time=0.2,
    )

    with patch.object(llm._async_client, "completions", new=AsyncMock(return_value=mock_resp)):
        import asyncio
        result = asyncio.run(llm.acomplete("test"))
        assert result.text == "async result"
