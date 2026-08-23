"""Comprehensive unit tests for the distllm-crewai integration.

All external HTTP calls are mocked via the DistLLM SDK clients (for the LLM
and Embedder) and via ``httpx`` (for the KnowledgeSource and Tools provider),
so no live DistLLM server is required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chat_object_response(content: str) -> MagicMock:
    """Build a response object shaped like ``ChatCompletionResponse``."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_chat_dict_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _make_embed_object_response(vectors) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=v) for v in vectors]
    return resp


def _make_embed_dict_response(vectors) -> dict:
    return {"data": [{"embedding": v} for v in vectors]}


def _async_stream(chunks):
    async def _gen(*args, **kwargs):
        for c in chunks:
            yield c
    return _gen


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def test_imports():
    """All four classes and the package import cleanly."""
    from distllm_crewai import (
        DistLLMCrewEmbedder,
        DistLLMCrewLLM,
        DistLLMKnowledgeSource,
        DistLLMToolProvider,
    )

    assert DistLLMCrewLLM is not None
    assert DistLLMCrewEmbedder is not None
    assert DistLLMToolProvider is not None
    assert DistLLMKnowledgeSource is not None


# ---------------------------------------------------------------------------
# LLM (sync + async)
# ---------------------------------------------------------------------------

def test_llm_init_defaults():
    from distllm_crewai import DistLLMCrewLLM

    llm = DistLLMCrewLLM(base_url="http://example:8000", model="m1", temperature=0.3)
    assert llm.base_url == "http://example:8000"
    assert llm.model == "m1"
    assert llm.temperature == 0.3
    assert llm.model_name == "m1"


def test_llm_generate_response_object():
    from distllm_crewai import DistLLMCrewLLM

    llm = DistLLMCrewLLM(base_url="http://localhost:8000")
    mock_client = MagicMock()
    mock_client.chat_completions.return_value = _make_chat_object_response("hello object")
    llm._client = mock_client

    out = llm.generate_response([{"role": "user", "content": "hi"}])
    assert out == "hello object"
    mock_client.chat_completions.assert_called_once()


def test_llm_generate_response_dict():
    from distllm_crewai import DistLLMCrewLLM

    llm = DistLLMCrewLLM(base_url="http://localhost:8000")
    mock_client = MagicMock()
    mock_client.chat_completions.return_value = _make_chat_dict_response("hello dict")
    llm._client = mock_client

    out = llm.generate_response([{"role": "user", "content": "hi"}])
    assert out == "hello dict"


def test_llm_generate_stream():
    from distllm_crewai import DistLLMCrewLLM

    llm = DistLLMCrewLLM(base_url="http://localhost:8000")
    mock_client = MagicMock()
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    mock_client.chat_completions_stream.return_value = iter(chunks)
    llm._client = mock_client

    out = "".join(llm.generate_stream([{"role": "user", "content": "hi"}]))
    assert out == "Hello world"


def test_llm_max_tokens_resolution():
    from distllm_crewai.llm import _resolve_max_tokens

    assert _resolve_max_tokens(10, 5) == 10
    assert _resolve_max_tokens(None, 5) == 5
    assert _resolve_max_tokens(None, None) == 256
    assert _resolve_max_tokens(None, None, 512) == 512


def test_llm_agenerate_response():
    from distllm_crewai import DistLLMCrewLLM

    llm = DistLLMCrewLLM(base_url="http://localhost:8000")
    mock_async = MagicMock()
    mock_async.chat_completions = AsyncMock(
        return_value=_make_chat_object_response("async hello")
    )
    llm._async_client = mock_async

    out = asyncio.run(
        llm.agenerate_response([{"role": "user", "content": "hi"}])
    )
    assert out == "async hello"


def test_llm_agenerate_stream():
    from distllm_crewai import DistLLMCrewLLM

    llm = DistLLMCrewLLM(base_url="http://localhost:8000")
    mock_async = MagicMock()
    chunks = [
        {"choices": [{"delta": {"content": "Async"}}]},
        {"choices": [{"delta": {"content": " stream"}}]},
    ]
    mock_async.chat_completions_stream = _async_stream(chunks)
    llm._async_client = mock_async

    async def _collect():
        return "".join(
            [c async for c in llm.agenerate_stream([{"role": "user", "content": "hi"}])]
        )

    assert asyncio.run(_collect()) == "Async stream"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

def test_embedder_embed_text_object():
    from distllm_crewai import DistLLMCrewEmbedder

    emb = DistLLMCrewEmbedder(base_url="http://localhost:8000")
    mock_client = MagicMock()
    mock_client.embeddings.return_value = _make_embed_object_response([[0.1, 0.2, 0.3]])
    emb._client = mock_client

    assert emb.embed_text("hello") == [0.1, 0.2, 0.3]
    assert emb.get_embedding_model() == "distributed-llm"


def test_embedder_embed_batch_dict():
    from distllm_crewai import DistLLMCrewEmbedder

    emb = DistLLMCrewEmbedder(base_url="http://localhost:8000")
    mock_client = MagicMock()
    mock_client.embeddings.return_value = _make_embed_dict_response(
        [[0.1, 0.2], [0.3, 0.4]]
    )
    emb._client = mock_client

    out = emb.embed_batch(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]


# ---------------------------------------------------------------------------
# KnowledgeSource (httpx mocked)
# ---------------------------------------------------------------------------

def test_knowledge_source_init():
    from distllm_crewai import DistLLMKnowledgeSource

    ks = DistLLMKnowledgeSource("src-1", base_url="http://localhost:8000")
    assert ks.source_id == "src-1"
    assert ks.base_url == "http://localhost:8000"
    assert ks.content == []


def test_knowledge_source_load_content():
    from distllm_crewai import DistLLMKnowledgeSource

    ks = DistLLMKnowledgeSource("src-1", base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"documents": [{"id": "d1", "text": "doc"}]}
    with patch("httpx.get", return_value=mock_resp):
        ks.load_content()
    assert ks.content == [{"id": "d1", "text": "doc"}]


def test_knowledge_source_query():
    from distllm_crewai import DistLLMKnowledgeSource

    ks = DistLLMKnowledgeSource("src-1", base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"score": 0.9, "text": "hit"}]}
    with patch("httpx.post", return_value=mock_resp):
        results = ks.query("find me", top_k=3)
    assert results == [{"score": 0.9, "text": "hit"}]


def test_knowledge_source_query_error_returns_empty():
    from distllm_crewai import DistLLMKnowledgeSource

    ks = DistLLMKnowledgeSource("src-1", base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {}
    with patch("httpx.post", return_value=mock_resp):
        results = ks.query("find me")
    assert results == []


# ---------------------------------------------------------------------------
# Tools provider (httpx mocked)
# ---------------------------------------------------------------------------

def test_tool_provider_default_tools():
    from distllm_crewai import DistLLMToolProvider

    provider = DistLLMToolProvider(base_url="http://localhost:8000")
    defaults = provider.default_tools()
    assert len(defaults) == 3
    names = {t["name"] for t in defaults}
    assert {"distllm_chat", "distllm_complete", "distllm_embed"} <= names


def test_tool_provider_call_tool():
    from distllm_crewai import DistLLMToolProvider

    provider = DistLLMToolProvider(base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"result": {"answer": 42}}
    with patch("httpx.post", return_value=mock_resp):
        out = provider.call_tool("distllm_chat", messages=[])
    assert '"answer": 42' in out


def test_tool_provider_get_tools_returns_list():
    from distllm_crewai import DistLLMToolProvider

    provider = DistLLMToolProvider(base_url="http://localhost:8000")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"name": "my_tool", "description": "does a thing"},
        ]
    }
    with patch("httpx.get", return_value=mock_resp):
        tools = provider.get_tools()
    assert isinstance(tools, list)


def test_tool_provider_builds_crew_tools():
    try:
        from crewai.tools import BaseTool  # noqa: F401
        from pydantic import BaseModel  # noqa: F401
    except Exception as e:
        pytest.skip(f"crewai not fully available: {e}")
    from distllm_crewai import DistLLMToolProvider

    provider = DistLLMToolProvider(base_url="http://localhost:8000")
    mock_get = MagicMock()
    mock_get.status_code = 200
    mock_get.json.return_value = {
        "data": [{"name": "my_tool", "description": "does a thing"}]
    }
    mock_post = MagicMock()
    mock_post.status_code = 200
    mock_post.json.return_value = {"result": "ok"}

    with patch("httpx.get", return_value=mock_get), patch(
        "httpx.post", return_value=mock_post
    ):
        tools = provider.get_tools()

    assert isinstance(tools, list)
    assert len(tools) >= 1
    # Exercise the dynamically built tool's run path (mocked HTTP).
    result = tools[0]._run(foo="bar")
    assert isinstance(result, str)
    assert '"result": "ok"' in result
