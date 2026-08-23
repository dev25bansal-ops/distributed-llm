"""Advanced tests for distllm-langchain DistLLMChat (no live server).

All HTTP is mocked by monkeypatching the underlying DistLLM SDK client methods
on the chat-model instance. Covers:
  * all message types (system / user / assistant / tool)
  * streaming (sync + async)
  * structured output (dict + pydantic, with/without raw)
  * tool binding
  * error states
  * usage / latency tracking
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat() -> "DistLLMChat":
    from distllm_langchain.chat_models import DistLLMChat

    return DistLLMChat(model="test-model", base_url="http://localhost:8000")


def _openai_dict(content: str, **extra: Any) -> dict:
    """Build a minimal OpenAI-style chat completion response dict."""
    choice: dict = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
    choice["message"].update(extra)
    return {
        "id": "chatcmpl-1",
        "model": "test-model",
        "choices": [choice],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


# ---------------------------------------------------------------------------
# Message type conversion
# ---------------------------------------------------------------------------


def test_system_and_user_message_conversion():
    from distllm_langchain.chat_models import _convert_message_to_dict

    s = _convert_message_to_dict(SystemMessage(content="be brief"))
    assert s == {"role": "system", "content": "be brief"}
    u = _convert_message_to_dict(HumanMessage(content="hello"))
    assert u == {"role": "user", "content": "hello"}


def test_assistant_tool_calls_roundtrip():
    from distllm_langchain.chat_models import (
        _convert_dict_to_message,
        _convert_message_to_dict,
    )

    ai = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"loc": "NY"}}],
    )
    d = _convert_message_to_dict(ai)
    assert d["role"] == "assistant"
    assert d["tool_calls"][0]["id"] == "call_1"
    assert d["tool_calls"][0]["function"]["name"] == "get_weather"
    assert d["tool_calls"][0]["function"]["arguments"] == {"loc": "NY"}

    # Back from dict
    back = _convert_dict_to_message(
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": {}}}],
        }
    )
    assert isinstance(back, AIMessage)
    assert back.additional_kwargs["tool_calls"][0]["function"]["name"] == "f"


def test_tool_message_conversion():
    from distllm_langchain.chat_models import (
        _convert_dict_to_message,
        _convert_message_to_dict,
    )

    tm = ToolMessage(content="22C", tool_call_id="call_1")
    d = _convert_message_to_dict(tm)
    assert d == {"role": "tool", "content": "22C", "tool_call_id": "call_1"}

    back = _convert_dict_to_message({"role": "tool", "content": "x", "tool_call_id": "t1"})
    assert isinstance(back, ToolMessage)
    assert back.tool_call_id == "t1"


def test_build_payload_includes_all_message_types():
    llm = _make_chat()
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "f", "args": {}}]),
        ToolMessage(content="result", tool_call_id="c1"),
    ]
    payload = llm._build_payload(messages, stop=None, kwargs={})
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert payload["model"] == "test-model"
    assert payload["messages"][2]["tool_calls"][0]["id"] == "c1"


# ---------------------------------------------------------------------------
# Sync generate with every message type
# ---------------------------------------------------------------------------


def test_sync_generate_all_message_types():
    llm = _make_chat()
    llm._client.chat_completions = MagicMock(return_value=_openai_dict("final answer"))

    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "f", "args": {}}]),
        ToolMessage(content="result", tool_call_id="c1"),
        HumanMessage(content="again"),
    ]
    result = llm.invoke(messages)
    assert isinstance(result, AIMessage)
    assert result.content == "final answer"

    # Verify the SDK was called with all four roles forwarded
    call_kwargs = llm._client.chat_completions.call_args.kwargs
    roles = [m["role"] for m in call_kwargs["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]


def test_usage_and_latency_tracked_in_llm_output():
    llm = _make_chat()
    llm._client.chat_completions = MagicMock(return_value=_openai_dict("hi"))

    # _generate sets llm_output on the ChatResult; inspect it directly
    full = llm._generate([HumanMessage(content="hey")], run_manager=None)
    out = full.llm_output
    assert out["token_usage"]["prompt_tokens"] == 5
    assert out["token_usage"]["completion_tokens"] == 7
    assert out["model_name"] == "test-model"
    assert "distllm_latency_ms" in out


# ---------------------------------------------------------------------------
# Streaming (sync)
# ---------------------------------------------------------------------------


def test_sync_stream_basic_concatenation():
    llm = _make_chat()
    llm.streaming = True
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    llm._client.chat_completions_stream = MagicMock(return_value=iter(chunks))

    out = ""
    for chunk in llm.stream([HumanMessage(content="hi")]):
        out += chunk.content
    assert out == "Hello world"


def test_sync_stream_skips_empty_deltas():
    llm = _make_chat()
    llm.streaming = True
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},  # no content -> skipped
        {"choices": [{"delta": {"content": "x"}}]},
        {"choices": [{"delta": {}}]},  # empty delta -> skipped
    ]
    llm._client.chat_completions_stream = MagicMock(return_value=iter(chunks))

    collected = [c.content for c in llm.stream([HumanMessage(content="hi")])]
    # langchain appends a final empty AIMessageChunk (chunk_position="last")
    assert [c for c in collected if c] == ["x"]


def test_sync_stream_pipeline_metadata():
    llm = _make_chat()
    llm.streaming = True
    chunks = [
        {
            "choices": [{"delta": {"content": "step"}}],
            "pipeline_stage": "attention",
            "node_id": "node-3",
            "latency_ms": 12.5,
        }
    ]
    llm._client.chat_completions_stream = MagicMock(return_value=iter(chunks))

    chunk = next(llm.stream([HumanMessage(content="hi")]))
    assert chunk.content == "step"
    # stream() yields AIMessageChunk objects (not ChatGenerationChunk)
    assert chunk.additional_kwargs["pipeline_stage"] == "attention"
    assert chunk.additional_kwargs["node_id"] == "node-3"
    assert chunk.additional_kwargs["latency_ms"] == 12.5


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def test_structured_output_dict_schema():
    llm = _make_chat()
    payload_dict = {"name": "Alice", "age": 30}
    llm._client.chat_completions = MagicMock(return_value=_openai_dict(json.dumps(payload_dict)))

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    chain = llm.with_structured_output(schema)
    result = chain.invoke([HumanMessage(content="who?")])
    assert isinstance(result, AIMessage)
    assert json.loads(result.content)["name"] == "Alice"


def test_structured_output_include_raw():
    llm = _make_chat()
    llm._client.chat_completions = MagicMock(return_value=_openai_dict('{"ok": true}'))

    chain = llm.with_structured_output({"type": "object"}, include_raw=True)
    result = chain.invoke([HumanMessage(content="go")])
    assert isinstance(result, dict)
    assert result["raw"].content == '{"ok": true}'
    assert result["parsed"] == {"ok": True}


def test_structured_output_pydantic_schema():
    class Person(BaseModel):
        name: str
        age: int

    llm = _make_chat()
    llm._client.chat_completions = MagicMock(return_value=_openai_dict('{"name": "Bob", "age": 42}'))

    chain = llm.with_structured_output(Person)
    result = chain.invoke([HumanMessage(content="person?")])
    assert isinstance(result, AIMessage)
    assert json.loads(result.content)["age"] == 42


# ---------------------------------------------------------------------------
# Tool binding
# ---------------------------------------------------------------------------


def test_bind_tools_forwards_tools_to_client():
    llm = _make_chat()
    llm._client.chat_completions = MagicMock(return_value=_openai_dict("used tool"))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "get weather",
                "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}},
            },
        }
    ]
    bound = llm.bind_tools(tools)
    result = bound.invoke([HumanMessage(content="weather in NY?")])
    assert result.content == "used tool"

    call_kwargs = llm._client.chat_completions.call_args.kwargs
    assert "tools" in call_kwargs
    assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# Error states
# ---------------------------------------------------------------------------


def test_sync_generate_propagates_error():
    from distllm_sdk.errors import AuthenticationError

    llm = _make_chat()
    llm._client.chat_completions = MagicMock(
        side_effect=AuthenticationError("bad key", request_id="r1")
    )
    with pytest.raises(AuthenticationError):
        llm.invoke([HumanMessage(content="hi")])


def test_stream_propagates_error():
    llm = _make_chat()
    llm._client.chat_completions_stream = MagicMock(side_effect=RuntimeError("stream down"))
    with pytest.raises(RuntimeError):
        list(llm.stream([HumanMessage(content="hi")]))


# ---------------------------------------------------------------------------
# Async paths
# ---------------------------------------------------------------------------


async def test_async_generate():
    llm = _make_chat()
    llm._async_client.chat_completions = AsyncMock(return_value=_openai_dict("async hi"))

    result = await llm.ainvoke([HumanMessage(content="hi")])
    assert isinstance(result, AIMessage)
    assert result.content == "async hi"


async def test_async_stream():
    async def _gen():
        yield {"choices": [{"delta": {"content": "a"}}]}
        yield {"choices": [{"delta": {"content": "b"}}]}

    llm = _make_chat()
    llm._async_client.chat_completions_stream = MagicMock(return_value=_gen())

    out = ""
    async for chunk in llm.astream([HumanMessage(content="hi")]):
        out += chunk.content
    assert out == "ab"
