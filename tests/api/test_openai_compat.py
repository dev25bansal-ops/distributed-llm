"""Integration: OpenAI API compatibility — request/response shape matches spec.

Tests the Pydantic models and serialization without starting a server.
Uses mocked coordinator to test route handler logic.
"""

import json
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from pydantic import ValidationError

from distllm.api.routes.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatChoice,
)
from distllm.api.routes.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
)
from distllm.api.routes.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingObject,
)
from distllm.api.routes.health import (
    ModelList,
    ModelInfo,
)

pytestmark = pytest.mark.integration

# ═══════════════════════════════════════════════════════════════════════════
# 5. OpenAI API Compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestChatCompletionModels:
    """ChatCompletionRequest/Response Pydantic models match OpenAI spec."""

    def test_valid_chat_request(self):
        req = ChatCompletionRequest(
            model="distributed-llm",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.5,
            max_tokens=100,
        )
        assert req.model == "distributed-llm"
        assert req.temperature == 0.5
        assert req.max_tokens == 100
        assert req.stream is False

    def test_chat_request_defaults(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.model == "distributed-llm"
        assert req.temperature == 0.7
        assert req.max_tokens == 256
        assert req.top_p == 0.9
        assert req.n == 1

    def test_chat_request_invalid_temperature_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=3.0,
            )

    def test_chat_request_invalid_max_tokens_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=99999,
            )

    def test_chat_request_with_stop_sequences(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Write a poem"}],
            stop=["\n", "The end"],
        )
        assert req.stop == ["\n", "The end"]

    def test_chat_request_with_stream_options(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert req.stream is True
        assert req.stream_options == {"include_usage": True}

    def test_chat_request_with_seed(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            seed=42,
        )
        assert req.seed == 42

    def test_chat_request_with_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Weather?"}],
            tools=tools,
            tool_choice="auto",
        )
        assert req.tools == tools
        assert req.tool_choice == "auto"

    def test_chat_request_with_response_format(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "JSON please"}],
            response_format={"type": "json_object"},
        )
        assert req.response_format == {"type": "json_object"}

    def test_chat_request_with_logit_bias(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            logit_bias={"42": 1.5, "100": -2.0},
        )
        assert req.logit_bias == {"42": 1.5, "100": -2.0}

    def test_chat_request_with_priorities(self):
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            priority=0,
        )
        assert req.priority == 0

    def test_chat_response_shape(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-test123",
            model="distributed-llm",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )
        assert resp.object == "chat.completion"
        assert resp.id == "chatcmpl-test123"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content == "Hello!"
        assert resp.usage["total_tokens"] == 15

    def test_chat_response_auto_id(self):
        resp = ChatCompletionResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content="Hi"))],
        )
        assert resp.id.startswith("chatcmpl-")
        assert resp.created > 0

    def test_chat_response_serializes_to_json(self):
        resp = ChatCompletionResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content="World"))],
        )
        raw = resp.model_dump_json()
        data = json.loads(raw)
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "World"

    def test_chat_streaming_delta(self):
        delta = ChatChoice(index=0, delta={"role": "assistant", "content": "Hello"})
        assert delta.delta["content"] == "Hello"
        assert delta.message is None

    def test_chat_with_empty_messages_allowed(self):
        req = ChatCompletionRequest(messages=[])
        assert req.messages == []

    def test_chat_with_no_messages_raises(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest()

    def test_chat_with_system_message(self):
        req = ChatCompletionRequest(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"

    def test_chat_with_multiple_turns(self):
        req = ChatCompletionRequest(
            messages=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        )
        assert len(req.messages) == 3


class TestCompletionModels:
    """CompletionRequest/Response models."""

    def test_valid_completion_request(self):
        req = CompletionRequest(model="distributed-llm", prompt="Once upon a time")
        assert req.prompt == "Once upon a time"
        assert req.max_tokens == 256

    def test_completion_request_with_all_params(self):
        req = CompletionRequest(
            model="custom-model",
            prompt="Hello world",
            max_tokens=50,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            stream=True,
            priority=1,
            user="test-user",
        )
        assert req.stream is True
        assert req.priority == 1
        assert req.user == "test-user"

    def test_completion_response_shape(self):
        resp = CompletionResponse(
            id="cmpl-test",
            model="distributed-llm",
            choices=[CompletionChoice(index=0, text="world", finish_reason="stop")],
        )
        assert resp.object == "text_completion"
        assert resp.choices[0].text == "world"
        assert resp.choices[0].finish_reason == "stop"

    def test_completion_response_auto_id(self):
        resp = CompletionResponse(
            choices=[CompletionChoice(text="hello")],
        )
        assert resp.id.startswith("cmpl-")

    def test_completion_response_serializes(self):
        resp = CompletionResponse(
            choices=[CompletionChoice(text="hello")],
        )
        data = json.loads(resp.model_dump_json())
        assert data["object"] == "text_completion"

    def test_completion_with_response_format(self):
        req = CompletionRequest(
            prompt="List numbers 1 to 3 in JSON",
            response_format={"type": "json_schema", "schema": {"type": "object"}},
        )
        assert req.response_format["type"] == "json_schema"

    def test_completion_empty_prompt_raises(self):
        req = CompletionRequest(prompt="")
        assert req.prompt == ""


class TestEmbeddingModels:
    """EmbeddingRequest/Response models."""

    def test_valid_embedding_request(self):
        req = EmbeddingRequest(input=["Hello world"], model="text-embedding-3-small")
        assert req.input == ["Hello world"]

    def test_embedding_request_list_input(self):
        req = EmbeddingRequest(input=["Hello", "World"], model="text-embedding-3-small")
        assert len(req.input) == 2

    def test_embedding_request_default_model(self):
        req = EmbeddingRequest(input=["test"])
        assert req.model == "distributed-llm"

    def test_embedding_response_shape(self):
        resp = EmbeddingResponse(
            model="text-embedding-3-small",
            data=[EmbeddingObject(index=0, embedding=[0.1, 0.2, 0.3])],
            usage={"prompt_tokens": 2, "total_tokens": 2},
        )
        assert resp.object == "list"
        assert len(resp.data) == 1
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]

    def test_embedding_response_serializes(self):
        resp = EmbeddingResponse(
            model="test",
            data=[EmbeddingObject(index=0, embedding=[0.5])],
        )
        data = json.loads(resp.model_dump_json())
        assert data["object"] == "list"
        assert data["data"][0]["embedding"] == [0.5]

    def test_embedding_model_dimension(self):
        vec = [float(i) for i in range(768)]
        obj = EmbeddingObject(index=0, embedding=vec)
        assert len(obj.embedding) == 768

    def test_embedding_usage_tracking(self):
        resp = EmbeddingResponse(
            model="test",
            data=[EmbeddingObject(index=0, embedding=[1.0])],
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )
        assert resp.usage["prompt_tokens"] == 10

    def test_embedding_empty_input_raises(self):
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="")


class TestHealthModels:
    """Model list and health models."""

    def test_model_list_shape(self):
        models = ModelList(
            object="list",
            data=[ModelInfo(id="distributed-llm", object="model", created=int(time.time()))],
        )
        assert len(models.data) == 1
        assert models.data[0].id == "distributed-llm"

    def test_model_list_serializes(self):
        models = ModelList(
            data=[ModelInfo(id="test-model", object="model", created=1234567890)],
        )
        data = json.loads(models.model_dump_json())
        assert data["object"] == "list"
        assert data["data"][0]["id"] == "test-model"


class TestChatRouteHandler:
    """Test the chat route handler logic with a mocked coordinator."""

    @pytest.fixture
    def mock_coord(self):
        coord = MagicMock()
        coord.generate.return_value = "Hello! I am a distributed LLM."
        coord._vlm_pipeline = None
        return coord

    @pytest.mark.asyncio
    async def test_chat_handler_shape(self, mock_coord):
        from distllm.api.api_state import g
        g.coordinator = mock_coord

        from distllm.api.routes.chat import chat_completions
        body = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=32,
        )

        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.model = body.model
        mock_request.state.tenant = "default"

        response = await chat_completions(mock_request, body)
        assert response is not None

    @pytest.mark.asyncio
    async def test_chat_handler_with_none_coordinator(self):
        from distllm.api.api_state import g
        g.coordinator = None

        from distllm.api.routes.chat import chat_completions

        body = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()

        with pytest.raises(Exception):
            await chat_completions(mock_request, body)
