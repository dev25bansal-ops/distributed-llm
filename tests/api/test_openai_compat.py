"""Tests for Feature 29: OpenAI Compatibility Layer."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from fastapi.testclient import TestClient
from distllm.api.server import app


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    """Disable auth to prevent middleware ordering issues."""
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def client():
    """Create test client with mocked coordinator."""
    return TestClient(app)


@pytest.fixture
def mock_coordinator():
    """Create a fully mocked coordinator."""
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.metrics_exporter = None
    coord.scheduler = None
    coord.prefix_cache = None
    coord.tokenizer = MagicMock()
    # Make tokenizer.encode return a real tensor
    real_tensor = torch.tensor([[1, 2, 3]])
    coord.tokenizer.encode.return_value = real_tensor
    coord.tokenizer.decode.return_value = "test"  # Return string for streaming tokens
    coord.tokenizer.eos_token_id = 2
    coord.list_models.return_value = ["test-model", "distributed-llm"]
    coord.generate.return_value = "Hello! This is a test response."
    coord.generate_async.return_value = "test-request-id"
    coord.wait_for_result.return_value = "Hello! This is a test response."
    coord.get_metrics.return_value = {"test_metric": 42}
    coord._param_update_channel = MagicMock()
    coord._param_update_channel.register = MagicMock()
    coord._param_update_channel.unregister = MagicMock()
    coord._param_update_channel.update.return_value = None
    coord.adapter_manager = None

    # Prevent MagicMock from auto-creating attributes that trigger wrong code paths
    coord._vlm_pipeline = None
    coord._spec_decoder = None

    # Prevent BackpressureMiddleware from thinking service is shutting down
    # (MagicMock attributes are truthy by default)
    coord._shutting_down = False

    # Mock local partitioner for embeddings endpoint
    coord.local_partitioner = MagicMock()

    class MockModelOutputs:
        """Simple class to hold model outputs without MagicMock async behavior."""
        def __init__(self):
            self.last_hidden_state = torch.randn(1, 3, 768)
            self.hidden_states = (torch.randn(1, 3, 768),)
            self.logits = torch.randn(1, 3, 1000)  # [batch, seq_len, vocab]
            self.past_key_values = None

    class MockModel:
        """Simple mock model that returns outputs object."""
        def __call__(self, *args, **kwargs):
            return MockModelOutputs()

        def parameters(self):
            return iter([torch.randn(10, 10)])

    coord.local_partitioner.full_model = MockModel()

    return coord


# --- ChatMessage Extensions ---

class TestChatMessageExtensions:
    def test_chat_message_with_tool_calls(self):
        from distllm.api.server import ChatMessage

        msg = ChatMessage(
            role="assistant",
            content="I'll call the weather function.",
            tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "London"}'}
            }]
        )
        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["function"]["name"] == "get_weather"

    def test_chat_message_with_tool_call_id(self):
        from distllm.api.server import ChatMessage

        msg = ChatMessage(
            role="tool",
            content='{"temperature": 22}',
            tool_call_id="call_123"
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"

    def test_chat_message_with_function_call_deprecated(self):
        from distllm.api.server import ChatMessage

        msg = ChatMessage(
            role="assistant",
            content=None,
            function_call={"name": "get_weather", "arguments": '{"location": "London"}'}
        )
        assert msg.function_call["name"] == "get_weather"

    def test_chat_message_with_name(self):
        from distllm.api.server import ChatMessage

        msg = ChatMessage(role="user", content="Hello", name="alice")
        assert msg.name == "alice"

    def test_chat_message_content_optional(self):
        from distllm.api.server import ChatMessage

        msg = ChatMessage(role="tool", content=None, tool_call_id="call_1")
        assert msg.content is None


# --- ChatCompletionRequest Extensions ---

class TestChatCompletionRequestExtensions:
    def test_request_with_tools(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="What's the weather?")],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }],
            tool_choice="auto"
        )
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"

    def test_request_with_stop_sequences(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Write a story")],
            stop=["\n\n", "THE END"]
        )
        assert req.stop == ["\n\n", "THE END"]

    def test_request_with_penalties(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Be creative")],
            presence_penalty=0.5,
            frequency_penalty=0.3
        )
        assert req.presence_penalty == 0.5
        assert req.frequency_penalty == 0.3

    def test_request_with_seed(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Deterministic response")],
            seed=42
        )
        assert req.seed == 42

    def test_request_with_user(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            user="user-123"
        )
        assert req.user == "user-123"

    def test_request_with_logit_bias(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            logit_bias={"42": 1.5, "100": -2.0}
        )
        assert req.logit_bias == {"42": 1.5, "100": -2.0}

    def test_request_with_stream_options(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Stream me")],
            stream=True,
            stream_options={"include_usage": True}
        )
        assert req.stream_options["include_usage"] is True

    def test_request_with_logprobs(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            logprobs=True,
            top_logprobs=5
        )
        assert req.logprobs is True
        assert req.top_logprobs == 5

    def test_request_with_n(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            n=3
        )
        assert req.n == 3

    def test_request_with_deprecated_functions(self):
        from distllm.api.server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="What's the weather?")],
            functions=[{"name": "get_weather", "parameters": {}}],
            function_call="auto"
        )
        assert len(req.functions) == 1
        assert req.function_call == "auto"


# --- Embedding Models ---

class TestEmbeddingModels:
    def test_embedding_request_defaults(self):
        from distllm.api.server import EmbeddingRequest

        req = EmbeddingRequest(input=["Hello world"])
        assert req.model == "distributed-llm"
        assert req.encoding_format == "float"
        assert req.dimensions is None

    def test_embedding_request_with_dimensions(self):
        from distllm.api.server import EmbeddingRequest

        req = EmbeddingRequest(
            input=["Test"],
            dimensions=384
        )
        assert req.dimensions == 384

    def test_embedding_response_structure(self):
        from distllm.api.server import EmbeddingResponse, EmbeddingObject

        resp = EmbeddingResponse(
            model="test-model",
            data=[
                EmbeddingObject(index=0, embedding=[0.1, 0.2, 0.3]),
                EmbeddingObject(index=1, embedding=[0.4, 0.5, 0.6]),
            ],
            usage={"prompt_tokens": 10, "total_tokens": 10}
        )
        assert resp.object == "list"
        assert len(resp.data) == 2
        assert resp.data[0].embedding == [0.1, 0.2, 0.3]
        assert resp.usage["prompt_tokens"] == 10


# --- API Endpoint Tests ---

class TestEmbeddingsEndpoint:
    def test_embeddings_success(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/embeddings", json={
            "input": ["Hello world", "Test sentence"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert "usage" in data
        assert len(data["data"]) == 2

    def test_embeddings_no_coordinator(self, client):
        from distllm.api import server
        server.coordinator = None

        response = client.post("/v1/embeddings", json={"input": ["Hello"]})
        assert response.status_code == 503

    def test_embeddings_multiple_inputs(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/embeddings", json={
            "input": ["First", "Second", "Third"]
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        for i, item in enumerate(data["data"]):
            assert item["index"] == i


class TestChatCompletionsEndpoint:
    def test_chat_with_tools(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What's the weather in London?"}],
            "tools": [{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {}}
            }],
            "tool_choice": "auto"
        })
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_with_tool_response(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "tool", "content": '{"temp": 22}', "tool_call_id": "call_1"}
            ]
        })
        assert response.status_code == 200

    def test_chat_with_stop_sequences(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Write a short story"}],
            "stop": ["\n\n", "THE END"]
        })
        assert response.status_code == 200

    def test_chat_with_penalties(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Be creative"}],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3
        })
        assert response.status_code == 200

    def test_chat_with_seed(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "seed": 42
        })
        assert response.status_code == 200

    def test_chat_stream_with_usage(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True}
        })
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_chat_with_user_field(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "user": "user-abc-123"
        })
        assert response.status_code == 200


class TestModelsEndpoint:
    def test_list_models_extended(self, client, mock_coordinator):
        from distllm.api import server
        server.coordinator = mock_coordinator

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        # Check ModelInfo has OpenAI fields
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert "created" in model
        assert "owned_by" in model
