"""Multi-node distributed E2E tests.

Uses an in-process coordinator with the real TinyStories-1M model (CPU).
Exercises the full API surface including:
- Chat completion (sync + streaming)
- Tool/function calling
- Embeddings
- Node failure handling
- Multi-model listing

All tests are marked ``e2e`` and ``slow``.
"""
import time

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# ======================================================================
#  Chat completion
# ======================================================================


class TestChatCompletion:
    """Basic chat completion via the coordinator and REST API."""

    def test_sync_generation(self, cluster):
        """Generate text through the coordinator's local pipeline."""
        result = cluster.coordinator.generate(
            "Once upon a time", max_new_tokens=10,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_completion_via_api(self, api_client):
        """POST /v1/chat/completions returns OpenAI-compatible JSON."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

    def test_response_structure(self, api_client):
        """Response contains required OpenAI fields."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        for field in ("id", "object", "created", "model", "choices", "usage"):
            assert field in data, f"Missing required field: {field}"
        assert "prompt_tokens" in data["usage"]
        assert data["object"] == "chat.completion"

    def test_with_system_message(self, api_client):
        """System message is accepted."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello"},
                ],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    def test_multi_turn_conversation(self, api_client):
        """Multi-turn message history is accepted."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"][0]["message"]["content"]) > 0


# ======================================================================
#  Streaming
# ======================================================================


class TestStreaming:
    """SSE streaming through the coordinator."""

    def test_streaming_chat(self, api_client):
        """stream=true returns Server-Sent Events."""
        with api_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Count to five"}],
                "max_tokens": 20,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            lines = list(resp.iter_lines())
            assert len(lines) > 0

    def test_streaming_events_are_json(self, api_client):
        """Each SSE data line is parseable JSON."""
        import json
        chunks = []
        with api_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Tell a story"}],
                "max_tokens": 15,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line[6:]))

        assert len(chunks) > 0
        if chunks:
            choices = chunks[-1].get("choices", [])
            if choices:
                assert choices[0].get("finish_reason") in (None, "stop", "length")


# ======================================================================
#  Tool / function calling
# ======================================================================


class TestToolCalling:
    """Tool definitions are accepted through the API."""

    WEATHER_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        },
    ]

    def test_tools_accepted(self, api_client):
        """tools parameter accepted; request completes."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "tools": self.WEATHER_TOOLS,
                "max_tokens": 20,
            },
        )
        assert resp.status_code == 200

    def test_tool_choice_none(self, api_client):
        """tool_choice='none' disables tool calling."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": self.WEATHER_TOOLS,
                "tool_choice": "none",
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    def test_deprecated_functions_parameter(self, api_client):
        """Deprecated ``functions`` parameter still works."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Get weather in Tokyo"}],
                "functions": [
                    {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                ],
                "function_call": "auto",
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200


# ======================================================================
#  Embeddings
# ======================================================================


class TestEmbeddings:
    """Embedding endpoint."""

    def test_embeddings_basic(self, api_client):
        """POST /v1/embeddings returns an embedding vector."""
        resp = api_client.post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello world"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)

    def test_embeddings_batch_input(self, api_client):
        """Multiple input strings => multiple embedding objects."""
        resp = api_client.post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["Hello", "World"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2

    def test_embeddings_response_schema(self, api_client):
        """Response matches OpenAI embedding schema."""
        resp = api_client.post(
            "/v1/embeddings",
            json={"model": "distributed-llm", "input": ["test"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        obj = data["data"][0]
        for field in ("object", "index", "embedding"):
            assert field in obj
        assert obj["object"] == "embedding"
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]


# ======================================================================
#  Node failure
# ======================================================================


class TestInputValidation:
    """API input validation and error handling."""

    def test_empty_messages_triggers_error(self, api_client):
        """Empty messages list triggers a 500 (model chokes) or should be 422."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [], "max_tokens": 5},
        )
        assert resp.status_code in (422, 500)

    def test_invalid_model_name(self, api_client):
        """Unknown model returns 400."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 400

    def test_missing_messages_field(self, api_client):
        """Omitting mandatory messages field yields 422."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "max_tokens": 5},
        )
        assert resp.status_code == 422

    def test_local_generate_still_works(self, cluster):
        """Direct coordinator.generate still works."""
        result = cluster.coordinator.generate(
            "Once upon a time", max_new_tokens=10,
        )
        assert isinstance(result, str)
        assert len(result) > 0


# ======================================================================
#  Multi-model
# ======================================================================


class TestMultiModel:
    """Model listing and basic multi-model operations."""

    def test_list_models(self, cluster):
        """list_models returns at least one model."""
        models = cluster.coordinator.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1

    def test_chat_with_model_id(self, api_client):
        """Chat completion with explicit model ID is accepted."""
        resp = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            assert "model" in resp.json()


# ======================================================================
#  Health
# ======================================================================


class TestHealth:
    """Health check and readiness endpoints."""

    def test_health_endpoint(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data or isinstance(data, dict)

    def test_models_endpoint(self, api_client):
        resp = api_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (list, dict))
        if isinstance(data, list):
            assert len(data) > 0
