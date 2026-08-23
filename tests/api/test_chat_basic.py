"""Basic chat completion tests: single-turn, multi-turn, parameter validation, response format."""

import json as _json
import os
import secrets
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.core.api_key_store import reset_api_key_store
from distllm.api.server import app


# ---------------------------------------------------------------------------
# Shared helpers (duplicated so each file is self-contained)
# ---------------------------------------------------------------------------

def _make_client():
    """Create a TestClient with API key auth pre-configured."""
    test_api_key = secrets.token_urlsafe(32)
    os.environ.pop("API_KEY_WAS_SET", None)
    os.environ["API_KEY"] = test_api_key
    reset_api_key_store()
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {test_api_key}"
    return client


def _cleanup_auth():
    os.environ.pop("API_KEY", None)
    os.environ.pop("API_KEY_WAS_SET", None)
    reset_api_key_store()


def make_mock_coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None

    def encode_fn(text, **kwargs):
        tokens = [1, 2, 3, 4, 5]
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.side_effect = encode_fn
    def decode_side_effect(tokens, **kwargs):
        if isinstance(tokens, int):
            token_list = [tokens]
        elif isinstance(tokens, list):
            token_list = tokens
        else:
            token_list = tokens.tolist()
        return " ".join(f"tok-{t}" for t in token_list)
    coord.tokenizer.decode.side_effect = decode_side_effect
    coord.tokenizer.eos_token_id = 0
    coord.generate.return_value = "Hello! This is a test response."

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 5, 1000)
    mock_output.past_key_values = MagicMock()
    mock_model.return_value = mock_output
    coord.local_partitioner = MagicMock()
    coord.local_partitioner.full_model = mock_model
    coord.list_models.return_value = ["test-model"]
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    coord._model_router = None
    coord._shutting_down = False
    # Ensure TemplateEngine falls through to naive join fallback
    coord.tokenizer.chat_template = None
    return coord


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestChatBasic:
    """Basic non-streaming chat completion."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def test_single_user_message_returns_200(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.status_code == 200

    def test_response_has_choices(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) >= 1

    def test_response_object_type(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.json()["object"] == "chat.completion"

    def test_choice_has_message(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        choice = resp.json()["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]

    def test_response_has_id(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.json()["id"].startswith("chatcmpl-")

    def test_response_has_created_timestamp(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.json()["created"] > 0

    def test_response_model_matches_request(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.json()["model"] == "distributed-llm"

    def test_finish_reason_present(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.json()["choices"][0]["finish_reason"] is not None

    def test_system_message_accepted(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    def test_multi_turn_conversation(self):
        resp = self.client.post(
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

    def test_temperature_within_range(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.5,
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    def test_temperature_at_max(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 2.0,
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    def test_temperature_at_min(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.0,
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    def test_top_p_within_range(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 0.9,
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200

    def test_max_tokens_respected(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5},
        )
        assert resp.status_code == 200

    def test_min_max_tokens(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1},
        )
        assert resp.status_code == 200

    def test_without_coordinator_returns_503(self):
        # Use a distinct request body to avoid dedup cache collision
        original = g.coordinator
        g.coordinator = None
        client = _make_client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "distributed-llm", "messages": [{"role": "user", "content": "503 test - no coordinator"}], "max_tokens": 10},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestChatMultiTurn:
    """Multi-turn conversation tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self._coord = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def test_history_reflected_in_prompt(self):
        # Use unique message to avoid dedup cache hit
        messages = [
            {"role": "user", "content": "Hello from test_history"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What do you think?"},
        ]
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": messages,
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200
        prompt_arg = self._coord.generate.call_args[0][0]
        assert "Hello from test_history" in prompt_arg
        assert "Hi there!" in prompt_arg
        assert "What do you think?" in prompt_arg
        assert prompt_arg.index("Hello from test_history") < prompt_arg.index("Hi there!")
        assert prompt_arg.index("Hi there!") < prompt_arg.index("What do you think?")

    def test_system_prompt_prepended(self):
        # Use unique message to avoid dedup cache hit
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {"role": "system", "content": "Be very concise."},
                    {"role": "user", "content": "System prompt test"},
                ],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 200
        prompt_arg = self._coord.generate.call_args[0][0]
        assert "Be very concise." in prompt_arg
        assert prompt_arg.index("Be very concise.") < prompt_arg.index("System prompt test")


class TestChatResponseFormat:
    """Response format tests (JSON mode, JSON schema)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        coord = make_mock_coordinator()
        g.coordinator = coord
        self._coord = coord
        self.client = _make_client()
        yield
        g.coordinator = None
        _cleanup_auth()

    def test_adapter_returns_200(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Give me JSON"}],
                "response_format": {"type": "json_object"},
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200

    def test_json_object_accepted(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Give me JSON"}],
                "response_format": {"type": "json_object"},
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200

    def test_json_object_finish_reason(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Give me JSON"}],
                "response_format": {"type": "json_object"},
                "max_tokens": 50,
            },
        )
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_json_schema_returns_200(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Give me JSON"}],
                "response_format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200

    def test_json_schema_response_matches_schema(self):
        coord = make_mock_coordinator()
        coord.generate.return_value = '{"name": "Alice", "age": 30}'
        original = g.coordinator
        g.coordinator = coord
        client = _make_client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [{"role": "user", "content": "Give me JSON with name and age"}],
                    "response_format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                            "required": ["name", "age"],
                        },
                    },
                    "max_tokens": 50,
                },
            )
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = _json.loads(content)
            assert isinstance(parsed, dict)
            assert "name" in parsed
            assert "age" in parsed
        finally:
            g.coordinator = original

class TestChatPriority:
    """Priority scheduling via the priority parameter (0=critical, 1=high, 2=normal, 3=low)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def _req(self, priority=None):
        body = {"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}
        if priority is not None:
            body["priority"] = priority
        return self.client.post("/v1/chat/completions", json=body)

    def test_priority_critical(self):
        resp = self._req(priority=0)
        assert resp.status_code == 200

    def test_priority_high(self):
        resp = self._req(priority=1)
        assert resp.status_code == 200

    def test_priority_low(self):
        resp = self._req(priority=3)
        assert resp.status_code == 200

    def test_priority_defaults_to_normal(self):
        resp = self._req()
        assert resp.status_code == 200

    def test_priority_below_range_rejected(self):
        resp = self._req(priority=-1)
        assert resp.status_code == 422

    def test_priority_above_range_rejected(self):
        resp = self._req(priority=4)
        assert resp.status_code == 422


class TestChatMaxTokens:
    """max_tokens parameter bounds validation (1-8192)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def _req(self, max_tokens):
        return self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": max_tokens},
        )

    def test_zero_returns_immediately(self):
        resp = self._req(max_tokens=0)
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "length"
        assert resp.json()["choices"][0]["message"]["content"] == ""

    def test_negative_rejected(self):
        resp = self._req(max_tokens=-1)
        assert resp.status_code == 422

    def test_above_max_rejected(self):
        resp = self._req(max_tokens=9999)
        assert resp.status_code == 422

    def test_min_boundary(self):
        resp = self._req(max_tokens=1)
        assert resp.status_code == 200

    def test_max_boundary(self):
        resp = self._req(max_tokens=8192)
        assert resp.status_code == 200


class TestChatHybridRouting:
    """Hybrid model routing via coordinator._model_router."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        model_router = MagicMock()
        model_router.is_hybrid_model.side_effect = lambda m: m == "hybrid-model"
        model_router.resolve.return_value = "target-model"
        coord._model_router = model_router
        coord.list_models.return_value = ["test-model", "target-model"]
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def test_hybrid_model_routes_to_target(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "hybrid-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.status_code == 200

    def test_non_hybrid_model_uses_default(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.status_code == 200

    def test_unknown_model_without_router_rejected(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        client = _make_client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "unknown-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
            )
            assert resp.status_code == 400
        finally:
            g.coordinator = original


class TestChatTemperatureBounds:
    """temperature bounds validation (ge=0, le=2.0)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def _req(self, temperature):
        return self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "temperature": temperature, "max_tokens": 10},
        )

    def test_negative_rejected(self):
        resp = self._req(temperature=-0.1)
        assert resp.status_code == 422

    def test_above_max_rejected(self):
        resp = self._req(temperature=2.1)
        assert resp.status_code == 422

    def test_min_boundary(self):
        resp = self._req(temperature=0)
        assert resp.status_code == 200

    def test_max_boundary(self):
        resp = self._req(temperature=2.0)
        assert resp.status_code == 200


class TestChatTopPBounds:
    """top_p bounds validation (ge=0, le=1.0)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def _req(self, top_p):
        return self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "top_p": top_p, "max_tokens": 10},
        )

    def test_negative_rejected(self):
        resp = self._req(top_p=-0.5)
        assert resp.status_code == 422

    def test_above_max_rejected(self):
        resp = self._req(top_p=1.5)
        assert resp.status_code == 422

    def test_min_boundary(self):
        resp = self._req(top_p=0)
        assert resp.status_code == 200

    def test_max_boundary(self):
        resp = self._req(top_p=1.0)
        assert resp.status_code == 200


class TestChatStopSequences:
    """Stop sequences schema validation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def test_stop_list_accepted(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "stop": ["\n", "END"]},
        )
        assert resp.status_code == 200

    def test_single_stop_accepted(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "stop": ["END"]},
        )
        assert resp.status_code == 200

    def test_empty_stop_list_accepted(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "stop": []},
        )
        assert resp.status_code == 200

    def test_none_stop_accepted(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
        )
        assert resp.status_code == 200


class TestChatLogprobs:
    """Logprobs schema validation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def test_logprobs_true_accepted(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "logprobs": True},
        )
        assert resp.status_code == 200

    def test_logprobs_false_accepted(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "logprobs": False},
        )
        assert resp.status_code == 200

    def test_top_logprobs_without_logprobs(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "top_logprobs": 5},
        )
        assert resp.status_code == 200

    def test_top_logprobs_with_logprobs(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "logprobs": True, "top_logprobs": 5},
        )
        assert resp.status_code == 200

    def test_top_logprobs_above_max_rejected(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "top_logprobs": 21},
        )
        assert resp.status_code == 422

    def test_top_logprobs_negative_rejected(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "top_logprobs": -1},
        )
        assert resp.status_code == 422


class TestChatSeed:
    """Seed (deterministic mode) schema validation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def _req(self, overrides=None):
        body = {"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}
        if overrides:
            body.update(overrides)
        return self.client.post("/v1/chat/completions", json=body)

    def test_seed_accepted(self):
        resp = self._req({"seed": 42})
        assert resp.status_code == 200

    def test_seed_zero_accepted(self):
        resp = self._req({"seed": 0})
        assert resp.status_code == 200

    def test_seed_negative_accepted(self):
        resp = self._req({"seed": -1})
        assert resp.status_code == 200

    def test_seed_large_value_accepted(self):
        resp = self._req({"seed": 999999999})
        assert resp.status_code == 200

    def test_seed_optional(self):
        resp = self._req()
        assert resp.status_code == 200

    def test_seed_deterministic_same_output(self):
        body = {"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10, "seed": 42}
        resp1 = self.client.post("/v1/chat/completions", json=body)
        resp2 = self.client.post("/v1/chat/completions", json=body)
        assert resp1.json()["choices"][0]["message"]["content"] == resp2.json()["choices"][0]["message"]["content"]


class TestChatEmptyPrompt:
    """Empty or edge-case prompts."""

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def test_empty_string_content(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": ""}], "max_tokens": 10},
        )
        assert resp.status_code == 200

    def test_empty_messages_list_handled(self):
        # Empty messages are rejected at validation (min_length=1) —
        # matches OpenAI-compatible behavior and test_input_validation.
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [], "max_tokens": 10},
        )
        assert resp.status_code == 422

    def test_missing_messages_returns_422(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "max_tokens": 10},
        )
        assert resp.status_code == 422
