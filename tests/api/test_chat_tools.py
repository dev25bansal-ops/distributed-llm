"""Tool calling tests."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


# ---------------------------------------------------------------------------
# Shared helpers (duplicated so each file is self-contained)
# ---------------------------------------------------------------------------

def disable_auth():
    os.environ.pop("API_KEY", None)
    os.environ.pop("API_KEY_WAS_SET", None)
    os.environ["DISABLE_AUTH"] = "1"
    os.environ["DISTLLM_DEV_MODE"] = "1"


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
    coord._shutting_down = False
    return coord


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestChatToolCalling:
    """Tool calling tests."""

    TOOL_GET_WEATHER = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }

    TOOL_CALL_RESPONSE = '{"name": "get_weather", "arguments": {"location": "New York"}}'

    @pytest.fixture(autouse=True)
    def setup(self):
        disable_auth()
        coord = make_mock_coordinator()
        coord.generate.side_effect = [
            self.TOOL_CALL_RESPONSE,
            "The weather in New York is sunny.",
        ]
        original = g.coordinator
        g.coordinator = coord
        self._coord = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_tool_calling_returns_200(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200

    def test_tool_calling_finish_reason(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"

    def test_tool_calling_has_tool_calls_in_message(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        message = resp.json()["choices"][0]["message"]
        assert "tool_calls" in message
        assert len(message["tool_calls"]) >= 1

    def test_tool_calls_have_function_name(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        tc = resp.json()["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"

    def test_tool_calls_have_arguments(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        tc = resp.json()["choices"][0]["message"]["tool_calls"][0]
        args = tc["function"]["arguments"]
        assert '"location"' in args

    def test_tool_calls_content_is_null(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        message = resp.json()["choices"][0]["message"]
        assert message["content"] is None

    def test_tool_calling_two_generations(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200
        assert self._coord.generate.call_count == 2

    def test_parallel_tool_calls(self):
        PARALLEL_RESPONSE = (
            '[{"name": "get_weather", "arguments": {"location": "NYC"}},'
            '{"name": "get_weather", "arguments": {"location": "London"}}]'
        )
        coord = make_mock_coordinator()
        coord.generate.side_effect = [PARALLEL_RESPONSE, "Done."]
        original = g.coordinator
        g.coordinator = coord
        try:
            resp = TestClient(app).post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [{"role": "user", "content": "Weather in NYC and London?"}],
                    "tools": [self.TOOL_GET_WEATHER],
                    "max_tokens": 50,
                },
            )
            assert resp.status_code == 200
            tcs = resp.json()["choices"][0]["message"]["tool_calls"]
            assert len(tcs) == 2
            assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
        finally:
            g.coordinator = original

    def test_tool_calling_without_tools_no_tool_calls(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            },
        )
        message = resp.json()["choices"][0]["message"]
        assert message.get("tool_calls") is None
        assert resp.json()["choices"][0]["finish_reason"] == "stop"

    def test_tool_choice_none_skips_tools(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "tool_choice": "none",
                "max_tokens": 50,
            },
        )
        assert resp.json()["choices"][0]["finish_reason"] == "stop"
        message = resp.json()["choices"][0]["message"]
        assert message.get("tool_calls") is None
        assert message["content"] is not None

    def test_multiple_tools_choice(self):
        TOOL_SEND_EMAIL = {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                    },
                    "required": ["to", "subject"],
                },
            },
        }
        MULTI_TOOL_RESPONSE = '{"name": "send_email", "arguments": {"to": "a@b.com", "subject": "Hello"}}'
        coord = make_mock_coordinator()
        coord.generate.side_effect = [MULTI_TOOL_RESPONSE, "Email sent."]
        original = g.coordinator
        g.coordinator = coord
        try:
            resp = TestClient(app).post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [{"role": "user", "content": "Send an email"}],
                    "tools": [self.TOOL_GET_WEATHER, TOOL_SEND_EMAIL],
                    "max_tokens": 50,
                },
            )
            assert resp.status_code == 200
            tc = resp.json()["choices"][0]["message"]["tool_calls"][0]
            assert tc["function"]["name"] in ("get_weather", "send_email")
            assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
        finally:
            g.coordinator = original

    def test_tool_calling_with_additional_messages(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [
                    {"role": "system", "content": "Always use tools."},
                    {"role": "user", "content": "What is the weather in New York?"},
                ],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
