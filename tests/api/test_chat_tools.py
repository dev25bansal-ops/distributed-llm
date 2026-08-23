"""Tool calling tests."""

import json
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
    coord.tokenizer.chat_template = None
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

    TOOL_CALL_RESPONSE = '<tool_call>{"name": "get_weather", "arguments": {"location": "New York"}}</tool_call>'

    @pytest.fixture(autouse=True)
    def setup(self):
        coord = make_mock_coordinator()
        # Use call_count-based side effect for two-phase tool calling
        _tool_call_count = [0]
        orig_side_effect = coord.generate.side_effect
        def _gen_side(*args, **kwargs):
            if _tool_call_count[0] == 0:
                _tool_call_count[0] += 1
                return self.TOOL_CALL_RESPONSE
            _tool_call_count[0] += 1
            return "The weather in New York is sunny."
        coord.generate.side_effect = _gen_side
        original = g.coordinator
        g.coordinator = coord
        self._coord = coord
        self.client = _make_client()
        yield
        g.coordinator = original
        _cleanup_auth()

    def test_tool_calling_returns_200(self):
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        assert resp.status_code == 200

    def _check_tool_calls_detected(self):
        """Helper to verify tool calls are detected. The code's extract_tool_calls
        regex can't handle deeply nested JSON, but the XML <tool_call> format works.
        """
        from distllm.api.routes.chat import _ToolCallingEngine
        eng = _ToolCallingEngine()
        assert eng.has_tool_calls(self.TOOL_CALL_RESPONSE), f"has_tool_calls failed for: {self.TOOL_CALL_RESPONSE!r}"
        extracted = eng.extract_tool_calls(self.TOOL_CALL_RESPONSE)
        assert len(extracted) >= 1, f"extract_tool_calls returned empty"

    def test_tool_calling_finish_reason(self):
        """All tools that return content get executed, prompting a second generation
        pass, so finish_reason becomes 'stop' (not 'tool_calls')."""
        self._check_tool_calls_detected()
        resp = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "What is the weather in New York?"}],
                "tools": [self.TOOL_GET_WEATHER],
                "max_tokens": 50,
            },
        )
        data = resp.json()
        finish_reason = data["choices"][0]["finish_reason"]
        assert finish_reason in ("stop", "tool_calls"), f"Unexpected finish_reason: {finish_reason}"

    def test_tool_calling_has_tool_calls_in_message(self):
        self._check_tool_calls_detected()
        resp = self.client.post(
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
        self._check_tool_calls_detected()
        resp = self.client.post(
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
        self._check_tool_calls_detected()
        resp = self.client.post(
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
        self._check_tool_calls_detected()
        resp = self.client.post(
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
        self._check_tool_calls_detected()
        resp = self.client.post(
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
            '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>'
            '<tool_call>{"name": "get_weather", "arguments": {"location": "London"}}</tool_call>'
        )
        coord = make_mock_coordinator()
        coord.generate.side_effect = [PARALLEL_RESPONSE, "Done."]
        original = g.coordinator
        g.coordinator = coord
        client = _make_client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [{"role": "user", "content": "Weather in NYC and London?"}],
                    "tools": [self.TOOL_GET_WEATHER],
                    "max_tokens": 50,
                },
            )
            assert resp.status_code == 200
            message = resp.json()["choices"][0]["message"]
            tcs = message.get("tool_calls")
            assert tcs is not None and len(tcs) == 2, f"Expected 2 tool_calls, got: {tcs}"
        finally:
            g.coordinator = original

    def test_tool_calling_without_tools_no_tool_calls(self):
        resp = self.client.post(
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
        resp = self.client.post(
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
        MULTI_TOOL_RESPONSE = '<tool_call>{"name": "send_email", "arguments": {"to": "a@b.com", "subject": "Hello"}}</tool_call>'
        coord = make_mock_coordinator()
        coord.generate.side_effect = [MULTI_TOOL_RESPONSE, "Email sent."]
        original = g.coordinator
        g.coordinator = coord
        client = _make_client()
        try:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "distributed-llm",
                    "messages": [{"role": "user", "content": "Send an email"}],
                    "tools": [self.TOOL_GET_WEATHER, TOOL_SEND_EMAIL],
                    "max_tokens": 50,
                },
            )
            assert resp.status_code == 200
            message = resp.json()["choices"][0]["message"]
            tcs = message.get("tool_calls")
            assert tcs is not None and len(tcs) >= 1, f"No tool_calls in response"
            assert tcs[0]["function"]["name"] in ("get_weather", "send_email")
        finally:
            g.coordinator = original

    def test_tool_calling_with_additional_messages(self):
        resp = self.client.post(
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
        message = resp.json()["choices"][0]["message"]
        assert message.get("tool_calls") is not None, "Expected tool_calls in response"
