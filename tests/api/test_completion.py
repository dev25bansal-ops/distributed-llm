"""Text completion tests: POST /v1/completions."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app


def make_mock_coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None
    coord.tokenizer = MagicMock()

    def encode_fn(text, **kwargs):
        tokens = list(range(1, len(text.split()) + 1))
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens
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
    coord.generate.return_value = "This is a completed text response."

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 5, 1000)
    mock_output.past_key_values = MagicMock()
    mock_model.return_value = mock_output
    coord.local_partitioner = MagicMock()
    coord.local_partitioner.full_model = mock_model
    coord._shutting_down = False
    return coord


class TestTextCompletionBasic:
    """Basic text completion via /v1/completions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_text_completion_returns_200(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Once upon a time", "max_tokens": 50},
        )
        assert resp.status_code == 200

    def test_response_has_choices(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Once upon a time", "max_tokens": 50},
        )
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) >= 1

    def test_choice_has_text(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Once upon a time", "max_tokens": 50},
        )
        data = resp.json()
        assert "text" in data["choices"][0]

    def test_response_object_type(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Once upon a time", "max_tokens": 50},
        )
        data = resp.json()
        assert data["object"] == "text_completion"

    def test_response_has_id(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Once upon a time", "max_tokens": 50},
        )
        data = resp.json()
        assert "id" in data
        assert data["id"].startswith("cmpl-")

    def test_response_has_created_timestamp(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Once upon a time", "max_tokens": 50},
        )
        data = resp.json()
        assert "created" in data
        assert isinstance(data["created"], int)

    def test_response_model_matches_request(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "Once upon a time", "max_tokens": 50},
        )
        data = resp.json()
        assert data["model"] == "test-model"

    def test_finish_reason_present(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Once upon a time", "max_tokens": 50},
        )
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_generation_time_present(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Once upon a time", "max_tokens": 50},
        )
        data = resp.json()
        assert "generation_time" in data
        assert isinstance(data["generation_time"], float)

    def test_temperature_within_range(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 10, "temperature": 0.5},
        )
        assert resp.status_code == 200

    def test_without_coordinator_returns_503(self):
        original = g.coordinator
        g.coordinator = None
        try:
            resp = TestClient(app).post(
                "/v1/completions",
                json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 10},
            )
            assert resp.status_code == 503
        finally:
            g.coordinator = original


class TestTextCompletionPriority:
    """Priority scheduling for text completions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_priority_critical(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 10, "priority": 0},
        )
        assert resp.status_code == 200

    def test_priority_above_range_rejected(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 10, "priority": 4},
        )
        assert resp.status_code == 422


class TestTextCompletionMaxTokens:
    """max_tokens validation for text completions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_zero_returns_immediately(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 0},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "length"
        assert resp.json()["choices"][0]["text"] == ""

    def test_negative_rejected(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": -1},
        )
        assert resp.status_code == 422

    def test_above_max_rejected(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 9999},
        )
        assert resp.status_code == 422

    def test_min_boundary(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 1},
        )
        assert resp.status_code == 200

    def test_max_boundary(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 8192},
        )
        assert resp.status_code == 200


class TestTextCompletionStreaming:
    """Streaming text completion (stream=true → SSE)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_streaming_returns_200(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 5, "stream": True},
        )
        assert resp.status_code == 200

    def test_streaming_content_type(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 5, "stream": True},
        )
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_has_data_events(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 5, "stream": True},
        )
        assert "data:" in resp.text

    def test_streaming_has_done_signal(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 5, "stream": True},
        )
        assert "[DONE]" in resp.text

    def test_streaming_has_text_delta(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 5, "stream": True},
        )
        assert '"text"' in resp.text

    def test_streaming_finish_reason(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 5, "stream": True},
        )
        assert "finish_reason" in resp.text


class TestTextCompletionResponseFormat:
    """Response format for text completions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_json_object_accepted(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Give me JSON", "max_tokens": 50, "response_format": {"type": "json_object"}},
        )
        assert resp.status_code == 200

    def test_json_schema_accepted(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={
                "model": "distributed-llm",
                "prompt": "Give me JSON",
                "max_tokens": 50,
                "response_format": {"type": "json_schema", "schema": {"type": "object", "properties": {"name": {"type": "string"}}}},
            },
        )
        assert resp.status_code == 200

    def test_empty_response_format_ignored(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "Hello", "max_tokens": 10, "response_format": {}},
        )
        assert resp.status_code == 200


class TestTextCompletionEmptyPrompt:
    """Empty prompt edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ.pop("API_KEY", None)
        os.environ.pop("API_KEY_WAS_SET", None)
        os.environ["DISABLE_AUTH"] = "1"
        os.environ["DISTLLM_DEV_MODE"] = "1"
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_empty_prompt_accepted(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "prompt": "", "max_tokens": 10},
        )
        assert resp.status_code == 200

    def test_missing_prompt_returns_422(self):
        resp = TestClient(app).post(
            "/v1/completions",
            json={"model": "distributed-llm", "max_tokens": 10},
        )
        assert resp.status_code == 422
