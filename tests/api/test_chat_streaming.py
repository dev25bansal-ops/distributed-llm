"""SSE streaming chat completion tests."""

import json
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

class TestChatStreaming:
    """Streaming chat completion (stream=true -> SSE)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        disable_auth()
        coord = make_mock_coordinator()
        original = g.coordinator
        g.coordinator = coord
        yield
        g.coordinator = original
        os.environ.pop("DISABLE_AUTH", None)
        os.environ.pop("DISTLLM_DEV_MODE", None)

    def test_streaming_returns_200(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        assert resp.status_code == 200

    def test_streaming_content_type(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_has_data_events(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        assert "data:" in resp.text

    def test_streaming_has_done_signal(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        assert "[DONE]" in resp.text

    def test_streaming_has_role_event(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        assert '"role": "assistant"' in resp.text

    def test_streaming_has_content_events(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        assert '"content"' in resp.text

    def test_streaming_has_finish_reason(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        assert "finish_reason" in resp.text

    def test_streaming_response_id_consistent(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        lines = [l for l in resp.text.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
        ids = set()
        for line in lines:
            ids.add(json.loads(line[6:])["id"])
        assert len(ids) == 1

    def test_streaming_has_multiple_events(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        data_lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
        assert len(data_lines) >= 2

    def test_streaming_usage_when_requested(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "model": "distributed-llm",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status_code == 200
        assert "usage" in resp.text

    def test_streaming_usage_not_included_by_default(self):
        resp = TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "distributed-llm", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5, "stream": True},
        )
        data_lines = [l for l in resp.text.split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
        has_usage = any('"usage"' in l for l in data_lines)
        assert not has_usage
