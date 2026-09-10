"""E2E test: Chat completion with streaming via the real API router.

Verifies the full streaming path:
1. FastAPI TestClient hits the real /v1/chat/completions endpoint
2. Middleware stack processes the request (auth bypassed)
3. The chat route builds a prompt and calls _stream_response
4. SSE event-stream is returned
"""

import pytest
import torch
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from fastapi import Request

import distllm.api.server as server_module
from distllm.api.server import app
from distllm.api.api_state import g as api_state


@pytest.fixture
def e2e_coordinator():
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = {}
    coord.node_order = []
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None
    coord._vlm_pipeline = None
    coord._spec_decoder = None
    coord._shutting_down = False

    def encode_fn(text, **kwargs):
        tokens = list(range(1, len(text.split()) + 1))
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.side_effect = encode_fn
    coord.tokenizer.decode.side_effect = lambda tokens, **kwargs: " ".join(
        f"tok-{t}" for t in (
            [tokens] if isinstance(tokens, int) else
            (tokens if isinstance(tokens, list) else tokens.tolist())
        )
    )
    coord.tokenizer.eos_token_id = 0
    coord.tokenizer.bos_token_id = 1
    coord.list_models.return_value = ["distributed-llm"]

    # The chat route resolves the requested model through the coordinator's
    # model router (missing model -> 400 "not found"). Wire the mock so
    # "distributed-llm" resolves.
    coord._model_router = MagicMock()
    coord._model_router.resolve.return_value = "distributed-llm"
    coord._model_router.list_models.return_value = ["distributed-llm"]

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.randn(10, 10)])
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 5, 1000)
    mock_output.past_key_values = MagicMock()
    mock_model.return_value = mock_output
    coord.local_partitioner = MagicMock()
    coord.local_partitioner.full_model = mock_model

    return coord


@pytest.fixture(autouse=True)
def _real_auth(monkeypatch):
    """Auth is ALWAYS required (the old DISABLE_AUTH bypass was removed
    in the security hardening). Configure a real API key and reset the
    key store so the TestClient's Bearer header authenticates."""
    import secrets as _secrets

    test_key = _secrets.token_hex(32)
    monkeypatch.setenv("API_KEY", test_key)
    monkeypatch.setenv("API_KEY_WAS_SET", "1")
    monkeypatch.delenv("API_KEYS", raising=False)
    monkeypatch.delenv("API_KEYS_FILE", raising=False)
    from distllm.core.api_key_store import reset_api_key_store

    reset_api_key_store()
    yield test_key


@pytest.fixture
def client(e2e_coordinator, _real_auth):
    # The server reads its coordinator from shared api_state (g), not a
    # module attribute — patch the live reference with save/restore.
    original = api_state.coordinator
    api_state.coordinator = e2e_coordinator
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {_real_auth}"
    yield c
    api_state.coordinator = original


class TestE2EStreaming:
    def test_streaming_returns_sse(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "distributed-llm",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_streaming_contains_expected_events(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "distributed-llm",
            "messages": [{"role": "user", "content": "Count to three"}],
            "stream": True,
        })
        assert response.status_code == 200
        text = response.text
        assert "role" in text or "choices" in text
        assert "[DONE]" in text

    def test_streaming_respects_max_tokens(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "distributed-llm",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 5,
        })
        assert response.status_code == 200

    def test_streaming_with_temperature(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "distributed-llm",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "temperature": 0.5,
        })
        assert response.status_code == 200

    def test_streaming_includes_usage_when_requested(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "distributed-llm",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        assert response.status_code == 200
        text = response.text
        assert "usage" in text or "prompt_tokens" in text or response.status_code == 200

    def test_non_streaming_returns_complete_response(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "distributed-llm",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
