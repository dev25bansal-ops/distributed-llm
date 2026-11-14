"""Regression: oversized prompts must be rejected with a clean 400.

Defect: ``/v1/chat/completions`` never compared the prompt's token count
against the model's context window.  An oversized prompt reached the backend,
which failed there and surfaced as an opaque 500.  Separately, the distributed
generation path read ``c.config.max_context_length``, an attribute
``CoordinatorConfig`` does not define — so every distributed generation raised
``AttributeError`` at ``request_pipeline.py:231``.

Two contracts are pinned here:
  1. API: prompt_tokens + max_tokens over the window -> HTTP 400 with
     ``code == "context_length_exceeded"``, never a 500.
  2. Pipeline: reading the context window never raises on a real
     ``CoordinatorConfig``.
"""

from __future__ import annotations

import json

import pytest
import torch
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.server import app
from distllm.core.api_key_store import reset_api_key_store


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _Tok:
    """Tokenizer whose token count is whatever the test asks for."""

    eos_token_id = 0

    def __init__(self, tokens_per_char: float = 1.0, fixed_len: int | None = None):
        self._tpc = tokens_per_char
        self._fixed_len = fixed_len

    def encode(self, text, **kwargs):
        if self._fixed_len is not None:
            n = self._fixed_len
        else:
            n = max(1, int(len(text) * self._tpc))
        tokens = [1] * n
        if kwargs.get("return_tensors") == "pt":
            return torch.tensor([tokens])
        return tokens

    def decode(self, ids, **kwargs):
        return "ok"


class _Coord:
    """Coordinator exposing a small context window."""

    def __init__(self, window: int):
        self.model_name = "test-model"
        self.nodes = {}
        self.node_order = []
        self.local_partitioner = None
        self._shutting_down = False
        self.tokenizer = _Tok(fixed_len=100)
        self.model_info = {"max_position_embeddings": window}
        self._model_router = None
        self.scheduler = None

    def list_models(self):
        return ["test-model"]

    def generate(self, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9,
                 **kwargs):
        return "should not be reached"


# ---------------------------------------------------------------------------
# 1. API-level guard
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    key = "test-key-for-context-guard"
    monkeypatch.setenv("API_KEY", key)
    monkeypatch.delenv("API_KEY_WAS_SET", raising=False)
    reset_api_key_store()
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {key}"
    yield c
    reset_api_key_store()


def _chat_payload(max_tokens: int) -> dict:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "x" * 4000}],
        "max_tokens": max_tokens,
        "stream": False,
    }


def test_oversized_prompt_returns_400_not_500(client):
    """100 prompt tokens + 8000 max_tokens > 4096 window -> 400."""
    original = g.coordinator
    g.coordinator = _Coord(window=4096)
    try:
        resp = client.post("/v1/chat/completions", json=_chat_payload(8000))
    finally:
        g.coordinator = original

    assert resp.status_code == 400, (resp.status_code, resp.text)
    body = resp.json()
    err = body.get("error", body)
    # OpenAI-compatible error envelope.
    assert err.get("code") == "context_length_exceeded", err
    assert err.get("type") == "invalid_request_error", err
    message = err.get("message", "")
    assert "4096" in message, message  # names the limit
    assert "context" in message.lower(), message


def test_prompt_alone_exceeding_window_returns_400(client):
    """Even with tiny max_tokens, a prompt larger than the window is rejected."""
    original = g.coordinator
    g.coordinator = _Coord(window=10)  # window smaller than the 100-token prompt
    try:
        resp = client.post("/v1/chat/completions", json=_chat_payload(1))
    finally:
        g.coordinator = original

    assert resp.status_code == 400, (resp.status_code, resp.text)
    body = resp.json()
    err = body.get("error", body)
    assert err.get("code") == "context_length_exceeded", err


def test_request_that_fits_is_not_rejected(client):
    """A request inside the window proceeds past the guard (no 400)."""
    coord = _Coord(window=4096)
    original = g.coordinator
    g.coordinator = coord
    try:
        resp = client.post("/v1/chat/completions", json=_chat_payload(256))
    finally:
        g.coordinator = original

    # The bogus backend may 500; what matters is that the *guard* did not
    # reject a request that fits the window.
    assert resp.status_code != 400, resp.text
    if resp.status_code >= 400:
        detail = resp.text
        assert "context_length_exceeded" not in detail, detail


def test_streaming_request_also_guarded(client):
    """Streaming requests get the same guard, before the SSE stream starts."""
    original = g.coordinator
    g.coordinator = _Coord(window=4096)
    try:
        payload = _chat_payload(8000)
        payload["stream"] = True
        resp = client.post("/v1/chat/completions", json=payload)
    finally:
        g.coordinator = original

    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert resp.json()["error"]["code"] == "context_length_exceeded"


# ---------------------------------------------------------------------------
# 2. Pipeline: reading the window must not blow up on a real config
# ---------------------------------------------------------------------------


def test_coordinator_config_exposes_context_window():
    """The pipeline reads ``config.max_context_length``; the field must exist."""
    from distllm.core.coordinator_config import CoordinatorConfig

    cfg = CoordinatorConfig(model_name="x")
    assert hasattr(cfg, "max_context_length"), (
        "request_pipeline.py requires max_context_length on CoordinatorConfig"
    )
    assert isinstance(cfg.max_context_length, int)
    assert cfg.max_context_length > 0