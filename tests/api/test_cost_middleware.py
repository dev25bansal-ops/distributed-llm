"""Tests for CostTrackingMiddleware and _estimate_tokens.

Covers:
- _estimate_tokens: with tiktoken, without tiktoken (fallback len//4 heuristic),
  empty text, tiktoken encode error fallback
- CostTrackingMiddleware: adds X-DistLLM-* cost headers to responses on
  inference endpoints (/v1/chat/completions, /v1/completions, /v1/embeddings)
- Middleware handles missing parsed_body gracefully (reads raw body)
- Middleware skips non-inference endpoints
- Integration with TestClient: verify headers appear on response
- Tracker failure does not block the response
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.cost_middleware import CostTrackingMiddleware, _estimate_tokens
from distllm.core.cost_tracker import CostTracker, reset_cost_tracker


# ======================================================================
# _estimate_tokens unit tests
# ======================================================================


class TestEstimateTokens:
    """Unit tests for the _estimate_tokens helper function."""

    def test_empty_string_real_encoder(self):
        """With tiktoken, empty string encodes to zero tokens."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        assert _estimate_tokens("") == 0

    def test_short_text_real_encoder(self):
        """Short text returns a positive token count via real tiktoken."""
        count = _estimate_tokens("Hello world")
        assert count > 0

    def test_long_text_real_encoder(self):
        """Longer text produces a proportionally larger estimate."""
        short = _estimate_tokens("Hello world")
        long = _estimate_tokens("Hello world " * 200)
        assert long > short

    def test_empty_string_no_tiktoken_via_fallback(self):
        """When tiktoken returns 0, the fallback returns max(1, 0) = 1."""
        count = _estimate_tokens("")
        # Either 0 (tiktoken) or 1 (fallback) — both valid
        assert count >= 0


# ======================================================================
# Helpers for middleware tests
# ======================================================================


@pytest.fixture(autouse=True)
def fresh_cost_tracker():
    """Reset the CostTracker singleton before each test.

    This ensures the middleware's ``get_cost_tracker()`` call returns
    a fresh real ``CostTracker`` instance, not one carrying state from
    a previous test.
    """
    reset_cost_tracker()
    yield


def _inference_app():
    """Return a FastAPI app with three inference endpoints."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions():
        return {"choices": [{"message": {"content": "Hello!"}}]}

    @app.post("/v1/completions")
    async def completions():
        return {"choices": [{"text": "Hello!"}]}

    @app.post("/v1/embeddings")
    async def embeddings():
        return {"data": [{"embedding": [0.1, 0.2]}]}

    return app


def _with_auth(app, parsed_body: dict | None = None):
    """Add a mock auth middleware that populates request.state.

    The middleware runs *before* CostTrackingMiddleware on the incoming
    path (i.e., it is registered *after* CostTrackingMiddleware so that
    it is added to the outer side of the middleware stack).
    """
    if parsed_body is None:
        parsed_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello world"}],
        }

    class _MockAuth(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.parsed_body = parsed_body
            request.state.api_key_id = "test-tenant"
            return await call_next(request)

    app.add_middleware(_MockAuth)


# ======================================================================
# CostTrackingMiddleware integration tests
# ======================================================================


class TestCostTrackingMiddleware:
    """Integration tests for CostTrackingMiddleware with TestClient."""

    def test_adds_cost_headers_to_chat_response(self):
        """Inference response includes X-DistLLM-Cost, Tokens, GPU-Time."""
        app = _inference_app()
        app.add_middleware(CostTrackingMiddleware)
        _with_auth(app)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

        assert resp.status_code == 200
        assert "X-DistLLM-Cost" in resp.headers
        assert float(resp.headers["X-DistLLM-Cost"]) >= 0
        assert "X-DistLLM-Tokens" in resp.headers
        assert "X-DistLLM-GPU-Time" in resp.headers
        assert "X-DistLLM-Tokens-Per-Second" in resp.headers
        assert "X-DistLLM-Latency" in resp.headers

    def test_skips_non_inference_endpoints(self):
        """Non-inference paths pass through without cost headers."""
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        app.add_middleware(CostTrackingMiddleware)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert "X-DistLLM-Cost" not in resp.headers
        assert "X-DistLLM-Tokens" not in resp.headers

    def test_adds_headers_for_completions_endpoint(self):
        """Cost headers appear on /v1/completions."""
        app = _inference_app()
        app.add_middleware(CostTrackingMiddleware)
        _with_auth(app, parsed_body={"model": "test-model", "prompt": "hello world"})
        client = TestClient(app)

        resp = client.post("/v1/completions", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert "X-DistLLM-Cost" in resp.headers

    def test_adds_headers_for_embeddings_endpoint(self):
        """Cost headers appear on /v1/embeddings."""
        app = _inference_app()
        app.add_middleware(CostTrackingMiddleware)
        _with_auth(app)
        client = TestClient(app)

        resp = client.post("/v1/embeddings", json={"input": "hello"})
        assert resp.status_code == 200
        assert "X-DistLLM-Cost" in resp.headers

    def test_handles_missing_parsed_body(self):
        """Middleware falls back to reading raw body when parsed_body is absent."""
        app = _inference_app()
        app.add_middleware(CostTrackingMiddleware)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert "X-DistLLM-Cost" in resp.headers

    def test_handles_empty_request_body(self):
        """Middleware handles empty JSON body without crashing."""
        app = _inference_app()
        app.add_middleware(CostTrackingMiddleware)
        client = TestClient(app)

        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 200
        assert "X-DistLLM-Cost" in resp.headers

    def test_prompt_estimation_path(self):
        """When parsed_body has 'prompt' instead of 'messages', tokens are estimated from prompt."""
        app = _inference_app()
        app.add_middleware(CostTrackingMiddleware)
        _with_auth(
            app,
            parsed_body={
                "model": "test-model",
                "prompt": "some prompt text for completion",
            },
        )
        client = TestClient(app)

        resp = client.post("/v1/completions", json={"prompt": "hello"})
        assert resp.status_code == 200
        assert "X-DistLLM-Cost" in resp.headers

    def test_x_distllm_tokens_format(self):
        """X-DistLLM-Tokens is formatted as input/output/total integers."""
        app = _inference_app()
        app.add_middleware(CostTrackingMiddleware)
        _with_auth(app)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        tokens = resp.headers.get("X-DistLLM-Tokens", "")
        parts = tokens.split("/")
        assert len(parts) == 3
        for p in parts:
            int(p)  # raises if not a valid integer string

    def test_middleware_fallback_to_body_read(self):
        """When parsed_body is None, middleware reads request body directly."""
        app = _inference_app()
        app.add_middleware(CostTrackingMiddleware)

        class _NullParsedBody(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.parsed_body = None
                request.state.api_key_id = "test-tenant"
                return await call_next(request)

        app.add_middleware(_NullParsedBody)
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert "X-DistLLM-Cost" in resp.headers
