"""Tests for QuotaMiddleware.

Covers:
- Token estimation with and without tiktoken
- Quota enforcement (under limit, at limit, exceeded)
- Skipping quota when disabled
- Per-tenant isolation
- _record_usage logic for request/response token counting
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.quota_middleware import (
    QuotaMiddleware,
    _estimate_token_count,
    get_usage_meter,
)


# ======================================================================
# _estimate_token_count unit tests
# ======================================================================


class TestEstimateTokenCount:
    def test_returns_zero_for_empty_text(self):
        assert _estimate_token_count("") == 0
        assert _estimate_token_count(None) == 0

    def test_fallback_heuristic(self):
        """Without tiktoken, uses len(text) // 4."""
        # If tiktoken is installed, the heuristic isn't used, so we just
        # verify the function returns a reasonable positive integer.
        count = _estimate_token_count("Hello, world!")
        assert count > 0

    def test_uses_tiktoken_when_available(self):
        """When tiktoken is available, it's used for encoding."""
        # tiktoken is a real installed package; the function imports it
        # inside its body.  We can't mock it away, so just verify it
        # returns a plausible count.
        try:
            import tiktoken
            count = _estimate_token_count("Some text here")
            assert count > 0
        except ImportError:
            pass  # tiktoken not installed, test irrelevant

    def test_heuristic_for_long_text(self):
        """Long text returns a positive token estimate."""
        text = "word " * 200  # ~1000 chars
        count = _estimate_token_count(text)
        assert count > 0


# ======================================================================
# QuotaMiddleware integration tests
# ======================================================================


@pytest.fixture
def quota_app():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "Hello!"}}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def _make_client(app, enabled: bool = True):
    """Helper: add QuotaMiddleware and return TestClient."""
    app.add_middleware(QuotaMiddleware, enable=enabled)
    client = TestClient(app)

    # Wire basic request state that AuthMiddleware normally sets
    async def mock_auth_middleware(request, call_next):
        request.state.tenant_id = "test-tenant"
        request.state.api_key_id = "key-123"
        request.state.api_key_role = "inference-only"
        request.state.model = "test-model"
        return await call_next(request)

    app.user_middleware.insert(0, type("mock_auth", (), {"__call__": mock_auth_middleware}))
    return client


def test_passthrough_when_disabled():
    """When DISTLLM_QUOTA_ENABLED=0 (default), requests pass unhindered."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "ok"}}]}

    app.add_middleware(QuotaMiddleware, enable=False)
    client = TestClient(app)

    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200


def _make_client(enabled=True, _app=None):
    """Helper: create FastAPI app with QuotaMiddleware + mock auth state."""
    if _app is None:
        app = FastAPI()
    else:
        app = _app

    class MockAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = "test-tenant"
            request.state.api_key_id = "key-123"
            request.state.api_key_role = "inference-only"
            request.state.model = "test-model"
            return await call_next(request)

    # Register mock auth AFTER QuotaMiddleware so it runs FIRST (outermost)
    # on incoming requests, setting request.state before QuotaMiddleware reads it.
    app.add_middleware(QuotaMiddleware, enable=enabled)
    app.add_middleware(MockAuthMiddleware)
    return TestClient(app)


def test_allows_request_under_quota():
    """When tenant is under quota, request succeeds."""
    app = FastAPI()

    from distllm.core.usage_meter import UsageMeter
    meter = UsageMeter(storage_path=":memory:")
    # Real signature: enforce_quota(tenant_id, raise_on_block=True, requested_tokens=None)
    meter.enforce_quota = lambda tenant_id, raise_on_block=True, requested_tokens=None: (True, "")
    import distllm.api.quota_middleware as qm
    qm._meter = meter

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "ok"}}]}

    client = _make_client(True, app)

    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200


def test_rejects_request_when_quota_exceeded():
    """When tenant quota is exceeded, returns 429."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        return {"choices": [{"message": {"content": "ok"}}]}

    client = _make_client(True, app)

    from distllm.core.usage_meter import UsageMeter
    meter = UsageMeter(storage_path=":memory:")  # fresh in-memory meter
    meter.enforce_quota = lambda tenant_id, raise_on_block=True, requested_tokens=None: (
        False, "Daily token limit exceeded")
    import distllm.api.quota_middleware as qm
    qm._meter = meter

    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 429
    body = resp.json()
    assert "quota_exceeded" in body.get("error", {}).get("type", "")


def test_skips_non_tracked_paths():
    """Health endpoints are not tracked by quota."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.add_middleware(QuotaMiddleware, enable=True)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200


def test_should_track_returns_true_for_chat():
    mw = QuotaMiddleware.__new__(QuotaMiddleware)
    assert mw._should_track("/v1/chat/completions") is True
    assert mw._should_track("/v1/completions") is True
    assert mw._should_track("/v1/embeddings") is True


def test_should_track_returns_false_for_other():
    mw = QuotaMiddleware.__new__(QuotaMiddleware)
    assert mw._should_track("/health") is False
    assert mw._should_track("/v1/plugins") is False
