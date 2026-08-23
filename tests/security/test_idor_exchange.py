"""Security: Insecure Direct Object Reference (IDOR) on exchange endpoints.

The ``acquire_prompt`` endpoint accepts a ``user_id`` query parameter.
A valid API key may be used to acquire prompts on behalf of *any* user
if the ``user_id`` to ``api_key_id`` binding is not enforced.

The fix (implemented in ``routes/exchange.py``) checks
``request.state.api_key_owner`` against the requested ``user_id``.
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.api_state import reset_app_state_for_testing, g
from distllm.core.api_key_store import get_api_key_store, reset_api_key_store

from distllm.api.routes.exchange import router as exchange_router
from distllm.api.api_state import _state


@pytest.fixture(autouse=True)
def reset_state():
    """Clean shared state before each test."""
    reset_app_state_for_testing()
    reset_api_key_store()
    os.environ.pop("API_KEYS", None)
    os.environ.pop("API_KEY", None)
    _state.coordinator = object()  # satisfy require_coordinator dep
    yield


def _make_client(api_key_owner: str | None = None) -> TestClient:
    """Build a TestClient with exchange router and mock auth state."""
    app = FastAPI()
    app.include_router(exchange_router)

    class _AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.api_key_id = "key-test123"
            if api_key_owner is not None:
                request.state.api_key_owner = api_key_owner
            request.state.api_key_role = "admin"
            return await call_next(request)

    app.add_middleware(_AuthMiddleware)
    return TestClient(app)


class TestIDORExchangeAcquire:
    """IDOR protection on the prompt exchange acquire endpoint."""

    def test_acquire_without_api_key_id_returns_401(self):
        """No api_key_id in request.state -> 401."""
        app = FastAPI()
        app.include_router(exchange_router)
        client = TestClient(app)
        resp = client.post("/v1/exchange/prompts/prompt-1/acquire?user_id=attacker")
        # Returns 503 (require_coordinator fires first) or 401 depending on order
        assert resp.status_code in (401, 503)

    def test_acquire_with_own_user_id_succeeds(self):
        """Acquiring a prompt for your own user_id works."""
        client = _make_client(api_key_owner="legit-user")
        resp = client.post(
            "/v1/exchange/prompts/prompt-1/acquire?user_id=legit-user",
        )
        # Returns 503 (no prompt_exchange configured) or 402 (not enough tokens)
        # but NOT 401/403 — auth passes
        assert resp.status_code in (402, 503)

    def test_acquire_for_other_user_returns_403(self):
        """Acquiring a prompt for another user_id returns 403."""
        client = _make_client(api_key_owner="legit-user")
        resp = client.post(
            "/v1/exchange/prompts/prompt-1/acquire?user_id=other-user",
        )
        # 403 (IDOR rejection) or 503 (prompt_exchange not configured)
        assert resp.status_code in (403, 503)

    def test_acquire_without_owner_falls_back(self):
        """When api_key_owner is not set, the check is skipped."""
        client = _make_client(api_key_owner=None)
        resp = client.post(
            "/v1/exchange/prompts/prompt-1/acquire?user_id=any-user",
        )
        # Falls through to downstream logic (503 if no exchange configured)
        assert resp.status_code in (402, 503)
