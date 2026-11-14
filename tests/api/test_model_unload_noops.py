"""Regression: ``POST /api/models/unload`` must not report success when
nothing was unloaded.

Defect: ``Coordinator`` always has ``hot_swap_unload`` (it returns False when
no hot-swap manager is configured) and never has ``unload_model``.  The route
therefore took the ``hot_swap_unload`` branch, got ``False`` back, and still
returned ``{"status": "ok", ...}`` — a 200 that unloads nothing.  A caller
watching the response cannot tell the unload did not happen.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from distllm.api.api_state import g
from distllm.api.routes import model_registry


class _CoordinatorNoHotSwap:
    """Shape of a real Coordinator without hot-swap configured."""

    def hot_swap_unload(self, name: str) -> bool:
        return False


class _CoordinatorHotSwapOK:
    """Hot-swap manager that actually unloads the model."""

    def __init__(self) -> None:
        self.unloaded: list[str] = []

    def hot_swap_unload(self, name: str) -> bool:
        self.unloaded.append(name)
        return True


class _CoordinatorHotSwapMissingModel:
    """Hot-swap manager that has never heard of the requested model."""

    def hot_swap_unload(self, name: str) -> bool:
        return False


class _CoordinatorRaisingHotSwap:
    """``hot_swap_unload`` raises — must surface as a clean error, not a 200."""

    def hot_swap_unload(self, name: str) -> bool:
        raise RuntimeError("manager boom")


@pytest.fixture
def app_client():
    """Mount the registry router on a bare app with an admin identity injected.

    ``require_role(...)`` returns a fresh closure on every call, so
    ``dependency_overrides`` cannot key it.  Stamp the role onto
    ``request.state`` before routing instead, which is exactly what the real
    AuthMiddleware does.
    """
    app = FastAPI()

    @app.middleware("http")
    async def _inject_role(request, call_next):
        request.state.api_key_role = "admin"
        request.state.api_key_id = "test-key"
        return await call_next(request)

    app.include_router(model_registry.router)
    yield TestClient(app)


def _set_coordinator(coord):
    original = g.coordinator
    g.coordinator = coord

    def restore():
        g.coordinator = original

    return restore


def test_unload_returns_error_when_no_hot_swap_manager(app_client):
    """No hot-swap manager -> explicit failure, not ``status: ok``."""
    restore = _set_coordinator(_CoordinatorNoHotSwap())
    try:
        resp = app_client.post(
            "/api/models/unload", json={"model": "llama-3-8b"}
        )
        body = resp.json()
        assert "error" in body, body
        assert body.get("status") != "ok", body
        assert resp.status_code >= 400, (resp.status_code, body)
    finally:
        restore()


def test_unload_returns_error_when_model_not_loaded(app_client):
    """Hot-swap manager returns False (unknown model) -> explicit failure."""
    restore = _set_coordinator(_CoordinatorHotSwapMissingModel())
    try:
        resp = app_client.post(
            "/api/models/unload", json={"model": "not-loaded"}
        )
        body = resp.json()
        assert "error" in body, body
        assert body.get("status") != "ok", body
        assert resp.status_code >= 400, (resp.status_code, body)
    finally:
        restore()


def test_unload_success_reports_ok(app_client):
    """A real unload still returns 200 + ``status: ok``."""
    coord = _CoordinatorHotSwapOK()
    restore = _set_coordinator(coord)
    try:
        resp = app_client.post(
            "/api/models/unload", json={"model": "llama-3-8b"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model"] == "llama-3-8b"
        assert coord.unloaded == ["llama-3-8b"]
    finally:
        restore()


def test_unload_exception_is_clean_error_not_200(app_client):
    """An exception inside the coordinator yields an error, never a fake ok."""
    restore = _set_coordinator(_CoordinatorRaisingHotSwap())
    try:
        resp = app_client.post(
            "/api/models/unload", json={"model": "llama-3-8b"}
        )
        body = resp.json()
        assert body.get("status") != "ok", body
        assert "error" in body, body
        assert resp.status_code >= 400, (resp.status_code, body)
    finally:
        restore()


# ---------------------------------------------------------------------------
# Status-code contract for the failure paths.
#
# Defect: with no coordinator, load/unload returned HTTP 200 with a bare
# {"error": "coordinator not available"} body, and the method-missing
# variants did the same.  The repo's own CLI (distllm model load/unload,
# distllm deploy) only checks the HTTP status via raise_for_status() and
# then prints "Model loaded: ..." / "Deployment request accepted." -- so a
# status-code-only client was told the operation succeeded when nothing
# happened.
# ---------------------------------------------------------------------------


class _CoordinatorWithoutLoadMethods:
    """Shape of a coordinator that exposes neither load nor unload method."""


def test_load_without_coordinator_is_5xx_not_200_error_body(app_client):
    restore = _set_coordinator(None)
    try:
        resp = app_client.post("/api/models/load", json={"model": "llama-x"})
        body = resp.json()
        assert resp.status_code >= 400, (resp.status_code, body)
        assert "coordinator not available" in body["error"]["message"], body
    finally:
        restore()


def test_unload_without_coordinator_is_5xx_not_200_error_body(app_client):
    restore = _set_coordinator(None)
    try:
        resp = app_client.post("/api/models/unload", json={"model": "llama-x"})
        body = resp.json()
        assert resp.status_code >= 400, (resp.status_code, body)
        assert "coordinator not available" in body["error"]["message"], body
    finally:
        restore()


def test_load_without_method_is_5xx_not_200_error_body(app_client):
    restore = _set_coordinator(_CoordinatorWithoutLoadMethods())
    try:
        resp = app_client.post("/api/models/load", json={"model": "llama-x"})
        body = resp.json()
        assert resp.status_code >= 400, (resp.status_code, body)
        assert "no load_model method available" in body["error"]["message"], body
    finally:
        restore()


def test_unload_without_method_is_5xx_not_200_error_body(app_client):
    restore = _set_coordinator(_CoordinatorWithoutLoadMethods())
    try:
        resp = app_client.post("/api/models/unload", json={"model": "llama-x"})
        body = resp.json()
        assert resp.status_code >= 400, (resp.status_code, body)
        assert "no unload_model method available" in body["error"]["message"], body
    finally:
        restore()


def test_registry_detail_missing_model_is_404(app_client):
    """A failed lookup must not be a 200 carrying an error object.

    Defect: GET /api/models/registry/{model_id} returned HTTP 200 with
    {"error": "model not found", ...}; clients keying on response.ok /
    curl --fail read the miss as a successful detail fetch.
    """
    from distllm.api.routes import model_registry as reg

    # The registry cache is module-global and TTL-based; force an empty view.
    reg._get_registry(force=True)
    resp = app_client.get("/api/models/registry/not-a-real-model")
    assert resp.status_code == 404, (resp.status_code, resp.text)
    body = resp.json()
    # This bare app has no error-envelope handler: the detail lives either
    # under "detail" (FastAPI default) or the structured envelope.
    detail = body.get("detail") or body.get("error", {}).get("message", "")
    assert "not found" in detail, body