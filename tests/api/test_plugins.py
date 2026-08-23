"""Tests for plugin management endpoints (routes/plugins.py).

Covers:
- GET /v1/plugins — list installed plugins
- GET /v1/plugins/registry — built-in + installed registry listing
- GET /v1/plugins/{plugin_name} — plugin details or 404
- POST /v1/plugins/{plugin_name}/enable  — enable a plugin
- POST /v1/plugins/{plugin_name}/disable — disable a plugin
- Auth: admin-only access with required coordinator
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.api_state import _state
from distllm.core.api_key_store import get_api_key_store, reset_api_key_store

# ── Test admin key used throughout ────────────────────────────────────────
_TEST_ADMIN_KEY = "test-plugin-admin-key-12345"


# ── Mock classes (no MagicMock) ───────────────────────────────────────────


class PluginMock:
    """A simple plugin stub — defines callable version() and hook methods."""

    def __init__(self, name: str, version: str = "1.0.0", doc: str = "A test plugin"):
        self._name = name
        self._version = version
        self.__doc__ = doc

    def version(self) -> str:
        return self._version

    def on_request(self):
        pass

    def on_response(self):
        pass


class PluginSystemMock:
    """Simulates a ``_plugin_system`` that exposes ``list_plugins()``."""

    def __init__(self, plugins: list | None = None):
        self._plugins = plugins or []

    def list_plugins(self):
        return self._plugins


class CoordinatorMock:
    """Minimal coordinator stub carrying an optional plugin system."""

    def __init__(self, plugin_system: PluginSystemMock | None = None):
        self._plugin_system = plugin_system


# ── Helpers ───────────────────────────────────────────────────────────────


def _build_client() -> TestClient:
    """Create a TestClient for the plugins router with auth middleware."""
    from distllm.api.routes.plugins import router

    app = FastAPI()

    class _TestAuthMiddleware(BaseHTTPMiddleware):
        """Simulate AuthMiddleware: validate Bearer token, set api_key_role."""

        async def dispatch(self, request: Request, call_next):
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                token = auth[7:]
                store = get_api_key_store()
                result = store.authenticate(token)
                if result:
                    request.state.api_key_role = result[1]
                    request.state.api_key_id = result[0]
            return await call_next(request)

    app.add_middleware(_TestAuthMiddleware)
    app.include_router(router)
    return TestClient(app)


def _coordinator_with_plugins(
    plugins: list | None = None,
) -> CoordinatorMock:
    """Build a coordinator with a populated plugin system."""
    return CoordinatorMock(plugin_system=PluginSystemMock(plugins or []))


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset the singleton key store before each test (re-reads from env)."""
    os.environ["API_KEYS"] = (
        '{"keys": ['
        '{"key": "' + _TEST_ADMIN_KEY + '", "role": "admin", "label": "test-admin-plugin"},'
        '{"key": "test-read-only", "role": "read-only", "label": "test-read"}'
        "]}"
    )
    reset_api_key_store()
    yield
    reset_api_key_store()


@pytest.fixture
def auth_header():
    """Valid admin Authorization header."""
    return {"Authorization": f"Bearer {_TEST_ADMIN_KEY}"}


@pytest.fixture
def client():
    """TestClient with a coordinator that has two installed plugins."""
    rate_limit = PluginMock("rate-limit", version="2.0.0", doc="Rate limiting plugin")
    audit_log = PluginMock("audit-log", version="1.5.0", doc="Audit logging plugin")
    _state.coordinator = _coordinator_with_plugins([("rate-limit", rate_limit), ("audit-log", audit_log)])
    yield _build_client()
    _state.coordinator = None  # cleanup to avoid leaking into other test modules


# ── Auth & coordinator checks ─────────────────────────────────────────────


class TestPluginAuth:
    """Verify auth and coordinator-guard behaviour."""

    def test_no_auth_returns_401(self):
        """Missing Authorization header -> 401."""
        client = _build_client()
        resp = client.get("/v1/plugins")
        assert resp.status_code == 401

    def test_read_only_key_returns_403(self):
        """A key with role 'read-only' cannot access admin-guarded endpoints."""
        client = _build_client()
        resp = client.get(
            "/v1/plugins",
            headers={"Authorization": "Bearer test-read-only"},
        )
        assert resp.status_code == 403

    def test_no_coordinator_returns_503(self, auth_header):
        """All plugin endpoints require a running coordinator -> 503."""
        client = _build_client()
        resp = client.get("/v1/plugins", headers=auth_header)
        assert resp.status_code == 503
        assert "coordinator" in resp.text.lower()


# ── GET /v1/plugins ───────────────────────────────────────────────────────


class TestListPlugins:
    """GET /v1/plugins — list installed plugins."""

    def test_returns_installed_plugins(self, client, auth_header):
        resp = client.get("/v1/plugins", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        names = [p["name"] for p in data["plugins"]]
        assert "rate-limit" in names
        assert "audit-log" in names

    def test_plugin_fields(self, client, auth_header):
        """Each plugin in the listing carries expected metadata and hooks."""
        resp = client.get("/v1/plugins", headers=auth_header)
        plugin = next(p for p in resp.json()["plugins"] if p["name"] == "rate-limit")
        assert plugin["version"] == "2.0.0"
        assert plugin["description"] == "Rate limiting plugin"
        assert plugin["state"] == "active"
        assert "on_request" in plugin["hooks"]
        assert "on_response" in plugin["hooks"]

    def test_empty_when_no_plugin_system(self, auth_header):
        """Coordinator exists but ``_plugin_system`` is None -> empty list."""
        _state.coordinator = _coordinator_with_plugins(None)
        client = _build_client()
        resp = client.get("/v1/plugins", headers=auth_header)
        assert resp.status_code == 200
        assert resp.json() == {"plugins": [], "total": 0}


# ── GET /v1/plugins/registry ──────────────────────────────────────────────


class TestPluginRegistry:
    """GET /v1/plugins/registry — list built-in and discovered plugins."""

    def test_returns_builtin_and_installed(self, client, auth_header):
        resp = client.get("/v1/plugins/registry", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "builtin" in data
        assert "installed" in data
        # All three built-in plugins are documented
        assert "rate-limit" in data["builtin"]
        assert "audit-log" in data["builtin"]
        assert "metrics" in data["builtin"]
        # Two installed plugins are reflected in the installed list
        assert "rate-limit" in data["installed"]
        assert "audit-log" in data["installed"]


# ── GET /v1/plugins/{plugin_name} ─────────────────────────────────────────


class TestGetPlugin:
    """GET /v1/plugins/{plugin_name} — plugin details."""

    def test_get_existing_plugin(self, client, auth_header):
        resp = client.get("/v1/plugins/rate-limit", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "rate-limit"
        assert data["version"] == "2.0.0"
        assert data["description"] == "Rate limiting plugin"
        assert data["state"] == "active"

    def test_get_builtin_plugin(self, client, auth_header):
        """A plugin that is only in PLUGIN_DOCS returns state='available'."""
        resp = client.get("/v1/plugins/metrics", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "metrics"
        assert data["state"] == "available"
        assert "Plugin health" in data["description"]
        assert len(data["hooks"]) == 5  # on_start, on_request, on_response, on_error, on_model_load

    def test_get_unknown_plugin_returns_404(self, client, auth_header):
        resp = client.get("/v1/plugins/nonexistent-plugin", headers=auth_header)
        assert resp.status_code == 404
        assert "not found" in resp.text.lower()


# ── POST /v1/plugins/{plugin_name}/enable ────────────────────────────────


class TestEnablePlugin:
    """POST /v1/plugins/{plugin_name}/enable."""

    def test_enable_known_plugin_rate_limit(self, client, auth_header):
        resp = client.post("/v1/plugins/rate-limit/enable", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "DISTLLM_PLUGIN_RATELIMIT_ENABLED" in data["message"]

    def test_enable_known_plugin_audit_log(self, client, auth_header):
        resp = client.post("/v1/plugins/audit-log/enable", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "DISTLLM_AUDIT_LOG" in data["message"]

    def test_enable_unknown_plugin(self, client, auth_header):
        """Unknown plugins cannot be enabled via the API."""
        resp = client.post("/v1/plugins/custom-plugin/enable", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "cannot be enabled" in data["message"]


# ── POST /v1/plugins/{plugin_name}/disable ───────────────────────────────


class TestDisablePlugin:
    """POST /v1/plugins/{plugin_name}/disable."""

    def test_disable_plugin(self, client, auth_header):
        resp = client.post("/v1/plugins/rate-limit/disable", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "Restart required" in data["message"]

    def test_disable_any_plugin_name_returns_success(self, client, auth_header):
        """Even non-existent plugin names return success with a restart note."""
        resp = client.post("/v1/plugins/completely-fake/disable", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "Restart required" in data["message"]
