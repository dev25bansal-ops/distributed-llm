"""Tests for SSO middleware wiring (I3).

Verifies that:
- ``setup_sso(app)`` is called by ``distllm.api.server`` and the SSO
  middleware + ``/v1/auth/*`` routes are actually mounted (previously
  defined but unwired).
- An SSO JWT is accepted and sets ``request.state.auth_method == "sso"``
  (so ``AuthMiddleware`` skips API-key validation).
- Unknown Bearer tokens fall through to API-key auth (pass-through).
- Revoked tokens are rejected with 401.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from distllm.api.sso_auth import SSOUserInfo
from distllm.api.sso_middleware import SsoMiddleware, _create_jwt, setup_sso

from distllm.api import server as server_module

AUTH_PATHS = {"/v1/auth/token", "/v1/auth/refresh", "/v1/auth/revoke"}


def _minimal_app() -> FastAPI:
    """A tiny app with the SSO middleware + an echo route that reports
    the auth method, so dispatch behavior can be tested in isolation."""
    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {"auth_method": getattr(request.state, "auth_method", None)}

    setup_sso(app)
    return app


def test_sso_middleware_is_wired_into_server(monkeypatch):
    """The real server app mounts SsoMiddleware (I3: was unwired)."""
    assert any(
        m.cls is SsoMiddleware for m in server_module.app.user_middleware
    ), "SsoMiddleware should be registered on distllm.api.server.app"


def test_auth_routes_are_mounted():
    """POST /v1/auth/{token,refresh,revoke} exist on the real server."""
    mounted = {
        getattr(r, "path", None) for r in server_module.app.routes
    }
    assert AUTH_PATHS.issubset(mounted)


def test_token_endpoint_rejects_empty_body():
    app = _minimal_app()
    with TestClient(app) as client:
        resp = client.post("/v1/auth/token", json={})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "missing_code"


def test_token_endpoint_rejects_unknown_provider():
    app = _minimal_app()
    with TestClient(app) as client:
        resp = client.post(
            "/v1/auth/token",
            json={"provider": "no_such_provider", "code": "abc", "state": "s"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "unknown_provider"


def test_valid_sso_jwt_sets_auth_method(monkeypatch):
    monkeypatch.setenv("SSO_JWT_SECRET", "a" * 64)
    token = _create_jwt(SSOUserInfo(sub="user1", roles=["user"]))

    app = _minimal_app()
    with TestClient(app) as client:
        resp = client.get("/echo", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["auth_method"] == "sso"


def test_unknown_bearer_falls_through():
    """A non-SSO Bearer token must pass through (API-key auth path)."""
    app = _minimal_app()
    with TestClient(app) as client:
        resp = client.get(
            "/echo", headers={"Authorization": "Bearer not-a-sso-jwt"}
        )
        assert resp.status_code == 200
        assert resp.json()["auth_method"] is None


def test_revoked_token_rejected(monkeypatch):
    monkeypatch.setenv("SSO_JWT_SECRET", "b" * 64)
    token = _create_jwt(SSOUserInfo(sub="user2", roles=["user"]))

    app = _minimal_app()
    with TestClient(app) as client:
        # Trigger middleware build so it is stashed on app.state.
        client.get("/echo")
        mw = getattr(app.state, "_sso_middleware", None)
        assert mw is not None
        mw.revoke_token_str(token)

        resp = client.get("/echo", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401
