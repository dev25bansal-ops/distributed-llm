"""Tests for SSO middleware wiring (I3).

Verifies that:
- ``setup_sso(app)`` is called by ``distllm.api.server`` and the SSO
  middleware + ``/v1/auth/*`` routes are actually mounted (previously
  defined but unwired).
- The auth routes (/login, /token, /refresh, /revoke) are reachable
  without a pre-existing API key on the real server (AuthMiddleware
  exemption — previously every one of them 401'd on the shipped app).
- ``GET /v1/auth/login`` mints state/nonce/PKCE and the resulting code
  exchange succeeds end-to-end; a state replay is refused.
- An SSO JWT is accepted and sets ``request.state.auth_method == "sso"``
  (so ``AuthMiddleware`` skips API-key validation).
- Unknown Bearer tokens fall through to API-key auth (pass-through).
- Revoked tokens are rejected with 401 — including externally-validated
  (OIDC) tokens revoked by their hash (previously a silent no-op).
- SAML / OAuth2 providers can be registered and driven through the
  middleware (previously an AttributeError / TypeError / unknown_provider).
"""

from __future__ import annotations

import base64
import json
import time
import types
import urllib.parse
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from distllm.api.sso_auth import (
    GenericOAuth2Handler,
    OIDCHandler,
    SSOUserInfo,
)
from distllm.api.sso_middleware import SsoMiddleware, _create_jwt, setup_sso

from distllm.api import server as server_module

AUTH_PATHS = {
    "/v1/auth/login",
    "/v1/auth/token",
    "/v1/auth/refresh",
    "/v1/auth/revoke",
}


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
    from tests.api._route_helpers import route_paths

    assert AUTH_PATHS.issubset(route_paths(server_module.app))


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


# ======================================================================
# Finding: auth routes blocked by the shipped AuthMiddleware
# ======================================================================


def test_auth_routes_reachable_without_api_key_on_real_server():
    """POST /v1/auth/token on the shipped app must reach the route handler
    (400 missing_code), not 401 "invalid API key": enterprise users
    completing IdP login by definition have no API key yet.

    Uses the audit's exact repro vehicle (TestClient WITHOUT the lifespan
    context manager) so the assertion targets the AuthMiddleware exemption.
    """
    client = TestClient(server_module.app)
    resp = client.post("/v1/auth/token", json={})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "missing_code"


def test_revoke_route_reachable_without_api_key_on_real_server():
    """A holder of an SSO JWT (no API key) must be able to revoke it."""
    client = TestClient(server_module.app)
    resp = client.post("/v1/auth/revoke", json={})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "missing_token"


def test_login_route_exists_on_real_server():
    """GET /v1/auth/login is mounted (mints the state /token needs)."""
    client = TestClient(server_module.app)
    resp = client.get("/v1/auth/login", follow_redirects=False)
    # No provider configured in the test environment -> 400, but the
    # request must reach the route (not 401 from AuthMiddleware).
    assert resp.status_code in (302, 400)
    if resp.status_code == 400:
        assert resp.json()["error"]["code"] in (
            "unknown_provider",
            "sso_not_configured",
        )


# ======================================================================
# Finding: callback state was unmintable outside get_login_url()
# ======================================================================


def _make_bare_oidc_handler() -> OIDCHandler:
    """OIDCHandler without network discovery (no JWKS; unverified opt-in)."""
    import threading

    h = OIDCHandler.__new__(OIDCHandler)
    h._client_id = "cid"
    h._client_secret = "sec"
    h._authority = "https://idp.example"
    h._callback_url = "https://app/cb"
    h._jwks_url = None
    h._authorization_endpoint = "https://idp.example/authorize"
    h._token_endpoint = "https://idp.example/token"
    h._userinfo_endpoint = ""
    h._discovered_jwks_url = None
    h._allow_unverified_id_token = True
    h._state_store = {}
    h._nonce_store = {}
    h._nonce_ttl = 600.0
    h._pkce_store = {}
    h._jwks_cache_ttl_s = 300.0
    h._jwks_cache_lock = threading.Lock()
    h._jwks_cache = {}
    h._jwks_inflight = {}
    h._jwks_last_error = {}
    return h


def _b64u(obj: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


class _FakeAsyncResp:
    def __init__(self, body: dict, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict:
        return self._body


class _FakeAsyncClient:
    """Async client stub returning a canned token response."""

    def __init__(self, token_body: dict, userinfo_status: int = 200):
        self.token_body = token_body
        self.userinfo_status = userinfo_status

    async def post(self, url, **kwargs):
        return _FakeAsyncResp(self.token_body)

    async def get(self, url, **kwargs):
        if self.userinfo_status != 200:
            return _FakeAsyncResp({}, status_code=self.userinfo_status)
        return _FakeAsyncResp({"sub": "user-1"})


def _app_with_oidc_provider(handler: OIDCHandler) -> tuple[FastAPI, TestClient]:
    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {"auth_method": getattr(request.state, "auth_method", None)}

    setup_sso(app)
    client = TestClient(app)
    client.get("/echo")  # force middleware construction
    mw = getattr(app.state, "_sso_middleware", None)
    assert mw is not None
    mw._providers["default"] = handler
    return app, client


def test_login_mints_state_and_exchange_succeeds_end_to_end():
    """Full login flow: GET /login redirects with state+nonce, POST /token
    with that state mints a working local JWT, and replaying the state is
    refused.  Before the fix nothing ever called get_login_url(), so every
    real callback was rejected 'state not found'."""
    handler = _make_bare_oidc_handler()
    app, client = _app_with_oidc_provider(handler)

    resp = client.get("/v1/auth/login", follow_redirects=False)
    assert resp.status_code == 302
    loc = resp.headers["location"]
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(loc).query)
    state, nonce = qs["state"][0], qs["nonce"][0]
    assert state and nonce
    # PKCE verifier bound to the minted state
    assert handler._pkce_store[state]

    id_token = (
        _b64u({"alg": "none", "typ": "JWT"})
        + "."
        + _b64u({"sub": "user-1", "nonce": nonce, "email": "u@example.com"})
        + "."
    )

    with patch(
        "distllm.api.sso_auth._get_async_client",
        lambda: _FakeAsyncClient({"access_token": "at", "id_token": id_token}),
    ):
        resp = client.post(
            "/v1/auth/token",
            json={"provider": "default", "code": "authcode", "state": state},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user"]["sub"] == "user-1"

    # The minted local JWT authenticates and flips auth_method to "sso".
    echo = client.get(
        "/echo", headers={"Authorization": f"Bearer {body['access_token']}"}
    )
    assert echo.status_code == 200
    assert echo.json()["auth_method"] == "sso"

    # The state is single-use: replaying it must fail (state popped).
    with patch(
        "distllm.api.sso_auth._get_async_client",
        lambda: _FakeAsyncClient({"access_token": "at", "id_token": id_token}),
    ):
        replay = client.post(
            "/v1/auth/token",
            json={"provider": "default", "code": "authcode", "state": state},
        )
    assert replay.status_code == 401


def test_login_with_nonce_mismatch_rejected():
    """An id_token whose nonce does not match the state's stored nonce is
    rejected — replay binding holds end-to-end through the route."""
    handler = _make_bare_oidc_handler()
    app, client = _app_with_oidc_provider(handler)

    resp = client.get("/v1/auth/login", follow_redirects=False)
    qs = urllib.parse.parse_qs(
        urllib.parse.urlparse(resp.headers["location"]).query
    )
    state = qs["state"][0]
    bad_token = (
        _b64u({"alg": "none", "typ": "JWT"})
        + "."
        + _b64u({"sub": "user-1", "nonce": "wrong-nonce"})
        + "."
    )
    with patch(
        "distllm.api.sso_auth._get_async_client",
        lambda: _FakeAsyncClient({"access_token": "at", "id_token": bad_token}),
    ):
        resp = client.post(
            "/v1/auth/token",
            json={"provider": "default", "code": "authcode", "state": state},
        )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "sso_auth_failed"


def test_login_unknown_provider_rejected():
    app, client = _app_with_oidc_provider(_make_bare_oidc_handler())
    resp = client.get("/v1/auth/login?provider=nope", follow_redirects=False)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unknown_provider"


# ======================================================================
# Finding: revoke was a silent no-op for externally-validated tokens
# ======================================================================


def test_revoked_external_oidc_token_rejected():
    """A stolen OIDC access token revoked via revoke_token_str() must be
    rejected on the next request — the external-provider validation path
    previously never consulted the blacklist."""
    async def _validate(token: str):
        return SSOUserInfo(sub="v", provider="oidc") if token == "tok" else None

    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {"auth_method": getattr(request.state, "auth_method", None)}

    setup_sso(app)
    with TestClient(app) as client:
        client.get("/echo")
        mw = app.state._sso_middleware
        mw.register_provider(
            "p", types.SimpleNamespace(_authority="x", async_validate_token=_validate)
        )

        # Sanity: the external token authenticates before revocation.
        ok = client.get("/echo", headers={"Authorization": "Bearer tok"})
        assert ok.status_code == 200
        assert ok.json()["auth_method"] == "sso"

        mw.revoke_token_str("tok")
        revoked = client.get("/echo", headers={"Authorization": "Bearer tok"})
        assert revoked.status_code == 401
        assert revoked.json()["error"]["code"] == "token_revoked"


def test_revoke_route_revokes_external_token_on_minimal_app():
    """POST /v1/auth/revoke on a middleware-bearing app revokes a token
    whose value is known to the holder (hash-based revocation)."""
    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {"ok": True}

    setup_sso(app)
    with TestClient(app) as client:
        client.get("/echo")
        resp = client.post("/v1/auth/revoke", json={"token": "opaque-external-tok"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mw = app.state._sso_middleware
        import hashlib

        h = hashlib.sha256(b"opaque-external-tok").hexdigest()
        assert mw._blacklist.is_revoked(h)


# ======================================================================
# Finding: SAML / OAuth2 unusable through the middleware
# ======================================================================


def test_auto_detect_registers_oauth2_as_default(monkeypatch):
    """DISTLLM_SSO_PROVIDER=oauth2 must produce a registered 'default'
    provider so /v1/auth/token can perform the exchange (previously
    400 unknown_provider despite is_enabled=True)."""
    from distllm.api import sso_auth

    monkeypatch.setenv("DISTLLM_SSO_PROVIDER", "oauth2")
    monkeypatch.setenv("DISTLLM_SSO_CLIENT_ID", "c")
    monkeypatch.setenv("DISTLLM_SSO_AUTHORIZE_URL", "https://x/a")
    monkeypatch.setenv("DISTLLM_SSO_TOKEN_URL", "https://x/t")
    monkeypatch.setenv("DISTLLM_SSO_USERINFO_URL", "https://x/u")
    saved = sso_auth._handler_singleton
    sso_auth._handler_singleton = None
    try:
        app = FastAPI()

        @app.get("/echo")
        async def echo(request: Request):
            return {"ok": True}

        setup_sso(app)
        with TestClient(app) as client:
            client.get("/echo")
            mw = app.state._sso_middleware
            assert isinstance(mw.get_provider("default"), GenericOAuth2Handler)
            # state present -> the exchange attempt reaches the handler and
            # fails on the unknown state, not on unknown_provider.
            resp = client.post(
                "/v1/auth/token",
                json={"provider": "default", "code": "x", "state": "y"},
            )
            assert resp.status_code == 401
            assert resp.json()["error"]["code"] == "sso_auth_failed"
    finally:
        sso_auth._handler_singleton = saved


def test_register_provider_accepts_saml_handler():
    """The documented sso.register_provider('corp', SAMLHandler(...)) call
    must not raise AttributeError ('_authority')."""
    from distllm.api.sso_auth import SAMLHandler

    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {"ok": True}

    setup_sso(app)
    with TestClient(app) as client:
        client.get("/echo")
        mw = app.state._sso_middleware
        with patch.object(SAMLHandler, "_initialize", lambda self: None):
            handler = SAMLHandler(metadata_url="https://idp/m", callback_url="/cb")
        mw.register_provider("corp", handler)  # must not raise
        assert mw.get_provider("corp") is handler


def test_token_route_handles_saml_handler_without_typeerror():
    """Forcing a SAML handler into _providers must yield 401
    sso_auth_failed (validation failure), not an HTTP 500 TypeError."""
    from distllm.api.sso_auth import SAMLHandler

    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {"ok": True}

    setup_sso(app)
    with TestClient(app) as client:
        client.get("/echo")
        mw = app.state._sso_middleware
        with patch.object(SAMLHandler, "_initialize", lambda self: None):
            handler = SAMLHandler(metadata_url="https://idp/m", callback_url="/cb")
        mw._providers["corp"] = handler

        resp = client.post(
            "/v1/auth/token",
            json={"provider": "corp", "code": "<samlp:Response/>", "state": "s"},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "sso_auth_failed"


def test_token_route_requires_state_only_for_code_flow():
    """OIDC/OAuth2 still require state (CSRF); SAML assertions do not."""
    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {"ok": True}

    setup_sso(app)
    with TestClient(app) as client:
        client.get("/echo")
        mw = app.state._sso_middleware
        mw._providers["oidc-prov"] = _make_bare_oidc_handler()
        resp = client.post(
            "/v1/auth/token", json={"provider": "oidc-prov", "code": "x"}
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "missing_state"
