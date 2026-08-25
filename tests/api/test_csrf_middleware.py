"""Tests for CSRF protection (CSRFSameOriginMiddleware) and SSO CSRF state.

Threat model note: DistLLM authenticates via headers browsers never attach
automatically (``Authorization: Bearer``, ``X-API-Key``, ``X-Cluster-Key``)
and issues no session cookies, so classic synchronizer-token CSRF does not
apply.  The CSRF layer therefore validates Origin/Referer on every
state-changing request (OWASP defense-in-depth for browser-borne forgery),
while OAuth2/OIDC flows are protected by server-side ``state`` binding.

Coverage map asserted here:

* The middleware is registered on the real server app.
* Every state-changing (POST/PUT/PATCH/DELETE) route on the server is
  subject to the middleware — none sits on an EXEMPT_PATH entry.
* Foreign-origin POST → 403; allowed-origin POST → passes; missing
  Origin+Referer → passes (non-browser clients).
* Wildcard-origin rules cannot be bypassed via suffix domains
  (``app.example.com.evil.io``) — regression for the unanchored-regex bug.
* A present-but-malformed Referer fails CLOSED (was previously allowed).
* OAuth2/OIDC ``state`` issued in one login flow cannot be redeemed in
  another (cross-session state reuse → rejected).

Run: python -m pytest tests/api/test_csrf_middleware.py -v
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from distllm.api import server as server_module
from distllm.api.csrf_middleware import (
    CSRFSameOriginMiddleware,
    _normalize_origin,
    _origin_matches_allowed,
)
from distllm.api.sso_auth import GenericOAuth2Handler, OIDCHandler


# ── Helpers ────────────────────────────────────────────────────────────────────


def _minimal_app() -> FastAPI:
    """Tiny app with the CSRF middleware and a state-changing echo route."""
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request):
        return {"ok": True}

    @app.delete("/echo")
    async def delete_echo():
        return {"ok": True}

    app.add_middleware(CSRFSameOriginMiddleware)
    return app


def _client() -> TestClient:
    return TestClient(_minimal_app())


# ── Wiring / coverage map ──────────────────────────────────────────────────────


class TestCsrfWiring:
    """The middleware must actually be mounted on the production app."""

    def test_csrf_middleware_registered_on_server(self):
        assert any(
            m.cls is CSRFSameOriginMiddleware for m in server_module.app.user_middleware
        ), "CSRFSameOriginMiddleware should be registered on distllm.api.server.app"

    def test_no_state_changing_route_on_exempt_path(self):
        """No POST/PUT/PATCH/DELETE route may sit on an EXEMPT_PATH entry.

        Exempt paths are monitoring/docs/dashboard surfaces; if a mutating
        route ever lands on one it silently escapes CSRF validation.
        """
        exempt = CSRFSameOriginMiddleware.EXEMPT_PATHS
        mutating_methods = {"POST", "PUT", "PATCH", "DELETE"}
        offenders = []
        for route in server_module.app.routes:
            methods = getattr(route, "methods", set()) or set()
            path = getattr(route, "path", "")
            if methods & mutating_methods and path in exempt:
                offenders.append(f"{sorted(methods & mutating_methods)} {path}")
        assert not offenders, (
            f"State-changing routes on CSRF-exempt paths: {offenders}"
        )

    def test_server_has_state_changing_routes(self):
        """Sanity: the coverage assertion above has something to cover."""
        mutating_methods = {"POST", "PUT", "PATCH", "DELETE"}
        count = sum(
            1
            for r in server_module.app.routes
            if (getattr(r, "methods", set()) or set()) & mutating_methods
        )
        assert count > 20


# ── Core behavior: reject / accept ─────────────────────────────────────────────


class TestCsrfCoreBehavior:
    def test_foreign_origin_post_rejected_403(self):
        """Request WITHOUT a valid CSRF origin (hostile cross-site Origin)
        must be rejected with 403 before reaching the handler."""
        with _client() as client:
            resp = client.post(
                "/echo",
                json={"x": 1},
                headers={"Origin": "https://evil.attacker.io"},
            )
            assert resp.status_code == 403
            body = resp.json()
            assert body["error"]["type"] == "csrf_error"
            # Handler never executed
            assert "ok" not in body

    def test_foreign_referer_post_rejected_403(self):
        """Origin absent but hostile Referer → rejected."""
        with _client() as client:
            resp = client.post(
                "/echo",
                json={"x": 1},
                headers={"Referer": "https://evil.attacker.io/login"},
            )
            assert resp.status_code == 403
            assert resp.json()["error"]["type"] == "csrf_error"

    def test_same_origin_local_post_accepted(self):
        """Localhost dev origins are always allowed."""
        with _client() as client:
            resp = client.post(
                "/echo",
                json={"x": 1},
                headers={"Origin": "http://localhost:8000"},
            )
            assert resp.status_code == 200
            assert resp.json() == {"ok": True}

    def test_no_origin_no_referer_passes_through(self):
        """Non-browser clients (curl, SDKs, node-to-node) omit both headers
        and must not break — they also cannot carry browser credentials."""
        with _client() as client:
            resp = client.post("/echo", json={"x": 1})
            assert resp.status_code == 200

    def test_get_request_bypasses_check(self):
        with _client() as client:
            resp = client.get("/echo", headers={"Origin": "https://evil.attacker.io"})
            # GET is exempt; route only defines POST/DELETE so 405 proves the
            # request reached routing (i.e., was NOT blocked by CSRF 403).
            assert resp.status_code == 405

    def test_options_preflight_bypasses_check(self):
        with _client() as client:
            resp = client.options(
                "/echo", headers={"Origin": "https://evil.attacker.io"}
            )
            assert resp.status_code != 403

    def test_delete_with_foreign_origin_rejected(self):
        """All state-changing methods are covered, not just POST."""
        with _client() as client:
            resp = client.delete("/echo", headers={"Origin": "https://evil.attacker.io"})
            assert resp.status_code == 403


class TestAllowedOriginsEnv:
    """DISTLLM_CSRF_ALLOWED_ORIGINS configuration."""

    @pytest.fixture
    def app_factory(self, monkeypatch):
        def make(allowed: str):
            monkeypatch.setenv("DISTLLM_CSRF_ALLOWED_ORIGINS", allowed)
            return _minimal_app()

        return make

    def test_configured_origin_allowed(self, app_factory):
        app = app_factory("https://console.distllm.dev")
        with TestClient(app) as client:
            ok = client.post(
                "/echo", json={}, headers={"Origin": "https://console.distllm.dev"}
            )
            blocked = client.post(
                "/echo", json={}, headers={"Origin": "https://other.io"}
            )
        assert ok.status_code == 200
        assert blocked.status_code == 403

    def test_wildcard_subdomain_matches_child_only(self, app_factory):
        app = app_factory("https://*.example.com")
        with TestClient(app) as client:
            child = client.post(
                "/echo", json={}, headers={"Origin": "https://app.example.com"}
            )
            deep = client.post(
                "/echo", json={}, headers={"Origin": "https://a.b.example.com"}
            )
        assert child.status_code == 200
        assert deep.status_code == 200

    def test_wildcard_suffix_domain_bypass_blocked(self, app_factory):
        """REGRESSION: ``*.example.com`` used to accept
        ``https://app.example.com.evil.io`` because re.match was unanchored
        at the end.  Attacker-controlled suffix domains must be rejected."""
        app = app_factory("https://*.example.com")
        with TestClient(app) as client:
            resp = client.post(
                "/echo",
                json={},
                headers={"Origin": "https://app.example.com.evil.io"},
            )
        assert resp.status_code == 403

    def test_wildcard_does_not_match_bare_domain_or_scheme_swap(self, app_factory):
        app = app_factory("https://*.example.com")
        with TestClient(app) as client:
            bare = client.post(
                "/echo", json={}, headers={"Origin": "https://example.com"}
            )
            swapped = client.post(
                "/echo", json={}, headers={"Origin": "http://app.example.com"}
            )
        assert bare.status_code == 403
        assert swapped.status_code == 403

    def test_origin_matching_is_case_insensitive(self, app_factory):
        app = app_factory("https://Console.Example.com/")
        with TestClient(app) as client:
            resp = client.post(
                "/echo", json={}, headers={"Origin": "https://console.example.COM"}
            )
        assert resp.status_code == 200


class TestMalformedRefererFailClosed:
    """A present-but-unparseable Referer must fail CLOSED."""

    def test_malformed_referer_rejected(self):
        with _client() as client:
            resp = client.post(
                "/echo", json={}, headers={"Referer": "not-a-url-at-all"}
            )
            assert resp.status_code == 403

    def test_referer_without_host_rejected(self):
        with _client() as client:
            resp = client.post("/echo", json={}, headers={"Referer": "file://"})
            assert resp.status_code == 403


# ── Unit-level: matcher function ───────────────────────────────────────────────


class TestOriginMatcherUnit:
    def test_exact_match(self):
        assert _origin_matches_allowed(
            "https://a.com", ["https://a.com", "https://b.com"]
        )

    def test_no_match(self):
        assert not _origin_matches_allowed("https://a.com", ["https://b.com"])

    def test_empty_allowed_list_strict(self):
        assert not _origin_matches_allowed("https://a.com", [])

    def test_safe_origins_always_allowed(self):
        assert _origin_matches_allowed("http://localhost:3000", [])
        assert _origin_matches_allowed("http://127.0.0.1:8080", [])

    def test_suffix_attack_unit(self):
        assert not _origin_matches_allowed(
            "https://sub.example.com.evil.io", ["https://*.example.com"]
        )

    def test_normalize_origin_strips_trailing_slash_and_cases(self):
        assert _normalize_origin("HTTPS://App.Example.COM/") == "https://app.example.com"
        assert _normalize_origin(" https://a.com ") == "https://a.com"
        assert _normalize_origin("garbage") == "garbage"


# ── SSO/OAuth2 state: cross-session reuse must be rejected ─────────────────────


class TestOAuthStateCrossSession:
    """The OAuth2 'state' parameter IS the per-session CSRF token for the
    login flow.  A state issued for one login session must never validate a
    callback belonging to another (single-use, server-side binding)."""

    def test_oidc_state_from_other_session_rejected(self):
        handler = OIDCHandler.__new__(OIDCHandler)
        handler._authority = "https://auth.example.com"
        handler._client_id = "c"
        handler._client_secret = "s"
        handler._callback_url = "/cb"
        handler._jwks_url = None
        handler._discovered_jwks_url = None
        handler._userinfo_endpoint = None
        handler._token_endpoint = ""
        handler._nonce_store = {}
        handler._nonce_ttl = 600.0
        # Session A obtained this state from get_login_url():
        handler._state_store = {"session-A-state": time.time() + 600}

        # Session B replays it → must be rejected (and must NOT consume A's
        # state, which stays bound to its own flow until used or expired).
        result = handler.handle_callback("code", expected_state="session-B-state")
        assert result is None
        assert "session-A-state" in handler._state_store

    def test_oidc_state_single_use_second_replay_rejected(self):
        """Even the legitimate session cannot redeem the same state twice."""
        handler = OIDCHandler.__new__(OIDCHandler)
        handler._state_store = {"st": time.time() + 600}
        handler._nonce_store = {}
        handler._token_endpoint = ""  # forces ImportError/failure after pop

        first = handler.handle_callback("code1", expected_state="st")
        assert first is None          # exchange fails (no endpoint) …
        second = handler.handle_callback("code1", expected_state="st")
        assert second is None         # … but the state is consumed regardless
        assert "st" not in handler._state_store

    def test_oauth2_state_missing_entirely_rejected_at_handler_level(self):
        handler = OIDCHandler.__new__(OIDCHandler)
        handler._state_store = {}
        handler._nonce_store = {}
        result = handler.handle_callback("code", expected_state="")
        assert result is None

    def test_generic_oauth2_expired_state_rejected(self):
        handler = GenericOAuth2Handler.__new__(GenericOAuth2Handler)
        handler._state_store = {"old": time.time() - 1}
        result = handler.handle_callback("code", expected_state="old")
        assert result is None


# ── Middleware-level exemptions sanity ─────────────────────────────────────────


class TestExemptions:
    @pytest.mark.parametrize(
        "path", ["/health", "/ready", "/live", "/metrics", "/docs", "/openapi.json"]
    )
    def test_exempt_paths_skip_validation(self, path):
        """Exempt paths reach routing even with a hostile Origin (they are
        read-only monitors; a 405/404 from the minimal app proves the CSRF
        middleware did not intercept with 403)."""
        app = FastAPI()

        @app.get(path)
        async def probe():
            return {"ok": True}

        app.add_middleware(CSRFSameOriginMiddleware)
        with TestClient(app) as client:
            resp = client.get(path, headers={"Origin": "https://evil.attacker.io"})
        assert resp.status_code == 200

    def test_websocket_upgrade_skips_validation(self):
        app = FastAPI()
        app.add_middleware(CSRFSameOriginMiddleware)

        # A raw ASGI check: dispatch short-circuits on Upgrade: websocket.
        # Simulate via scope directly is overkill; assert class behavior via
        # a POST carrying the upgrade header reaches routing (405, not 403).
        with TestClient(app) as client:
            resp = client.post(
                "/anything",
                headers={
                    "Origin": "https://evil.attacker.io",
                    "Upgrade": "websocket",
                },
            )
        assert resp.status_code != 403
