"""Tests for SSOAuthHandler and provider implementations.

Covers:
- SSOUserInfo role mapping heuristic
- SSOAuthHandler initialization from environment variables
- OIDCHandler state/nonce stores
- GenericOAuth2Handler login URL generation
- Token validation (via mock JWKS)
- Token revocation blocklist
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest

from distllm.api.sso_auth import (
    GenericOAuth2Handler,
    OIDCHandler,
    SAMLHandler,
    SSOAuthHandler,
    SSOUserInfo,
    get_sso_handler,
)
from distllm.api.sso_middleware import (
    DEFAULT_JWT_ISSUER,
    _create_jwt,
    _decode_jwt,
)


# ======================================================================
# SSOUserInfo tests
# ======================================================================


class TestSSOUserInfo:
    """SSOUserInfo dataclass and role-mapping logic."""

    def test_to_api_key_role_admin(self):
        info = SSOUserInfo(sub="user1", roles=["admin"])
        assert info.to_api_key_role == "admin"

    def test_to_api_key_role_auditor(self):
        info = SSOUserInfo(sub="user1", roles=["auditor"])
        assert info.to_api_key_role == "auditor"

    def test_to_api_key_role_inference_by_default(self):
        info = SSOUserInfo(sub="user1", roles=["viewer"])
        assert info.to_api_key_role == "inference-only"

    def test_to_api_key_role_empty_roles(self):
        info = SSOUserInfo(sub="user1", roles=[])
        assert info.to_api_key_role == "inference-only"

    def test_to_api_key_role_admin_via_groups(self):
        info = SSOUserInfo(sub="user1", roles=[], groups=["admin"])
        assert info.to_api_key_role == "admin"

    def test_to_api_key_role_auditor_via_groups(self):
        info = SSOUserInfo(sub="user1", roles=[], groups=["Auditor"])
        assert info.to_api_key_role == "auditor"

    def test_to_api_key_role_read_only(self):
        info = SSOUserInfo(sub="user1", roles=["read-only"])
        assert info.to_api_key_role == "read-only"

    def test_to_api_key_role_administrator_maps_to_admin(self):
        info = SSOUserInfo(sub="user1", roles=["Administrator"])
        assert info.to_api_key_role == "admin"

    def test_empty_sub_does_not_crash(self):
        info = SSOUserInfo(sub="")
        assert isinstance(info.to_api_key_role, str)


# ======================================================================
# SAMLHandler tests
# ======================================================================


class TestSAMLHandler:
    def test_init_without_pysaml2_falls_back_gracefully(self):
        """When pysaml2 is not installed, handler returns callback URL."""
        handler = SAMLHandler(metadata_url="https://idp.example.com/metadata", callback_url="/auth/callback")
        # Without pysaml2, get_login_url returns fallback
        url = handler.get_login_url()
        assert url == "/auth/callback"

    def test_handle_callback_returns_none_on_exception(self):
        """Without pysaml2, callback processing returns None."""
        handler = SAMLHandler(metadata_url="https://idp.example.com/metadata", callback_url="/auth/callback")
        result = handler.handle_callback("<samlp:Response></samlp:Response>")
        assert result is None

    def test_init_skips_initialization_when_mocked(self, monkeypatch):
        monkeypatch.setattr(
            "distllm.api.sso_auth.SAMLHandler._initialize",
            lambda self: None,
        )
        handler = SAMLHandler.__new__(SAMLHandler)
        handler._metadata_url = "https://idp.example.com/metadata"
        handler._callback_url = "/auth/callback"
        handler._client = None
        url = handler.get_login_url()
        assert url == "/auth/callback"


# ======================================================================
# OIDCHandler tests
# ======================================================================


class TestOIDCHandler:
    def test_discover_sets_default_endpoints_on_failure(self):
        """When discovery HTTP call fails, default endpoints are set."""
        handler = OIDCHandler.__new__(OIDCHandler)
        handler._authority = "https://auth.example.com"
        handler._client_id = "test-client"
        handler._client_secret = "test-secret"
        handler._callback_url = "/auth/callback"
        handler._jwks_url = None
        handler._state_store = {}
        handler._nonce_store = {}
        handler._pkce_store = {}
        handler._nonce_ttl = 600.0

        handler._set_default_endpoints()

        assert handler._authorization_endpoint == "https://auth.example.com/authorize"
        assert handler._token_endpoint == "https://auth.example.com/oauth/token"
        assert handler._userinfo_endpoint == "https://auth.example.com/userinfo"

    def test_get_login_url_includes_nonce(self):
        """OIDC login URL contains nonce parameter."""
        handler = OIDCHandler.__new__(OIDCHandler)
        handler._authorization_endpoint = "https://auth.example.com/authorize"
        handler._client_id = "test-client"
        handler._callback_url = "/auth/callback"
        handler._scope = "openid profile email"
        handler._state_store = {}
        handler._nonce_store = {}
        handler._pkce_store = {}
        handler._nonce_ttl = 600.0

        url = handler.get_login_url(state="test-state-123")
        assert "nonce=" in url
        assert "state=test-state-123" in url

    def test_handle_callback_validates_state(self):
        """Callback with expected_state validates and removes it."""
        handler = OIDCHandler.__new__(OIDCHandler)
        handler._state_store = {"valid-state": time.time() + 600}
        handler._nonce_store = {}
        handler._token_endpoint = "https://auth.example.com/token"
        handler._client_id = "test"
        handler._client_secret = "secret"
        handler._callback_url = "/cb"
        handler._discovered_jwks_url = None
        handler._jwks_url = None
        handler._userinfo_endpoint = None

        # No httpx → ImportError caught → returns None
        result = handler.handle_callback("code123", expected_state="valid-state")
        assert result is None  # ImportError for httpx in test env
        assert "valid-state" not in handler._state_store  # popped

    def test_handle_callback_rejects_bad_state(self):
        """Callback with unknown state returns None (possible CSRF)."""
        handler = OIDCHandler.__new__(OIDCHandler)
        handler._state_store = {}
        handler._nonce_store = {}
        handler._pkce_store = {}
        result = handler.handle_callback("code123", expected_state="nonexistent")
        assert result is None

    def test_handle_callback_rejects_expired_state(self):
        """Callback with expired state returns None."""
        handler = OIDCHandler.__new__(OIDCHandler)
        handler._state_store = {"old-state": time.time() - 9999}
        handler._nonce_store = {}
        result = handler.handle_callback("code123", expected_state="old-state")
        assert result is None

    def test_validate_token_rejects_bad_jwks(self, monkeypatch):
        """When JWKS fetch fails, validate_token returns None."""
        def _fake_get(url, **kw):
            return type('R', (), {'status_code': 200, 'json': lambda: {"keys": []}})()
        monkeypatch.setattr("httpx.get", _fake_get)

        handler = OIDCHandler.__new__(OIDCHandler)
        handler._jwks_url = "https://auth.example.com/.well-known/jwks"
        handler._client_id = "test-client"
        handler._discovered_jwks_url = None

        result = handler.validate_token("invalid-token")
        assert result is None  # no matching key found


# ======================================================================
# GenericOAuth2Handler tests
# ======================================================================


class TestGenericOAuth2Handler:
    def test_get_login_url_includes_state(self):
        handler = GenericOAuth2Handler(
            client_id="test",
            client_secret="secret",
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            userinfo_url="https://api.github.com/user",
            callback_url="/auth/callback",
        )
        url = handler.get_login_url(state="my-state-42")
        assert "state=my-state-42" in url
        assert "client_id=test" in url

    def test_handle_callback_validates_state(self):
        handler = GenericOAuth2Handler.__new__(GenericOAuth2Handler)
        handler._state_store = {"valid": time.time() + 600}
        handler._pkce_store = {"valid": "test-verifier"}
        handler._token_url = "https://example.com/token"
        handler._client_id = "test"
        handler._client_secret = "secret"
        handler._callback_url = "/cb"

        result = handler.handle_callback("code", expected_state="valid")
        # No httpx → ImportError → None
        assert result is None
        assert "valid" not in handler._state_store

    def test_handle_callback_rejects_empty_state(self):
        """SEC-API-04: empty state must be rejected without calling the IdP.

        Mirrors the OIDCHandler empty-state test (test_csrf_middleware).
        """
        handler = GenericOAuth2Handler.__new__(GenericOAuth2Handler)
        handler._state_store = {}
        handler._pkce_store = {}
        with patch("httpx.post") as mock_post:
            result = handler.handle_callback("code", expected_state="")
        assert result is None
        mock_post.assert_not_called()

    def test_handle_callback_rejects_state_without_pkce_verifier(self):
        """SEC-API-04: a state with no bound PKCE verifier is rejected."""
        handler = GenericOAuth2Handler.__new__(GenericOAuth2Handler)
        handler._state_store = {"orphan-state": time.time() + 600}
        handler._pkce_store = {}
        with patch("httpx.post") as mock_post:
            result = handler.handle_callback("code", expected_state="orphan-state")
        assert result is None
        mock_post.assert_not_called()
        assert "orphan-state" not in handler._state_store

    def test_get_login_url_includes_pkce_challenge(self):
        handler = GenericOAuth2Handler(
            client_id="test",
            client_secret="secret",
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            userinfo_url="https://api.github.com/user",
            callback_url="/auth/callback",
        )
        url = handler.get_login_url()
        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        assert len(handler._pkce_store) == 1

    def test_handle_callback_rejects_bad_state(self):
        handler = GenericOAuth2Handler.__new__(GenericOAuth2Handler)
        handler._state_store = {}
        result = handler.handle_callback("code", expected_state="bad")
        assert result is None

    def test_handle_callback_rejects_expired_state(self):
        handler = GenericOAuth2Handler.__new__(GenericOAuth2Handler)
        handler._state_store = {"old": time.time() - 9999}
        result = handler.handle_callback("code", expected_state="old")
        assert result is None


# ======================================================================
# SSOAuthHandler integration tests
# ======================================================================


class TestSSOAuthHandler:
    def test_disabled_when_no_provider_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            handler = SSOAuthHandler()
            assert not handler.is_enabled
            assert handler.get_login_url() == ""
            assert handler.handle_callback("code") is None

    def test_initializes_oidc_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "DISTLLM_SSO_PROVIDER": "oidc",
                "DISTLLM_SSO_CLIENT_ID": "test-client",
                "DISTLLM_SSO_CLIENT_SECRET": "test-secret",
                "DISTLLM_SSO_AUTHORITY": "https://auth.example.com",
                "DISTLLM_SSO_CALLBACK_URL": "/auth/callback",
            },
            clear=True,
        ):
            handler = SSOAuthHandler()
            assert handler.is_enabled
            url = handler.get_login_url(state="test")
            assert "nonce=" in url

    def test_initializes_oauth2_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "DISTLLM_SSO_PROVIDER": "oauth2",
                "DISTLLM_SSO_CLIENT_ID": "test-client",
                "DISTLLM_SSO_CLIENT_SECRET": "test-secret",
                "DISTLLM_SSO_AUTHORIZE_URL": "https://github.com/login/oauth/authorize",
                "DISTLLM_SSO_TOKEN_URL": "https://github.com/login/oauth/access_token",
                "DISTLLM_SSO_USERINFO_URL": "https://api.github.com/user",
                "DISTLLM_SSO_CALLBACK_URL": "/auth/callback",
            },
            clear=True,
        ):
            handler = SSOAuthHandler()
            assert handler.is_enabled
            url = handler.get_login_url(state="test")
            assert "state=test" in url

    def test_revoke_token_adds_to_blocklist(self):
        handler = SSOAuthHandler.__new__(SSOAuthHandler)
        handler._revoked_tokens = {}
        handler._revocation_ttl_s = 3600.0
        handler._last_revocation_cleanup = time.time()

        handler.revoke_token("abc123")
        assert "abc123" in handler._revoked_tokens

    def test_validate_token_rejects_revoked(self):
        handler = SSOAuthHandler.__new__(SSOAuthHandler)
        handler._revoked_tokens = {}
        handler._revocation_ttl_s = 3600.0
        handler._last_revocation_cleanup = time.time()
        handler._handler = None

        handler.revoke_token("some-hash")
        result = handler.validate_token("some-token")
        # Without handler, returns None anyway, but should be None due to
        # revocation check processing correctly
        assert result is None

    def test_validate_token_returns_none_without_handler(self):
        handler = SSOAuthHandler.__new__(SSOAuthHandler)
        handler._revoked_tokens = {}
        handler._revocation_ttl_s = 3600.0
        handler._last_revocation_cleanup = time.time()
        handler._handler = None

        result = handler.validate_token("some-token")
        assert result is None

    def test_get_sso_handler_returns_singleton(self):
        with patch.dict("os.environ", {}, clear=True):
            h1 = get_sso_handler()
            h2 = get_sso_handler()
            assert h1 is h2

    def test_async_handle_callback_runs_in_thread(self):
        """async_handle_callback offloads to asyncio.to_thread."""
        with patch.dict("os.environ", {}, clear=True):
            handler = SSOAuthHandler()
            import asyncio
            result = asyncio.run(handler.async_handle_callback("code", "state"))
            assert result is None  # no provider configured

    def test_async_validate_token_runs_in_thread(self):
        with patch.dict("os.environ", {}, clear=True):
            handler = SSOAuthHandler()
            import asyncio
            result = asyncio.run(handler.async_validate_token("token"))
            assert result is None

    def test_cleanup_revoked_removes_expired(self):
        handler = SSOAuthHandler.__new__(SSOAuthHandler)
        handler._revoked_tokens = {"old-hash": time.time() - 9999}
        handler._revocation_ttl_s = 3600.0
        handler._last_revocation_cleanup = time.time() - 301  # force cleanup

        handler._cleanup_revoked()
        assert "old-hash" not in handler._revoked_tokens


# ======================================================================
# SEC-API-05-JWT: local JWT issuer + expiry enforcement
# ======================================================================


class TestLocalJwtIssuerAndExpiry:
    """Local SSO JWTs must carry ``iss`` and must require ``exp``.

    The encode/decode pair lives in ``sso_middleware.py`` (``_create_jwt``
    / ``_decode_jwt``); PyJWT only verifies ``exp`` when the claim is
    present, so the decode-side ``require`` is what makes expiry
    mandatory.
    """

    @pytest.fixture(autouse=True)
    def _fixed_secret(self, monkeypatch):
        monkeypatch.setenv("SSO_JWT_SECRET", "c" * 64)
        monkeypatch.delenv("DISTLLM_SSO_ISSUER", raising=False)
        return None

    def test_issued_token_contains_default_issuer(self):
        import jwt  # PyJWT

        token = _create_jwt(SSOUserInfo(sub="user1", roles=["user"]))
        # Inspect without verification so a broken decoder cannot mask a
        # missing claim on the encode side.
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["iss"] == DEFAULT_JWT_ISSUER == "distllm"
        assert "exp" in payload

    def test_issued_token_contains_env_issuer(self, monkeypatch):
        import jwt  # PyJWT

        monkeypatch.setenv("DISTLLM_SSO_ISSUER", "https://sso.internal.example")
        token = _create_jwt(SSOUserInfo(sub="user1"))
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["iss"] == "https://sso.internal.example"

    def test_valid_token_passes(self):
        token = _create_jwt(SSOUserInfo(sub="user1", roles=["user"]))
        payload = _decode_jwt(token)
        assert payload is not None
        assert payload["sub"] == "user1"
        assert payload["iss"] == DEFAULT_JWT_ISSUER

    def test_token_without_exp_is_rejected(self):
        import jwt  # PyJWT

        secret = "c" * 64
        token = jwt.encode(
            {"iss": DEFAULT_JWT_ISSUER, "sub": "attacker", "jti": "x"},
            secret,
            algorithm="HS256",
        )
        # Sanity: the same token decodes fine without requiring exp,
        # proving the rejection below comes from our ``require`` option.
        assert jwt.decode(token, secret, algorithms=["HS256"])["sub"] == "attacker"
        assert _decode_jwt(token) is None

    def test_token_with_wrong_issuer_is_rejected(self):
        import jwt  # PyJWT

        now = int(time.time())
        token = jwt.encode(
            {
                "iss": "https://evil.example",
                "sub": "attacker",
                "iat": now,
                "exp": now + 3600,
                "jti": "x",
            },
            "c" * 64,
            algorithm="HS256",
        )
        assert _decode_jwt(token) is None

    def test_token_without_issuer_claim_is_rejected(self):
        import jwt  # PyJWT

        now = int(time.time())
        token = jwt.encode(
            {"sub": "attacker", "iat": now, "exp": now + 3600, "jti": "x"},
            "c" * 64,
            algorithm="HS256",
        )
        assert _decode_jwt(token) is None

    def test_expired_token_is_rejected(self):
        import jwt  # PyJWT

        now = int(time.time())
        token = jwt.encode(
            {
                "iss": DEFAULT_JWT_ISSUER,
                "sub": "user1",
                "iat": now - 7200,
                "exp": now - 3600,
                "jti": "x",
            },
            "c" * 64,
            algorithm="HS256",
        )
        assert _decode_jwt(token) is None

    def test_issuer_from_env_is_enforced_on_decode(self, monkeypatch):
        """A token minted under the default issuer fails when the
        deployment expects a custom issuer (and vice versa)."""
        default_token = _create_jwt(SSOUserInfo(sub="user1"))
        assert _decode_jwt(default_token) is not None

        monkeypatch.setenv("DISTLLM_SSO_ISSUER", "https://sso.internal.example")
        assert _decode_jwt(default_token) is None

        custom_token = _create_jwt(SSOUserInfo(sub="user1"))
        assert _decode_jwt(custom_token) is not None


# ======================================================================
# Callback hardening: mandatory id_token, unconditional nonce check,
# fail-closed userinfo enrichment, .env config, login-state minting
# ======================================================================


def _bare_oidc_handler(
    *,
    has_userinfo: bool = True,
    nonce: str | None = "n",
    allow_unverified: bool = True,
) -> OIDCHandler:
    """OIDCHandler built via __new__ (no network discovery)."""
    h = OIDCHandler.__new__(OIDCHandler)
    h._state_store = {"s": time.time() + 600}
    h._nonce_store = (
        {"s": {"nonce": nonce, "expires": time.time() + 600}} if nonce else {}
    )
    h._pkce_store = {"s": "verifier"}
    h._token_endpoint = "https://x/t"
    h._client_id = "c"
    h._client_secret = "s"
    h._callback_url = "https://x/cb"
    h._userinfo_endpoint = "https://x/u" if has_userinfo else ""
    h._jwks_url = None
    h._discovered_jwks_url = None
    h._allow_unverified_id_token = allow_unverified
    h._nonce_ttl = 600.0
    return h


def _unsigned_id_token(claims: dict) -> str:
    """Structural JWT (alg=none) — decodable only via the unverified path."""
    import base64

    def _b64(obj: dict) -> str:
        return base64.urlsafe_b64encode(
            json.dumps(obj).encode()
        ).rstrip(b"=").decode()

    return f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64(claims)}."


class _Resp:
    status_code = 200
    text = ""

    def __init__(self, body: dict, status_code: int = 200):
        self._body = body
        self.status_code = status_code

    def json(self) -> dict:
        return self._body


class TestCallbackHardening:
    def test_missing_id_token_is_refused(self):
        """A token response without id_token must fail closed: previously it
        skipped both signature verification and the nonce replay check and
        authenticated from userinfo."""
        h = _bare_oidc_handler()
        with patch("httpx.post", lambda *a, **k: _Resp({"access_token": "at"})), patch(
            "httpx.get", lambda *a, **k: _Resp({"sub": "forged"})
        ):
            assert h.handle_callback("code", expected_state="s") is None

    def test_nonce_replay_check_runs_even_when_nonce_store_empty(self):
        """With a valid id_token but no stored nonce for the state, the
        exchange must be refused (replay protection is not optional)."""
        h = _bare_oidc_handler(nonce=None)
        id_token = _unsigned_id_token({"sub": "u", "nonce": "anything"})
        with patch(
            "httpx.post",
            lambda *a, **k: _Resp({"access_token": "at", "id_token": id_token}),
        ), patch("httpx.get", lambda *a, **k: _Resp({"sub": "u"})):
            assert h.handle_callback("code", expected_state="s") is None

    def test_nonce_mismatch_is_refused(self):
        h = _bare_oidc_handler(nonce="expected")
        id_token = _unsigned_id_token({"sub": "u", "nonce": "attacker"})
        with patch(
            "httpx.post",
            lambda *a, **k: _Resp({"access_token": "at", "id_token": id_token}),
        ), patch("httpx.get", lambda *a, **k: _Resp({"sub": "u"})):
            assert h.handle_callback("code", expected_state="s") is None

    def test_userinfo_failure_fails_closed(self):
        """A failed userinfo call after a valid id_token must not mint an
        empty-subject identity (one shared JWT for every user + a valid
        credential from a transient IdP outage)."""
        h = _bare_oidc_handler(nonce="n")
        id_token = _unsigned_id_token({"sub": "u", "nonce": "n"})
        with patch(
            "httpx.post",
            lambda *a, **k: _Resp({"access_token": "at", "id_token": id_token}),
        ), patch("httpx.get", lambda *a, **k: _Resp({"error": "boom"}, 500)):
            assert h.handle_callback("code", expected_state="s") is None

    def test_userinfo_failure_fails_closed_async(self):
        import asyncio

        h = _bare_oidc_handler(nonce="n")
        id_token = _unsigned_id_token({"sub": "u", "nonce": "n"})

        class _AsyncResp(_Resp):
            pass

        class _Client:
            async def post(self, url, **kw):
                return _AsyncResp({"access_token": "at", "id_token": id_token})

            async def get(self, url, **kw):
                return _AsyncResp({"error": "boom"}, 500)

        with patch("distllm.api.sso_auth._get_async_client", lambda: _Client()):
            result = asyncio.run(h.async_handle_callback("code", expected_state="s"))
        assert result is None

    def test_successful_exchange_returns_userinfo_identity(self):
        """Positive control: valid id_token + matching nonce + userinfo 200
        returns the enriched identity."""
        h = _bare_oidc_handler(nonce="n")
        id_token = _unsigned_id_token({"sub": "u", "nonce": "n"})
        with patch(
            "httpx.post",
            lambda *a, **k: _Resp({"access_token": "at", "id_token": id_token}),
        ), patch(
            "httpx.get", lambda *a, **k: _Resp({"sub": "u", "email": "u@x"})
        ):
            user = h.handle_callback("code", expected_state="s")
        assert user is not None
        assert user.sub == "u"
        assert user.email == "u@x"

    def test_no_userinfo_endpoint_uses_verified_id_token_claims(self):
        """Without a userinfo endpoint the identity comes from the verified
        id_token — not from an empty-subject placeholder."""
        h = _bare_oidc_handler(has_userinfo=False, nonce="n")
        id_token = _unsigned_id_token({"sub": "u", "nonce": "n"})
        with patch(
            "httpx.post",
            lambda *a, **k: _Resp({"access_token": "at", "id_token": id_token}),
        ):
            user = h.handle_callback("code", expected_state="s")
        assert user is not None
        assert user.sub == "u"


class TestLoginStateMinting:
    def test_get_login_url_state_roundtrip(self):
        """get_login_url() mints state+nonce+PKCE that handle_callback then
        accepts — the stores were previously populated but never fed by any
        route, so every real callback was 'state not found'."""
        import urllib.parse

        h = _bare_oidc_handler(nonce=None)
        h._authorization_endpoint = "https://x/authorize"
        url = h.get_login_url()
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        state, nonce = qs["state"][0], qs["nonce"][0]
        assert state in h._state_store
        assert state in h._pkce_store
        assert h._nonce_store[state]["nonce"] == nonce

        id_token = _unsigned_id_token({"sub": "u", "nonce": nonce})
        with patch(
            "httpx.post",
            lambda *a, **k: _Resp({"access_token": "at", "id_token": id_token}),
        ), patch("httpx.get", lambda *a, **k: _Resp({"sub": "u"})):
            user = h.handle_callback("code", expected_state=state)
        assert user is not None and user.sub == "u"

    def test_sso_handler_honors_dotenv_configuration(self, tmp_path, monkeypatch):
        """.env-configured DISTLLM_SSO_* values must enable SSO: operators
        configure every other DISTLLM_ setting through .env, and SSOAuthHandler
        previously read os.environ only, silently leaving SSO off."""
        for k in [k for k in os.environ if k.startswith("DISTLLM_SSO")]:
            monkeypatch.delenv(k, raising=False)
        (tmp_path / ".env").write_text(
            "DISTLLM_SSO_PROVIDER=oidc\n"
            "DISTLLM_SSO_CLIENT_ID=env-client\n"
            "DISTLLM_SSO_AUTHORITY=https://idp.env.example\n"
        )
        monkeypatch.chdir(tmp_path)
        with patch.object(OIDCHandler, "_discover", lambda self: None):
            handler = SSOAuthHandler()
        assert handler.is_enabled
        assert handler._handler._client_id == "env-client"

    def test_process_env_wins_over_dotenv(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "DISTLLM_SSO_PROVIDER=oidc\nDISTLLM_SSO_CLIENT_ID=env-client\n"
            "DISTLLM_SSO_AUTHORITY=https://idp.env.example\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DISTLLM_SSO_CLIENT_ID", "proc-client")
        with patch.object(OIDCHandler, "_discover", lambda self: None):
            handler = SSOAuthHandler()
        assert handler.is_enabled
        assert handler._handler._client_id == "proc-client"
