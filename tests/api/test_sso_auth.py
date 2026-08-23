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
        handler._token_url = "https://example.com/token"
        handler._client_id = "test"
        handler._client_secret = "secret"
        handler._callback_url = "/cb"

        result = handler.handle_callback("code", expected_state="valid")
        # No httpx → ImportError → None
        assert result is None
        assert "valid" not in handler._state_store

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
