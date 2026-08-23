"""Unified SSO authentication — SAML 2.0, OpenID Connect, OAuth2.

Provides a single ``SSOAuthHandler`` that delegates to the correct
provider based on ``DISTLLM_SSO_PROVIDER``.  Callers use::

    from distllm.api.auth import get_sso_handler

    handler = get_sso_handler()
    url = handler.get_login_url()
    info = await handler.async_handle_callback(code, state)
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time

from loguru import logger

from .models import SSOUserInfo

# Provider handlers are imported lazily in _initialize to avoid
# forcing optional dependencies (pysaml2, httpx, PyJWT) at import time.
# Lazy-import helpers:
_HANDLER_TYPES: dict[str, type] = {}


def _get_handler_class(provider: str):
    """Import and return the handler class for *provider*."""
    if provider in _HANDLER_TYPES:
        return _HANDLER_TYPES[provider]
    if provider == "saml":
        from .saml import SAMLHandler as cls
    elif provider == "oidc":
        from .oidc import OIDCHandler as cls
    elif provider == "oauth2":
        from .oauth2 import GenericOAuth2Handler as cls
    else:
        return None
    _HANDLER_TYPES[provider] = cls
    return cls


# ── SSOAuthHandler ───────────────────────────────────────────────────────


class SSOAuthHandler:
    """Unified SSO authentication handler — routes to the right provider.

    Configured via environment variables. Call ``get_sso_handler()``
    to get the singleton instance.
    """

    def __init__(self):
        self._provider: str = os.environ.get("DISTLLM_SSO_PROVIDER", "").lower()
        self._handler = None
        self._initialize()
        # Local token revocation blocklist
        self._revoked_tokens: dict[str, float] = {}
        self._revocation_ttl_s: float = 3600.0
        self._last_revocation_cleanup: float = time.time()

    def _initialize(self) -> None:
        """Initialize the appropriate SSO handler based on configuration."""
        callback_url = os.environ.get("DISTLLM_SSO_CALLBACK_URL", "")
        cls = _get_handler_class(self._provider)
        if cls is None:
            logger.info(
                "No SSO provider configured. Set DISTLLM_SSO_PROVIDER to "
                "'saml', 'oidc', or 'oauth2' to enable enterprise single sign-on."
            )
            return

        if self._provider == "saml":
            metadata_url = os.environ.get("DISTLLM_SSO_METADATA_URL", "")
            if metadata_url:
                self._handler = cls(metadata_url=metadata_url, callback_url=callback_url)
                logger.info("SSO: SAML 2.0 handler initialized")

        elif self._provider == "oidc":
            client_id = os.environ.get("DISTLLM_SSO_CLIENT_ID", "")
            client_secret = os.environ.get("DISTLLM_SSO_CLIENT_SECRET", "")
            authority = os.environ.get("DISTLLM_SSO_AUTHORITY", "")
            jwks_url = os.environ.get("DISTLLM_SSO_JWKS_URL", "")
            if client_id and authority:
                self._handler = cls(
                    client_id=client_id,
                    client_secret=client_secret,
                    authority=authority,
                    callback_url=callback_url,
                    jwks_url=jwks_url or None,
                )
                logger.info(f"SSO: OIDC handler initialized (authority={authority})")

        elif self._provider == "oauth2":
            client_id = os.environ.get("DISTLLM_SSO_CLIENT_ID", "")
            client_secret = os.environ.get("DISTLLM_SSO_CLIENT_SECRET", "")
            auth_url = os.environ.get("DISTLLM_SSO_AUTHORIZE_URL", "")
            token_url = os.environ.get("DISTLLM_SSO_TOKEN_URL", "")
            user_url = os.environ.get("DISTLLM_SSO_USERINFO_URL", "")
            if client_id and auth_url and token_url:
                self._handler = cls(
                    client_id=client_id,
                    client_secret=client_secret,
                    authorize_url=auth_url,
                    token_url=token_url,
                    userinfo_url=user_url,
                    callback_url=callback_url,
                )
                logger.info("SSO: OAuth2 handler initialized")

    @property
    def is_enabled(self) -> bool:
        return self._handler is not None

    def get_login_url(self, state: str = "") -> str:
        if self._handler is None:
            return ""
        return self._handler.get_login_url(state)

    async def async_handle_callback(self, code: str, state: str = "") -> SSOUserInfo | None:
        """Async wrapper for handle_callback — runs in a thread to avoid blocking the event loop."""
        return await asyncio.to_thread(self.handle_callback, code, state)

    def handle_callback(self, code: str, state: str = "") -> SSOUserInfo | None:
        if self._handler is None:
            return None
        from .oauth2 import GenericOAuth2Handler
        from .oidc import OIDCHandler
        if isinstance(self._handler, (OIDCHandler, GenericOAuth2Handler)):
            return self._handler.handle_callback(code, expected_state=state)
        return self._handler.handle_callback(code)

    def revoke_token(self, token_hash: str) -> None:
        """Revoke a token by its SHA-256 hash."""
        self._revoked_tokens[token_hash] = time.time()
        logger.info(f"Token revoked (hash prefix: {token_hash[:16]}...)")

    def _cleanup_revoked(self) -> None:
        """Periodic cleanup of expired revocation entries."""
        now = time.time()
        if now - self._last_revocation_cleanup < 300:
            return
        cutoff = now - self._revocation_ttl_s
        self._revoked_tokens = {h: ts for h, ts in self._revoked_tokens.items() if ts > cutoff}
        self._last_revocation_cleanup = now

    async def async_validate_token(self, access_token: str) -> SSOUserInfo | None:
        """Async wrapper for validate_token."""
        return await asyncio.to_thread(self.validate_token, access_token)

    def validate_token(self, access_token: str) -> SSOUserInfo | None:
        """Validate an access token (OIDC only for now)."""
        token_hash = hashlib.sha256(access_token.encode()).hexdigest()
        self._cleanup_revoked()
        if token_hash in self._revoked_tokens:
            logger.debug(f"Token rejected — revoked (hash prefix: {token_hash[:16]}...)")
            return None
        from .oidc import OIDCHandler
        if isinstance(self._handler, OIDCHandler):
            return self._handler.validate_token(access_token)
        return None


# ── Singleton ───────────────────────────────────────────────────────────

_handler_singleton: SSOAuthHandler | None = None
_handler_lock = threading.Lock()


def get_sso_handler() -> SSOAuthHandler:
    """Get or create the SSO auth handler singleton."""
    global _handler_singleton
    if _handler_singleton is None:
        with _handler_lock:
            if _handler_singleton is None:
                _handler_singleton = SSOAuthHandler()
    return _handler_singleton
