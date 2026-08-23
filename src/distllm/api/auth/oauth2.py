"""Generic OAuth2 token exchange (GitHub, GitLab, etc.)."""

from __future__ import annotations

import json
import os
import time
import urllib.parse

import httpx
from loguru import logger

from .models import SSOUserInfo


class GenericOAuth2Handler:
    """Generic OAuth2 token exchange (GitHub, GitLab, etc.).

    Useful for self-hosted deployments that use an existing
    OAuth2 provider for authentication.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        authorize_url: str,
        token_url: str,
        userinfo_url: str,
        callback_url: str,
        scope: str = "read:user",
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._authorize_url = authorize_url
        self._token_url = token_url
        self._userinfo_url = userinfo_url
        self._callback_url = callback_url
        self._scope = scope
        # OAuth2 state store for CSRF protection
        self._state_store: dict[str, float] = {}
        self._state_ttl = 600.0  # 10 minutes
        # PKCE code verifier store — maps state → code_verifier
        self._pkce_store: dict[str, str] = {}

    def get_login_url(self, state: str = "") -> str:
        """Generate OAuth2 authorization URL with CSRF-protected state and PKCE."""
        state = state or os.urandom(16).hex()
        self._state_store[state] = time.time() + self._state_ttl

        # PKCE: generate verifier + S256 challenge to prevent code interception.
        pkce_verifier = os.urandom(32).hex()
        pkce_challenge = _b.urlsafe_b64encode(_h.sha256(pkce_verifier.encode()).digest()).rstrip(b"=").decode()
        self._pkce_store[state] = pkce_verifier

        params = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._callback_url,
            "scope": self._scope,
            "state": state,
            "code_challenge": pkce_challenge,
            "code_challenge_method": "S256",
        })
        return f"{self._authorize_url}?{params}"

    def handle_callback(self, code: str, expected_state: str = "") -> SSOUserInfo | None:
        """Exchange code for token. Validates state if provided."""
        if expected_state:
            stored_expiry = self._state_store.pop(expected_state, None)
            if stored_expiry is None:
                logger.error("OAuth2 state not found — possible CSRF attack")
                return None
            if time.time() > stored_expiry:
                logger.error("OAuth2 state expired")
                return None
        try:
            # Exchange code for token with PKCE code_verifier
            code_verifier = self._pkce_store.pop(expected_state, "")
            token_data = {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "code": code,
                "redirect_uri": self._callback_url,
            }
            if code_verifier:
                token_data["code_verifier"] = code_verifier
            resp = httpx.post(
                self._token_url,
                data=token_data,
                headers={"Accept": "application/json"},
                timeout=15.0,
            )
            if resp.status_code != 200:
                logger.error(f"OAuth2 token exchange failed: {resp.status_code}")
                return None

            token_data = resp.json()
            access_token = token_data.get("access_token", "")

            # Get user info
            user_resp = httpx.get(
                self._userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
            if user_resp.status_code != 200:
                return None

            user_data = user_resp.json()
            return SSOUserInfo(
                sub=str(user_data.get("id", user_data.get("login", ""))),
                email=user_data.get("email", ""),
                name=user_data.get("name", user_data.get("login", "")),
                roles=user_data.get("roles", []),
                provider="oauth2",
                raw_attributes=user_data,
            )

        except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError, KeyError,
                TypeError, AttributeError) as e:
            logger.error(f"OAuth2 callback failed: {e}")
            return None
