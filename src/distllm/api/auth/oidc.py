"""OpenID Connect authentication provider.

Supports any OIDC-compliant IdP (Auth0, Okta OIDC, Azure AD, Google).
Uses the OIDC discovery URL to auto-configure endpoints.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse

import httpx
import jwt
import hashlib as _hashlib
import base64 as _b64
from jwt.exceptions import InvalidTokenError
from loguru import logger

from .models import SSOUserInfo


class OIDCHandler:
    """OpenID Connect authentication (Auth0, Okta OIDC, Azure AD, Google).

    Uses the OIDC discovery URL to auto-configure endpoints.
    Falls back to manual configuration if discovery fails.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        authority: str,
        callback_url: str,
        jwks_url: str | None = None,
        allow_unverified_id_token: bool = False,
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._authority = authority.rstrip("/")
        self._callback_url = callback_url
        self._jwks_url = jwks_url
        # Fail-closed by default: if the id_token signature cannot be
        # cryptographically verified (no JWKS discovered) we reject the
        # authentication rather than trusting an unverified subject. Opt-in
        # only for deployments that intentionally trust the IdP token
        # endpoint over authenticated TLS.
        self._allow_unverified_id_token = allow_unverified_id_token

        # Discovered endpoints
        self._authorization_endpoint = ""
        self._token_endpoint = ""
        self._userinfo_endpoint = ""
        self._discovered_jwks_url = ""

        # OAuth2 state and OIDC nonce stores for CSRF protection
        self._state_store: dict[str, float] = {}
        self._nonce_store: dict[str, str] = {}
        self._nonce_ttl = 600.0  # 10 minutes

        # PKCE code verifier store — maps state → code_verifier.
        # PKCE prevents authorization code interception attacks when the
        # OAuth2 callback is delivered over an untrusted channel.
        self._pkce_store: dict[str, str] = {}
        self._discover()

    def _discover(self) -> None:
        """Discover OIDC endpoints from the well-known configuration."""
        try:
            discovery_url = f"{self._authority}/.well-known/openid-configuration"
            resp = httpx.get(discovery_url, timeout=10.0)
            if resp.status_code == 200:
                config = resp.json()
                self._authorization_endpoint = config.get("authorization_endpoint", "")
                self._token_endpoint = config.get("token_endpoint", "")
                self._userinfo_endpoint = config.get("userinfo_endpoint", "")
                self._discovered_jwks_url = config.get("jwks_uri", "")
                logger.info(f"OIDC endpoints discovered from {discovery_url}")
            else:
                logger.warning(f"OIDC discovery failed: {resp.status_code}")
                self._set_default_endpoints()
        except ImportError:
            logger.warning("httpx not installed, using default OIDC endpoints")
            self._set_default_endpoints()
        except Exception as e:
            # Network/HTTP/JSON errors only; auth failures must not be masked
            # as "discovery errors". Re-raise anything that isn't I/O related.
            if isinstance(e, (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError)):
                logger.warning(f"OIDC discovery error: {e}")
                self._set_default_endpoints()
            else:
                raise

    def _set_default_endpoints(self) -> None:
        """Set default OIDC endpoint paths based on the authority URL."""
        base = self._authority
        self._authorization_endpoint = f"{base}/authorize"
        self._token_endpoint = f"{base}/oauth/token"
        self._userinfo_endpoint = f"{base}/userinfo"

    def get_login_url(self, state: str = "") -> str:
        """Generate the OIDC authorization URL with CSRF-protected state and nonce.

        The state parameter is stored server-side and validated on callback
        to prevent CSRF attacks on the OAuth2 redirect flow.
        """
        state = state or os.urandom(16).hex()
        self._state_store[state] = time.time() + self._nonce_ttl
        # OIDC nonce for replay protection — store the actual nonce (not just
        # an expiry timestamp) so the callback can bind the ID token to the
        # authorization request (RFC 8252).
        nonce = os.urandom(16).hex()
        self._nonce_store[state] = nonce

        # PKCE: generate code_verifier and code_challenge (S256) to prevent
        # authorization code interception attacks when the callback travels
        # over an untrusted channel.
        pkce_verifier = os.urandom(32).hex()
        pkce_challenge = _hashlib.sha256(pkce_verifier.encode()).digest()
        code_challenge = _b64.urlsafe_b64encode(pkce_challenge).rstrip(b"=").decode()
        self._pkce_store[state] = pkce_verifier

        params = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._callback_url,
            "scope": "openid profile email",
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        })
        return f"{self._authorization_endpoint}?{params}"

    def handle_callback(self, code: str, expected_state: str = "", expected_nonce: str = "") -> SSOUserInfo | None:
        """Exchange authorization code for tokens and get user info.

        If *expected_state* is provided, validates it against the stored
        state parameter to prevent OAuth2 CSRF attacks.

        If *expected_nonce* is provided, validates the nonce in the ID token
        to prevent authorization code replay (OIDC nonce binding, :rfc:`8252`).
        """

        if expected_state:
            stored_expiry = self._state_store.pop(expected_state, None)
            if stored_expiry is None:
                logger.error("OAuth state not found — possible CSRF attack")
                return None
            if time.time() > stored_expiry:
                logger.error("OAuth state expired")
                return None

        # Validate the OIDC nonce (single-use, bound to the state) to prevent
        # authorization-code replay.
        if expected_nonce:
            stored_nonce = self._nonce_store.pop(expected_state, None)
            if stored_nonce is None or stored_nonce != expected_nonce:
                logger.error("OIDC nonce mismatch — possible authorization-code replay")
                return None
        try:
            import httpx

            # Exchange code for tokens with PKCE code_verifier
            code_verifier = self._pkce_store.pop(expected_state, "")
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "redirect_uri": self._callback_url,
            }
            if code_verifier:
                token_data["code_verifier"] = code_verifier
            token_resp = httpx.post(
                self._token_endpoint,
                data=token_data,
                timeout=15.0,
            )
            if token_resp.status_code != 200:
                logger.error(f"Token exchange failed: {token_resp.status_code} {token_resp.text}")
                return None

            tokens = token_resp.json()
            access_token = tokens.get("access_token", "")
            id_token = tokens.get("id_token", "")

            # Validate OIDC nonce if provided — prevents authorization code replay.
            if expected_nonce and id_token:
                import jwt as pyjwt
                try:
                    decoded = pyjwt.decode(id_token, options={"verify_signature": False})
                    received_nonce = decoded.get("nonce", "")
                    if received_nonce != expected_nonce:
                        logger.error("OIDC nonce mismatch — possible authorization code replay attack")
                        return None
                    logger.debug("OIDC nonce validated")
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    logger.error(f"OIDC nonce validation failed: {exc}")
                    return None

            # Fail-closed: if the id_token signature cannot be cryptographically
            # verified (no JWKS discovered) and unverified tokens are not
            # explicitly allowed, reject the authentication. Trusting an
            # unverified token would let an attacker impersonate any subject.
            if (
                id_token
                and not self._discovered_jwks_url
                and not self._allow_unverified_id_token
            ):
                logger.error(
                    "OIDC id_token signature cannot be verified (JWKS not "
                    "discovered) and unverified tokens are not allowed — "
                    "rejecting authentication"
                )
                return None

            # Verify ID token signature when JWKS is available
            if id_token and self._discovered_jwks_url:
                try:
                    validated = self.validate_token(id_token)
                    if validated is None:
                        logger.error("OIDC ID token validation failed — rejecting authentication")
                        return None
                    logger.debug("OIDC ID token signature verified")
                except InvalidTokenError as e:
                    # Malformed/corrupt/forged id_token: never swallow as a
                    # generic callback error — reject and surface the specific
                    # signature/verification failure.
                    logger.error(f"OIDC ID token validation failed with exception: {e}")
                    return None

            # Get user info
            if access_token and self._userinfo_endpoint:
                user_resp = httpx.get(
                    self._userinfo_endpoint,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10.0,
                )
                if user_resp.status_code == 200:
                    user_data = user_resp.json()
                    return SSOUserInfo(
                        sub=user_data.get("sub", ""),
                        email=user_data.get("email", ""),
                        name=user_data.get("name", ""),
                        roles=user_data.get("roles", []),
                        groups=user_data.get("groups", user_data.get("groups_v2", [])),
                        provider="oidc",
                        raw_attributes=user_data,
                    )

            return SSOUserInfo(sub=tokens.get("sub", ""), provider="oidc")

        except ImportError:
            logger.error("httpx required for OIDC callback handling")
            return None
        except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError, KeyError,
                TypeError, AttributeError, InvalidTokenError) as e:
            logger.error(f"OIDC callback handling failed: {e}")
            return None

    def validate_token(self, access_token: str) -> SSOUserInfo | None:
        """Validate an OIDC access token using JWKS."""

        jwks_url = self._jwks_url or self._discovered_jwks_url
        if not jwks_url:
            logger.warning("No JWKS URL configured — cannot validate token locally")
            return None

        try:
            jwks_url = self._jwks_url or self._discovered_jwks_url
            jwks_resp = httpx.get(jwks_url, timeout=10.0)
            if jwks_resp.status_code != 200:
                return None

            jwks_data = jwks_resp.json()
            # Find the signing key
            unverified_header = jwt.get_unverified_header(access_token)
            kid = unverified_header.get("kid", "")

            public_key = None
            for key_data in jwks_data.get("keys", []):
                if key_data.get("kid") == kid:
                    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(key_data))
                    break

            if public_key is None:
                logger.warning("No matching JWK found for token kid")
                return None

            # Validate and decode
            payload = jwt.decode(
                access_token,
                public_key,
                algorithms=["RS256", "RS384", "RS512"],
                audience=self._client_id,
            )

            return SSOUserInfo(
                sub=payload.get("sub", ""),
                email=payload.get("email", ""),
                name=payload.get("name", ""),
                roles=payload.get("roles", []),
                groups=payload.get("groups", []),
                provider="oidc",
                raw_attributes=payload,
            )

        except ImportError:
            logger.error("PyJWT and httpx required for token validation")
            return None
        except (httpx.HTTPError, json.JSONDecodeError, OSError, ValueError, KeyError,
                TypeError, AttributeError, InvalidTokenError) as e:
            logger.error(f"Token validation failed: {e}")
            return None
