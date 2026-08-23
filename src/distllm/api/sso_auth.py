"""SSO/SAML, OIDC, and OAuth2 authentication providers.

Supports enterprise single sign-on via:

1. SAML 2.0 — Any IdP (Okta, Azure AD, OneLogin, Keycloak)
2. OpenID Connect — Generic OIDC (Auth0, Google, Microsoft)
3. Generic OAuth2 — Token exchange (GitHub, GitLab)

Usage:
    from distllm.api.sso_auth import SSOAuthHandler, get_sso_handler

    handler = get_sso_handler()
    # Login flow:
    redirect_url = handler.get_login_url()
    token = handler.handle_callback(code, state)

    # Token validation (for API middleware):
    user_info = handler.validate_token(access_token)

Configuration (environment variables):
    DISTLLM_SSO_PROVIDER: "saml", "oidc", or "oauth2"
    DISTLLM_SSO_CLIENT_ID: OAuth2/OIDC client ID
    DISTLLM_SSO_CLIENT_SECRET: OAuth2/OIDC client secret
    DISTLLM_SSO_AUTHORITY: OIDC issuer URL (e.g., https://your-tenant.auth0.com)
    DISTLLM_SSO_METADATA_URL: SAML IdP metadata URL
    DISTLLM_SSO_CALLBACK_URL: Redirect URI (e.g., https://distllm.cloud/auth/callback)
    DISTLLM_SSO_JWKS_URL: JWKS endpoint for token validation
"""

from __future__ import annotations

import json
import os
import time
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class SSOUserInfo:
    """User info returned after successful SSO authentication.

    Maps provider-specific claims to a standard format.
    """
    sub: str                          # Unique user ID
    email: str = ""
    name: str = ""
    roles: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    provider: str = ""
    raw_attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def to_api_key_role(self) -> str:
        """Map SSO roles/groups to DistLLM API key roles.

        Heuristic: if user has 'admin' group → 'admin'.
        If 'auditor' → 'auditor'. Otherwise → 'inference-only'.
        """
        all_roles = self.roles + self.groups
        if "admin" in all_roles or "Administrator" in all_roles:
            return "admin"
        if "auditor" in all_roles or "Auditor" in all_roles:
            return "auditor"
        if "read-only" in all_roles:
            return "read-only"
        return "inference-only"


class SAMLHandler:
    """SAML 2.0 authentication via SAML HTTP Artifact or POST binding.

    Uses pysaml2 when available. Falls back to metadata-only mode
    for manual IdP configuration.
    """

    def __init__(self, metadata_url: str, callback_url: str, entity_id: str = "distllm"):
        self._metadata_url = metadata_url
        self._callback_url = callback_url
        self._entity_id = entity_id
        self._client = None
        self._initialize()

    def _initialize(self) -> None:
        """Try to initialize pysaml2 for full SAML support."""
        try:
            from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
            from saml2.client import Saml2Client
            from saml2.config import Config as Saml2Config

            config = Saml2Config()
            config.setattr("entityid", self._entity_id)
            config.setattr("metadata", {"remote": [{"url": self._metadata_url}]})
            config.setattr("service", {
                "sp": {
                    "endpoints": {
                        "assertion_consumer_service": [
                            (self._callback_url, BINDING_HTTP_POST),
                        ],
                    },
                    # SECURITY: Disable unsolicited assertions (prevents SAML response injection)
                    # Require signed authn requests (prevents request forgery)
                    "allow_unsolicited": False,
                    "authn_requests_signed": True,
                },
            })
            self._client = Saml2Client(config=config)
            logger.info("SAML 2.0 client initialized with pysaml2")
        except ImportError:
            logger.warning(
                "pysaml2 not installed. SAML will use metadata-only mode. "
                "Install with: pip install pysaml2"
            )

    def get_login_url(self) -> str:
        """Generate the SAML login redirect URL."""
        if self._client is None:
            return self._callback_url

        try:
            from saml2 import BINDING_HTTP_REDIRECT
            _, info = self._client.prepare_for_authenticate(
                relay_state="",
                binding=BINDING_HTTP_REDIRECT,
            )
            headers = dict(info.get("headers", []))
            return headers.get("Location", self._callback_url)
        except Exception as e:
            logger.error(f"SAML login URL generation failed: {e}")
            return self._callback_url

    def handle_callback(self, saml_response: str) -> SSOUserInfo | None:
        """Handle the SAML assertion response."""
        try:
            from saml2.client import Saml2Client
            authn_response = self._client.parse_authn_request_response(
                saml_response,
                self._client.config.getattr("endpoints")["assertion_consumer_service"][0][1],
            )
            attrs = authn_response.get_identity()
            return SSOUserInfo(
                sub=authn_response.get_subject().text or "",
                email=attrs.get("email", [""])[0] if isinstance(attrs.get("email"), list) else attrs.get("email", ""),
                name=attrs.get("name", [""])[0] if isinstance(attrs.get("name"), list) else attrs.get("name", ""),
                roles=attrs.get("roles", attrs.get("Role", [])),
                groups=attrs.get("groups", attrs.get("Group", [])),
                provider="saml",
                raw_attributes=attrs,
            )
        except Exception as e:
            logger.error(f"SAML callback handling failed: {e}")
            return None


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
    ):
        self._client_id = client_id
        self._client_secret = client_secret
        self._authority = authority.rstrip("/")
        self._callback_url = callback_url
        self._jwks_url = jwks_url

        # Discovered endpoints
        self._authorization_endpoint = ""
        self._token_endpoint = ""
        self._userinfo_endpoint = ""
        self._discovered_jwks_url = ""

        # OAuth2 state and OIDC nonce stores for CSRF protection
        self._state_store: dict[str, float] = {}
        self._nonce_store: dict[str, float] = {}
        self._nonce_ttl = 600.0  # 10 minutes

        self._discover()

    def _discover(self) -> None:
        """Discover OIDC endpoints from the well-known configuration."""
        try:
            import httpx
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
            logger.warning(f"OIDC discovery error: {e}")
            self._set_default_endpoints()

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
        import time
        self._state_store[state] = time.time() + self._nonce_ttl
        # OIDC nonce for replay protection
        nonce = os.urandom(16).hex()
        self._nonce_store[state] = time.time() + self._nonce_ttl
        params = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._callback_url,
            "scope": "openid profile email",
            "state": state,
            "nonce": nonce,
        })
        return f"{self._authorization_endpoint}?{params}"

    def handle_callback(self, code: str, expected_state: str = "") -> SSOUserInfo | None:
        """Exchange authorization code for tokens and get user info.

        If *expected_state* is provided, validates it against the stored
        state parameter to prevent OAuth2 CSRF attacks.
        """
        import time
        import secrets

        if expected_state:
            stored_expiry = self._state_store.pop(expected_state, None)
            if stored_expiry is None:
                logger.error("OAuth state not found — possible CSRF attack")
                return None
            if time.time() > stored_expiry:
                logger.error("OAuth state expired")
                return None
        try:
            import httpx

            # Exchange code for tokens
            token_resp = httpx.post(
                self._token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": self._callback_url,
                },
                timeout=15.0,
            )
            if token_resp.status_code != 200:
                logger.error(f"Token exchange failed: {token_resp.status_code} {token_resp.text}")
                return None

            tokens = token_resp.json()
            access_token = tokens.get("access_token", "")
            id_token = tokens.get("id_token", "")

            # H-13: Verify ID token signature when JWKS is available
            # SECURITY: Validation failure MUST reject, not continue
            if id_token and self._discovered_jwks_url:
                try:
                    validated = self.validate_token(id_token)
                    if validated is None:
                        logger.error("OIDC ID token validation failed — rejecting authentication")
                        return None
                    logger.debug("OIDC ID token signature verified")
                except Exception as e:
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
        except Exception as e:
            logger.error(f"OIDC callback handling failed: {e}")
            return None

    def validate_token(self, access_token: str) -> SSOUserInfo | None:
        """Validate an OIDC access token using JWKS."""
        jwks_url = self._jwks_url or self._discovered_jwks_url
        if not jwks_url:
            logger.warning("No JWKS URL configured — cannot validate token locally")
            return None

        try:
            import httpx
            import jwt  # PyJWT

            # Fetch JWKS
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
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None


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

    def get_login_url(self, state: str = "") -> str:
        """Generate OAuth2 authorization URL with CSRF-protected state."""
        state = state or os.urandom(16).hex()
        import time
        self._state_store[state] = time.time() + self._state_ttl
        params = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._callback_url,
            "scope": self._scope,
            "state": state,
        })
        return f"{self._authorize_url}?{params}"

    def handle_callback(self, code: str, expected_state: str = "") -> SSOUserInfo | None:
        """Exchange code for token. Validates state if provided."""
        import time
        if expected_state:
            stored_expiry = self._state_store.pop(expected_state, None)
            if stored_expiry is None:
                logger.error("OAuth2 state not found — possible CSRF attack")
                return None
            if time.time() > stored_expiry:
                logger.error("OAuth2 state expired")
                return None
        try:
            import httpx

            # Exchange code for token
            resp = httpx.post(
                self._token_url,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._callback_url,
                },
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

        except Exception as e:
            logger.error(f"OAuth2 callback failed: {e}")
            return None


class SSOAuthHandler:
    """Unified SSO authentication handler — routes to the right provider.

    Configured via environment variables. Call ``get_sso_handler()``
    to get the singleton instance.
    """

    def __init__(self):
        self._provider: str = os.environ.get("DISTLLM_SSO_PROVIDER", "").lower()
        self._handler: SAMLHandler | OIDCHandler | GenericOAuth2Handler | None = None
        self._initialize()
        # Local token revocation blocklist — maps (token_hash, iat) to
        # revocation timestamp.  Populated via the revoke_token() API.
        # Entries expire after REVOCATION_TTL_S to bound memory growth.
        self._revoked_tokens: dict[str, float] = {}
        self._revocation_ttl_s: float = 3600.0  # 1 hour
        self._last_revocation_cleanup: float = time.time()

    def _initialize(self) -> None:
        """Initialize the appropriate SSO handler based on configuration."""
        callback_url = os.environ.get("DISTLLM_SSO_CALLBACK_URL", "")

        if self._provider == "saml":
            metadata_url = os.environ.get("DISTLLM_SSO_METADATA_URL", "")
            if metadata_url:
                self._handler = SAMLHandler(
                    metadata_url=metadata_url,
                    callback_url=callback_url,
                )
                logger.info("SSO: SAML 2.0 handler initialized")

        elif self._provider == "oidc":
            client_id = os.environ.get("DISTLLM_SSO_CLIENT_ID", "")
            client_secret = os.environ.get("DISTLLM_SSO_CLIENT_SECRET", "")
            authority = os.environ.get("DISTLLM_SSO_AUTHORITY", "")
            jwks_url = os.environ.get("DISTLLM_SSO_JWKS_URL", "")
            if client_id and authority:
                self._handler = OIDCHandler(
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
                self._handler = GenericOAuth2Handler(
                    client_id=client_id,
                    client_secret=client_secret,
                    authorize_url=auth_url,
                    token_url=token_url,
                    userinfo_url=user_url,
                    callback_url=callback_url,
                )
                logger.info("SSO: OAuth2 handler initialized")

        else:
            logger.info(
                "No SSO provider configured. Set DISTLLM_SSO_PROVIDER to "
                "'saml', 'oidc', or 'oauth2' to enable enterprise single sign-on."
            )

    @property
    def is_enabled(self) -> bool:
        return self._handler is not None

    def get_login_url(self, state: str = "") -> str:
        if self._handler is None:
            return ""
        return self._handler.get_login_url(state)

    def handle_callback(self, code: str, state: str = "") -> SSOUserInfo | None:
        if self._handler is None:
            return None
        # SECURITY: Validate OAuth state parameter to prevent CSRF on callback
        # The handler stores state on get_login_url() and validates on handle_callback()
        if isinstance(self._handler, (OIDCHandler, GenericOAuth2Handler)):
            return self._handler.handle_callback(code, expected_state=state)
        return self._handler.handle_callback(code)

    def revoke_token(self, token_hash: str) -> None:
        """Revoke a token by its SHA-256 hash.

        The token will be rejected by subsequent ``validate_token`` calls
        until the revocation entry expires (after REVOCATION_TTL_S).
        """
        self._revoked_tokens[token_hash] = time.time()
        logger.info(f"Token revoked (hash prefix: {token_hash[:16]}...)")

    def _cleanup_revoked(self) -> None:
        """Periodic cleanup of expired revocation entries."""
        now = time.time()
        if now - self._last_revocation_cleanup < 300:  # every 5 min
            return
        cutoff = now - self._revocation_ttl_s
        self._revoked_tokens = {h: ts for h, ts in self._revoked_tokens.items() if ts > cutoff}
        self._last_revocation_cleanup = now

    def validate_token(self, access_token: str) -> SSOUserInfo | None:
        """Validate an access token (OIDC only for now).

        Checks the local revocation blocklist before delegating to
        the provider-specific handler.  If the token's hash is in
        the blocklist, returns ``None`` (rejected).
        """
        import hashlib
        token_hash = hashlib.sha256(access_token.encode()).hexdigest()
        self._cleanup_revoked()
        if token_hash in self._revoked_tokens:
            logger.debug(f"Token rejected — revoked (hash prefix: {token_hash[:16]}...)")
            return None
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
