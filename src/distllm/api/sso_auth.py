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
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# ─── Shared pooled HTTP client (PERF-01) ────────────────────────────────────
#
# Sync httpx helpers (httpx.get/post) build a fresh Client — plus a fresh TCP
# + TLS handshake — for every request, and block the event loop when invoked
# from async contexts (SsoMiddleware.dispatch, /v1/auth/token). A module-level
# httpx.AsyncClient with an explicit Limits pool reuses keep-alive connections
# across requests, keeping p99 latency flat under concurrent login bursts.
# Sync entry points keep their existing semantics and offload to the async
# path internally, so there is exactly one HTTP code path to audit.
_HTTP_LIMITS = None
_HTTP_TIMEOUT = None


def _get_async_client():
    """Return the process-wide pooled AsyncClient for SSO IdP calls."""
    global _HTTP_LIMITS, _HTTP_TIMEOUT
    import httpx
    if _HTTP_LIMITS is None:
        _HTTP_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=50)
        _HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
    client = getattr(_get_async_client, "_client", None)
    if client is None:
        client = httpx.AsyncClient(limits=_HTTP_LIMITS, timeout=_HTTP_TIMEOUT)
        _get_async_client._client = client
    return client


# ── JWKS cache policy (SEC-A9) ─────────────────────────────────────────────────
#
# Unauthenticated garbage tokens used to trigger a fresh outbound JWKS fetch
# per provider per request (latency DoS + request flood against the IdP).
# Responses are now cached per issuer endpoint with a TTL, failed fetches are
# negatively cached briefly, and concurrent validations share a single
# in-flight request (single-flight).

DEFAULT_JWKS_CACHE_TTL_S = 300.0   # successful JWKS responses live 5 minutes
JWKS_NEGATIVE_CACHE_TTL_S = 60.0   # failed JWKS fetches retry at most 1x/minute


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
        self._nonce_store: dict[str, float] = {}
        self._nonce_ttl = 600.0  # 10 minutes

        # PKCE code verifier store — maps state -> code_verifier.
        # PKCE prevents authorization code interception attacks when the
        # callback is delivered over an untrusted channel.
        self._pkce_store: dict[str, str] = {}

        # SEC-A9: JWKS response cache + single-flight, keyed by the JWKS URL.
        # The cache key is ALWAYS an operator-configured or discovery-derived
        # endpoint — never anything taken from a token (no ``jku``/``x5u``
        # support), so tokens cannot steer where we fetch from.
        self._jwks_cache_ttl_s = DEFAULT_JWKS_CACHE_TTL_S
        self._jwks_cache_lock = threading.Lock()
        self._jwks_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._jwks_inflight: dict[str, threading.Event] = {}
        self._jwks_last_error: dict[str, tuple[float, Exception]] = {}

        self._discover()

    async def async_discover(self) -> None:
        """ASYNC OIDC discovery via the pooled AsyncClient (PERF-01)."""
        try:
            import httpx

            discovery_url = f"{self._authority}/.well-known/openid-configuration"
            resp = await _get_async_client().get(discovery_url, timeout=10.0)
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
        except (httpx.HTTPError, ValueError) as e:
            logger.warning(f"OIDC discovery error: {e}")
            self._set_default_endpoints()

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
            # Network/HTTP/JSON errors only; auth failures must not be masked
            # as "discovery errors". Re-raise anything that isn't I/O related.
            import httpx as _httpx

            if isinstance(e, (_httpx.HTTPError, ValueError, OSError)):
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
        import time
        self._state_store[state] = time.time() + self._nonce_ttl
        # OIDC nonce for replay protection — store the VALUE so the callback
        # can verify the id_token's ``nonce`` claim against it.
        nonce = os.urandom(16).hex()
        self._nonce_store[state] = {
            "nonce": nonce,
            "expires": time.time() + self._nonce_ttl,
        }
        # PKCE: bind a code_verifier to this state so a stolen/intercepted
        # authorization code is unusable without the verifier (RFC 7636).
        # S256 challenge is advertised in the URL; the verifier is stored
        # server-side and presented at the token exchange.
        import base64
        import hashlib

        code_verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode("ascii")
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        pkce_store = getattr(self, "_pkce_store", None)
        if pkce_store is None:
            pkce_store = self._pkce_store = {}
        pkce_store[state] = code_verifier

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

    async def async_handle_callback(
        self, code: str, expected_state: str = "", expected_nonce: str = ""
    ) -> SSOUserInfo | None:
        """ASYNC OIDC callback via the pooled AsyncClient (PERF-01).

        Token/userinfo calls use the shared pooled AsyncClient so they
        never block the event loop and reuse keep-alive connections.
        CSRF state, mandatory ID-token verification (H-13), and nonce
        replay protection are identical to the sync version.
        """
        import time

        # SECURITY: state is mandatory for the authorization-code flow (OAuth2
        # CSRF protection).  A callback without a state cannot be bound to a
        # login the user actually initiated.
        if not expected_state:
            logger.error("OAuth state missing — refusing code exchange")
            return None
        stored_expiry = self._state_store.pop(expected_state, None)
        if stored_expiry is None:
            logger.error("OAuth state not found — possible CSRF attack")
            return None
        if time.time() > stored_expiry:
            logger.error("OAuth state expired")
            return None

        # Caller-supplied nonce (RFC 8252 binding): when present it must
        # match the nonce issued for this state. The stored entry is kept
        # (not popped) — single-use semantics come from the state store,
        # and the id_token's nonce claim is checked against the same entry
        # after signature validation.
        if expected_nonce:
            stored_nonce = self._nonce_store.get(expected_state)
            if isinstance(stored_nonce, dict):
                stored_nonce = stored_nonce.get("nonce")
            if stored_nonce is None or stored_nonce != expected_nonce:
                logger.error("OIDC nonce mismatch — possible authorization-code replay")
                return None
        try:
            import httpx

            client = _get_async_client()

            # Exchange code for tokens (with PKCE code_verifier when the
            # login URL issued one). Fail closed: a state with no bound
            # verifier (already consumed, or a login that never issued
            # PKCE) must not silently downgrade to a verifier-less
            # exchange (SEC-API-04 / RFC 7636; mirrors GenericOAuth2Handler).
            code_verifier = self._pkce_store.pop(expected_state, None)
            if not code_verifier:
                logger.error("OIDC PKCE verifier missing — possible code interception")
                return None
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "redirect_uri": self._callback_url,
            }
            token_data["code_verifier"] = code_verifier
            token_resp = await client.post(
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

            if id_token:
                try:
                    validated = self.validate_token(id_token)
                except Exception as e:
                    logger.error(f"OIDC ID token validation failed with exception: {e}")
                    return None
                if validated is None:
                    logger.error(
                        "OIDC ID token validation failed (signature/JWKS) — "
                        "rejecting authentication"
                    )
                    return None
                logger.debug("OIDC ID token signature verified")

                stored_nonce = self._nonce_store.pop(expected_state, None)
                if stored_nonce is None or time.time() > stored_nonce.get("expires", 0):
                    logger.error("OIDC nonce missing or expired — possible replay attack")
                    return None
                token_nonce = (getattr(validated, "raw_attributes", {}) or {}).get("nonce")
                if not token_nonce or token_nonce != stored_nonce.get("nonce"):
                    logger.error("OIDC nonce mismatch — possible replay attack")
                    return None

            if access_token and self._userinfo_endpoint:
                user_resp = await client.get(
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
            logger.error(f"OIDC async callback handling failed: {e}")
            return None

    def handle_callback(
        self, code: str, expected_state: str = "", expected_nonce: str = ""
    ) -> SSOUserInfo | None:
        """Exchange authorization code for tokens and get user info.

        If *expected_state* is provided, validates it against the stored
        state parameter to prevent OAuth2 CSRF attacks.

        If *expected_nonce* is provided, validates the nonce in the ID token
        to prevent authorization code replay (OIDC nonce binding, :rfc:`8252`).
        """
        import secrets
        import time

        # SECURITY: state is mandatory for the authorization-code flow (OAuth2
        # CSRF protection).  A callback without a state cannot be bound to a
        # login the user actually initiated.
        # SECURITY: state is mandatory for the authorization-code flow (OAuth2
        # CSRF protection).  A callback without a state cannot be bound to a
        # login the user actually initiated.
        if not expected_state:
            logger.error("OAuth state missing — refusing code exchange")
            return None

        stored_expiry = self._state_store.pop(expected_state, None)
        if stored_expiry is None:
            logger.error("OAuth state not found — possible CSRF attack")
            return None
        if time.time() > stored_expiry:
            logger.error("OAuth state expired")
            return None

        # Caller-supplied nonce (RFC 8252 binding): when present it must
        # match the nonce issued for this state. The stored entry is kept
        # (not popped) — single-use semantics come from the state store,
        # and the id_token's nonce claim is checked against the same entry
        # after signature validation.
        if expected_nonce:
            stored_nonce = self._nonce_store.get(expected_state)
            if isinstance(stored_nonce, dict):
                stored_nonce = stored_nonce.get("nonce")
            if stored_nonce is None or stored_nonce != expected_nonce:
                logger.error("OIDC nonce mismatch — possible authorization-code replay")
                return None
        try:
            import httpx

            # Exchange code for tokens (with PKCE code_verifier when the
            # login URL issued one). Fail closed: a state with no bound
            # verifier (already consumed, or a login that never issued
            # PKCE) must not silently downgrade to a verifier-less
            # exchange (SEC-API-04 / RFC 7636; mirrors GenericOAuth2Handler).
            code_verifier = self._pkce_store.pop(expected_state, None)
            if not code_verifier:
                logger.error("OIDC PKCE verifier missing — possible code interception")
                return None
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "redirect_uri": self._callback_url,
            }
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

            # H-13: Verify ID token signature.  SECURITY: this is MANDATORY —
            # an id_token must always be cryptographically validated, and the
            # exchange fails closed when validation cannot be performed (e.g.
            # JWKS unavailable).  Previously the check was skipped when OIDC
            # discovery had failed, letting forged/unsigned claims through.
            if id_token:
                try:
                    validated = self.validate_token(id_token)
                except Exception as e:
                    logger.error(f"OIDC ID token validation failed with exception: {e}")
                    return None
                if validated is None:
                    logger.error(
                        "OIDC ID token validation failed (signature/JWKS) — "
                        "rejecting authentication"
                    )
                    return None
                logger.debug("OIDC ID token signature verified")

                # SECURITY: validate the OIDC nonce claim (replay protection).
                stored_nonce = self._nonce_store.pop(expected_state, None)
                if stored_nonce is None or time.time() > stored_nonce.get("expires", 0):
                    logger.error("OIDC nonce missing or expired — possible replay attack")
                    return None
                token_nonce = (getattr(validated, "raw_attributes", {}) or {}).get("nonce")
                if not token_nonce or token_nonce != stored_nonce.get("nonce"):
                    logger.error("OIDC nonce mismatch — possible replay attack")
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

    def _fetch_jwks_cached(self, jwks_url: str) -> dict[str, Any]:
        """Fetch the provider's JWKS document through the TTL cache.

        Guarantees (SEC-A9):

        * ``jwks_url`` is derived solely from trusted operator config or OIDC
          discovery of that config — callers must never pass a token-derived
          URL.  This handler reads no ``jku``/``x5u`` headers, so tokens can
          never steer this fetch.
        * Repeated validations inside the cache TTL make **zero** network
          calls; a failed fetch is negatively cached for
          ``JWKS_NEGATIVE_CACHE_TTL_S`` so an outage cannot be amplified into
          a per-request retry storm against the IdP.
        * Concurrent validations for the same endpoint share one in-flight
          HTTP request (single-flight via ``threading.Event``).

        Raises on network/parsing failure (after consulting the negative
        cache) — ``validate_token`` catches and rejects the token.
        """
        now = time.time()

        with self._jwks_cache_lock:
            entry = self._jwks_cache.get(jwks_url)
            if entry is not None and entry[0] > now:
                return entry[1]
            # Negative cache: a recent failure means "IdP unreachable / bad
            # response" — fail fast without another outbound request.
            err_entry = self._jwks_last_error.get(jwks_url)
            if err_entry is not None and err_entry[0] > now:
                raise err_entry[1]

        # Single-flight: only one thread fetches per URL; others wait on the
        # same Event and then read the result from the shared cache.
        with self._jwks_cache_lock:
            event = self._jwks_inflight.get(jwks_url)
            if event is None:
                event = threading.Event()
                self._jwks_inflight[jwks_url] = event
                leader = True
            else:
                leader = False

        if not leader:
            # Wait bounded time for the in-flight fetch to finish.
            if not event.wait(timeout=10.0):
                # Leader hung (e.g., DNS black hole). Surface as failure;
                # do NOT issue our own duplicate request.
                raise TimeoutError(
                    f"timed out waiting for in-flight JWKS fetch ({jwks_url})"
                )
            with self._jwks_cache_lock:
                entry = self._jwks_cache.get(jwks_url)
                if entry is not None and entry[0] > time.time():
                    return entry[1]
                err_entry = self._jwks_last_error.get(jwks_url)
                if err_entry is not None:
                    raise err_entry[1]
                raise RuntimeError("JWKS fetch completed but no result recorded")

        try:
            import httpx
            resp = httpx.get(jwks_url, timeout=10.0)
            if resp.status_code != 200:
                raise RuntimeError(f"JWKS fetch failed with status {resp.status_code}")
            jwks_data = resp.json()
            if not isinstance(jwks_data, dict):
                raise ValueError("JWKS document is not a JSON object")
            with self._jwks_cache_lock:
                self._jwks_cache[jwks_url] = (
                    time.time() + self._jwks_cache_ttl_s,
                    jwks_data,
                )
                self._jwks_last_error.pop(jwks_url, None)
            return jwks_data
        except Exception as e:
            with self._jwks_cache_lock:
                self._jwks_last_error[jwks_url] = (
                    time.time() + JWKS_NEGATIVE_CACHE_TTL_S,
                    e,
                )
            raise
        finally:
            with self._jwks_cache_lock:
                self._jwks_inflight.pop(jwks_url, None)
            event.set()

    async def async_validate_token(self, access_token: str) -> SSOUserInfo | None:
        """ASYNC OIDC token validation (PERF-01): JWKS fetch through the
        pooled AsyncClient, identical verification semantics.

        SEC-A9 preserved: the JWKS document is read through the same TTL +
        negative-cache + single-flight ``_fetch_jwks_cached`` path (executed
        off-loop), so repeated bad tokens cannot fan out one outbound request
        each.  Only a cache miss performs an HTTP fetch.
        """
        jwks_url = self._jwks_url or self._discovered_jwks_url
        if not jwks_url:
            logger.warning("No JWKS URL configured — cannot validate token locally")
            return None
        try:
            import asyncio

            import jwt  # PyJWT

            # Cache hit is instant; a miss runs the blocking (single-flight)
            # fetch in a worker thread so the event loop never stalls.
            jwks_data = await asyncio.to_thread(self._fetch_jwks_cached, jwks_url)
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
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None

    def validate_token(self, access_token: str) -> SSOUserInfo | None:
        """Validate an OIDC access token using JWKS."""
        import jwt  # PyJWT

        jwks_url = self._jwks_url or self._discovered_jwks_url
        if not jwks_url:
            # Fail-closed: without a JWKS URL the signature cannot be
            # cryptographically verified, so reject — unless the operator
            # explicitly opted into trusting unverified id_tokens (deployments
            # that intentionally trust the IdP token endpoint over TLS).
            allow = getattr(self, "_allow_unverified_id_token", False)
            if not allow:
                logger.warning(
                    "No JWKS URL configured — cannot validate token locally; "
                    "rejecting (fail-closed)"
                )
                return None
            logger.warning(
                "No JWKS URL configured — accepting unverified token "
                "(allow_unverified_id_token=True)"
            )
            payload = jwt.decode(
                access_token, options={"verify_signature": False}
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

        try:
            import jwt  # PyJWT

            # SEC-A9: fetch through the TTL + single-flight cache. The URL is
            # trusted config/discovery output only — never token-controlled.
            jwks_data = self._fetch_jwks_cached(jwks_url)

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
        # PKCE code verifier store — maps state -> code_verifier.
        # PKCE prevents authorization code interception attacks when the
        # redirect is intercepted (RFC 7636). S256 challenge is advertised
        # in get_login_url; the verifier is bound to the state and must be
        # presented at the token exchange.
        self._pkce_store: dict[str, str] = {}

    def get_login_url(self, state: str = "") -> str:
        """Generate OAuth2 authorization URL with CSRF-protected state and PKCE."""
        state = state or os.urandom(16).hex()
        import base64
        import hashlib

        self._state_store[state] = time.time() + self._state_ttl

        # PKCE: generate code_verifier (43-128 chars per RFC 7636) and the
        # S256 challenge. The verifier is stored server-side, keyed by state.
        code_verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode("ascii")
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        self._pkce_store[state] = code_verifier

        params = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._callback_url,
            "scope": self._scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        })
        return f"{self._authorize_url}?{params}"

    async def async_handle_callback(
        self, code: str, expected_state: str = ""
    ) -> SSOUserInfo | None:
        """ASYNC OAuth2 callback via the pooled AsyncClient (PERF-01)."""
        import time

        if not expected_state:
            logger.error("OAuth2 state missing — refusing code exchange")
            return None
        stored_expiry = self._state_store.pop(expected_state, None)
        if stored_expiry is None:
            logger.error("OAuth2 state not found — possible CSRF attack")
            return None
        if time.time() > stored_expiry:
            logger.error("OAuth2 state expired")
            return None
        code_verifier = self._pkce_store.pop(expected_state, None)
        if not code_verifier:
            logger.error("OAuth2 PKCE verifier missing — possible code interception")
            return None
        try:
            import httpx

            client = _get_async_client()
            resp = await client.post(
                self._token_url,
                data={
                    "grant_type": "authorization_code",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._callback_url,
                    "code_verifier": code_verifier,
                },
                headers={"Accept": "application/json"},
                timeout=15.0,
            )
            if resp.status_code != 200:
                logger.error(f"OAuth2 token exchange failed: {resp.status_code}")
                return None
            token_data_resp = resp.json()
            access_token = token_data_resp.get("access_token", "")
            user_resp = await client.get(
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
            logger.error(f"OAuth2 async callback failed: {e}")
            return None

    def handle_callback(self, code: str, expected_state: str = "") -> SSOUserInfo | None:
        """Exchange code for token. Requires non-empty state and validates PKCE.

        SECURITY (SEC-API-04): state is mandatory — a callback with an
        empty or missing state cannot be bound to a login the user
        initiated and is rejected. The state must have been issued by
        :meth:`get_login_url` (and must not have expired), and the PKCE
        code_verifier bound to that state must be present and is sent to
        the token endpoint. Both are single-use: state and verifier are
        consumed (popped) on first use, so replays are rejected.
        """
        import secrets
        import time

        # SECURITY: state is mandatory for the authorization-code flow
        # (OAuth2 CSRF protection).  A callback without a state cannot be
        # bound to a login the user actually initiated.
        if not expected_state:
            logger.error("OAuth2 state missing — refusing code exchange")
            return None

        stored_expiry = self._state_store.pop(expected_state, None)
        if stored_expiry is None:
            logger.error("OAuth2 state not found — possible CSRF attack")
            return None
        if time.time() > stored_expiry:
            logger.error("OAuth2 state expired")
            return None

        # SECURITY: PKCE is mandatory. The verifier bound to this state at
        # get_login_url() must exist; if the login never issued a verifier
        # (e.g. a forged/forged-state attempt), refuse the exchange.
        code_verifier = self._pkce_store.pop(expected_state, None)
        if not code_verifier:
            logger.error("OAuth2 PKCE verifier missing — possible code interception")
            return None

        try:
            import httpx

            # Exchange code for token (with PKCE code_verifier)
            resp = httpx.post(
                self._token_url,
                data={
                    "grant_type": "authorization_code",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "code": code,
                    "redirect_uri": self._callback_url,
                    "code_verifier": code_verifier,
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

    async def async_handle_callback(self, code: str, state: str = "") -> SSOUserInfo | None:
        """ASYNC callback via the provider's pooled AsyncClient (PERF-01).

        Prefers the provider's native ``async_handle_callback`` so the
        event loop is never blocked and connections are pooled; falls
        back to a worker thread only for providers without one (SAML).
        """
        if self._handler is None:
            return None
        native = getattr(self._handler, "async_handle_callback", None)
        if native is not None:
            return await native(code, expected_state=state)
        import asyncio

        return await asyncio.to_thread(self.handle_callback, code, state)

    async def async_validate_token(self, access_token: str) -> SSOUserInfo | None:
        """ASYNC token validation (PERF-01): revocation check, then the
        provider's native async path, else a worker-thread fallback."""
        import hashlib

        token_hash = hashlib.sha256(access_token.encode()).hexdigest()
        self._cleanup_revoked()
        if token_hash in self._revoked_tokens:
            return None
        native = getattr(self._handler, "async_validate_token", None)
        if native is not None:
            return await native(access_token)
        import asyncio

        return await asyncio.to_thread(self.validate_token, access_token)

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
