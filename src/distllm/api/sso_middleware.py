"""SSO middleware — enterprise single sign-on for the API pipeline.

Integrates with ``sso_auth.py`` handlers (``OIDCHandler``,
``GenericOAuth2Handler``, ``SAMLHandler``).  Tries local JWT
validation first, then falls through to external OIDC provider
validation, and finally to the existing API-key auth middleware.

Usage
-----
In ``server.py``, at the end of the middleware registration block::

    from distllm.api.sso_middleware import setup_sso

    sso = setup_sso(app)

    # Optionally register additional OIDC providers at startup:
    # from distllm.api.sso_auth import OIDCHandler
    # sso.register_provider("auth0", OIDCHandler(...))

Endpoints added
---------------
- ``POST /v1/auth/token``   — exchange SSO authorisation code for a local JWT
- ``POST /v1/auth/refresh``  — refresh an expiring JWT (rotation model)
- ``POST /v1/auth/revoke``   — revoke a JWT (in-memory blacklist with TTL)

Integration notes
-----------------
``AuthMiddleware`` (in ``middleware.py``) is modified to skip API-key
validation when ``request.state.auth_method == "sso"``, so the two
middlewares compose cleanly.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

try:
    import jwt as pyjwt
except ImportError:  # pragma: no cover
    pyjwt = None  # type: ignore[assignment]

from distllm.api.sso_auth import (
    GenericOAuth2Handler,
    OIDCHandler,
    SAMLHandler,
    SSOUserInfo,
    get_sso_handler,
)


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_JWT_TTL_S = 3600             # 1 hour
DEFAULT_REFRESH_TTL_S = 86_400       # 24 hours
REVOCATION_TTL_S = 3600.0            # 1 hour for blacklist entries
REVOCATION_CLEANUP_INTERVAL_S = 300  # 5 minutes between cleanup passes

AUTH_ROUTES: frozenset[str] = frozenset({
    "/v1/auth/token",
    "/v1/auth/refresh",
    "/v1/auth/revoke",
})

HEALTH_ROUTES: frozenset[str] = frozenset({
    "/health",
    "/ready",
    "/live",
    "/metrics",
})

# Live middleware instance and the deferred handle returned by setup_sso().
# Starlette only constructs SsoMiddleware when it builds the app's
# middleware stack (on the first request), so provider registrations made
# via setup_sso() before that point are buffered in _setup_handle and
# flushed onto the live instance once it exists.
_active: "SsoMiddleware | None" = None
_setup_handle: "_SsoHandle | None" = None


# ── In-memory token blacklist with TTL-based auto-cleanup ──────────────────────

class TokenBlacklist:
    """In-memory JWT blacklist with TTL-based cleanup.

    Thread-safe for single-process use (dict operations are atomic in
    CPython under the GIL).  **Not** shared across processes — for
    multi-worker deployments, use a Redis-backed blacklist instead.
    """

    def __init__(self, ttl_s: float = REVOCATION_TTL_S) -> None:
        self._data: dict[str, float] = {}
        self._ttl_s = ttl_s
        self._lock = __import__("threading").RLock()
        self._last_cleanup = 0.0

    def revoke(self, jti: str) -> None:
        """Add *jti* to the blacklist with an expiry."""
        with self._lock:
            self._data[jti] = time.time() + self._ttl_s

    def is_revoked(self, jti: str) -> bool:
        """Return ``True`` if *jti* is (still) in the blacklist."""
        self._maybe_cleanup()
        with self._lock:
            return jti in self._data

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < REVOCATION_CLEANUP_INTERVAL_S:
            return
        cutoff = now
        with self._lock:
            stale = [k for k, v in self._data.items() if v < cutoff]
            for k in stale:
                del self._data[k]
        self._last_cleanup = now


# ── JWT helper functions ───────────────────────────────────────────────────────

def _get_jwt_secret() -> str:
    """Return the local JWT signing secret from env or generate one.

    The secret is cached in the environment variable after generation
    so that multiple calls within the same process return the same value.
    Restarting the process without ``SSO_JWT_SECRET`` set invalidates all
    existing tokens — set it to a persistent 64-char hex string in prod.
    """
    secret = os.environ.get("SSO_JWT_SECRET")
    if secret:
        return secret
    generated = secrets.token_hex(32)
    os.environ["SSO_JWT_SECRET"] = generated
    logger.warning(
        "SSO_JWT_SECRET not set — generated ephemeral signing key. "
        "All existing SSO tokens will be invalidated on restart. "
        "Set SSO_JWT_SECRET to a persistent 64-char hex string."
    )
    return generated


def _create_jwt(
    user: SSOUserInfo,
    ttl_s: int = DEFAULT_JWT_TTL_S,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed local JWT containing SSO user information.

    The returned token is a signed HS256 JWT with standard claims
    (``sub``, ``iat``, ``exp``, ``jti``) plus custom claims:
    ``email``, ``name``, ``roles``, ``groups``, ``provider``, ``tenant_id``.

    Raises ``RuntimeError`` if PyJWT is not installed.
    """
    if pyjwt is None:
        raise RuntimeError(
            "PyJWT is required for SSO token generation. "
            "Install with: pip install pyjwt"
        )

    now = datetime.now(tz=timezone.utc)
    jti = secrets.token_hex(16)
    payload: dict[str, Any] = {
        "sub": user.sub,
        "email": user.email,
        "name": user.name,
        "roles": list(user.roles),
        "groups": list(user.groups),
        "provider": user.provider,
        "tenant_id": user.raw_attributes.get("tenant_id", "default"),
        "jti": jti,
        "iat": now,
        "exp": now + timedelta(seconds=ttl_s),
    }
    if extra_claims:
        payload.update(extra_claims)

    secret = _get_jwt_secret()
    return pyjwt.encode(payload, secret, algorithm="HS256")


def _decode_jwt(token: str) -> dict[str, Any] | None:
    """Validate and decode a local JWT.

    Returns the decoded payload dict on success, or ``None`` if the
    token is invalid, expired, or tampered with.
    """
    if pyjwt is None:
        return None
    try:
        secret = _get_jwt_secret()
        return pyjwt.decode(token, secret, algorithms=["HS256"])
    except pyjwt.ExpiredSignatureError:
        logger.debug("SSO JWT expired")
        return None
    except pyjwt.InvalidTokenError as e:
        logger.debug("SSO JWT invalid: {}", e)
        return None
    except Exception as e:
        logger.warning("SSO JWT decode error: {}", e)
        return None


# ── Refresh-token store (in-memory) ────────────────────────────────────────────

# SHA-256(token) -> {"user": SSOUserInfo, "expires": float}
_refresh_store: dict[str, dict[str, Any]] = {}


def _store_refresh_token(user: SSOUserInfo) -> str:
    """Generate, store, and return an opaque refresh token."""
    token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    _refresh_store[token_hash] = {
        "user": user,
        "expires": time.time() + DEFAULT_REFRESH_TTL_S,
    }
    return token


def _consume_refresh_token(token: str) -> dict[str, Any] | None:
    """Look up and consume (pop) a refresh token.

    Returns ``{"user": SSOUserInfo, "expires": float}`` on success
    or ``None`` if the token is not found or expired.  Single-use
    semantics enforce token rotation.
    """
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    entry = _refresh_store.pop(token_hash, None)
    if entry is None:
        return None
    if time.time() > entry["expires"]:
        return None
    return entry


def _cleanup_expired_refresh_tokens() -> None:
    """Remove expired entries from ``_refresh_store``."""
    now = time.time()
    stale = [k for k, v in _refresh_store.items() if v["expires"] < now]
    for k in stale:
        del _refresh_store[k]
    if stale:
        logger.debug("Cleaned up {} expired refresh tokens", len(stale))


# ── SsoMiddleware ──────────────────────────────────────────────────────────────

class SsoMiddleware(BaseHTTPMiddleware):
    """Enterprise SSO authentication middleware.

    Tries SSO authentication first using local JWTs and registered
    OIDC providers.  Falls through to the API-key auth middleware
    (``AuthMiddleware``) when no valid SSO token is found.

    The middleware is deliberately a no-op for:

    * Health-check endpoints (``/health``, ``/ready``, etc.)
    * ``OPTIONS`` (CORS preflight) requests
    * Auth endpoints (``/v1/auth/*``)

    Multiple OIDC providers are supported via ``register_provider()``.
    On startup the middleware auto-detects the default provider from
    the ``get_sso_handler()`` singleton (configured through
    ``DISTLLM_SSO_*`` environment variables).
    """

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)
        self._providers: dict[str, OIDCHandler] = {}
        self._blacklist = TokenBlacklist()
        self._auto_detect_providers()
        # NOTE: Starlette passes a wrapped downstream app (an
        # ExceptionMiddleware) to BaseHTTPMiddleware.__init__, so we cannot
        # stash this instance on app.state here. It is stashed lazily on the
        # first dispatch instead (the middleware runs before every route,
        # including the /v1/auth/* routes).
        global _active
        _active = self
        if _setup_handle is not None:
            for name, handler in _setup_handle._pending.items():
                self.register_provider(name, handler)
            _setup_handle._pending.clear()

    # ── Public configuration API ─────────────────────────────────────────────

    def register_provider(self, name: str, handler: OIDCHandler) -> None:
        """Register an OIDC provider for JWT validation.

        Args:
            name:    Provider identifier (used in API requests to ``/v1/auth/token``).
            handler: An ``OIDCHandler`` instance configured for the provider.
        """
        self._providers[name] = handler
        logger.info("SSO: provider '{}' registered (authority={})", name, handler._authority)

    def get_provider(self, name: str) -> OIDCHandler | None:
        """Get a registered OIDC provider by name, or ``None``."""
        return self._providers.get(name)

    @property
    def registered_providers(self) -> dict[str, OIDCHandler]:
        """Return a snapshot of registered providers."""
        return dict(self._providers)

    # ── Token revocation API (used by route handlers) ────────────────────────

    def revoke_jti(self, jti: str) -> None:
        """Revoke a JWT by its unique identifier (``jti`` claim)."""
        self._blacklist.revoke(jti)

    def revoke_token_str(self, token: str) -> None:
        """Revoke a raw JWT string by decoding and extracting its ``jti``.

        If the token cannot be decoded (e.g. it is a non-local token),
        falls back to hash-based revocation using SHA-256 of the token.
        """
        payload = _decode_jwt(token)
        if payload is not None:
            jti = payload.get("jti", "")
            if jti:
                self._blacklist.revoke(jti)
                logger.debug("Token revoked by JTI (jti={})", jti[:16])
                return
        # Fallback: hash-based revocation
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        self._blacklist.revoke(token_hash)
        logger.debug("Token revoked by hash (hash={})", token_hash[:16])

    # ── Provider auto-detection ──────────────────────────────────────────────

    def _auto_detect_providers(self) -> None:
        """Import the default provider from ``get_sso_handler()``.

        This only fires when ``DISTLLM_SSO_PROVIDER`` is set in the
        environment; silently no-ops otherwise.
        """
        try:
            handler = get_sso_handler()
            if not handler.is_enabled:
                return
            inner = handler._handler
            if isinstance(inner, OIDCHandler):
                self._providers["default"] = inner
                logger.info(
                    "SSO: auto-registered default OIDC provider (authority={})",
                    inner._authority,
                )
            elif isinstance(inner, SAMLHandler):
                logger.info("SSO: SAML provider detected (JWT flow unavailable)")
            elif isinstance(inner, GenericOAuth2Handler):
                logger.info("SSO: OAuth2 provider detected")
        except Exception as exc:
            logger.warning("SSO: auto-detection failed: {}", exc)

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        """Attempt SSO auth first; pass through on failure."""
        # Make the live instance reachable by the /v1/auth/* route handlers
        # (request.app is the FastAPI app; its state persists).
        request.app.state._sso_middleware = self  # type: ignore[attr-defined]
        # Skip health / CORS / auth routes
        if request.url.path in HEALTH_ROUTES:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path in AUTH_ROUTES:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer ") or len(auth_header) <= 8:
            return await call_next(request)

        token = auth_header[7:]

        # 1. Local JWT validation (fast path — no network)
        payload = _decode_jwt(token)
        if payload is not None:
            jti = payload.get("jti", "")
            if jti and self._blacklist.is_revoked(jti):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Token has been revoked",
                            "type": "auth_error",
                            "code": "token_revoked",
                        }
                    },
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                )

            _set_state_from_payload(request, payload)
            logger.info(
                "SSO auth ok | provider=local-jwt sub={} ip={} path={}",
                payload.get("sub", "?"),
                request.client.host if request.client else "?",
                request.url.path,
            )
            return await call_next(request)

        # 2. External OIDC provider validation (network call to JWKS endpoint)
        for provider_name, handler in self._providers.items():
            user = handler.validate_token(token)
            if user is not None:
                _set_state_from_user(request, user, provider_name)
                logger.info(
                    "SSO auth ok | provider={} sub={} ip={} path={}",
                    provider_name,
                    user.sub,
                    request.client.host if request.client else "?",
                    request.url.path,
                )
                return await call_next(request)

        # 3. Fall through — AuthMiddleware will try API key auth
        logger.debug(
            "Bearer token not recognised by any SSO provider — "
            "delegating to API-key auth"
        )
        return await call_next(request)


# ── request.state helpers ──────────────────────────────────────────────────────

# Defined as module-level functions so they remain accessible even if
# the middleware instance is garbage-collected (defensive).

def _set_state_from_payload(request: Request, payload: dict[str, Any]) -> None:
    """Populate ``request.state`` from a decoded local JWT payload."""
    user = SSOUserInfo(
        sub=payload.get("sub", ""),
        email=payload.get("email", ""),
        name=payload.get("name", ""),
        roles=payload.get("roles", []),
        groups=payload.get("groups", []),
        provider=payload.get("provider", "sso"),
        raw_attributes=payload,
    )
    request.state.user = user
    request.state.tenant_id = payload.get("tenant_id", "default")
    request.state.auth_method = "sso"
    request.state.api_key_role = user.to_api_key_role
    request.state.api_key_id = f"sso:{user.provider}:{user.sub}"


def _set_state_from_user(
    request: Request,
    user: SSOUserInfo,
    provider_name: str,
) -> None:
    """Populate ``request.state`` from a validated ``SSOUserInfo`` object."""
    raw: dict[str, Any] = user.raw_attributes
    request.state.user = user
    request.state.tenant_id = raw.get("tenant_id", raw.get("tid", "default"))
    request.state.auth_method = "sso"
    request.state.api_key_role = user.to_api_key_role
    request.state.api_key_id = f"sso:{provider_name}:{user.sub}"


# ── Route handlers ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1/auth", tags=["auth"])


async def _get_sso_middleware(request: Request) -> SsoMiddleware:
    """Look up the ``SsoMiddleware`` instance on the app."""
    mw = getattr(request.app.state, "_sso_middleware", None)
    if mw is None:
        raise RuntimeError("SSO middleware not configured")
    return mw  # type: ignore[return-value]


@router.post("/token")
async def auth_token(request: Request) -> JSONResponse:
    """Exchange an SSO authorisation code for a local JWT.

    Request body (JSON)::

        {
            "provider": "default",
            "code":     "authorization_code_from_provider",
            "state":    "optional_csrf_state",
            "redirect_uri": "optional_redirect_uri"
        }

    Returns::

        {
            "access_token":  "<signed HS256 JWT>",
            "token_type":    "bearer",
            "expires_in":    3600,
            "refresh_token": "<opaque 256-bit token>",
            "user": {
                "sub":     "...",
                "email":   "...",
                "name":    "...",
                "provider": "oidc"
            }
        }

    Error responses:
        * 400 — missing/invalid body field
        * 401 — SSO code exchange failed
        * 503 — SSO middleware not configured
    """
    try:
        middleware = await _get_sso_middleware(request)
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "SSO middleware not configured on this server",
                    "type": "server_error",
                    "code": "sso_not_configured",
                }
            },
        )

    body = await request.json()

    provider_name = body.get("provider", "default")
    code = body.get("code", "")
    state = body.get("state", "")

    if not code:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Missing required field: 'code'",
                    "type": "invalid_request",
                    "code": "missing_code",
                }
            },
        )

    if not state:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Missing required field: 'state' (OAuth2 CSRF protection)",
                    "type": "invalid_request",
                    "code": "missing_state",
                }
            },
        )

    handler = middleware.get_provider(provider_name)
    if handler is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Unknown SSO provider: '{provider_name}'",
                    "type": "invalid_request",
                    "code": "unknown_provider",
                }
            },
        )

    _cleanup_expired_refresh_tokens()

    # Exchange the authorisation code for identity information.
    user = handler.handle_callback(code, expected_state=state)
    if user is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "SSO authentication failed — invalid code or state mismatch",
                    "type": "auth_error",
                    "code": "sso_auth_failed",
                }
            },
        )

    # Create local JWT
    try:
        access_token = _create_jwt(user)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "code": "jwt_generation_failed",
                }
            },
        )

    # Issue a refresh token (single-use / rotation model)
    refresh_token = _store_refresh_token(user)

    return JSONResponse(
        content={
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": DEFAULT_JWT_TTL_S,
            "refresh_token": refresh_token,
            "user": {
                "sub": user.sub,
                "email": user.email,
                "name": user.name,
                "provider": user.provider,
            },
        }
    )


@router.post("/refresh")
async def auth_refresh(request: Request) -> JSONResponse:
    """Refresh an expiring JWT using a single-use refresh token.

    The old refresh token is consumed (rotation model).  A new
    access token and a new refresh token are returned.

    Request body (JSON)::

        {
            "refresh_token": "<opaque token from /token response>"
        }

    Returns::

        {
            "access_token":  "<new signed HS256 JWT>",
            "token_type":    "bearer",
            "expires_in":    3600,
            "refresh_token": "<new opaque 256-bit token>"
        }

    Error responses:
        * 400 — missing ``refresh_token`` field
        * 401 — invalid or expired refresh token
    """
    body = await request.json()
    refresh_token = body.get("refresh_token", "")

    if not refresh_token:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Missing required field: 'refresh_token'",
                    "type": "invalid_request",
                    "code": "missing_refresh_token",
                }
            },
        )

    entry = _consume_refresh_token(refresh_token)
    if entry is None:
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Invalid or expired refresh token",
                    "type": "auth_error",
                    "code": "invalid_refresh_token",
                }
            },
        )

    user: SSOUserInfo = entry["user"]

    try:
        access_token = _create_jwt(user)
    except RuntimeError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "code": "jwt_generation_failed",
                }
            },
        )

    new_refresh_token = _store_refresh_token(user)

    return JSONResponse(
        content={
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": DEFAULT_JWT_TTL_S,
            "refresh_token": new_refresh_token,
        }
    )


@router.post("/revoke")
async def auth_revoke(request: Request) -> JSONResponse:
    """Revoke an SSO JWT.

    The token is added to an in-memory blacklist.  Subsequent requests
    carrying this token will be rejected with HTTP 401.

    Idempotent — revoking an already-revoked token returns 200 OK.

    Request body (JSON)::

        {
            "token": "<JWT or opaque token string to revoke>"
        }

    Returns ``{"status": "ok"}`` on success.
    """
    body = await request.json()
    token = body.get("token", "")

    if not token:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Missing required field: 'token'",
                    "type": "invalid_request",
                    "code": "missing_token",
                }
            },
        )

    try:
        middleware = await _get_sso_middleware(request)
        middleware.revoke_token_str(token)
    except RuntimeError:
        logger.warning("Cannot revoke token: SSO middleware not configured")

    return JSONResponse(content={"status": "ok"})


# ── Server integration helper ──────────────────────────────────────────────────

class _SsoHandle:
    """Deferred handle to the live :class:`SsoMiddleware` instance.

    ``setup_sso`` returns one of these because Starlette constructs the
    real middleware only when the app's middleware stack is built (on the
    first request). Provider registrations made before that point are
    buffered and flushed onto the live instance when it comes alive.
    """

    def __init__(self) -> None:
        self._pending: dict[str, OIDCHandler] = {}

    def register_provider(self, name: str, handler: OIDCHandler) -> None:
        if _active is not None:
            _active.register_provider(name, handler)
        else:
            self._pending[name] = handler

    def get_provider(self, name: str) -> OIDCHandler | None:
        if _active is not None:
            return _active.get_provider(name)
        return self._pending.get(name)

    @property
    def registered_providers(self) -> dict[str, OIDCHandler]:
        if _active is not None:
            return _active.registered_providers
        return dict(self._pending)


def setup_sso(app: FastAPI) -> _SsoHandle:
    """Register the SSO middleware and auth routes on a FastAPI application.

    This function:

    1. Adds ``SsoMiddleware`` to the middleware stack (outermost position,
       so it processes requests before every other middleware).
    2. Includes the ``/v1/auth/*`` route router (token exchange, refresh,
       revocation).
    3. Returns a deferred handle for runtime provider registration
       (buffered until Starlette constructs the middleware at first
       request, then forwarded to the live instance).

    Call it **after** all other ``app.add_middleware(...)`` calls (or at
    least after ``AuthMiddleware``).

    The middleware auto-detects the default OIDC provider from
    ``get_sso_handler()`` (configured via ``DISTLLM_SSO_*`` env vars).
    Additional providers can be registered on the returned handle::

        sso = setup_sso(app)
        sso.register_provider("auth0", OIDCHandler(...))
    """
    app.add_middleware(SsoMiddleware)
    app.include_router(router)
    global _setup_handle
    _setup_handle = _SsoHandle()
    return _setup_handle
