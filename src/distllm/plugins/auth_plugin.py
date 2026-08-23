"""Authentication and authorization plugin for the distributed LLM system.

Provides JWT token validation, role-based access control (RBAC), and
per-role rate limiting as a plugin that hooks into the request pipeline.

Integrates with ``ApiKeyStore`` for API key authentication and adds
JWT validation when ``DISTLLM_AUTH_SECRET`` is configured.

Roles (highest to lowest privilege):

* ``admin`` — full access, no rate limits
* ``model-admin`` — manage models (load, unload, configure)
* ``auditor`` — read-only + audit log access
* ``inference-only`` — chat/completions/embeddings only (1000 req/hr)
* ``read-only`` — health, metrics, list models (100 req/hr)

Configuration (environment variables):

* ``DISTLLM_AUTH_ENABLED`` — enable the plugin (default: "1")
* ``DISTLLM_AUTH_SECRET`` — HMAC secret for JWT validation
* ``DISTLLM_AUTH_JWT_AUDIENCE`` — expected JWT audience claim
* ``DISTLLM_AUTH_JWT_ISSUER`` — expected JWT issuer claim
* ``DISTLLM_AUTH_RATELIMIT_ADMIN`` — admin rate limit per hour (default: 0 = unlimited)
* ``DISTLLM_AUTH_RATELIMIT_MODEL_ADMIN`` — model-admin rate limit (default: 5000)
* ``DISTLLM_AUTH_RATELIMIT_AUDITOR`` — auditor rate limit (default: 2000)
* ``DISTLLM_AUTH_RATELIMIT_INFERENCE_ONLY`` — inference rate limit (default: 1000)
* ``DISTLLM_AUTH_RATELIMIT_READ_ONLY`` — read-only rate limit (default: 100)
* ``DISTLLM_AUTH_RATELIMIT_WINDOW`` — window in seconds (default: 3600)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import base64
from typing import Any

from loguru import logger

from distllm.core.plugin_system import PluginBase


# ── Optional PyJWT import ──────────────────────────────────────────────────

try:
    import jwt as _pyjwt

    _HAS_PYJWT = True
except ImportError:
    _pyjwt = None  # type: ignore[assignment]
    _HAS_PYJWT = False


# ── Constants ──────────────────────────────────────────────────────────────

#: All recognised roles in privilege order (highest first).
ROLE_PRIVILEGES: dict[str, int] = {
    "admin": 6,
    "user-admin": 5,
    "model-admin": 4,
    "auditor": 3,
    "inference-only": 2,
    "read-only": 1,
}

#: Map of HTTP methods to the minimum role required for that method.
METHOD_ROLE_MAP: dict[str, str] = {
    "DELETE": "admin",
    "PUT": "model-admin",
    "PATCH": "model-admin",
    "POST": "inference-only",
    "GET": "read-only",
}

#: Paths exempt from authorization checks (already handled by AuthMiddleware).
_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/ready",
    "/live",
    "/metrics",
})

#: Default rate limits per role (requests per window).
_DEFAULT_ROLE_LIMITS: dict[str, int] = {
    "admin": 0,  # 0 = unlimited
    "model-admin": 5000,
    "auditor": 2000,
    "inference-only": 1000,
    "read-only": 100,
}


# ── Sliding window counter ────────────────────────────────────────────────


class _SlidingWindowCounter:
    """Thread-safe sliding-window counter for rate limiting.

    Mirrors the implementation in ``builtin.RateLimitPlugin`` to keep
    the plugin self-contained without cross-module coupling.
    """

    def __init__(self, max_requests: int, window_seconds: float = 3600.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def allow(self) -> bool:
        now = time.time()
        with self._lock:
            self._prune(now)
            if len(self._timestamps) >= self.max_requests:
                return False
            self._timestamps.append(now)
            return True

    def remaining(self) -> int:
        now = time.time()
        with self._lock:
            self._prune(now)
            return max(0, self.max_requests - len(self._timestamps))

    def retry_after(self) -> int:
        """Return seconds until the oldest request in the window expires."""
        now = time.time()
        with self._lock:
            if not self._timestamps:
                return 0
            self._prune(now)
            if not self._timestamps:
                return 0
            return max(1, int(self.window_seconds - (now - self._timestamps[0])))


# ── JWT validation ─────────────────────────────────────────────────────────


def _b64_decode(data: str) -> bytes:
    """Decode a base64url-encoded string, adding padding as needed."""
    padding = 4 - len(data) % 4
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def _validate_jwt_hs256(token: str, secret: str) -> dict[str, Any] | None:
    """Validate an HS256 JWT without PyJWT (pure-Python fallback).

    Returns the decoded payload on success, ``None`` on failure.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}".encode()
        expected = hmac.new(secret.encode(), message, hashlib.sha256).digest()
        actual = _b64_decode(signature_b64)
        if not hmac.compare_digest(expected, actual):
            return None

        # Decode payload
        payload_bytes = _b64_decode(payload_b64)
        payload = json.loads(payload_bytes)

        # Check expiry
        now = time.time()
        if "exp" in payload and payload["exp"] < now:
            logger.warning("JWT token has expired")
            return None

        # Check not-before
        if "nbf" in payload and payload["nbf"] > now:
            logger.warning("JWT token is not yet valid (nbf)")
            return None

        return payload
    except (json.JSONDecodeError, ValueError, KeyError, Exception) as exc:
        logger.debug(f"JWT fallback validation failed: {exc}")
        return None


def validate_jwt(
    token: str,
    secret: str,
    audience: str | None = None,
    issuer: str | None = None,
) -> dict[str, Any] | None:
    """Validate a JWT token and return the payload.

    Uses PyJWT if available for full algorithm support; otherwise falls
    back to a pure-Python HS256 implementation.

    Returns the decoded payload dict on success, ``None`` on failure.
    """
    if _HAS_PYJWT:
        try:
            options: dict[str, Any] = {
                "verify_signature": True,
                "require": ["exp"],
            }
            # Auto-detect: if secret looks like a PEM public key, use RS256;
            # otherwise use HS256 (shared secret).
            _is_pem = secret.strip().startswith("-----BEGIN") if isinstance(secret, str) else False
            _algorithms = ["RS256", "RS384", "RS512"] if _is_pem else ["HS256"]

            payload = _pyjwt.decode(
                token,
                secret,
                algorithms=_algorithms,
                audience=audience or None,
                issuer=issuer or None,
                options=options,
            )
            return payload
        except _pyjwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except _pyjwt.InvalidAudienceError:
            logger.warning("JWT audience mismatch")
            return None
        except _pyjwt.InvalidIssuerError:
            logger.warning("JWT issuer mismatch")
            return None
        except _pyjwt.InvalidTokenError as exc:
            logger.warning(f"Invalid JWT token: {exc}")
            return None

    # Fallback: manual HS256 validation
    payload = _validate_jwt_hs256(token, secret)
    if payload is None:
        return None

    # Manual audience/issuer checks for the fallback path
    if audience and payload.get("aud") and payload["aud"] != audience:
        logger.warning("JWT audience mismatch (fallback)")
        return None
    if issuer and payload.get("iss") and payload["iss"] != issuer:
        logger.warning("JWT issuer mismatch (fallback)")
        return None

    return payload


# ── Auth plugin ────────────────────────────────────────────────────────────


class AuthPlugin(PluginBase):
    """JWT validation, RBAC enforcement, and per-role rate limiting.

    Runs as an ``on_request`` hook after ``AuthMiddleware`` has resolved
    ``api_key_role`` and ``api_key_id`` into the request context.

    When ``DISTLLM_AUTH_SECRET`` is set, the plugin also validates
    ``Authorization: Bearer <jwt>`` tokens carried in the ``_auth_header``
    context field (populated by a custom middleware or proxy).

    Activated when ``DISTLLM_AUTH_ENABLED=1`` (default).
    """

    def name(self) -> str:
        return "auth"

    def version(self) -> str:
        return "1.0.0"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_init(self, context: dict[str, Any]) -> None:
        self._enabled = os.environ.get("DISTLLM_AUTH_ENABLED", "1") == "1"
        if not self._enabled:
            logger.info("AuthPlugin: disabled via DISTLLM_AUTH_ENABLED=0")
            return

        self._jwt_secret = os.environ.get("DISTLLM_AUTH_SECRET", "")
        self._jwt_audience = os.environ.get("DISTLLM_AUTH_JWT_AUDIENCE", "")
        self._jwt_issuer = os.environ.get("DISTLLM_AUTH_JWT_ISSUER", "")

        self._validate_config()
        self._init_rate_limiters()

        jwt_status = "PyJWT" if _HAS_PYJWT else "fallback HS256"
        logger.info(
            f"AuthPlugin: enabled, JWT={'configured' if self._jwt_secret else 'disabled'}, "
            f"validator={jwt_status}, "
            f"rate limits for {len(self._role_limiters)} roles"
        )

    def _validate_config(self) -> None:
        """Log warnings for insecure or missing configuration."""
        if self._jwt_secret and len(self._jwt_secret) < 32:
            logger.warning(
                "AuthPlugin: DISTLLM_AUTH_SECRET is shorter than 32 characters. "
                "Use a strong secret in production."
            )
        if self._jwt_secret and not _HAS_PYJWT:
            logger.info(
                "AuthPlugin: PyJWT not installed; using built-in HS256 fallback. "
                "Install PyJWT for full algorithm support: pip install PyJWT"
            )

    def _init_rate_limiters(self) -> None:
        """Read per-role rate limits from environment variables."""
        try:
            window = int(os.environ.get("DISTLLM_AUTH_RATELIMIT_WINDOW", "3600"))
        except (ValueError, TypeError):
            window = 3600

        self._rate_window = window
        self._role_limiters: dict[str, _SlidingWindowCounter] = {}

        for role, default_limit in _DEFAULT_ROLE_LIMITS.items():
            env_key = f"DISTLLM_AUTH_RATELIMIT_{role.upper().replace('-', '_')}"
            try:
                limit = int(os.environ.get(env_key, str(default_limit)))
            except (ValueError, TypeError):
                limit = default_limit

            if limit <= 0:
                # Unlimited — no counter needed
                continue
            self._role_limiters[role] = _SlidingWindowCounter(limit, window)

    # ── Request hook ──────────────────────────────────────────────────────

    def on_request(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Validate authentication and authorization for the request.

        Returns a rejection dict when access is denied, or ``None`` to
        allow the request to proceed.  The rejection dict follows the
        ``_reject`` convention understood by ``PluginHookMiddleware``.
        """
        if not self._enabled:
            return None

        path = context.get("path", "")
        if path in _EXEMPT_PATHS:
            return None

        # Step 1: Optional JWT validation
        jwt_role = self._validate_jwt_from_context(context)
        if jwt_role is not None:
            # JWT was present and valid — override the API key role
            context["api_key_role"] = jwt_role
            context["auth_method"] = "jwt"

        # Step 2: RBAC enforcement
        rbac_rejection = self._enforce_rbac(context)
        if rbac_rejection is not None:
            return rbac_rejection

        # Step 3: Per-role rate limiting
        rate_rejection = self._enforce_rate_limit(context)
        if rate_rejection is not None:
            return rate_rejection

        return None

    # ── JWT validation ────────────────────────────────────────────────────

    def _validate_jwt_from_context(self, context: dict[str, Any]) -> str | None:
        """Validate JWT if a bearer token is present in the context.

        Returns the resolved role string on success, ``None`` if no JWT
        is present or validation fails.

        The JWT ``role`` claim overrides the API key role.  When the
        ``role`` claim is absent, the API key role is preserved.
        """
        if not self._jwt_secret:
            return None

        auth_header = context.get("_auth_header", "")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]
        if not token:
            return None

        payload = validate_jwt(
            token,
            self._jwt_secret,
            audience=self._jwt_audience or None,
            issuer=self._jwt_issuer or None,
        )
        if payload is None:
            return None

        # JWT is valid — extract the role claim from the token.
        # The caller (on_request) will use this to override api_key_role,
        # giving the JWT role higher authority than the underlying API key.
        jwt_role = payload.get("role", "")
        if jwt_role in ROLE_PRIVILEGES:
            return jwt_role

        # JWT is valid but has no recognized role claim — fall back to
        # the API key role if one exists, otherwise grant minimum access
        # (read-only) rather than denying the request outright.
        # SECURITY: The API key role was already validated by AuthMiddleware
        # and is from a trusted source (not attacker-supplied).
        api_key_role = context.get("api_key_role", "")
        if api_key_role in ROLE_PRIVILEGES:
            return api_key_role

        return "read"

    # ── RBAC enforcement ──────────────────────────────────────────────────

    def _enforce_rbac(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Check whether the caller's role permits the HTTP method.

        Returns a rejection dict if access is denied, ``None`` otherwise.
        """
        role = context.get("api_key_role", "")
        method = context.get("method", "GET").upper()

        if not role:
            return {
                "_reject": {
                    "reason": "No authenticated role. Provide a valid API key or JWT.",
                    "status": 401,
                }
            }

        if role not in ROLE_PRIVILEGES:
            return {
                "_reject": {
                    "reason": f"Unknown role: {role}",
                    "status": 403,
                }
            }

        required = METHOD_ROLE_MAP.get(method, "read-only")
        if not self._role_has_access(role, required):
            return {
                "_reject": {
                    "reason": (
                        f"Role '{role}' is not authorized for {method} requests. "
                        f"Requires '{required}' or higher."
                    ),
                    "status": 403,
                }
            }

        return None

    @staticmethod
    def _role_has_access(actual: str, required: str) -> bool:
        """Return ``True`` if *actual* role satisfies *required* role.

        Uses a privilege-level comparison so that higher roles implicitly
        satisfy lower requirements (e.g. ``admin`` satisfies everything).
        """
        if actual == required:
            return True
        return ROLE_PRIVILEGES.get(actual, 0) >= ROLE_PRIVILEGES.get(required, 0)

    # ── Rate limiting ─────────────────────────────────────────────────────

    def _enforce_rate_limit(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Apply per-role rate limiting.

        Admins are unlimited by default.  Returns a rejection dict when
        the role's quota is exhausted, ``None`` otherwise.
        """
        role = context.get("api_key_role", "read-only")
        limiter = self._role_limiters.get(role)

        if limiter is None:
            # Unlimited role (e.g. admin) — no rate limit
            return None

        if not limiter.allow():
            retry_after = limiter.retry_after()
            return {
                "_reject": {
                    "reason": f"Role '{role}' rate limit exceeded",
                    "status": 429,
                    "retry_after": retry_after,
                }
            }

        return None
