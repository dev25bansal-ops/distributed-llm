"""CSRF protection middleware for the API layer.

Validates Origin and Referer headers on all non-GET, non-OPTIONS requests
to prevent Cross-Site Request Forgery attacks.

Design note (why Origin/Referer instead of synchronizer tokens): DistLLM's
API is authenticated exclusively via headers the browser never attaches
automatically (``Authorization: Bearer …``, ``X-API-Key``, ``X-Cluster-Key``),
and no session cookies are issued.  A cross-site form POST therefore cannot
carry credentials at all; the residual risk is credential-bearing requests
fired from same-browser JS, which modern browsers always tag with a truthful
``Origin`` header on cross-origin POSTs.  Validating Origin/Referer when
present is the OWASP-recommended secondary control for this architecture;
requests with neither header are allowed (non-browser clients such as curl,
SDKs, and inter-node traffic legitimately omit them).

Uses the Same-Origin check: if the Origin or Referer header is present,
it must match the server's expected origin. If neither header is present,
the request is allowed through (browsers always send Origin for cross-origin
POST requests in modern browsers).

Compatible with the existing middleware stack in server.py.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# Default allowed origins (can be overridden via env var)
_DEFAULT_ALLOWED_ORIGINS: list[str] = []

# Origins that are always permitted (local dev)
_SAFE_ORIGINS: set[str] = {
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:8000",
}


def _get_allowed_origins() -> list[str]:
    """Get allowed origins from the environment variable.

    Reads ``DISTLLM_CSRF_ALLOWED_ORIGINS`` as a comma-separated list.
    Falls back to an empty list (same-origin only).
    """
    raw = os.environ.get("DISTLLM_CSRF_ALLOWED_ORIGINS", "")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return list(_DEFAULT_ALLOWED_ORIGINS)


def _normalize_origin(origin: str) -> str:
    """Normalize an Origin/Referer-origin string for comparison.

    Lowercases scheme+host (origins are case-insensitive per RFC 6454) and
    strips any trailing slash so configured entries like
    ``https://App.Example.com/`` compare equal to a browser-sent
    ``https://app.example.com``.
    """
    origin = origin.strip()
    parsed = urlparse(origin)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if not scheme or not netloc:
        # Not parseable as scheme://host — fall back to a conservative
        # lowercase of the raw string.
        return origin.lower().rstrip("/")
    return f"{scheme}://{netloc}"


def _origin_matches_allowed(origin: str, allowed_origins: list[str]) -> bool:
    """Check if an origin matches any of the allowed origins.

    Supports exact match and subdomain wildcard via leading ``*.``
    (e.g. ``https://*.example.com`` matches ``https://app.example.com``
    but NOT ``https://example.com``, ``https://app.example.com.evil.io``,
    or ``https://evil.com/?x=*.example.com``).
    """
    normalized_origin = _normalize_origin(origin)

    # Always allow safe localhost origins
    if normalized_origin in {_normalize_origin(o) for o in _SAFE_ORIGINS}:
        return True

    for allowed in allowed_origins:
        normalized_allowed = _normalize_origin(allowed)

        if normalized_allowed.startswith(("http://*.", "https://*.")):
            scheme, _, domain = normalized_allowed.partition("://")
            wildcard_domain = domain[2:]  # strip "*."
            # SECURITY: anchored on BOTH ends, and the subdomain group uses
            # ``+`` (not ``*``) so the wildcard never matches the bare apex
            # domain — ``https://*.example.com`` means "subdomains only".
            # Without the trailing "$", ``*.example.com`` also matched
            # ``https://app.example.com.evil.io`` (re.match is a prefix
            # match), letting attacker-controlled suffix domains forge
            # same-origin requests.
            pattern = (
                rf"^{re.escape(scheme)}://([a-zA-Z0-9-]+\.)+"
                rf"{re.escape(wildcard_domain)}$"
            )
            if re.match(pattern, normalized_origin):
                return True
        elif normalized_origin == normalized_allowed:
            return True

    return False


class CSRFSameOriginMiddleware(BaseHTTPMiddleware):
    """Middleware that validates Origin/Referer headers on state-changing requests.

    The check is based on the Same-Origin policy:
    - If the ``Origin`` header is present, it must match the allowed origins.
    - If ``Origin`` is absent but ``Referer`` is present, the Referer origin
      must match the allowed origins.
    - If neither header is present, the request is allowed (browsers always
      send Origin for cross-origin POST in modern browsers, so this is safe).

    Exempted paths:
    - ``/health``, ``/ready``, ``/live``, ``/metrics`` (monitoring probes)
    - ``/docs``, ``/openapi.json``, ``/redoc`` (API docs)
    - WebSocket upgrade requests (``Upgrade: websocket``)

    Exempted methods:
    - ``GET``, ``HEAD``, ``OPTIONS`` (read-only or preflight)

    Environment variables:
    - ``DISTLLM_CSRF_ALLOWED_ORIGINS``: comma-separated list of allowed origins
      (default: empty = same-origin only).
    - ``DISTLLM_CSRF_DISABLED``: set to ``1`` to disable this middleware
      (not recommended for production).
    """

    EXEMPT_PATHS: frozenset[str] = frozenset({
        "/health", "/ready", "/live", "/healthz", "/readyz",
        "/metrics", "/docs", "/openapi.json", "/redoc",
        "/dashboard", "/ws", "/ws/metrics",
    })

    # Paths that are exempt because they receive unauthenticated requests
    # from external systems (e.g., K8s probes, monitoring, CORS preflight)
    EXEMPT_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})

    def __init__(self, app):
        super().__init__(app)
        self._allowed_origins = _get_allowed_origins()
        self._disabled = os.environ.get("DISTLLM_CSRF_DISABLED", "0") == "1"

        if not self._allowed_origins:
            logger.info(
                "CSRF middleware: no allowed origins configured — "
                "enforcing strict same-origin policy."
            )
        else:
            logger.info(
                f"CSRF middleware: allowed origins = {self._allowed_origins}"
            )

    async def dispatch(self, request: Request, call_next):
        # Skip if disabled via env var
        if self._disabled:
            return await call_next(request)

        # Skip exempt paths (monitoring, docs, WebSocket)
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Skip exempt methods (GET/HEAD/OPTIONS are read-only or preflight)
        if request.method in self.EXEMPT_METHODS:
            return await call_next(request)

        # Check for WebSocket upgrade
        upgrade = request.headers.get("Upgrade", "").lower()
        if upgrade == "websocket":
            return await call_next(request)

        # Validate Origin or Referer header
        origin = request.headers.get("Origin", "")
        referer = request.headers.get("Referer", "")

        # If Origin is present, validate it
        if origin:
            if not _origin_matches_allowed(origin, self._allowed_origins):
                logger.warning(
                    f"CSRF blocked request from origin={origin!r} "
                    f"method={request.method} path={request.url.path} "
                    f"ip={request.client.host if request.client else 'unknown'}"
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": {
                            "message": "CSRF validation failed: origin not allowed",
                            "type": "csrf_error",
                            "code": "403",
                        }
                    },
                )
            # Origin is valid
            return await call_next(request)

        # If Origin is absent but Referer is present, validate Referer.
        # SECURITY: fail CLOSED on an unparseable Referer — previously a
        # malformed value (no scheme/host) produced an empty origin string
        # and fell through to "allowed", bypassing the check entirely.
        if referer:
            referer_origin = _normalize_origin(referer)
            if (
                not referer_origin
                or "://" not in referer_origin
                or not _origin_matches_allowed(referer_origin, self._allowed_origins)
            ):
                logger.warning(
                    f"CSRF blocked request from referer={referer!r} "
                    f"method={request.method} path={request.url.path} "
                    f"ip={request.client.host if request.client else 'unknown'}"
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": {
                            "message": "CSRF validation failed: referer not allowed",
                            "type": "csrf_error",
                            "code": "403",
                        }
                    },
                )

        # Neither Origin nor Referer: allow through (browsers always send
        # Origin for cross-origin POST in modern browsers)
        return await call_next(request)
