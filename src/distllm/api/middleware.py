"""API middleware for distributed LLM inference."""

import os
import hmac
import time
import uuid
import secrets
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.errors import error_response
from distllm.core.api_key_store import get_api_key_store, role_satisfies


class _RateLimiter:
    """Sliding window rate limiter: tracks failed attempts per IP.

    Memory-bounded: evicts the oldest IP when the tracked IP count exceeds
    ``max_ips``. Each IP stores up to ``max_attempts`` timestamps within a
    ``window_seconds`` sliding window.
    """

    def __init__(self, max_attempts: int = 30, window_seconds: int = 60, max_ips: int = 10000):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.max_ips = max_ips
        self._attempts: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, ip: str) -> None:
        """Remove expired entries for *ip* and enforce the IP cap."""
        now = time.time()
        cutoff = now - self.window_seconds
        timestamps = self._attempts[ip]
        # Deque popleft is O(1) — much faster than list filter
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()

        # Remove empty entries
        if not timestamps:
            del self._attempts[ip]
            return

        # Enforce IP cap: evict oldest IP when over limit
        if len(self._attempts) > self.max_ips:
            oldest_ip = min(self._attempts, key=lambda k: self._attempts[k][-1])
            del self._attempts[oldest_ip]

    def is_rate_limited(self, ip: str) -> bool:
        """Return True if the IP has exceeded the rate limit."""
        self._prune(ip)
        if ip not in self._attempts:
            return False
        return len(self._attempts[ip]) >= self.max_attempts

    def retry_after(self, ip: str) -> int:
        """Return the number of seconds until the rate limit resets for this IP."""
        now = time.time()
        if ip in self._attempts and len(self._attempts[ip]) >= self.max_attempts:
            oldest = min(self._attempts[ip])
            return max(1, int(self.window_seconds - (now - oldest)))
        return 0

    def record_attempt(self, ip: str) -> None:
        """Record a failed auth attempt."""
        self._attempts[ip].append(time.time())
        self._prune(ip)


# Module-level rate limiter instances
_rate_limiter = _RateLimiter(max_attempts=30, window_seconds=60)

try:
    _rate_limit_req = int(os.environ.get("DISTLLM_RATE_LIMIT_REQUESTS", "1000"))
except (ValueError, TypeError):
    _rate_limit_req = 1000
_request_rate_limiter = _RateLimiter(
    max_attempts=_rate_limit_req,
    window_seconds=60,
)


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate API key from Authorization header using the ApiKeyStore.

    On success, sets ``request.state.api_key_role`` and
    ``request.state.api_key_id`` for downstream role checks.

    Security: Always requires a valid API key. Authentication fails-closed
    (returns 401) by default.
    """

    async def dispatch(self, request: Request, call_next):
        store = get_api_key_store()

        auth_header = request.headers.get("Authorization", "")
        if os.environ.get("DISTLLM_TRUST_PROXY_HEADERS") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
            client_ip = request.headers.get("X-Real-IP") or ""
            if not client_ip:
                forwarded = request.headers.get("X-Forwarded-For", "")
                parts = [p.strip() for p in forwarded.split(",") if p.strip()]
                client_ip = parts[-1] if parts else ""
        else:
            client_ip = ""
        client_ip = client_ip or (request.client.host if request.client else "unknown")

        # Check rate limit before validating
        if _rate_limiter.is_rate_limited(client_ip):
            retry_after = _rate_limiter.retry_after(client_ip)
            return error_response(
                status_code=429,
                error="Too Many Requests",
                message="Too many failed authentication attempts. Try again later.",
                type="auth_rate_limit",
                retry_after=retry_after,
                request_id=getattr(request.state, "request_id", None),
            )

        if not auth_header.startswith("Bearer ") or len(auth_header) < 8:
            _rate_limiter.record_attempt(client_ip)
            return error_response(
                status_code=401,
                error="Unauthorized",
                message="Unauthorized: missing API key",
                type="auth_error",
                request_id=getattr(request.state, "request_id", None),
            )

        token = auth_header[7:]
        result = store.authenticate(token)
        if result is None:
            _rate_limiter.record_attempt(client_ip)
            return error_response(
                status_code=401,
                error="Unauthorized",
                message="Unauthorized: invalid API key",
                type="auth_error",
                request_id=getattr(request.state, "request_id", None),
            )

        key_id, role = result
        request.state.api_key_role = role
        request.state.api_key_id = key_id
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate a unique request ID per request and propagate it.

    Reads X-Request-ID, X-Request-Timeout, and X-Priority headers
    and stores them in request.state for downstream use.
    """

    def _parse_priority(self, raw: str | None) -> int:
        if raw is None:
            return 2
        mapping = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        return mapping.get(raw.strip().lower(), 2)

    def _parse_timeout(self, raw: str | None) -> float | None:
        if raw is None:
            return None
        try:
            val = float(raw)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request.state.request_timeout = self._parse_timeout(
            request.headers.get("X-Request-Timeout")
        )
        request.state.request_priority = self._parse_priority(
            request.headers.get("X-Priority")
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestRateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP request rate limiting.

    Enabled when ``DISTLLM_RATE_LIMIT_REQUESTS`` is set to a positive
    integer (default: 1000 requests per 60-second window).
    Set to 0 to disable.
    """

    async def dispatch(self, request: Request, call_next):
        limit = int(os.environ.get("DISTLLM_RATE_LIMIT_REQUESTS", "1000"))
        if limit > 0:
            client_ip = request.client.host if request.client else "unknown"
            if _request_rate_limiter.is_rate_limited(client_ip):
                retry_after = _request_rate_limiter.retry_after(client_ip)
                return error_response(
                    status_code=429,
                    error="Too Many Requests",
                    message=f"Request rate limit exceeded. Retry after {retry_after}s.",
                    type="rate_limit",
                    retry_after=retry_after,
                    request_id=getattr(request.state, "request_id", None),
                )
            _request_rate_limiter.record_attempt(client_ip)
        return await call_next(request)
