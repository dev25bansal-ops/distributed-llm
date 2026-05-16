"""API middleware for distributed LLM inference."""

import os
import hmac
import time
import uuid
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class _RateLimiter:
    """Sliding window rate limiter: tracks failed attempts per IP."""

    def __init__(self, max_attempts: int = 30, window_seconds: int = 60):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def is_rate_limited(self, ip: str) -> bool:
        """Return True if the IP has exceeded the rate limit."""
        now = time.time()
        cutoff = now - self.window_seconds
        # Prune old entries
        self._attempts[ip] = [t for t in self._attempts[ip] if t > cutoff]
        return len(self._attempts[ip]) >= self.max_attempts

    def retry_after(self, ip: str) -> int:
        """Return the number of seconds until the rate limit resets for this IP."""
        now = time.time()
        if self._attempts.get(ip) and len(self._attempts[ip]) >= self.max_attempts:
            oldest = min(self._attempts[ip])
            return max(1, int(self.window_seconds - (now - oldest)))
        return 0

    def record_attempt(self, ip: str) -> None:
        """Record a failed auth attempt."""
        self._attempts[ip].append(time.time())


# Module-level rate limiter instance
_rate_limiter = _RateLimiter(max_attempts=30, window_seconds=60)


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate API key from Authorization header.

    Security: Requires API_KEY environment variable to be set.
    When not set, authentication fails-closed (returns 401) for security.
    Set DISABLE_AUTH=1 to bypass in development only.
    Includes rate limiting to prevent brute-force attacks.
    """

    async def dispatch(self, request: Request, call_next):
        api_key = os.environ.get("API_KEY")

        # Allow explicit opt-out for development only
        if os.environ.get("DISABLE_AUTH") == "1":
            if not getattr(self, "_warned", False):
                import logging
                logging.getLogger("distllm.security").warning(
                    "AUTHENTICATION DISABLED via DISABLE_AUTH=1. "
                    "This is a security risk and should only be used in development."
                )
                self._warned = True
            return await call_next(request)

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication not configured: set API_KEY environment variable"},
            )

        auth_header = request.headers.get("Authorization", "")
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit before validating
        if _rate_limiter.is_rate_limited(client_ip):
            retry_after = _rate_limiter.retry_after(client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many failed authentication attempts. Try again later."},
                headers={"Retry-After": str(retry_after)},
            )

        if not auth_header.startswith("Bearer ") or len(auth_header) < 8 or not hmac.compare_digest(auth_header[7:], api_key):
            _rate_limiter.record_attempt(client_ip)
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: invalid or missing API key"},
            )
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate a unique request ID per request and propagate it.

    Sets X-Request-ID on the response and stores it in request.state
    for downstream use (e.g. gRPC metadata propagation).
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
