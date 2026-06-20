"""Starlette middleware for token-bucket rate limiting.

Wraps ``RateLimiter`` to enforce per-client, per-endpoint request
throttling with standard ``X-RateLimit-*`` response headers.

Usage::

    from fastapi import FastAPI
    from distllm.api.rate_limiter import RateLimiter
    from distllm.api.rate_limit_middleware import RateLimitMiddleware

    app = FastAPI()
    limiter = RateLimiter(default_rpm=1000, burst_multiplier=1.5)
    app.add_middleware(RateLimitMiddleware, rate_limiter=limiter, enabled=True)
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from distllm.api.ip_utils import get_client_ip
from distllm.api.rate_limiter import RateLimiter

# Endpoints that are never rate-limited (health probes, metrics scrape).
_SKIP_PREFIXES = ("/health", "/metrics", "/ready", "/live")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-client, per-endpoint rate limits using a token bucket.

    Args:
        app: ASGI application.
        rate_limiter: ``RateLimiter`` instance with bucket configuration.
        enabled: When False the middleware is a no-op (all requests pass through).
    """

    def __init__(self, app, rate_limiter: RateLimiter, enabled: bool = True) -> None:
        super().__init__(app)
        self._limiter = rate_limiter
        self._enabled = enabled

    async def dispatch(self, request: Request, call_next):
        if not self._enabled:
            return await call_next(request)

        # Never rate-limit health or metrics endpoints
        path = request.url.path
        if any(path.startswith(prefix) for prefix in _SKIP_PREFIXES):
            return await call_next(request)

        # Identify the client
        client_ip = get_client_ip(request)

        endpoint = path

        if not self._limiter.is_allowed(client_ip, endpoint):
            limit, remaining, retry_after = self._limiter.get_limits(client_ip, endpoint)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "429",
                        "message": "Rate limit exceeded. Try again later.",
                        "type": "rate_limit_error",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(max(1, int(retry_after))),
                },
            )

        response = await call_next(request)

        # Attach rate-limit headers to successful responses
        limit, remaining, _ = self._limiter.get_limits(client_ip, endpoint)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
