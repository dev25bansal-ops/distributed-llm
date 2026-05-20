"""Rate limiting middleware for FastAPI."""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from distllm.api.errors import error_response
from distllm.api.rate_limiter import RateLimiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that applies rate limits per client per endpoint.

    Identifies clients by API key (if authenticated) or IP address.
    Returns 429 Too Many Requests when rate limited.
    """

    def __init__(self, app, rate_limiter: RateLimiter, enabled: bool = True):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Identify client
        client_id = self._get_client_id(request)
        endpoint = request.url.path

        # Skip rate limiting for health/metrics endpoints
        if endpoint in ("/health", "/metrics", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        # Check rate limit
        if not self.rate_limiter.is_allowed(client_id, endpoint):
            limit, remaining, retry_after = self.rate_limiter.get_limits(client_id, endpoint)
            return error_response(
                status_code=429,
                error="rate_limit_exceeded",
                message=f"Rate limit exceeded. Retry after {retry_after:.0f}s",
                type="rate_limit_error",
                request_id=getattr(request.state, "request_id", None),
                retry_after=retry_after,
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        limit, remaining, retry_after = self.rate_limiter.get_limits(client_id, endpoint)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from API key or IP address.

        Returns a client ID prefixed with 'auth:' for authenticated clients,
        allowing the rate limiter to apply different limits.
        """
        # Try API key first
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            # Prefix with 'auth:' to signal authenticated client
            return f"auth:{token}"

        # Fall back to IP address
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
