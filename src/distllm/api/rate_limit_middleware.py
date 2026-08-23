"""Rate-limiting middleware for the DistLLM API.

This module re-exports :class:`RequestRateLimitMiddleware` (defined in
``distllm.api.middleware``) as ``RateLimitMiddleware`` so that existing tests
and call sites that ``from distllm.api.rate_limit_middleware import
RateLimitMiddleware`` continue to work. The real implementation lives in
``distllm.api.middleware`` to avoid duplicating the Starlette
``BaseHTTPMiddleware`` plumbing.
"""

from distllm.api.rate_limiter_unified import RateLimitMiddleware

__all__ = ["RateLimitMiddleware"]
