"""API middleware for distributed LLM inference."""

import os
import hmac
import time
import uuid
import secrets
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.errors import error_response


def _get_or_generate_api_key() -> str | None:
    """Get API_KEY from env, or generate one if not set.

    If API_KEY is not set, generates a secure random key and logs it.
    This ensures auth is never disabled by default in production.
    """
    api_key = os.environ.get("API_KEY")
    if api_key:
        # Mark explicitly configured keys without reclassifying keys that
        # this middleware generated earlier in the same process.
        if "API_KEY_WAS_SET" not in os.environ:
            os.environ["API_KEY_WAS_SET"] = "1"
        return api_key

    # Generate a secure random API key if not set
    generated_key = secrets.token_urlsafe(48)
    os.environ["API_KEY"] = generated_key
    os.environ["API_KEY_WAS_SET"] = "0"
    logger.warning(
        "API_KEY not set. Generated a secure random API key for production use.\n"
        "Save this key and set API_KEY=<your-key> in your environment:\n"
        f"API_KEY={generated_key}"
    )
    return generated_key


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
        api_key = _get_or_generate_api_key()

        # Allow explicit opt-out for development only.
        # Requires BOTH DISABLE_AUTH=1 AND DISTLLM_DEV_MODE=1 to be set.
        # When API_KEY is configured, auth is NEVER bypassed.
        if (
            os.environ.get("DISABLE_AUTH") == "1"
            and os.environ.get("DISTLLM_DEV_MODE") == "1"
            and os.environ.get("API_KEY_WAS_SET") != "1"
        ):
            if not getattr(self, "_warned", False):
                logger.warning(
                    "AUTHENTICATION DISABLED via DISABLE_AUTH=1 + DISTLLM_DEV_MODE=1. "
                    "This is a security risk and should only be used in development."
                )
                self._warned = True
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        client_ip = request.client.host if request.client else "unknown"

        # Check rate limit before validating
        if _rate_limiter.is_rate_limited(client_ip):
            retry_after = _rate_limiter.retry_after(client_ip)
            return error_response(
                status_code=429,
                error="Too Many Requests",
                message="Too many failed authentication attempts. Try again later.",
                type="auth_rate_limit",
                request_id=getattr(request.state, "request_id", None),
            )

        if not auth_header.startswith("Bearer ") or len(auth_header) < 8 or not hmac.compare_digest(auth_header[7:], api_key):
            _rate_limiter.record_attempt(client_ip)
            return error_response(
                status_code=401,
                error="Unauthorized",
                message="Unauthorized: invalid or missing API key",
                type="auth_error",
                request_id=getattr(request.state, "request_id", None),
            )
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
