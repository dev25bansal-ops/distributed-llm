"""API middleware for distributed LLM inference."""

import os
import hmac
import time
import uuid
import secrets
from collections import OrderedDict, defaultdict, deque
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.errors import error_response
from distllm.core.api_key_store import get_api_key_store, role_satisfies


class _RateLimiter:
    """Sliding window rate limiter with dual keying (IP + API key).

    M-06: Can key by IP (default), API key, or both. When keying by both,
    a request is rate limited if EITHER its IP OR its API key exceeds the
    limit. This prevents NAT-shared IPs from causing false positives while
    still preventing per-IP abuse.

    Memory-bounded: evicts the LRU entry when the tracked count exceeds
    ``max_ips``. Uses an OrderedDict for O(1) LRU eviction instead of
    O(N) min-search.

    Each key stores up to ``max_attempts`` timestamps within a
    ``window_seconds`` sliding window.
    """

    KEY_BY_IP = "ip"
    KEY_BY_API_KEY = "api_key"
    KEY_BY_BOTH = "both"

    def __init__(self, max_attempts: int = 30, window_seconds: int = 60,
                 max_ips: int = 10000, key_by: str = "ip"):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.max_ips = max_ips
        self._key_by = key_by if key_by in (self.KEY_BY_IP, self.KEY_BY_API_KEY, self.KEY_BY_BOTH) else "ip"
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._access_order: OrderedDict[str, None] = OrderedDict()

    def set_key_by(self, mode: str) -> None:
        """Change rate limiting key mode at runtime.

        Args:
            mode: 'ip', 'api_key', or 'both'.
        """
        if mode in (self.KEY_BY_IP, self.KEY_BY_API_KEY, self.KEY_BY_BOTH):
            self._key_by = mode

    def _make_keys(self, ip: str, api_key: str | None = None) -> list[str]:
        """Generate rate limit keys based on current keying mode."""
        if self._key_by == self.KEY_BY_API_KEY and api_key:
            return [f"apikey:{api_key}"]
        elif self._key_by == self.KEY_BY_BOTH:
            keys = [f"ip:{ip}"]
            if api_key:
                keys.append(f"apikey:{api_key}")
            return keys
        return [f"ip:{ip}"]

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
            self._access_order.pop(ip, None)
            return

        # Touch LRU order
        self._access_order.pop(ip, None)
        self._access_order[ip] = None

        # Enforce IP cap: evict LRU IP when over limit (O(1) with OrderedDict)
        while len(self._attempts) > self.max_ips:
            oldest_ip, _ = next(iter(self._access_order.items()))
            del self._attempts[oldest_ip]
            del self._access_order[oldest_ip]

    def is_rate_limited(self, ip: str, api_key: str | None = None) -> bool:
        """Return True if the key has exceeded the rate limit.

        Args:
            ip: Client IP address.
            api_key: Optional API key for dual-keyed rate limiting.
        """
        keys = self._make_keys(ip, api_key)
        for key in keys:
            self._prune(key)
            if key in self._attempts and len(self._attempts[key]) >= self.max_attempts:
                return True
        return False

    def retry_after(self, ip: str, api_key: str | None = None) -> int:
        """Return the number of seconds until the rate limit resets.

        Args:
            ip: Client IP address.
            api_key: Optional API key for dual-keyed rate limiting.
        """
        now = time.time()
        keys = self._make_keys(ip, api_key)
        max_wait = 0
        for key in keys:
            if key in self._attempts and len(self._attempts[key]) >= self.max_attempts:
                oldest = min(self._attempts[key])
                wait = max(1, int(self.window_seconds - (now - oldest)))
                max_wait = max(max_wait, wait)
        return max_wait

    def record_attempt(self, ip: str, api_key: str | None = None) -> None:
        """Record a failed auth attempt."""
        keys = self._make_keys(ip, api_key)
        for key in keys:
            self._attempts[key].append(time.time())
            self._prune(key)


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

        no_auth = os.environ.get("DISTLLM_NO_AUTH") == "1"
        if no_auth:
            # SECURITY: DISTLLM_NO_AUTH is never allowed in production.
            # Only PYTEST_CURRENT_TEST can bypass auth — no env var override.
            logger.critical("DISTLLM_NO_AUTH=1 is ignored: authentication cannot be disabled via environment variable. "
                          "Use PYTEST_CURRENT_TEST for testing, or pass --no-auth as a CLI flag to the coordinator.")
            # Still allow PYTEST_CURRENT_TEST for test suites that set both vars
            no_auth = False

        # Skip auth for health endpoints (K8s probes, load balancers) and OPTIONS (CORS preflight)
        if request.url.path in ("/health", "/ready", "/live", "/metrics") or request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if os.environ.get("DISTLLM_TRUST_PROXY_HEADERS") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
            client_ip = request.headers.get("X-Real-IP") or ""
            if not client_ip:
                forwarded = request.headers.get("X-Forwarded-For", "")
                # C-05: Per RFC 7239, the leftmost IP is the original client.
                parts = [p.strip() for p in forwarded.split(",") if p.strip()]
                client_ip = parts[0] if parts else ""
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
        raw_id = request.headers.get("X-Request-ID")
        if raw_id:
            # Validate X-Request-ID format (UUID or alphanumeric, no control chars)
            import re
            if not re.match(r'^[a-zA-Z0-9\-]{1,64}$', raw_id):
                request_id = str(uuid.uuid4())
            else:
                request_id = raw_id
        else:
            request_id = str(uuid.uuid4())
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

    # M-09: Cache env var at class level instead of reading on every request
    _rate_limit_value = int(os.environ.get("DISTLLM_RATE_LIMIT_REQUESTS", "1000"))

    async def dispatch(self, request: Request, call_next):
        limit = self._rate_limit_value
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
