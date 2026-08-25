"""API middleware for distributed LLM inference."""

import asyncio
import os
import hmac
import threading
import time
import uuid
import secrets
from collections import OrderedDict, defaultdict, deque
from typing import Any, Callable, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from distllm.api.errors import error_response
from distllm.api.ip_utils import get_client_ip
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
        import threading
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.max_ips = max_ips
        self._key_by = key_by if key_by in (self.KEY_BY_IP, self.KEY_BY_API_KEY, self.KEY_BY_BOTH) else "ip"
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._access_order: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Clear all tracked attempts (used between test cases)."""
        with self._lock:
            self._attempts.clear()
            self._access_order.clear()

    def set_key_by(self, mode: str) -> None:
        """Change rate limiting key mode at runtime.

        Args:
            mode: 'ip', 'api_key', or 'both'.
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            keys = self._make_keys(ip, api_key)
            for key in keys:
                self._attempts[key].append(time.time())
                self._prune(key)

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
    os.environ["API_KEY_WAS_SET"] = "0"

    # SECURITY: Log a fingerprint, not the full key. Full keys in logs = credential leak.
    # Logs are written to disk, captured by log aggregators, and visible in CI output.
    import hashlib
    import sys
    fingerprint = generated_key[:8] + "..." + hashlib.sha256(generated_key.encode()).hexdigest()[:8]
    logger.warning(
        f"API_KEY not set. Generated a secure random API key.\n"
        f"Key fingerprint: {fingerprint}\n"
        "The full key is printed once to stdout (NOT via the logger), so it is "
        "not persisted in log files, aggregators, or CI. Set API_KEY=<key> in "
        "your environment and restart."
    )
    # Surface the one-time key on stdout only — never through the logger, so the
    # secret is not written to disk logs or captured by log aggregators.
    print(
        "\nGenerated API key (shown once — do NOT paste into logs/screenshots/CI):\n"
        f"{generated_key}\n"
        "Set API_KEY=<key> in your environment and restart.\n",
        file=sys.stdout,
        flush=True,
    )
    return generated_key


# Module-level rate limiter instances
_rate_limiter = _RateLimiter(max_attempts=30, window_seconds=60)
_get_or_generate_api_key()

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

        # AUTH BYPASS REMOVED — CRITICAL SECURITY FIX
        # DISTLLM_NO_AUTH and DISABLE_AUTH env vars are no longer supported.
        # Authentication is always required. Dev mode bypass was a security risk
        # because the env var could be set in production by accident.
        #
        # For development/testing, use DISTLLM_DEV_MODE=1 with a valid API key,
        # or run the test suite which sets PYTEST_CURRENT_TEST automatically.
        if os.environ.get("DISTLLM_NO_AUTH") == "1":
            logger.critical(
                "SECURITY: DISTLLM_NO_AUTH is no longer supported. "
                "Authentication is always required. Remove this env var."
            )
        if os.environ.get("DISABLE_AUTH") == "1":
            logger.critical(
                "SECURITY: DISABLE_AUTH is no longer supported. "
                "Authentication is always required. Remove this env var."
            )

        # Health endpoints are exempt — K8s probes, load balancers, and
        # monitoring systems need unauthenticated access for uptime checks.
        # The HA leader-election heartbeat is also exempt so peer coordinators
        # can exchange liveness without per-peer API keys; it is still gated by
        # the shared X-HA-Secret when DISTLLM_HA_SECRET is configured (see the
        # /api/v1/ha/heartbeat route).
        if request.url.path in (
            "/health",
            "/v1/health",
            "/v1/health/readiness",
            "/v1/health/liveness",
            "/ready",
            "/live",
            "/healthz",
            "/readyz",
            "/metrics",
            "/api/v1/ha/heartbeat",
        ):
            return await call_next(request)

        # OPTIONS (CORS preflight) is intentionally allowed without auth.
        #
        # Security note: The browser sends a CORS preflight (OPTIONS) before
        # any actual authenticated request. It carries no user data, sets no
        # cookies, and is entirely read-only. Requiring auth here would break
        # legitimate cross-origin workflows (e.g., a JS client on a different
        # origin) without providing any meaningful security gain — an attacker
        # who can send OPTIONS cannot do anything they could not already do
        # with a normal GET. The actual CORS origin/method/header validation
        # is enforced server-side by Starlette/FastAPI's CORSMiddleware.
        if request.method == "OPTIONS":
            # Log non-trivial OPTIONS paths for observability so operators
            # can detect unexpected OPTIONS traffic to auth-required endpoints.
            if request.url.path not in ("/", ""):
                logger.debug(
                    "OPTIONS request from {} to {} — CORS preflight (auth bypassed)",
                    get_client_ip(request),
                    request.url.path,
                )
            return await call_next(request)

        # If SsoMiddleware already authenticated, skip API-key validation.
        if getattr(request.state, "auth_method", None) == "sso":
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header:
            # Browser WebSocket clients cannot set arbitrary headers; the
            # convention is to pass the token as a subprotocol pair
            # ["Bearer", "<key>"] in Sec-WebSocket-Protocol.
            sec_ws = request.headers.get("Sec-WebSocket-Protocol", "")
            parts = [p.strip() for p in sec_ws.split(",") if p.strip()]
            if len(parts) >= 2 and parts[0] == "Bearer":
                auth_header = f"Bearer {parts[1]}"
        client_ip = get_client_ip(request)

        if not auth_header.startswith("Bearer ") or len(auth_header) < 8:
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
            _rate_limiter.record_attempt(client_ip)
            return error_response(
                status_code=401,
                error="Unauthorized",
                message="Unauthorized: invalid API key",
                type="auth_error",
                request_id=getattr(request.state, "request_id", None),
            )

        token = auth_header[7:]
        # B4-1 perf fix: ApiKeyStore.authenticate memoizes verdicts (sha256
        # token digest -> result, short TTL, invalidated on key rotation).
        # The cold path is still PBKDF2-100k (~30 ms per stored key), so run
        # it on a worker thread: cache hits cost one cheap thread hop, and
        # cache misses no longer stall the event loop for every request.
        result = await asyncio.to_thread(store.authenticate, token)
        if result is None:
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
    """Per-client request rate limiting.

    Enabled when ``DISTLLM_RATE_LIMIT_REQUESTS`` is set to a positive
    integer (default: 1000 requests per 60-second window).
    Set to 0 to disable.

    Keying policy (C15/H11): the limiter key is
    ``"<client_ip>|<api_key_id>"`` once the request carries an
    authenticated ``api_key_id`` on ``request.state``, falling back to the
    bare client IP while unauthenticated.  The composite key closes two
    bypass/false-positive classes at once:

    * NAT fairness -- distinct API keys behind one shared IP get their own
      budgets instead of collectively exhausting a single per-IP bucket.
    * Rotation bypass (H11) -- the IP half is resolved fail-closed via
      :func:`get_client_ip` (spoofable ``X-Forwarded-For`` is ignored for
      direct peers), so header rotation cannot mint fresh buckets, and the
      API-key half means an exhausted ``ip|old_key`` bucket stays limited
      even as new keys appear.
    """

    # M-09: Cache env var at class level instead of reading on every request
    _rate_limit_value = int(os.environ.get("DISTLLM_RATE_LIMIT_REQUESTS", "1000"))

    @staticmethod
    def _limit_key(client_ip: str, request: Any) -> str:
        """Build the limiter key: ``ip|api_key_id`` when authenticated.

        Unauthenticated (or not-yet-authenticated) requests key by the
        client IP alone.  Reads ``api_key_id`` defensively so behaviour is
        correct regardless of whether this middleware runs inside or
        outside ``AuthMiddleware`` in the stack.
        """
        api_key_id = getattr(getattr(request, "state", None), "api_key_id", None)
        if api_key_id:
            return f"{client_ip}|{api_key_id}"
        return client_ip

    async def dispatch(self, request: Request, call_next):
        limit = self._rate_limit_value
        if limit > 0:
            client_ip = get_client_ip(request)
            rl_key = self._limit_key(client_ip, request)
            if _request_rate_limiter.is_rate_limited(rl_key):
                retry_after = _request_rate_limiter.retry_after(rl_key)
                return error_response(
                    status_code=429,
                    error="Too Many Requests",
                    message=f"Request rate limit exceeded. Retry after {retry_after}s.",
                    type="rate_limit",
                    retry_after=retry_after,
                    request_id=getattr(request.state, "request_id", None),
                )
            _request_rate_limiter.record_attempt(rl_key)
        return await call_next(request)


# ── ContentModerationMiddleware ──────────────────────────────────────────

import enum as _enum
from distllm.security.content_moderation.pipeline import ContentModerationPipeline
from distllm.security.content_moderation.base import ModerationResult


class ModerationAction(str, _enum.Enum):
    """Action to take when moderation is triggered."""

    BLOCK = "block"
    SANITIZE = "sanitize"
    FLAG = "flag"


class ModerationScope(str, _enum.Enum):
    """Scope of moderation checking."""

    INPUTS = "inputs"
    OUTPUTS = "outputs"
    BOTH = "both"


class ContentModerationMiddleware(BaseHTTPMiddleware):
    """Intercept requests/responses and run the content moderation pipeline.

    Configurable action on violation:

    * ``BLOCK`` -- return a 451 response immediately.
    * ``SANITIZE`` -- allow the request through but replace the content
      with the redacted version.
    * ``FLAG`` -- pass the request through but add ``X-Moderation-Flag``
      and ``X-Moderation-Detail`` headers to the response.

    Configurable scope per endpoint (default ``BOTH``):

    * ``INPUTS`` -- only moderate request bodies.
    * ``OUTPUTS`` -- only moderate response bodies.
    * ``BOTH`` -- moderate both directions.

    Must be placed **after** ``AuthMiddleware`` so that
    ``request.state.api_key_role`` is populated, but **before** the route
    handlers so that blocked requests are rejected early.

    Environment variables:

    * ``DISTLLM_MODERATION_ACTION`` -- one of ``block``, ``sanitize``,
      ``flag`` (default: ``sanitize``).
    * ``DISTLLM_MODERATION_SCOPE`` -- one of ``inputs``, ``outputs``,
      ``both`` (default: ``both``).
    * ``DISTLLM_MODERATION_SKIP_PATHS`` -- comma-separated path prefixes
      to skip (default: ``/health,/ready,/live,/metrics,/docs,/openapi.json,/redoc``).
    """

    def __init__(
        self,
        app: Any,
        pipeline: ContentModerationPipeline | None = None,
        action: ModerationAction | str | None = None,
        scope: ModerationScope | str | None = None,
        skip_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._pipeline = pipeline or ContentModerationPipeline()

        raw_action = action or os.environ.get("DISTLLM_MODERATION_ACTION", "sanitize")
        self._action = (
            ModerationAction(raw_action)
            if isinstance(raw_action, str)
            else raw_action
        )

        raw_scope = scope or os.environ.get("DISTLLM_MODERATION_SCOPE", "both")
        self._scope = (
            ModerationScope(raw_scope)
            if isinstance(raw_scope, str)
            else raw_scope
        )

        raw_skip = os.environ.get(
            "DISTLLM_MODERATION_SKIP_PATHS",
            "/health,/ready,/live,/metrics,/docs,/openapi.json,/redoc",
        )
        self._skip_paths = skip_paths or {p.strip() for p in raw_skip.split(",") if p.strip()}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._skip_paths:
            return await call_next(request)

        # ── Input moderation ──────────────────────────────────────────
        if self._scope in (ModerationScope.INPUTS, ModerationScope.BOTH):
            result = await self._moderate_request(request)
            if result is not None:
                return result  # BLOCK action returned an error response

        # ── Process the request ──────────────────────────────────────
        response = await call_next(request)

        # ── Output moderation ─────────────────────────────────────────
        if self._scope in (ModerationScope.OUTPUTS, ModerationScope.BOTH):
            response = await self._moderate_response(request, response)

        return response

    async def _moderate_request(self, request: Request) -> Response | None:
        """Moderate the incoming request body.  Returns an error response if
        the action is BLOCK, otherwise modifies request state in place."""
        # Read the body if we can (it may already be cached by BodyCacheMiddleware).
        body_bytes = None
        try:
            body_bytes = await request.body()
        except Exception:
            pass
        if not body_bytes:
            return None

        # Try to parse JSON body.
        try:
            import json
            body = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        # Extract text content from common LLM request formats.
        content = self._extract_text(body)
        if not content:
            return None

        result = await self._pipeline.async_process(content)
        if result.passed:
            return None

        if self._action == ModerationAction.BLOCK:
            from fastapi.responses import JSONResponse as _JSONResponse
            detail = self._build_detail(result)
            return _JSONResponse(
                status_code=451,
                content={
                    "error": {
                        "message": f"Content moderation blocked: {detail}",
                        "type": "content_moderation",
                        "code": "451",
                    },
                },
            )

        if self._action == ModerationAction.SANITIZE:
            # Store the redacted text so route handlers can use it.
            request.state.moderation_redacted_text = result.redacted_text
            request.state.moderation_result = result

        if self._action == ModerationAction.FLAG:
            request.state.moderation_result = result

        return None

    async def _moderate_response(self, request: Request, response: Response) -> Response:
        """Moderate the outgoing response body (FLAG and SANITIZE only)."""
        # Only moderate JSON responses.
        content_type = response.headers.get("content-type", "")
        if "json" not in content_type:
            return response

        body = getattr(response, "body", None)
        if body is None:
            return response

        try:
            import json
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            return response

        # Extract output text from common LLM response formats.
        text = self._extract_output_text(payload)
        if not text:
            return response

        result = await self._pipeline.async_process(text)
        if result.passed:
            return response

        if self._action == ModerationAction.BLOCK:
            from fastapi.responses import JSONResponse as _JSONResponse
            detail = self._build_detail(result)
            return _JSONResponse(
                status_code=451,
                content={
                    "error": {
                        "message": f"Output moderation blocked: {detail}",
                        "type": "content_moderation",
                        "code": "451",
                    },
                },
            )

        if self._action == ModerationAction.SANITIZE:
            # Replace output text with redacted version.
            new_body = self._replace_output_text(payload, result.redacted_text)
            import json as _json
            from starlette.responses import Response as _Response
            response = _Response(
                content=_json.dumps(new_body),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        if self._action == ModerationAction.FLAG:
            detail = self._build_detail(result)
            response.headers["X-Moderation-Flag"] = "true"
            response.headers["X-Moderation-Detail"] = detail

        return response

    def _extract_text(self, body: dict[str, Any]) -> str:
        """Extract the primary text content from a request body dict.

        Handles chat completion, completion, and embedding formats.
        """
        # Chat messages format
        messages = body.get("messages", [])
        if messages and isinstance(messages, list):
            parts = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # Multimodal content -- extract text parts only.
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
            return " ".join(parts)

        # Prompt / input format
        prompt = body.get("prompt", body.get("input", ""))
        if isinstance(prompt, str):
            return prompt

        return ""

    def _extract_output_text(self, payload: dict[str, Any]) -> str:
        """Extract generated text from common LLM response formats."""
        # Chat completion format
        choices = payload.get("choices", [])
        if choices and isinstance(choices, list):
            parts = []
            for choice in choices:
                msg = choice.get("message", choice.get("delta", {}))
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
            return " ".join(parts)

        # Simple text response
        text = payload.get("text", payload.get("response", ""))
        if isinstance(text, str):
            return text

        return ""

    def _replace_output_text(
        self, payload: dict[str, Any], redacted_text: str
    ) -> dict[str, Any]:
        """Replace the generated text in a response payload with *redacted_text*."""
        choices = payload.get("choices", [])
        if choices and isinstance(choices, list):
            for choice in choices:
                msg = choice.get("message", choice.get("delta", {}))
                if "content" in msg:
                    msg["content"] = redacted_text
            return payload

        if "text" in payload:
            payload["text"] = redacted_text
        elif "response" in payload:
            payload["response"] = redacted_text

        return payload

    @staticmethod
    def _build_detail(result: ModerationResult) -> str:
        """Build a human-readable detail string from a moderation result."""
        parts: list[str] = []
        if result.toxicity and result.toxicity.toxic:
            parts.append(f"toxicity={result.toxicity.score:.2f}")
        if result.jailbreak and result.jailbreak.jailbreak_attempt:
            parts.append(f"jailbreak={result.jailbreak.confidence:.2f}")
        if result.topic_filter and not result.topic_filter.allowed:
            parts.append(f"topics={','.join(result.topic_filter.violated_policies)}")
        if result.pii and result.pii.entities_found:
            pii_types = sorted(set(e.type for e in result.pii.entities_found))
            parts.append(f"pii={','.join(pii_types)}")
        return "; ".join(parts) if parts else "unknown"
