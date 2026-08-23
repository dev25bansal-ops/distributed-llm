"""Unified rate limiter with strategy pattern, hierarchical limits, and per-endpoint configuration.

Provides:
- ``RateLimitStrategy`` — enum of available algorithms
- ``RateLimitConfig`` — frozen dataclass for per-scope configuration
- ``TokenBucket`` — in-memory per-key token bucket with automatic refill
- ``SlidingWindow`` — in-memory per-key sliding window counter with LRU eviction
- ``RedisRateLimiter`` — distributed rate limiter using Redis sorted sets
- ``RateLimiter`` — unified facade wrapping all strategies with hierarchical support
- ``RateLimitMiddleware`` — ASGI middleware integrating ``RateLimiter``

Usage::

    from distllm.api.rate_limiter_unified import (
        RateLimitConfig,
        RateLimitStrategy,
        RateLimiter,
        RateLimitMiddleware,
    )

    limiter = RateLimiter(
        default_config=RateLimitConfig(
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_second=10.0,
            burst_size=50,
        ),
        endpoint_configs={
            "/v1/chat/completions": RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                max_requests=200,
                window_seconds=60,
            ),
        },
        global_config=RateLimitConfig(requests_per_second=500.0, burst_size=1000),
    )

    app.add_middleware(RateLimitMiddleware, rate_limiter=limiter)
"""

from __future__ import annotations

import enum
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from fastapi import Request, Response
from distllm.api.errors import error_response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from distllm.api.ip_utils import get_client_ip


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class RateLimitStrategy(str, enum.Enum):
    """Supported rate limiting algorithms."""

    TOKEN_BUCKET = "token_bucket"
    """Classic token-bucket algorithm per key."""

    SLIDING_WINDOW = "sliding_window"
    """Sliding-window counter per key with LRU eviction."""

    REDIS = "redis"
    """Distributed rate limiting backed by Redis sorted sets."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_KEYS_DEFAULT = 10_000


@dataclass(frozen=True)
class RateLimitConfig:
    """Configuration for a single rate-limiting scope.

    Attributes:
        strategy: Which algorithm to use.
        requests_per_second: Token refill rate (used by token bucket).
        burst_size: Token bucket capacity (used by token bucket).
        window_seconds: Size of the sliding window in seconds (used by
            sliding window and Redis).
        max_requests: Maximum requests allowed in the window (used by
            sliding window and Redis).
        max_keys: Maximum tracked keys before LRU eviction (in-memory only).
        redis_prefix: Key prefix for Redis keys.
        redis_ttl: TTL in seconds for Redis keys.
    """

    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    requests_per_second: float = 1.0
    burst_size: int = 10
    window_seconds: int = 60
    max_requests: int = 1000
    max_keys: int = _MAX_KEYS_DEFAULT
    redis_prefix: str = "rate_limit:"
    redis_ttl: int = 120

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.burst_size < 1:
            raise ValueError("burst_size must be >= 1")
        if self.window_seconds < 1:
            raise ValueError("window_seconds must be >= 1")
        if self.max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if self.redis_ttl < self.window_seconds:
            # Widen TTL so Redis keys survive the full window even under clock
            # skew.  Freezing the dataclass prevents mutation, so we warn.
            import warnings

            warnings.warn(
                f"redis_ttl ({self.redis_ttl}s) < window_seconds "
                f"({self.window_seconds}s) — keys may expire early. "
                "Set redis_ttl >= window_seconds + grace.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Token-bucket strategy
# ---------------------------------------------------------------------------


class _TokenBucketEntry:
    """Mutable state for a single key's token bucket."""

    __slots__ = ("tokens", "last_refill", "max_tokens", "rate_per_second")

    def __init__(self, max_tokens: float, rate_per_second: float) -> None:
        self.tokens = max_tokens
        self.last_refill = time.monotonic()
        self.max_tokens = max_tokens
        self.rate_per_second = rate_per_second

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate_per_second)
            self.last_refill = now

    def consume(self) -> bool:
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def peek(self) -> int:
        self._refill()
        return int(self.tokens)

    def wait_seconds(self) -> float:
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / self.rate_per_second if self.rate_per_second > 0 else 0.0


class TokenBucket:
    """Per-key token bucket with automatic continuous refill.

    Thread-safe.  Each key gets its own logical bucket.  Entries are created
    lazily on first access.

    Args:
        config: Rate limit configuration for this bucket.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._buckets: dict[str, _TokenBucketEntry] = {}
        self._lock = threading.Lock()

    # -- helpers ---------------------------------------------------------

    def _entry(self, key: str) -> _TokenBucketEntry:
        entry = self._buckets.get(key)
        if entry is None:
            entry = _TokenBucketEntry(
                max_tokens=float(self._config.burst_size),
                rate_per_second=self._config.requests_per_second,
            )
            self._buckets[key] = entry
        return entry

    # -- public API ------------------------------------------------------

    def is_allowed(self, key: str) -> bool:
        """Atomically check and consume one token for *key*.

        Returns True if a token was consumed (request allowed).
        """
        with self._lock:
            return self._entry(key).consume()

    def remaining(self, key: str) -> int:
        """Return whole tokens currently available for *key*."""
        with self._lock:
            return self._entry(key).peek()

    def limit(self, key: str) -> int:
        """Return the burst capacity (max tokens) for *key*."""
        _ = key
        return self._config.burst_size

    def retry_after(self, key: str) -> float:
        """Seconds until a token becomes available for *key* (0.0 if ready)."""
        with self._lock:
            return self._entry(key).wait_seconds()

    def reset(self, key: str) -> None:
        """Remove the bucket for *key*, effectively resetting it."""
        with self._lock:
            self._buckets.pop(key, None)

    def reset_all(self) -> None:
        """Remove all buckets."""
        with self._lock:
            self._buckets.clear()


# ---------------------------------------------------------------------------
# Sliding-window strategy
# ---------------------------------------------------------------------------


class SlidingWindow:
    """Per-key sliding window counter with LRU eviction.

    Thread-safe.  Tracks timestamps in a ``deque`` per key and evicts the
    least-recently-used key when the total tracked keys exceeds *max_keys*.

    Args:
        config: Rate limit configuration for this window.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._access_order: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    # -- helpers ---------------------------------------------------------

    def _prune(self, key: str) -> None:
        """Remove expired entries for *key* and enforce key cap."""
        now = time.time()
        cutoff = now - self._config.window_seconds
        timestamps = self._windows.get(key)
        if timestamps is None:
            return
        # O(1) pop-left
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if not timestamps:
            del self._windows[key]
            self._access_order.pop(key, None)
            return
        # Touch LRU order
        self._access_order.pop(key, None)
        self._access_order[key] = None
        # Evict LRU when over limit
        while len(self._windows) > self._config.max_keys:
            oldest_key, _ = next(iter(self._access_order.items()))
            del self._windows[oldest_key]
            del self._access_order[oldest_key]

    # -- public API ------------------------------------------------------

    def is_allowed(self, key: str) -> bool:
        """Atomically check the limit and record the attempt.

        Returns True if *key* is under the limit **and** the attempt was
        recorded (one unit consumed).
        """
        with self._lock:
            self._prune(key)
            window = self._windows.get(key)
            if window is not None and len(window) >= self._config.max_requests:
                return False
            self._windows[key].append(time.time())
            self._prune(key)  # prune again so count stays bounded
            return True

    def remaining(self, key: str) -> int:
        """Return remaining slots in the window for *key*."""
        with self._lock:
            self._prune(key)
            window = self._windows.get(key)
            current = len(window) if window else 0
            return max(0, self._config.max_requests - current)

    def limit(self, key: str) -> int:
        """Return max requests per window."""
        _ = key
        return self._config.max_requests

    def retry_after(self, key: str) -> float:
        """Seconds until the window resets for *key* (0.0 if under limit)."""
        with self._lock:
            self._prune(key)
            window = self._windows.get(key)
            if window is None or len(window) < self._config.max_requests:
                return 0.0
            now = time.time()
            oldest = min(window)
            return max(1.0, self._config.window_seconds - (now - oldest))

    def reset(self, key: str) -> None:
        """Reset the window for *key*."""
        with self._lock:
            self._windows.pop(key, None)
            self._access_order.pop(key, None)

    def reset_all(self) -> None:
        """Reset all windows."""
        with self._lock:
            self._windows.clear()
            self._access_order.clear()


# ---------------------------------------------------------------------------
# Redis strategy  (lazy import, safe to use without redis-py installed)
# ---------------------------------------------------------------------------


class RedisRateLimiter:
    """Distributed rate limiter backed by Redis sorted sets.

    Uses a ZSET with timestamps as scores for precise sliding-window
    tracking.  Falls back to a local :class:`TokenBucket` when
    *redis_client* is ``None`` (a warning is logged on first use).

    Args:
        config: Rate limit configuration.
        redis_client: An instance of ``redis.Redis`` (or compatible).
            When ``None`` a local token-bucket fallback is used instead of
            distributed limiting.
    """

    def __init__(self, config: RateLimitConfig, redis_client: Any = None) -> None:
        self._config = config
        self._redis = redis_client
        self._fallback: TokenBucket | None = None
        if redis_client is None:
            logger.warning(
                "RedisRateLimiter: no redis_client provided -- using local "
                "token-bucket fallback. Distributed limiting will not work."
            )
            self._fallback = TokenBucket(config)

    # -- helpers ---------------------------------------------------------

    def _key(self, suffix: str) -> str:
        return f"{self._config.redis_prefix}{suffix}"

    def _check_and_record_redis(self, key: str) -> bool:
        """Atomic check-and-record via Redis pipeline.

        Returns True if under limit and the attempt was recorded, False
        if rate limited.  Fail-opens on Redis errors.
        """
        try:
            import redis as redis_module
        except ImportError:
            logger.warning("redis-py not installed — allowing request (fail open)")
            return True

        now = time.time()
        cutoff = now - self._config.window_seconds
        redis_key = self._key(key)
        try:
            pipe = self._redis.pipeline()  # type: ignore[union-attr]
            pipe.zremrangebyscore(redis_key, 0, cutoff)
            pipe.zcard(redis_key)
            results = pipe.execute()
            current_count = results[1] if results[1] is not None else 0
            if current_count >= self._config.max_requests:
                return False
            self._redis.zadd(redis_key, mapping={str(now): now})  # type: ignore[union-attr]
            self._redis.expire(redis_key, self._config.redis_ttl)  # type: ignore[union-attr]
            return True
        except redis_module.RedisError:
            logger.warning("Redis error in rate limiter, allowing request (fail open)")
            return True

    def _count_redis(self, key: str) -> int | None:
        """Query current count from Redis, or ``None`` on error."""
        try:
            import redis as redis_module
        except ImportError:
            return None

        now = time.time()
        cutoff = now - self._config.window_seconds
        redis_key = self._key(key)
        try:
            count = self._redis.zcount(redis_key, cutoff, now)  # type: ignore[union-attr]
            return int(count) if count is not None else 0
        except redis_module.RedisError:
            logger.warning("Redis error reading rate-limit state")
            return None

    def _oldest_redis(self, key: str) -> float | None:
        """Return the oldest timestamp for *key* from Redis, or ``None``."""
        try:
            import redis as redis_module
        except ImportError:
            return None

        now = time.time()
        redis_key = self._key(key)
        try:
            oldest = self._redis.zrange(redis_key, 0, 0, withscores=True)  # type: ignore[union-attr]
            if oldest:
                return oldest[0][1]
            return None
        except redis_module.RedisError:
            return None

    # -- public API ------------------------------------------------------

    def is_allowed(self, key: str) -> bool:
        """Atomically check and record an attempt for *key*.

        Returns True if under the limit and the attempt was recorded.
        """
        if self._fallback is not None:
            return self._fallback.is_allowed(key)
        return self._check_and_record_redis(key)

    def remaining(self, key: str) -> int:
        """Return remaining slots for *key*."""
        if self._fallback is not None:
            return self._fallback.remaining(key)
        count = self._count_redis(key)
        if count is None:
            return 0  # conservative estimate on error
        return max(0, self._config.max_requests - count)

    def limit(self, key: str) -> int:
        """Return max requests per window."""
        _ = key
        return self._config.max_requests

    def retry_after(self, key: str) -> float:
        """Seconds until the window resets for *key*."""
        if self._fallback is not None:
            return self._fallback.retry_after(key)
        count = self._count_redis(key)
        if count is None or count < self._config.max_requests:
            return 0.0
        oldest = self._oldest_redis(key)
        if oldest is not None:
            return max(1.0, self._config.window_seconds - (time.time() - oldest))
        return 1.0

    def reset(self, key: str) -> None:
        """Reset rate-limit state for *key*."""
        if self._fallback is not None:
            self._fallback.reset(key)
        else:
            try:
                self._redis.delete(self._key(key))  # type: ignore[union-attr]
            except Exception:
                pass

    def reset_all(self) -> None:
        """Reset all rate-limit state.

        .. note::
            For the Redis backend this is intentionally a no-op because
            scanning for all ``rate_limit:*`` keys is expensive.  Callers
            should use ``FLUSHDB`` or key-pattern deletion at the Redis
            level when a full reset is required.
        """
        if self._fallback is not None:
            self._fallback.reset_all()


# ---------------------------------------------------------------------------
# Strategy protocol & factory
# ---------------------------------------------------------------------------


class _Strategy(Protocol):
    """Interface satisfied by all strategy implementations."""

    def is_allowed(self, key: str) -> bool: ...
    def remaining(self, key: str) -> int: ...
    def limit(self, key: str) -> int: ...
    def retry_after(self, key: str) -> float: ...
    def reset(self, key: str) -> None: ...
    def reset_all(self) -> None: ...


def _build_strategy(config: RateLimitConfig, redis_client: Any = None) -> _Strategy:
    """Factory: return the strategy implementation matching *config.strategy*."""
    if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
        return TokenBucket(config)
    if config.strategy == RateLimitStrategy.SLIDING_WINDOW:
        return SlidingWindow(config)
    if config.strategy == RateLimitStrategy.REDIS:
        return RedisRateLimiter(config, redis_client=redis_client)
    raise ValueError(f"Unknown rate-limit strategy: {config.strategy!r}")


# ---------------------------------------------------------------------------
# Unified RateLimiter facade
# ---------------------------------------------------------------------------


class RateLimiter:
    """Unified rate limiter wrapping multiple strategies with hierarchical limits.

    Supports up to **four tiers** of checking, evaluated in order:

    1. **Global** — a system-wide cap applied to all requests.
    2. **Tenant** — a per-tenant cap (e.g. per organisation).
    3. **Model** — a per-model cap (e.g. per deployed model).
    4. **Endpoint per client** — a per-(client, endpoint) cap.

    A request must pass **all** configured tiers.  The first tier that
    fails immediately denies the request.

    Each tier can use a different strategy and configuration.

    Args:
        default_config: Default config for endpoints without an override.
        endpoint_configs: Per-endpoint config overrides keyed by URL path.
        global_config: Optional global-tier config.
        tenant_configs: Per-tenant config overrides keyed by tenant ID.
        model_configs: Per-model config overrides keyed by model name.
        tenant_default_config: Fallback config for tenants not in
            *tenant_configs*.  When ``None`` the tenant tier is skipped
            for unrecognised tenants.
        model_default_config: Fallback config for models not in
            *model_configs*.  When ``None`` the model tier is skipped
            for unrecognised models.
        redis_client: Redis client instance; required only when any config
            uses ``REDIS`` strategy.
    """

    def __init__(
        self,
        default_config: RateLimitConfig | None = None,
        *,
        endpoint_configs: dict[str, RateLimitConfig] | None = None,
        global_config: RateLimitConfig | None = None,
        tenant_configs: dict[str, RateLimitConfig] | None = None,
        model_configs: dict[str, RateLimitConfig] | None = None,
        tenant_default_config: RateLimitConfig | None = None,
        model_default_config: RateLimitConfig | None = None,
        redis_client: Any = None,
    ) -> None:
        self._default_config = default_config or RateLimitConfig()
        self._redis_client = redis_client

        # Build strategy instances
        self._default_strategy = _build_strategy(self._default_config, redis_client)

        self._endpoint_configs = endpoint_configs or {}
        self._endpoint_strategies: dict[str, _Strategy] = {
            ep: _build_strategy(cfg, redis_client)
            for ep, cfg in self._endpoint_configs.items()
        }

        self._global_config = global_config
        self._global_strategy: _Strategy | None = (
            _build_strategy(global_config, redis_client) if global_config else None
        )

        self._tenant_configs = tenant_configs or {}
        self._tenant_strategies: dict[str, _Strategy] = {
            t: _build_strategy(cfg, redis_client) for t, cfg in self._tenant_configs.items()
        }
        self._tenant_default_strategy: _Strategy | None = (
            _build_strategy(tenant_default_config, redis_client)
            if tenant_default_config
            else None
        )

        self._model_configs = model_configs or {}
        self._model_strategies: dict[str, _Strategy] = {
            m: _build_strategy(cfg, redis_client) for m, cfg in self._model_configs.items()
        }
        self._model_default_strategy: _Strategy | None = (
            _build_strategy(model_default_config, redis_client)
            if model_default_config
            else None
        )

    # -- strategy resolution ---------------------------------------------

    def _get_endpoint_strategy(self, endpoint: str) -> _Strategy:
        return self._endpoint_strategies.get(endpoint, self._default_strategy)

    def _get_tenant_strategy(self, tenant: str) -> _Strategy | None:
        strategy = self._tenant_strategies.get(tenant)
        if strategy is not None:
            return strategy
        return self._tenant_default_strategy

    def _get_model_strategy(self, model: str) -> _Strategy | None:
        strategy = self._model_strategies.get(model)
        if strategy is not None:
            return strategy
        return self._model_default_strategy

    # -- public API ------------------------------------------------------

    def is_allowed(
        self,
        client_id: str,
        endpoint: str,
        tenant: str | None = None,
        model: str | None = None,
    ) -> bool:
        """Check whether a request is allowed through all applicable tiers.

        This **consumes** capacity from each tier (token, window slot, etc.)
        when the request is allowed.  Do not call this more than once per
        request.

        Args:
            client_id: Unique client identifier (API-key ID, IP address, …).
            endpoint: Request path (e.g. ``/v1/chat/completions``).
            tenant: Optional tenant ID for tenant-tier checking.
            model: Optional model name for model-tier checking.

        Returns:
            True if the request passes all applicable rate limits (capacity
            is consumed from every tier).
        """
        # Tier 1 -- Global
        if self._global_strategy is not None:
            if not self._global_strategy.is_allowed("__global__"):
                return False

        # Tier 2 -- Tenant
        if tenant is not None:
            tenant_strat = self._get_tenant_strategy(tenant)
            if tenant_strat is not None:
                if not tenant_strat.is_allowed(f"tenant:{tenant}"):
                    return False

        # Tier 3 -- Model
        if model is not None:
            model_strat = self._get_model_strategy(model)
            if model_strat is not None:
                if not model_strat.is_allowed(f"model:{model}"):
                    return False

        # Tier 4 -- Endpoint per client
        strategy = self._get_endpoint_strategy(endpoint)
        key = f"{client_id}:{endpoint}"
        return strategy.is_allowed(key)

    def get_limits(
        self,
        client_id: str,
        endpoint: str,
        tenant: str | None = None,
        model: str | None = None,
    ) -> dict[str, dict[str, int | float]]:
        """Return current rate-limit state for all applicable tiers.

        Returns a dict keyed by tier name (``"global"``, ``"tenant"``,
        ``"model"``, ``"endpoint"``), each containing ``limit``,
        ``remaining``, and ``retry_after``.

        .. note::
            This does **not** consume capacity.  It is safe to call
            alongside ``is_allowed`` for observability.
        """
        result: dict[str, dict[str, int | float]] = {}

        if self._global_strategy is not None:
            result["global"] = {
                "limit": self._global_strategy.limit("__global__"),
                "remaining": self._global_strategy.remaining("__global__"),
                "retry_after": self._global_strategy.retry_after("__global__"),
            }

        if tenant is not None:
            tenant_strat = self._get_tenant_strategy(tenant)
            if tenant_strat is not None:
                result["tenant"] = {
                    "limit": tenant_strat.limit(f"tenant:{tenant}"),
                    "remaining": tenant_strat.remaining(f"tenant:{tenant}"),
                    "retry_after": tenant_strat.retry_after(f"tenant:{tenant}"),
                }

        if model is not None:
            model_strat = self._get_model_strategy(model)
            if model_strat is not None:
                result["model"] = {
                    "limit": model_strat.limit(f"model:{model}"),
                    "remaining": model_strat.remaining(f"model:{model}"),
                    "retry_after": model_strat.retry_after(f"model:{model}"),
                }

        strategy = self._get_endpoint_strategy(endpoint)
        key = f"{client_id}:{endpoint}"
        result["endpoint"] = {
            "limit": strategy.limit(key),
            "remaining": strategy.remaining(key),
            "retry_after": strategy.retry_after(key),
        }

        return result

    def retry_after(
        self,
        client_id: str,
        endpoint: str,
        tenant: str | None = None,
        model: str | None = None,
    ) -> float:
        """Return the maximum ``Retry-After`` across all tiers.

        Useful for populating the ``Retry-After`` response header when a
        request was denied.
        """
        max_wait = 0.0

        if self._global_strategy is not None:
            max_wait = max(max_wait, self._global_strategy.retry_after("__global__"))

        if tenant is not None:
            tenant_strat = self._get_tenant_strategy(tenant)
            if tenant_strat is not None:
                max_wait = max(max_wait, tenant_strat.retry_after(f"tenant:{tenant}"))

        if model is not None:
            model_strat = self._get_model_strategy(model)
            if model_strat is not None:
                max_wait = max(max_wait, model_strat.retry_after(f"model:{model}"))

        strategy = self._get_endpoint_strategy(endpoint)
        key = f"{client_id}:{endpoint}"
        max_wait = max(max_wait, strategy.retry_after(key))

        return max_wait

    def reset_client(self, client_id: str, endpoint: str | None = None) -> None:
        """Reset rate-limit state for *client_id*.

        Args:
            client_id: Client identifier to reset.
            endpoint: If provided, only this endpoint is reset for the client.
                Otherwise all stored endpoints for the client are reset.
        """
        if endpoint is not None:
            strategy = self._get_endpoint_strategy(endpoint)
            strategy.reset(f"{client_id}:{endpoint}")
        else:
            for ep, strategy in self._endpoint_strategies.items():
                strategy.reset(f"{client_id}:{ep}")
            self._default_strategy.reset(f"{client_id}:*")

    def reset_all(self) -> None:
        """Reset all rate-limit state across every tier."""
        if self._global_strategy is not None:
            self._global_strategy.reset_all()
        for strategy in self._tenant_strategies.values():
            strategy.reset_all()
        if self._tenant_default_strategy is not None:
            self._tenant_default_strategy.reset_all()
        for strategy in self._model_strategies.values():
            strategy.reset_all()
        if self._model_default_strategy is not None:
            self._model_default_strategy.reset_all()
        for strategy in self._endpoint_strategies.values():
            strategy.reset_all()
        self._default_strategy.reset_all()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate-limiting ASGI middleware backed by a :class:`RateLimiter`.

    This middleware is designed as a drop-in successor to both the legacy
    ``_RateLimiter`` auth-throttle and ``RequestRateLimitMiddleware`` from
    ``middleware.py``.

    Health/liveness endpoints and ``OPTIONS`` (CORS preflight) are excluded
    by default.

    Args:
        rate_limiter: A preconfigured :class:`RateLimiter` instance.
        get_client_id: Callable returning a client identifier from the
            request.  Default: ``request.state.api_key_id`` with a fallback
            to the client IP.
        get_tenant: Callable returning a tenant ID (or ``None``).
            Default: ``request.state.api_key_role`` with a fallback to the
            ``X-Tenant-ID`` header.
        get_model: Callable returning a model name (or ``None``).
            Default: ``X-Model-ID`` header.
        exclude_paths: Set of URL paths exempt from rate limiting.
    """

    def __init__(
        self,
        app: Any,
        rate_limiter: RateLimiter,
        *,
        enabled: bool = True,
        get_client_id: Callable[[Request], str] | None = None,
        get_tenant: Callable[[Request], str | None] | None = None,
        get_model: Callable[[Request], str | None] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._rate_limiter = rate_limiter
        self._enabled = enabled
        self._get_client_id = get_client_id or self._default_get_client_id
        self._get_tenant = get_tenant or self._default_get_tenant
        self._get_model = get_model or self._default_get_model
        self._exclude_paths = exclude_paths or {
            "/health",
            "/ready",
            "/live",
            "/metrics",
        }

    # -- default extraction helpers --------------------------------------

    @staticmethod
    def _default_get_client_id(request: Request) -> str:
        api_key_id = getattr(request.state, "api_key_id", None)
        if api_key_id is not None:
            return str(api_key_id)
        return get_client_ip(request)

    @staticmethod
    def _default_get_tenant(request: Request) -> str | None:
        role = getattr(request.state, "api_key_role", None)
        if role is not None:
            return str(role)
        return request.headers.get("X-Tenant-ID")

    @staticmethod
    def _default_get_model(request: Request) -> str | None:
        return request.headers.get("X-Model-ID")

    # -- dispatch --------------------------------------------------------

    async def dispatch(self, request: Request, call_next: Any) -> Response:  # type: ignore[override]
        # Disabled middleware passes everything through untouched.
        if not self._enabled:
            return await call_next(request)

        # Exempt health/probe endpoints
        if request.url.path in self._exclude_paths:
            return await call_next(request)

        # Exempt CORS preflight (see middleware.py for rationale)
        if request.method == "OPTIONS":
            return await call_next(request)

        client_id = self._get_client_id(request)
        endpoint = request.url.path
        tenant = self._get_tenant(request)
        model = self._get_model(request)

        if not self._rate_limiter.is_allowed(
            client_id=client_id,
            endpoint=endpoint,
            tenant=tenant,
            model=model,
        ):
            try:
                try:
                    _limits = self._rate_limiter.get_limits(
                        client_id=client_id, endpoint=endpoint,
                        tenant=tenant, model=model,
                    )
                except TypeError:
                    _limits = self._rate_limiter.get_limits(client_id, endpoint)
                if isinstance(_limits, tuple):
                    _l, _r, _ra = _limits
                    limits_info = {"limit": _l, "remaining": _r, "retry_after": _ra}
                else:
                    limits_info = _limits.get("endpoint", {})
            except Exception:
                limits_info = {}
            retry_after = self._rate_limiter.retry_after(
                client_id=client_id,
                endpoint=endpoint,
                tenant=tenant,
                model=model,
            )
            logger.warning(
                "Rate limit exceeded | client={} endpoint={} tenant={} model={} retry_after={}",
                client_id,
                endpoint,
                tenant,
                model,
                retry_after,
            )
            # Retry-After must be at least 1 second (HTTP semantics: whole
            # seconds; sub-second values read as "0" to clients).
            retry_after_s = max(1, int(retry_after))
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"Rate limit exceeded. Retry after {retry_after_s}s.",
                        "type": "rate_limit_error",
                        "param": None,
                        "code": "429",
                        "retry_after": retry_after_s,
                    }
                },
                headers={
                    "Retry-After": str(retry_after_s),
                    "X-RateLimit-Limit": str(limits_info.get("limit", "")),
                    "X-RateLimit-Remaining": str(limits_info.get("remaining", 0)),
                },
            )

        response = await call_next(request)

        # Attach standard rate-limit response headers
        try:
            try:
                limits = self._rate_limiter.get_limits(
                    client_id=client_id,
                    endpoint=endpoint,
                    tenant=tenant,
                    model=model,
                )
            except TypeError:
                # Flat limiter interface: get_limits(client_id, endpoint).
                limits = self._rate_limiter.get_limits(client_id, endpoint)
            if isinstance(limits, tuple):
                limit, remaining, retry_after = limits
                ep_limits = {
                    "limit": limit,
                    "remaining": remaining,
                    "retry_after": retry_after,
                }
            else:
                ep_limits = limits.get("endpoint", {})
            if ep_limits:
                response.headers["X-RateLimit-Limit"] = str(ep_limits.get("limit", ""))
                response.headers["X-RateLimit-Remaining"] = str(
                    ep_limits.get("remaining", "")
                )
                reset_at = time.time() + float(ep_limits.get("retry_after", 0))
                response.headers["X-RateLimit-Reset"] = str(int(reset_at))
        except Exception:
            logger.opt(exception=True).debug("Failed to attach rate-limit headers")

        return response
