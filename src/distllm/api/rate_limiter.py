"""Token-bucket rate limiter for per-client, per-endpoint request throttling.

Provides:
- ``TokenBucket`` — single-bucket rate limiter with burst support
- ``RateLimiter`` — per-client, per-endpoint limiter using token buckets
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class TokenBucket:
    """Token-bucket rate limiter with configurable rate and burst.

    Tokens refill continuously at ``rate_per_minute / 60`` tokens per second.
    The bucket capacity equals ``rate_per_minute * burst_multiplier``.

    Args:
        rate_per_minute: Sustained request rate allowed per minute.
        burst_multiplier: Multiplier for burst capacity (1.0 = no burst headroom).
    """

    def __init__(self, rate_per_minute: float, burst_multiplier: float = 1.0) -> None:
        self.rate_per_minute = rate_per_minute
        self.burst_multiplier = burst_multiplier
        self._rate_per_second = rate_per_minute / 60.0
        self.max_tokens: float = rate_per_minute * burst_multiplier
        self.tokens: float = self.max_tokens
        self._last_refill = time.monotonic()

    @property
    def burst_size(self) -> int:
        """Integer burst capacity (rate * burst_multiplier)."""
        return int(self.rate_per_minute * self.burst_multiplier)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self._rate_per_second)
            self._last_refill = now

    def consume(self) -> bool:
        """Try to consume one token.

        Returns:
            True if a token was consumed, False if the bucket is empty.
        """
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def get_remaining(self) -> int:
        """Return the number of whole tokens currently available."""
        self._refill()
        return int(self.tokens)

    def get_retry_after(self) -> float:
        """Return seconds until the next token becomes available.

        Returns 0.0 if tokens are currently available.
        """
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / self._rate_per_second if self._rate_per_second > 0 else 0.0


class RateLimiter:
    """Per-client, per-endpoint rate limiter using token buckets.

    Each (client_id, endpoint) pair gets its own ``TokenBucket``.
    Endpoint-specific limits override the default when configured.

    Args:
        default_rpm: Default sustained requests per minute.
        endpoint_limits: Mapping of endpoint path → RPM override.
        burst_multiplier: Burst multiplier applied to all buckets.
    """

    def __init__(
        self,
        default_rpm: float = 60.0,
        endpoint_limits: dict[str, float] | None = None,
        burst_multiplier: float = 1.0,
    ) -> None:
        self.default_rpm = default_rpm
        self.endpoint_limits = endpoint_limits or {}
        self.burst_multiplier = burst_multiplier
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, client_id: str, endpoint: str) -> TokenBucket:
        """Return or create the token bucket for (client_id, endpoint)."""
        key = (client_id, endpoint)
        bucket = self._buckets.get(key)
        if bucket is None:
            rpm = self.endpoint_limits.get(endpoint, self.default_rpm)
            bucket = TokenBucket(rate_per_minute=rpm, burst_multiplier=self.burst_multiplier)
            self._buckets[key] = bucket
        return bucket

    def is_allowed(self, client_id: str, endpoint: str, tenant: str = "", model: str = "") -> bool:
        """Check whether a request from *client_id* to *endpoint* is allowed.

        Consumes one token if available.  ``tenant``/``model`` are accepted
        for interface compatibility with the tiered limiter and ignored by
        this flat per-client bucket.

        Returns:
            True if the request should proceed, False if rate-limited.
        """
        with self._lock:
            bucket = self._get_bucket(client_id, endpoint)
            return bucket.consume()

    def get_limits(self, client_id: str, endpoint: str) -> tuple[int, int, float]:
        """Return current rate-limit info for (client_id, endpoint).

        Returns:
            Tuple of (limit, remaining, retry_after).
        """
        with self._lock:
            bucket = self._get_bucket(client_id, endpoint)
            limit = bucket.burst_size
            remaining = bucket.get_remaining()
            retry_after = bucket.get_retry_after()
            return limit, remaining, retry_after

    def retry_after(
        self, client_id: str, endpoint: str = "", tenant: str = "", model: str = ""
    ) -> float:
        """Seconds until the next request from *client_id* may proceed.

        Accepts (and ignores) tenant/model for interface compatibility with
        the tiered limiter used by the unified middleware.
        """
        with self._lock:
            bucket = self._get_bucket(client_id, endpoint)
            return bucket.get_retry_after()

    def reset_client(self, client_id: str) -> None:
        """Reset all buckets for a specific client."""
        with self._lock:
            keys_to_remove = [k for k in self._buckets if k[0] == client_id]
            for key in keys_to_remove:
                del self._buckets[key]

    def reset_all(self) -> None:
        """Reset all client buckets."""
        with self._lock:
            self._buckets.clear()


class HierarchicalRateLimiter:
    """Three-tier hierarchical rate limiter: global → tenant → model.

    Checks limits in order: global first, then tenant, then model.
    A request must pass ALL three tiers to be allowed.

    Usage::

        limiter = HierarchicalRateLimiter(
            global_rpm=10000,
            tenant_rpm=1000,
            model_rpm={"llama-70b": 100, "llama-8b": 500},
        )
        if limiter.is_allowed(tenant="acme", model="llama-70b", endpoint="/v1/chat"):
            process_request()
    """

    def __init__(
        self,
        global_rpm: float = 10000.0,
        tenant_rpm: float = 1000.0,
        model_rpm: dict[str, float] | None = None,
        burst_multiplier: float = 1.5,
    ):
        self._global = TokenBucket(rate_per_minute=global_rpm, burst_multiplier=burst_multiplier)
        self._tenant_rpm = tenant_rpm
        self._model_rpm = model_rpm or {}
        self._burst_multiplier = burst_multiplier
        self._tenant_buckets: dict[str, TokenBucket] = {}
        self._model_buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _get_tenant_bucket(self, tenant: str) -> TokenBucket:
        if tenant not in self._tenant_buckets:
            self._tenant_buckets[tenant] = TokenBucket(
                rate_per_minute=self._tenant_rpm,
                burst_multiplier=self._burst_multiplier,
            )
        return self._tenant_buckets[tenant]

    def _get_model_bucket(self, model: str) -> TokenBucket:
        if model not in self._model_buckets:
            rpm = self._model_rpm.get(model, self._tenant_rpm)
            self._model_buckets[model] = TokenBucket(
                rate_per_minute=rpm,
                burst_multiplier=self._burst_multiplier,
            )
        return self._model_buckets[model]

    def is_allowed(self, tenant: str, model: str, endpoint: str = "") -> bool:
        """Check if request passes all three tiers.

        Returns True only if global, tenant, AND model limits are OK.
        """
        with self._lock:
            # Tier 1: Global
            if not self._global.consume():
                return False

            # Tier 2: Tenant
            tenant_bucket = self._get_tenant_bucket(tenant)
            if not tenant_bucket.consume():
                return False

            # Tier 3: Model
            model_bucket = self._get_model_bucket(model)
            if not model_bucket.consume():
                return False

            return True

    def get_limits(self, tenant: str, model: str) -> dict:
        """Return current limits for all three tiers."""
        with self._lock:
            tenant_bucket = self._get_tenant_bucket(tenant)
            model_bucket = self._get_model_bucket(model)
            return {
                "global_remaining": self._global.get_remaining(),
                "global_limit": self._global.burst_size,
                "tenant_remaining": tenant_bucket.get_remaining(),
                "tenant_limit": tenant_bucket.burst_size,
                "model_remaining": model_bucket.get_remaining(),
                "model_limit": model_bucket.burst_size,
            }

    def reset_tenant(self, tenant: str) -> None:
        """Reset rate limits for a specific tenant."""
        with self._lock:
            self._tenant_buckets.pop(tenant, None)
