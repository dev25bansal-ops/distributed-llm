"""Rate Limiter with Leaky Bucket: smooth rate limiting for request throttling.

Replaces token bucket with a leaky bucket algorithm:

  - Fixed drain rate (tokens/second)
  - Burst capacity (maximum accumulated tokens)
  - Per-user, per-IP, or per-endpoint buckets
  - Leaky bucket produces smoother traffic vs token bucket bursts
  - Optional exponential backoff on repeated violations
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LeakyBucket:
    """A single leaky bucket instance."""
    key: str
    rate_per_sec: float       # Drain rate (tokens/second)
    burst: int                # Maximum capacity
    water: float = 0.0        # Current water level
    last_drip: float = field(default_factory=time.time)
    consecutive_violations: int = 0

    def _drip(self) -> None:
        """Drip water out based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_drip
        drained = elapsed * self.rate_per_sec
        self.water = max(0.0, self.water - drained)
        self.last_drip = now

    def try_add(self, tokens: float = 1.0) -> bool:
        """Try to add tokens to the bucket. Returns True if allowed."""
        self._drip()
        if self.water + tokens <= self.burst:
            self.water += tokens
            self.consecutive_violations = 0
            return True
        self.consecutive_violations += 1
        return False

    def wait_time(self) -> float:
        """Time until bucket can accept 1 token."""
        if self.water < self.burst:
            return 0.0
        excess = self.water + 1 - self.burst
        return excess / max(self.rate_per_sec, 0.001)

    def reset(self) -> None:
        self.water = 0.0
        self.last_drip = time.time()
        self.consecutive_violations = 0


class LeakyBucketRateLimiter:
    """Rate limiter using leaky bucket algorithm with per-key isolation.

    Features:
      - Per-user, per-IP, per-endpoint rate limiting
      - Burst capacity with smooth leak
      - Exponential backoff for repeat violators
      - Configurable default limits

    Usage:
        limiter = LeakyBucketRateLimiter(default_rate=10, default_burst=20)

        # Check a request
        if limiter.check("user:abc123"):
            process_request()
        else:
            return 429 Too Many Requests

        # Per-key custom limits
        limiter.set_limit("user:admin", rate=100, burst=200)
    """

    def __init__(
        self,
        default_rate: float = 10.0,     # requests/second
        default_burst: int = 20,
        enable_backoff: bool = True,
        backoff_multiplier: float = 2.0,
        backoff_max_sec: float = 300.0,
        cleanup_interval_s: float = 300.0,
    ):
        self._default_rate = default_rate
        self._default_burst = default_burst
        self._enable_backoff = enable_backoff
        self._backoff_mult = backoff_multiplier
        self._backoff_max = backoff_max_sec
        self._cleanup_interval = cleanup_interval_s

        self._buckets: dict[str, LeakyBucket] = {}
        self._custom_limits: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def check(self, key: str, tokens: float = 1.0) -> bool:
        """Check if a request should be allowed. Returns True if allowed."""
        now = time.time()
        with self._lock:
            self._maybe_cleanup(now)
            bucket = self._get_bucket(key)

            # Exponential backoff
            if self._enable_backoff and bucket.consecutive_violations > 3:
                backoff_sec = min(
                    self._backoff_mult ** (bucket.consecutive_violations - 3),
                    self._backoff_max,
                )
                if now - bucket.last_drip < backoff_sec:
                    return False

            allowed = bucket.try_add(tokens)

            if not allowed and self._enable_backoff:
                bucket.consecutive_violations += 1

            return allowed

    def check_many(self, keys: list[str], tokens: float = 1.0) -> dict[str, bool]:
        """Check multiple keys in one call."""
        return {key: self.check(key, tokens) for key in keys}

    def set_limit(self, key: str, rate: float, burst: int) -> None:
        """Set custom rate/burst limits for a key."""
        with self._lock:
            self._custom_limits[key] = (rate, burst)
            if key in self._buckets:
                bucket = self._buckets[key]
                bucket.rate_per_sec = rate
                bucket.burst = burst
                bucket.water = min(bucket.water, burst)

    def reset_key(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        with self._lock:
            if key in self._buckets:
                self._buckets[key].reset()

    def remaining(self, key: str) -> float:
        """Return remaining capacity for a key."""
        with self._lock:
            bucket = self._get_bucket(key)
            bucket._drip()
            return max(0.0, bucket.burst - bucket.water)

    def _get_bucket(self, key: str) -> LeakyBucket:
        """Get or create bucket for key."""
        if key not in self._buckets:
            rate, burst = self._custom_limits.get(
                key, (self._default_rate, self._default_burst)
            )
            self._buckets[key] = LeakyBucket(
                key=key, rate_per_sec=rate, burst=burst
            )
        return self._buckets[key]

    def _maybe_cleanup(self, now: float) -> None:
        """Remove stale buckets to prevent memory leak."""
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale_keys = [
            k for k, b in self._buckets.items()
            if now - b.last_drip > self._cleanup_interval * 2
        ]
        for k in stale_keys:
            del self._buckets[k]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total_buckets = len(self._buckets)
            active = sum(
                1 for b in self._buckets.values()
                if b.water > 0 or b.consecutive_violations > 0
            )
            violators = sum(
                1 for b in self._buckets.values()
                if b.consecutive_violations > 3
            )
            return {
                "total_keys": total_buckets,
                "active_keys": active,
                "violators_in_backoff": violators,
                "default_rate": self._default_rate,
                "default_burst": self._default_burst,
            }
