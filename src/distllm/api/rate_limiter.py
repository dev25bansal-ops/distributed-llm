"""Token bucket rate limiter for API requests."""

import time
from collections import defaultdict, OrderedDict


class TokenBucket:
    """Token bucket rate limiter for a single client.

    Tokens are added at a fixed rate (requests per minute).
    Each request consumes one token. Burst allows up to burst_size tokens.
    """

    def __init__(self, rate_per_minute: float, burst_multiplier: float = 1.5):
        self.rate_per_minute = rate_per_minute
        self.rate_per_second = rate_per_minute / 60.0
        self.burst_size = int(rate_per_minute * burst_multiplier)
        self.tokens = float(self.burst_size)
        self.max_tokens = float(self.burst_size)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate_per_second)
        self.last_refill = now

    def consume(self) -> bool:
        """Try to consume a token.

        Returns:
            True if request is allowed, False if rate limited.
        """
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def get_remaining(self) -> int:
        """Get remaining tokens."""
        self._refill()
        return int(self.tokens)

    def get_retry_after(self) -> float:
        """Get seconds until next token is available."""
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        return deficit / self.rate_per_second


class RateLimiter:
    """Per-client rate limiter using token buckets.

    Tracks rate limits per client (by API key or IP address).
    Includes LRU eviction to prevent unbounded memory growth.
    """

    def __init__(
        self,
        default_rpm: float = 60.0,
        endpoint_limits: dict[str, float] | None = None,
        burst_multiplier: float = 1.5,
        max_clients: int = 10000,
    ):
        self.default_rpm = default_rpm
        self.endpoint_limits = endpoint_limits or {}
        self.burst_multiplier = burst_multiplier
        self.max_clients = max_clients  # Security: Prevent unbounded memory growth
        self._buckets: dict[str, dict[str, TokenBucket]] = defaultdict(dict)
        self._access_order: OrderedDict[str, bool] = OrderedDict()

    def _get_bucket(self, client_id: str, endpoint: str) -> TokenBucket:
        """Get or create a token bucket for a client+endpoint."""
        if client_id not in self._buckets:
            # Security: LRU eviction when max clients reached
            if len(self._buckets) >= self.max_clients:
                oldest, _ = self._access_order.popitem(last=False)
                self._buckets.pop(oldest, None)
            self._access_order[client_id] = True

        if endpoint not in self._buckets[client_id]:
            rpm = self.endpoint_limits.get(endpoint, self.default_rpm)
            self._buckets[client_id][endpoint] = TokenBucket(
                rate_per_minute=rpm,
                burst_multiplier=self.burst_multiplier,
            )
        return self._buckets[client_id][endpoint]

    def is_allowed(self, client_id: str, endpoint: str) -> bool:
        """Check if a request is allowed.

        Args:
            client_id: Client identifier (API key or IP).
            endpoint: Request endpoint path.

        Returns:
            True if allowed, False if rate limited.
        """
        bucket = self._get_bucket(client_id, endpoint)
        # Update access order for LRU — O(1) with OrderedDict
        self._access_order.move_to_end(client_id)
        return bucket.consume()

    def get_limits(self, client_id: str, endpoint: str) -> tuple[int, int, float]:
        """Get rate limit info for headers.

        Returns:
            (limit, remaining, retry_after_seconds)
        """
        rpm = self.endpoint_limits.get(endpoint, self.default_rpm)
        bucket = self._get_bucket(client_id, endpoint)
        remaining = bucket.get_remaining()
        retry_after = bucket.get_retry_after()
        return int(rpm), remaining, retry_after

    def reset_client(self, client_id: str) -> None:
        """Reset all rate limits for a client."""
        self._buckets.pop(client_id, None)
        if client_id in self._access_order:
            self._access_order.remove(client_id)

    def reset_all(self) -> None:
        """Reset all rate limits."""
        self._buckets.clear()
        self._access_order.clear()

    @property
    def active_clients(self) -> int:
        """Get number of active clients being tracked."""
        return len(self._buckets)
