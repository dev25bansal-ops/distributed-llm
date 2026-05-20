"""Per-tenant rate limiter with configurable RPM limits."""

import time
from collections import defaultdict

from distllm.tenant.models import Tenant, TenantTier, ResourceQuota


class TokenBucket:
    def __init__(self, rate_per_minute: float, burst: int | None = None):
        self.rpm = rate_per_minute
        self.rps = rate_per_minute / 60.0
        self.max_tokens = burst or int(rate_per_minute * 1.5)
        self.tokens = float(self.max_tokens)
        self.last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rps)
        self.last_refill = now

    def consume(self) -> bool:
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def remaining(self) -> int:
        self._refill()
        return int(self.tokens)

    def retry_after(self) -> float:
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.rps


class TenantRateLimiter:
    """Per-tenant rate limiter using token buckets.

    Enforces max_rpm from the tenant's ResourceQuota.
    Tracks separate buckets for requests and tokens (TPM).
    """

    def __init__(self):
        self._req_buckets: dict[str, TokenBucket] = {}
        self._tpm_buckets: dict[str, TokenBucket] = {}

    def check_request(self, tenant_id: str, quota: ResourceQuota) -> bool:
        if tenant_id not in self._req_buckets:
            self._req_buckets[tenant_id] = TokenBucket(quota.max_rpm)
        return self._req_buckets[tenant_id].consume()

    def check_tokens(self, tenant_id: str, quota: ResourceQuota, estimated_tokens: int = 1) -> bool:
        if tenant_id not in self._tpm_buckets:
            self._tpm_buckets[tenant_id] = TokenBucket(quota.max_tpm)
        bucket = self._tpm_buckets[tenant_id]
        for _ in range(estimated_tokens):
            if not bucket.consume():
                return False
        return True

    def get_limits(self, tenant_id: str, quota: ResourceQuota) -> dict:
        rps = self._req_buckets.get(tenant_id)
        tps = self._tpm_buckets.get(tenant_id)
        return {
            "rpm_limit": quota.max_rpm,
            "tpm_limit": quota.max_tpm,
            "requests_remaining": rps.remaining() if rps else quota.max_rpm,
            "tokens_remaining": tps.remaining() if tps else quota.max_tpm,
            "retry_after_seconds": max(
                rps.retry_after() if rps else 0,
                tps.retry_after() if tps else 0,
            ),
        }

    def reset_tenant(self, tenant_id: str):
        self._req_buckets.pop(tenant_id, None)
        self._tpm_buckets.pop(tenant_id, None)
