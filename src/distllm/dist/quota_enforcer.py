"""Pipeline-level multi-tenant quota enforcement and fair scheduling.

Extends the scheduler-level ``TenantBudget`` with:
- Burst credits (token bucket with max burst)
- Per-tenant priority queuing (fair scheduling across tenants)
- Pipeline admission control (reject when quota exhausted)
- Cross-tenant fairness (weighted fair queuing)

Usage::

    enforcer = QuotaEnforcer()
    enforcer.set_tenant_quota("tenant-a", tokens_per_minute=10000, burst=5000)
    enforcer.set_tenant_quota("tenant-b", tokens_per_minute=5000, burst=2000)

    # Before processing a request:
    if enforcer.try_consume("tenant-a", tokens=128):
        process(request)
    else:
        enforcer.enqueue("tenant-a", request)  # queue for later

    # In the decode loop:
    batch = enforcer.select_next(available_slots=4)
    # -> returns a batch respecting tenant fairness and burst credits
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TenantQuota:
    """Per-tenant quota with token bucket and burst support."""
    tenant_id: str
    tokens_per_minute: float = 10000.0
    burst: float = 5000.0              # max accumulated credits
    weight: float = 1.0                # for weighted fair queuing

    # Token bucket state
    tokens: float = 5000.0             # current tokens (starts full)
    last_refill: float = field(default_factory=time.time)
    total_consumed: float = 0.0

    # Request queue for this tenant
    queued_requests: list[dict[str, Any]] = field(default_factory=list)

    def refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        added = elapsed * (self.tokens_per_minute / 60.0)
        self.tokens = min(self.burst, self.tokens + added)
        self.last_refill = now

    def can_consume(self, amount: float) -> bool:
        self.refill()
        return self.tokens >= amount

    def consume(self, amount: float) -> bool:
        self.refill()
        if self.tokens < amount:
            return False
        self.tokens -= amount
        self.total_consumed += amount
        return True

    @property
    def utilization(self) -> float:
        return min(self.total_consumed / max(self.tokens_per_minute, 1), 1.0)


class QuotaEnforcer:
    """Multi-tenant quota enforcement with fair scheduling.

    Manages per-tenant token buckets, burst credits, and request
    queuing.  The ``select_next()`` method builds a batch that
    respects tenant fairness via weighted deficit round-robin.
    """

    def __init__(self) -> None:
        self._quotas: dict[str, TenantQuota] = {}
        self._lock = threading.RLock()
        self._deficit: dict[str, float] = {}  # for DRR
        self._quantum: int = 256  # tokens per round

    # ── Quota configuration ────────────────────────────────────────────

    def set_tenant_quota(
        self,
        tenant_id: str,
        tokens_per_minute: float = 10000.0,
        burst: float = 5000.0,
        weight: float = 1.0,
    ) -> TenantQuota:
        """Set or update a tenant's quota.

        Args:
            tenant_id: Unique tenant identifier.
            tokens_per_minute: Sustained token throughput.
            burst: Maximum burst (accumulated credits).
            weight: Scheduling weight for fair queuing (1.0 = normal).

        Returns:
            The ``TenantQuota`` instance.
        """
        with self._lock:
            if tenant_id in self._quotas:
                q = self._quotas[tenant_id]
                q.tokens_per_minute = tokens_per_minute
                q.burst = burst
                q.weight = weight
            else:
                q = TenantQuota(
                    tenant_id=tenant_id,
                    tokens_per_minute=tokens_per_minute,
                    burst=burst,
                    weight=weight,
                )
                self._quotas[tenant_id] = q
                self._deficit[tenant_id] = 0.0
            return q

    def remove_tenant(self, tenant_id: str) -> None:
        with self._lock:
            self._quotas.pop(tenant_id, None)
            self._deficit.pop(tenant_id, None)

    def get_quota(self, tenant_id: str) -> TenantQuota | None:
        with self._lock:
            return self._quotas.get(tenant_id)

    def list_quotas(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "tenant_id": q.tenant_id,
                    "tokens_per_minute": q.tokens_per_minute,
                    "burst": q.burst,
                    "tokens": round(q.tokens, 1),
                    "utilization": round(q.utilization, 3),
                    "queued": len(q.queued_requests),
                }
                for q in self._quotas.values()
            ]

    # ── Token consumption ──────────────────────────────────────────────

    def try_consume(self, tenant_id: str, tokens: int = 1) -> bool:
        """Try to consume *tokens* for a tenant.

        Returns True if the tenant has sufficient quota.
        """
        with self._lock:
            q = self._quotas.get(tenant_id)
            if q is None:
                return True  # unknown tenants are not rate-limited
            return q.consume(tokens)

    def can_consume(self, tenant_id: str, tokens: int = 1) -> bool:
        """Check if a tenant can consume *tokens* without consuming."""
        with self._lock:
            q = self._quotas.get(tenant_id)
            if q is None:
                return True
            return q.can_consume(tokens)

    def refill_all(self) -> None:
        """Refill all token buckets (called periodically)."""
        with self._lock:
            for q in self._quotas.values():
                q.refill()

    # ── Request queuing ────────────────────────────────────────────────

    def enqueue(self, tenant_id: str, request: dict[str, Any]) -> None:
        """Queue a request for a tenant (when quota is exhausted).

        Queued requests will be returned by ``select_next()`` when
        the tenant's quota recovers.
        """
        with self._lock:
            q = self._quotas.get(tenant_id)
            if q is None:
                return
            q.queued_requests.append(request)

    def dequeue(self, tenant_id: str) -> dict[str, Any] | None:
        """Dequeue the next request for a tenant."""
        with self._lock:
            q = self._quotas.get(tenant_id)
            if q is None or not q.queued_requests:
                return None
            return q.queued_requests.pop(0)

    def queue_depth(self, tenant_id: str) -> int:
        with self._lock:
            q = self._quotas.get(tenant_id)
            return len(q.queued_requests) if q else 0

    # ── Fair scheduling (Weighted Deficit Round-Robin) ─────────────────

    def select_next(
        self,
        available_slots: int,
        min_tokens: int = 1,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Build a batch respecting tenant fairness.

        Uses weighted deficit round-robin across tenants with queued
        requests.  Tenants with higher weight get more slots.

        Args:
            available_slots: Maximum requests to return.
            min_tokens: Minimum tokens needed per request.

        Returns:
            List of ``(tenant_id, request)`` tuples.
        """
        batch: list[tuple[str, dict[str, Any]]] = []
        with self._lock:
            if not self._quotas:
                return batch

            candidates = [
                tid for tid, q in self._quotas.items()
                if q.queued_requests
            ]
            if not candidates:
                return batch

            # DRR: credit each tenant, serve those with enough credit
            for _ in range(available_slots):
                best_tid = None
                for tid in candidates:
                    q = self._quotas[tid]
                    self._deficit[tid] = self._deficit.get(tid, 0.0) + q.weight * self._quantum
                    if (self._deficit[tid] >= min_tokens
                            and q.queued_requests
                            and q.can_consume(min_tokens)):
                        if best_tid is None or self._deficit[tid] > self._deficit.get(best_tid, 0):
                            best_tid = tid

                if best_tid is None:
                    break

                req = self._quotas[best_tid].queued_requests.pop(0)
                self._deficit[best_tid] -= min_tokens
                batch.append((best_tid, req))

        return batch

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tenants": len(self._quotas),
                "total_queued": sum(len(q.queued_requests) for q in self._quotas.values()),
                "quotas": self.list_quotas(),
            }
