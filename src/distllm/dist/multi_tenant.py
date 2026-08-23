"""Multi-tenant SLO enforcement — per-tenant latency/throughput targets with priority scheduling.

Provides a priority queue layer that wraps the existing batch scheduler
with per-tenant rate limiting, latency SLO tracking, and priority boosting
for tenants approaching their SLO breach threshold.

Architecture::

    Request ──► TenantClassifier ──► Per-Tenant Bucket ──► PriorityQueue ──► BatchScheduler
                    │                       │                    │
                    │                   ┌───┴───┐          ┌────┴────┐
                    │                   │Rate   │          │ Priority│
                    │                   │Limiter│          │ Booster │
                    │                   └───┬───┘          └────┬────┘
                    │                       │                    │
                    ▼                       ▼                    ▼
            tenant_id from          admit/backpressure      boost priority if
            API key / JWT           based on RPM            SLO breach imminent

Usage::

    slo = MultiTenantSLOEnforcer()
    slo.register_tenant("tenant-a", max_rpm=1000, latency_slo_ms=500)
    slo.register_tenant("tenant-b", max_rpm=500, latency_slo_ms=2000)

    # Before scheduling:
    if slo.should_admit(tenant_id):
        slo.record_request_start(tenant_id)
        # ... schedule ...
        slo.record_request_end(tenant_id, latency_ms=120)

    # Get boost factor for priority queue:
    boost = slo.get_priority_boost(tenant_id)  # 0.0 - 2.0
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TenantSLO:
    """SLO configuration for a single tenant."""
    tenant_id: str
    max_rpm: float = 60.0          # max requests per minute
    latency_slo_ms: float = 1000.0  # p99 latency target
    max_concurrent: int = 10
    burst_multiplier: float = 2.0   # burst capacity relative to max_rpm
    priority_base: float = 1.0      # base scheduling priority


@dataclass
class TenantMetrics:
    """Live metrics for a single tenant."""
    total_requests: int = 0
    completed_requests: int = 0
    rejected_requests: int = 0
    total_latency_ms: float = 0.0
    latency_p99_ms: float = 0.0
    current_concurrent: int = 0
    rp1m: int = 0  # requests in the last minute
    slo_breaches: int = 0
    last_had_backpressure: bool = False

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.completed_requests, 1)

    @property
    def slo_compliance_pct(self) -> float:
        total = self.total_requests
        if total == 0:
            return 100.0
        breaches = self.slo_breaches
        return max(0.0, 100.0 * (1.0 - breaches / total))


class SlidingWindowRateCounter:
    """Sliding-window rate counter for per-tenant RPM tracking."""

    def __init__(self, window_s: float = 60.0):
        self._window_s = window_s
        self._slots: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def record(self, count: int = 1) -> None:
        now = time.time()
        with self._lock:
            self._slots.append((now, count))
            self._prune(now)

    def count(self) -> int:
        now = time.time()
        with self._lock:
            self._prune(now)
            return sum(c for _, c in self._slots)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._slots and self._slots[0][0] < cutoff:
            self._slots.popleft()


class MultiTenantSLOEnforcer:
    """Per-tenant rate limiting, SLO tracking, and priority boosting.

    Usage::

        enforcer = MultiTenantSLOEnforcer()

        # Register tenants with their SLAs.
        enforcer.register_tenant("acme-corp", max_rpm=1000, latency_slo_ms=500)
        enforcer.register_tenant("startup-inc", max_rpm=100, latency_slo_ms=2000)

        # Before processing a request:
        if not enforcer.should_admit(tenant_id):
            return 429  # rate limited

        # During scheduling:
        boost = enforcer.get_priority_boost(tenant_id)
        # (priority = base_priority * boost)

        # After completion:
        enforcer.record_request_end(tenant_id, latency_ms=145)
    """

    def __init__(self):
        self._slos: dict[str, TenantSLO] = {}
        self._metrics: dict[str, TenantMetrics] = {}
        self._rate_counters: dict[str, SlidingWindowRateCounter] = {}
        self._latency_histories: dict[str, deque] = {}
        self._lock = threading.Lock()

    # ── Tenant registration ───────────────────────────────────────────

    def register_tenant(
        self,
        tenant_id: str,
        max_rpm: float = 60.0,
        latency_slo_ms: float = 1000.0,
        max_concurrent: int = 10,
        priority_base: float = 1.0,
    ) -> None:
        """Register or update a tenant's SLO configuration."""
        with self._lock:
            self._slos[tenant_id] = TenantSLO(
                tenant_id=tenant_id,
                max_rpm=max_rpm,
                latency_slo_ms=latency_slo_ms,
                max_concurrent=max_concurrent,
                priority_base=priority_base,
            )
            if tenant_id not in self._metrics:
                self._metrics[tenant_id] = TenantMetrics()
            if tenant_id not in self._rate_counters:
                self._rate_counters[tenant_id] = SlidingWindowRateCounter()
            if tenant_id not in self._latency_histories:
                self._latency_histories[tenant_id] = deque(maxlen=1000)

    def remove_tenant(self, tenant_id: str) -> None:
        with self._lock:
            self._slos.pop(tenant_id, None)
            self._metrics.pop(tenant_id, None)
            self._rate_counters.pop(tenant_id, None)
            self._latency_histories.pop(tenant_id, None)

    # ── Admission control ─────────────────────────────────────────────

    def should_admit(self, tenant_id: str) -> bool:
        """Check if *tenant_id* is allowed to send a request now.

        Returns ``True`` if within rate limit and concurrent limit.
        Returns ``False`` (with backpressure logging) if either limit
        is exceeded.
        """
        slo = self._slos.get(tenant_id)
        if slo is None:
            return True  # unknown tenant = no limit

        metrics = self._metrics.get(tenant_id)
        if metrics is None:
            return True

        rate_counter = self._rate_counters.get(tenant_id)
        if rate_counter is None:
            return True

        # Check concurrent limit.
        if metrics.current_concurrent >= slo.max_concurrent:
            if not metrics.last_had_backpressure:
                logger.warning(f"Tenant {tenant_id}: concurrent limit reached "
                              f"({metrics.current_concurrent}/{slo.max_concurrent})")
                metrics.last_had_backpressure = True
            return False

        # Check rate limit (with burst allowance).
        recent = rate_counter.count()
        burst_capacity = slo.max_rpm * slo.burst_multiplier
        if recent >= burst_capacity:
            if not metrics.last_had_backpressure:
                logger.warning(f"Tenant {tenant_id}: rate limit reached "
                              f"({recent:.0f} rpm / {burst_capacity:.0f} burst)")
                metrics.last_had_backpressure = True
            return False

        metrics.last_had_backpressure = False
        return True

    def record_request_start(self, tenant_id: str) -> None:
        """Record that a request from *tenant_id* has started."""
        metrics = self._metrics.get(tenant_id)
        if metrics is None:
            return
        rate_counter = self._rate_counters.get(tenant_id)
        if rate_counter:
            rate_counter.record(1)

        with self._lock:
            metrics.total_requests += 1
            metrics.current_concurrent += 1

    def record_request_end(self, tenant_id: str, latency_ms: float) -> None:
        """Record a completed request."""
        slo = self._slos.get(tenant_id)
        metrics = self._metrics.get(tenant_id)
        if metrics is None:
            return

        with self._lock:
            metrics.completed_requests += 1
            metrics.current_concurrent = max(0, metrics.current_concurrent - 1)
            metrics.total_latency_ms += latency_ms

            # Track p99 latency via reservoir sampling.
            history = self._latency_histories.get(tenant_id)
            if history is not None:
                history.append(latency_ms)
                if len(history) >= 100:
                    sorted_lats = sorted(history)[:999]
                    p99_idx = int(len(sorted_lats) * 0.99)
                    metrics.latency_p99_ms = sorted_lats[p99_idx]

            # Check for SLO breach.
            if slo and metrics.latency_p99_ms > slo.latency_slo_ms:
                metrics.slo_breaches += 1

    # ── Priority boosting ─────────────────────────────────────────────

    def get_priority_boost(self, tenant_id: str) -> float:
        """Return a priority multiplier for *tenant_id*.

        Returns 1.0 for normal operation.  Returns >1.0 when a tenant
        is approaching its SLO breach, giving it a scheduling advantage.
        Returns <1.0 for tenants with headroom.

        Boost ranges from 0.5 (lots of headroom) to 2.0 (near SLO breach).
        """
        slo = self._slos.get(tenant_id)
        metrics = self._metrics.get(tenant_id)
        if slo is None or metrics is None:
            return 1.0

        # How close is p99 to SLO?
        if metrics.latency_p99_ms <= 0:
            return 1.0

        ratio = metrics.latency_p99_ms / slo.latency_slo_ms

        if ratio >= 1.0:
            return 2.0       # already breaching — max boost
        elif ratio >= 0.9:
            return 1.5       # approaching breach — significant boost
        elif ratio >= 0.75:
            return 1.2       # getting warm — modest boost
        elif ratio <= 0.3:
            return 0.8       # lots of headroom — yield to others
        return 1.0

    # ── Observability ─────────────────────────────────────────────────

    def get_tenant_status(self, tenant_id: str) -> dict[str, Any] | None:
        """Return full status for a single tenant."""
        slo = self._slos.get(tenant_id)
        metrics = self._metrics.get(tenant_id)
        if slo is None or metrics is None:
            return None

        rate_counter = self._rate_counters.get(tenant_id)

        return {
            "tenant_id": tenant_id,
            "slo": {
                "max_rpm": slo.max_rpm,
                "latency_slo_ms": slo.latency_slo_ms,
                "max_concurrent": slo.max_concurrent,
                "priority_base": slo.priority_base,
            },
            "metrics": {
                "total_requests": metrics.total_requests,
                "completed": metrics.completed_requests,
                "rejected": metrics.rejected_requests,
                "current_concurrent": metrics.current_concurrent,
                "rpm": rate_counter.count() if rate_counter else 0,
                "avg_latency_ms": round(metrics.avg_latency_ms, 1),
                "latency_p99_ms": round(metrics.latency_p99_ms, 1),
                "slo_breaches": metrics.slo_breaches,
                "slo_compliance_pct": round(metrics.slo_compliance_pct, 1),
                "priority_boost": self.get_priority_boost(tenant_id),
            },
        }

    def get_all_status(self) -> list[dict[str, Any]]:
        """Return status for all registered tenants."""
        return [
            status for tid in self._slos
            if (status := self.get_tenant_status(tid)) is not None
        ]

    def get_global_metrics(self) -> dict[str, Any]:
        """Aggregate metrics across all tenants."""
        with self._lock:
            total_requests = sum(m.total_requests for m in self._metrics.values())
            total_breaches = sum(m.slo_breaches for m in self._metrics.values())
            total_rejected = sum(m.rejected_requests for m in self._metrics.values())
            return {
                "tenants": len(self._slos),
                "total_requests": total_requests,
                "total_slo_breaches": total_breaches,
                "total_rejected": total_rejected,
                "overall_slo_compliance": round(
                    100.0 * (1.0 - total_breaches / max(total_requests, 1)), 1
                ),
            }
