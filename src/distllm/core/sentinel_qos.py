"""Multi-tenant QoS isolation -- token buckets, fair queuing, KV partitioning,
SLO violation detection, and admission control.

Provides a complete sentinel layer for enforcing per-tenant quality-of-service
guarantees in a shared inference system.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_REFILL_RATE: float = 100.0
DEFAULT_BURST_SIZE: int = 1000
DEFAULT_TENANT_WEIGHT: float = 1.0
DEFAULT_SLO_LATENCY_MS: float = 500.0
GLOBAL_KV_BUDGET: int = 10_000_000
ADMISSION_LOAD_THRESHOLD: float = 0.9
ALERT_COOLDOWN_S: float = 10.0


# ── TenantTokenBucket ─────────────────────────────────────────────────────────


@dataclass
class _TokenBucketState:
    """Internal state for a single tenant's token bucket."""
    tokens: float
    last_refill: float
    refill_rate: float
    burst_size: int


class TenantTokenBucket:
    """Per-tenant token bucket for rate limiting.

    Each tenant has an independent bucket configured with a *refill_rate*
    (tokens per second) and *burst_size* (maximum token capacity).

    Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, _TokenBucketState] = {}

    # ── Configuration ────────────────────────────────────────────────────

    def set_config(
        self,
        tenant_id: str,
        refill_rate: float = DEFAULT_REFILL_RATE,
        burst_size: int = DEFAULT_BURST_SIZE,
    ) -> None:
        """Set or update a tenant's token bucket configuration.

        If the tenant already has a bucket, accumulated tokens are preserved
        up to the new *burst_size*.
        """
        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                self._buckets[tenant_id] = _TokenBucketState(
                    tokens=float(burst_size),
                    last_refill=time.time(),
                    refill_rate=refill_rate,
                    burst_size=burst_size,
                )
            else:
                self._refill(bucket)
                bucket.refill_rate = refill_rate
                bucket.burst_size = burst_size
                bucket.tokens = min(bucket.tokens, float(burst_size))

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove a tenant's bucket entirely."""
        with self._lock:
            self._buckets.pop(tenant_id, None)

    # ── Core operation ───────────────────────────────────────────────────

    def consume(self, tenant_id: str, tokens: int) -> bool:
        """Attempt to consume *tokens* from *tenant_id*'s bucket.

        Returns ``True`` if the tokens were available and consumed,
        ``False`` if the tenant has exceeded its rate limit.
        """
        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                return True  # no limit configured -> allow

            self._refill(bucket)

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True
            return False

    def available(self, tenant_id: str) -> float:
        """Return the number of tokens currently available for *tenant_id*."""
        with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                return float("inf")
            self._refill(bucket)
            return bucket.tokens

    def configs(self) -> dict[str, dict[str, float | int]]:
        """Return the current configuration for all tenants."""
        with self._lock:
            return {
                tid: {
                    "refill_rate": b.refill_rate,
                    "burst_size": b.burst_size,
                    "available": b.tokens,
                }
                for tid, b in self._buckets.items()
            }

    # ── Internal ─────────────────────────────────────────────────────────

    def _refill(self, bucket: _TokenBucketState) -> None:
        now = time.time()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(
            bucket.tokens + elapsed * bucket.refill_rate,
            float(bucket.burst_size),
        )
        bucket.last_refill = now


# ── TenantQueueScheduler ──────────────────────────────────────────────────────


@dataclass(order=True)
class _QueuedRequest:
    """A request queued for a tenant, tagged with its virtual finish time.

    Ordering is by (finish_time, seq) so that the min-heap pops the
    WFQ-determined request deterministically.
    """
    finish_time: float
    tenant_id: str = field(compare=False)
    request: Any = field(compare=False)
    seq: int = field(compare=False)


class TenantQueueScheduler:
    """Weighted fair-queuing scheduler across tenants.

    Each tenant has an implicit FIFO queue.  When a request is enqueued,
    a virtual finish time is computed as::

        finish = max(virtual_now, last_finish[tenant]) + 1.0 / weight

    ``dequeue()`` pops the request with the smallest finish time,
    providing natural WFQ behaviour: a tenant with weight=2.0 receives
    roughly twice the throughput of a tenant with weight=1.0.

    Thread-safe.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heap: list[_QueuedRequest] = []
        self._last_finish: dict[str, float] = {}
        self._weights: dict[str, float] = {}
        self._counter = 0

    # ── Configuration ────────────────────────────────────────────────────

    def set_weight(self, tenant_id: str, weight: float = DEFAULT_TENANT_WEIGHT) -> None:
        """Set the WFQ weight for *tenant_id*.  Higher weight = more throughput."""
        with self._lock:
            self._weights[tenant_id] = max(weight, 0.01)

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove a tenant and all its pending requests from the scheduler."""
        with self._lock:
            self._weights.pop(tenant_id, None)
            self._last_finish.pop(tenant_id, None)
            # Filter out the tenant's requests and rebuild the heap
            self._heap = [r for r in self._heap if r.tenant_id != tenant_id]
            heapq.heapify(self._heap)

    # ── Operations ───────────────────────────────────────────────────────

    def enqueue(self, tenant_id: str, request: Any) -> None:
        """Enqueue a *request* for *tenant_id*."""
        with self._lock:
            weight = self._weights.get(tenant_id, DEFAULT_TENANT_WEIGHT)
            virtual_now = time.time()
            last = self._last_finish.get(tenant_id, 0.0)
            finish_time = max(virtual_now, last) + 1.0 / weight
            self._last_finish[tenant_id] = finish_time
            self._counter += 1
            heapq.heappush(
                self._heap,
                _QueuedRequest(finish_time, tenant_id, request, self._counter),
            )

    def dequeue(self) -> Any | None:
        """Pop and return the next request in WFQ order.

        Returns ``None`` if no requests are pending.
        """
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap).request

    # ── Introspection ────────────────────────────────────────────────────

    def front(self) -> Any | None:
        """Peek at the next request without removing it."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0].request

    def pending_count(self, tenant_id: str | None = None) -> int:
        """Return the number of pending requests for *tenant_id* (or total)."""
        with self._lock:
            if tenant_id is None:
                return len(self._heap)
            return sum(1 for r in self._heap if r.tenant_id == tenant_id)

    def pending_counts(self) -> dict[str, int]:
        """Return a dict mapping tenant_id -> pending request count."""
        with self._lock:
            counts: dict[str, int] = {}
            for r in self._heap:
                counts[r.tenant_id] = counts.get(r.tenant_id, 0) + 1
            return counts


# ── KVPartitionManager ────────────────────────────────────────────────────────


@dataclass
class _TenantKVState:
    """KV cache allocation state for a single tenant."""
    allocated_blocks: int = 0


class KVPartitionManager:
    """Partitions KV cache blocks per tenant.

    A global budget caps the total number of blocks that can be allocated
    across all tenants.  Individual tenants may also have a per-tenant soft
    limit (0 = unlimited beyond the global cap).

    Thread-safe.
    """

    def __init__(self, global_budget: int = GLOBAL_KV_BUDGET) -> None:
        self._lock = threading.Lock()
        self._global_budget = global_budget
        self._tenants: dict[str, _TenantKVState] = {}
        self._soft_limits: dict[str, int] = {}

    # ── Configuration ────────────────────────────────────────────────────

    def set_soft_limit(self, tenant_id: str, max_blocks: int) -> None:
        """Set a per-tenant soft limit on KV cache blocks (0 = no limit)."""
        with self._lock:
            self._soft_limits[tenant_id] = max_blocks

    def set_global_budget(self, blocks: int) -> None:
        """Update the global KV cache block budget."""
        with self._lock:
            self._global_budget = blocks

    @property
    def global_budget(self) -> int:
        return self._global_budget

    # ── Operations ───────────────────────────────────────────────────────

    def allocate(self, tenant_id: str, blocks: int) -> bool:
        """Allocate *blocks* for *tenant_id*.

        Returns ``True`` if the allocation is within limits, ``False`` if
        the global budget or per-tenant soft limit would be exceeded.
        """
        with self._lock:
            state = self._tenants.setdefault(tenant_id, _TenantKVState())

            # Check global budget
            total_used = sum(t.allocated_blocks for t in self._tenants.values())
            if total_used + blocks > self._global_budget:
                logger.warning(
                    f"KV global budget exceeded: {total_used + blocks} > "
                    f"{self._global_budget}"
                )
                return False

            # Check per-tenant soft limit
            soft = self._soft_limits.get(tenant_id, 0)
            if soft > 0 and state.allocated_blocks + blocks > soft:
                logger.warning(
                    f"KV soft limit for tenant {tenant_id}: "
                    f"{state.allocated_blocks + blocks} > {soft}"
                )
                return False

            state.allocated_blocks += blocks
            return True

    def release(self, tenant_id: str) -> None:
        """Release all KV cache blocks for *tenant_id*."""
        with self._lock:
            self._tenants.pop(tenant_id, None)
            self._soft_limits.pop(tenant_id, None)

    def release_blocks(self, tenant_id: str, blocks: int) -> None:
        """Release a specific number of *blocks* for *tenant_id*."""
        with self._lock:
            state = self._tenants.get(tenant_id)
            if state is not None:
                state.allocated_blocks = max(0, state.allocated_blocks - blocks)

    # ── Introspection ────────────────────────────────────────────────────

    def get_usage(self, tenant_id: str) -> int:
        """Return the number of KV blocks currently used by *tenant_id*."""
        with self._lock:
            state = self._tenants.get(tenant_id)
            return state.allocated_blocks if state else 0

    def total_usage(self) -> int:
        """Return the total number of KV blocks allocated across all tenants."""
        with self._lock:
            return sum(t.allocated_blocks for t in self._tenants.values())

    def remaining_global(self) -> int:
        """Return the number of KV blocks remaining in the global budget."""
        with self._lock:
            used = sum(t.allocated_blocks for t in self._tenants.values())
            return max(0, self._global_budget - used)

    def tenant_summary(self) -> dict[str, dict[str, int]]:
        """Return a summary of KV usage per tenant."""
        with self._lock:
            return {
                tid: {
                    "allocated_blocks": state.allocated_blocks,
                    "soft_limit": self._soft_limits.get(tid, 0),
                }
                for tid, state in self._tenants.items()
            }


# ── ViolationDetector ─────────────────────────────────────────────────────────


@dataclass
class ViolationEvent:
    """A recorded SLO violation."""
    tenant_id: str
    actual_latency_ms: float
    threshold_ms: float
    timestamp: float


@dataclass
class _TenantSLI:
    """Service Level Indicator for a single tenant."""
    total_requests: int = 0
    violations: int = 0

    @property
    def sli(self) -> float:
        """Return the SLI as a fraction [0.0, 1.0]."""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.violations / self.total_requests)


class ViolationDetector:
    """Detects SLO violations and tracks per-tenant SLI.

    Each tenant can have an independent latency SLO (in milliseconds).
    When a check fails, the violation is recorded and optionally logged
    as a warning.  Repeated alerts for the same tenant are rate-limited
    by *alert_cooldown_s*.

    Thread-safe.
    """

    def __init__(self, alert_cooldown_s: float = ALERT_COOLDOWN_S) -> None:
        self._lock = threading.Lock()
        self._slos: dict[str, float] = {}
        self._sli: dict[str, _TenantSLI] = {}
        self._violations: list[ViolationEvent] = []
        self._last_alert: dict[str, float] = {}
        self._alert_cooldown = alert_cooldown_s

    # ── Configuration ────────────────────────────────────────────────────

    def set_slo(self, tenant_id: str, latency_threshold_ms: float = DEFAULT_SLO_LATENCY_MS) -> None:
        """Set the latency SLO (ms) for *tenant_id*."""
        with self._lock:
            self._slos[tenant_id] = latency_threshold_ms

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove SLO config and SLI data for *tenant_id*."""
        with self._lock:
            self._slos.pop(tenant_id, None)
            self._sli.pop(tenant_id, None)
            self._last_alert.pop(tenant_id, None)

    # ── Core operation ───────────────────────────────────────────────────

    def check(self, tenant_id: str, latency_ms: float) -> bool:
        """Check if *latency_ms* violates *tenant_id*'s SLO.

        Returns ``True`` if the SLO is met (no violation), ``False`` if
        the latency exceeds the configured threshold (violation).
        """
        with self._lock:
            threshold = self._slos.get(tenant_id)
            if threshold is None:
                return True  # no SLO configured -> no violation

            sli = self._sli.setdefault(tenant_id, _TenantSLI())
            sli.total_requests += 1

            if latency_ms <= threshold:
                return True

            # ── Violation ────────────────────────────────────────────
            sli.violations += 1
            event = ViolationEvent(
                tenant_id=tenant_id,
                actual_latency_ms=latency_ms,
                threshold_ms=threshold,
                timestamp=time.time(),
            )
            self._violations.append(event)

            # Rate-limited alert
            now = time.time()
            last = self._last_alert.get(tenant_id, 0.0)
            if now - last >= self._alert_cooldown:
                logger.warning(
                    f"SLO violation for tenant {tenant_id}: "
                    f"{latency_ms:.1f}ms > {threshold:.0f}ms threshold "
                    f"(SLI={sli.sli:.3f})"
                )
                self._last_alert[tenant_id] = now

            return False

    # ── Introspection ────────────────────────────────────────────────────

    def track_sli(self, tenant_id: str | None = None) -> dict[str, dict[str, Any]]:
        """Return per-tenant SLI data as a dict.

        If *tenant_id* is ``None``, return data for all tenants.
        """
        with self._lock:
            if tenant_id is not None:
                sli = self._sli.get(tenant_id)
                if sli is None:
                    return {}
                return {
                    tenant_id: {
                        "total_requests": sli.total_requests,
                        "violations": sli.violations,
                        "sli": sli.sli,
                    }
                }

            return {
                tid: {
                    "total_requests": s.total_requests,
                    "violations": s.violations,
                    "sli": s.sli,
                }
                for tid, s in self._sli.items()
            }

    def recent_violations(
        self,
        limit: int = 100,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent violation events."""
        with self._lock:
            events = self._violations
            if tenant_id is not None:
                events = [ev for ev in events if ev.tenant_id == tenant_id]
            return [
                {
                    "tenant_id": ev.tenant_id,
                    "actual_latency_ms": ev.actual_latency_ms,
                    "threshold_ms": ev.threshold_ms,
                    "timestamp": ev.timestamp,
                }
                for ev in events[-limit:]
            ]


# ── AdmissionController ───────────────────────────────────────────────────────


@dataclass
class AdmissionResult:
    """Result of an admission decision."""
    admitted: bool
    reason: str = ""


class AdmissionController:
    """Admission control based on current system load and tenant limits.

    Tracks an abstract "load" representing the total estimated cost of
    in-flight requests across all tenants.  When system load exceeds the
    configured threshold, new requests are rejected.  Per-tenant concurrency
    limits are also enforced.

    Thread-safe.
    """

    def __init__(
        self,
        load_threshold: float = ADMISSION_LOAD_THRESHOLD,
    ) -> None:
        self._lock = threading.Lock()
        self._load_threshold = load_threshold
        self._current_load: float = 0.0
        self._max_load: float = 1.0
        self._in_flight: dict[str, int] = {}
        self._concurrency_limits: dict[str, int] = {}
        self._admitted_count: dict[str, int] = {}
        self._rejected_count: dict[str, int] = {}

    # ── Configuration ────────────────────────────────────────────────────

    def set_max_load(self, max_load: float) -> None:
        """Set the maximum load the system can sustain."""
        with self._lock:
            self._max_load = max(max_load, 0.01)

    @property
    def max_load(self) -> float:
        return self._max_load

    def set_load_threshold(self, threshold: float) -> None:
        """Set the fraction of ``max_load`` at which admission is denied."""
        with self._lock:
            self._load_threshold = max(min(threshold, 1.0), 0.0)

    def set_concurrency_limit(self, tenant_id: str, max_concurrent: int) -> None:
        """Set the maximum number of concurrent requests for *tenant_id*."""
        with self._lock:
            self._concurrency_limits[tenant_id] = max_concurrent

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove all tracking data for *tenant_id*."""
        with self._lock:
            self._in_flight.pop(tenant_id, None)
            self._concurrency_limits.pop(tenant_id, None)
            self._admitted_count.pop(tenant_id, None)
            self._rejected_count.pop(tenant_id, None)

    # ── Core operation ───────────────────────────────────────────────────

    def can_admit(self, tenant_id: str, estimated_cost: float = 0.0) -> AdmissionResult:
        """Check whether *tenant_id* may submit a request.

        Args:
            tenant_id: The tenant requesting admission.
            estimated_cost: Estimated load contribution of this request
                (e.g. estimated tokens or compute units).

        Returns:
            An ``AdmissionResult`` with ``admitted=True`` if the request
            can proceed, or ``admitted=False`` with a ``reason`` otherwise.
        """
        with self._lock:
            # Global load check
            effective_load = self._current_load + estimated_cost
            load_capacity = self._max_load * self._load_threshold
            if effective_load > load_capacity:
                self._rejected_count[tenant_id] = (
                    self._rejected_count.get(tenant_id, 0) + 1
                )
                return AdmissionResult(
                    admitted=False,
                    reason=(
                        f"system overloaded: load {effective_load:.2f} > "
                        f"capacity {load_capacity:.2f}"
                    ),
                )

            # Per-tenant concurrency check
            limit = self._concurrency_limits.get(tenant_id, 0)
            if limit > 0:
                current = self._in_flight.get(tenant_id, 0)
                if current >= limit:
                    self._rejected_count[tenant_id] = (
                        self._rejected_count.get(tenant_id, 0) + 1
                    )
                    return AdmissionResult(
                        admitted=False,
                        reason=(
                            f"tenant {tenant_id} concurrency limit "
                            f"({limit}) reached"
                        ),
                    )

            # Admit
            self._in_flight[tenant_id] = self._in_flight.get(tenant_id, 0) + 1
            self._current_load += estimated_cost
            self._admitted_count[tenant_id] = (
                self._admitted_count.get(tenant_id, 0) + 1
            )
            return AdmissionResult(admitted=True, reason="")

    def finish_request(self, tenant_id: str, estimated_cost: float = 0.0) -> None:
        """Release a concurrency slot and load contribution after a request completes."""
        with self._lock:
            current = self._in_flight.get(tenant_id, 0)
            self._in_flight[tenant_id] = max(0, current - 1)
            self._current_load = max(0.0, self._current_load - estimated_cost)

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def current_load(self) -> float:
        """Return the current aggregate load level."""
        with self._lock:
            return self._current_load

    @property
    def utilization(self) -> float:
        """Return load as a fraction [0.0, 1.0] of ``max_load``."""
        with self._lock:
            return self._current_load / self._max_load if self._max_load > 0 else 0.0

    def admission_counts(self) -> dict[str, dict[str, int]]:
        """Return per-tenant admission and rejection counts."""
        with self._lock:
            tenants = set(self._admitted_count) | set(self._rejected_count)
            return {
                tid: {
                    "admitted": self._admitted_count.get(tid, 0),
                    "rejected": self._rejected_count.get(tid, 0),
                }
                for tid in sorted(tenants)
            }


# ── Sentinel ──────────────────────────────────────────────────────────────────


class Sentinel:
    """Top-level QoS sentinel combining all isolation subsystems.

    Usage::

        sentinel = Sentinel(
            global_kv_budget=5_000_000,
            load_threshold=0.85,
        )

        # Configure tenants in a single call
        sentinel.configure_tenant("tenant-a",
            refill_rate=200.0, burst_size=2000, weight=2.0,
            kv_soft_limit=500_000, latency_slo_ms=300.0,
        )
        sentinel.configure_tenant("tenant-b",
            refill_rate=50.0, burst_size=500, weight=1.0,
            latency_slo_ms=1000.0,
        )

        # Request lifecycle
        if sentinel.token_bucket.consume("tenant-a", 1):
            result = sentinel.admission.can_admit("tenant-a", estimated_cost=0.1)
            if result.admitted:
                # ... inference ...
                sentinel.admission.finish_request("tenant-a", estimated_cost=0.1)

        sentinel.start()
        # ...
        sentinel.stop()

        print(sentinel.stats())
    """

    def __init__(
        self,
        global_kv_budget: int = GLOBAL_KV_BUDGET,
        load_threshold: float = ADMISSION_LOAD_THRESHOLD,
        alert_cooldown_s: float = ALERT_COOLDOWN_S,
    ) -> None:
        self.token_bucket = TenantTokenBucket()
        self.queue_scheduler = TenantQueueScheduler()
        self.kv_partition = KVPartitionManager(global_budget=global_kv_budget)
        self.violation_detector = ViolationDetector(alert_cooldown_s=alert_cooldown_s)
        self.admission = AdmissionController(load_threshold=load_threshold)
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the sentinel.

        Currently a no-op placeholder; future versions may activate
        background monitoring or alerting loops.
        """
        self._running = True
        logger.info("Sentinel started")

    def stop(self) -> None:
        """Stop the sentinel."""
        self._running = False
        logger.info("Sentinel stopped")

    @property
    def running(self) -> bool:
        """Return ``True`` iff the sentinel has been started."""
        return self._running

    # ── Convenience ──────────────────────────────────────────────────────

    def configure_tenant(
        self,
        tenant_id: str,
        *,
        refill_rate: float = DEFAULT_REFILL_RATE,
        burst_size: int = DEFAULT_BURST_SIZE,
        weight: float = DEFAULT_TENANT_WEIGHT,
        kv_soft_limit: int = 0,
        latency_slo_ms: float = DEFAULT_SLO_LATENCY_MS,
        concurrency_limit: int = 0,
    ) -> None:
        """Configure all subsystems for *tenant_id* in a single call.

        Args:
            tenant_id: Tenant identifier.
            refill_rate: Token bucket refill rate (tokens/sec).
            burst_size: Token bucket burst capacity.
            weight: WFQ scheduling weight.
            kv_soft_limit: Max KV cache blocks (0 = no per-tenant limit).
            latency_slo_ms: Latency SLO threshold in milliseconds.
            concurrency_limit: Max concurrent requests (0 = unlimited).
        """
        self.token_bucket.set_config(
            tenant_id, refill_rate=refill_rate, burst_size=burst_size,
        )
        self.queue_scheduler.set_weight(tenant_id, weight=weight)
        if kv_soft_limit > 0:
            self.kv_partition.set_soft_limit(tenant_id, kv_soft_limit)
        self.violation_detector.set_slo(
            tenant_id, latency_threshold_ms=latency_slo_ms,
        )
        if concurrency_limit > 0:
            self.admission.set_concurrency_limit(tenant_id, concurrency_limit)
        logger.debug(
            f"Tenant {tenant_id} configured: "
            f"refill={refill_rate}, burst={burst_size}, weight={weight}, "
            f"kv_limit={kv_soft_limit}, slo_ms={latency_slo_ms}, "
            f"concurrency={concurrency_limit}"
        )

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove a tenant from all subsystems."""
        self.token_bucket.remove_tenant(tenant_id)
        self.queue_scheduler.remove_tenant(tenant_id)
        self.kv_partition.release(tenant_id)
        self.violation_detector.remove_tenant(tenant_id)
        self.admission.remove_tenant(tenant_id)
        logger.debug(f"Tenant {tenant_id} removed from Sentinel")

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return aggregated per-tenant metrics from all subsystems.

        Returns a dict with keys:

        - **tenants**: per-tenant SLI data
        - **admissions**: admission / rejection counts
        - **violations**: recent SLO violation summary
        - **kv_cache**: global KV usage summary
        - **queues**: pending request counts per tenant
        - **system**: overall load and utilization metrics
        """
        return {
            "tenants": self.violation_detector.track_sli(),
            "admissions": self.admission.admission_counts(),
            "violations": {
                "recent": self.violation_detector.recent_violations(limit=20),
            },
            "kv_cache": {
                "per_tenant": self.kv_partition.tenant_summary(),
                "total_used": self.kv_partition.total_usage(),
                "global_budget": self.kv_partition.global_budget,
                "remaining": self.kv_partition.remaining_global(),
            },
            "queues": self.queue_scheduler.pending_counts(),
            "system": {
                "current_load": self.admission.current_load,
                "max_load": self.admission.max_load,
                "utilization": self.admission.utilization,
            },
        }


# ── __all__ ───────────────────────────────────────────────────────────────────

__all__ = [
    "AdmissionController",
    "AdmissionResult",
    "KVPartitionManager",
    "Sentinel",
    "TenantQueueScheduler",
    "TenantTokenBucket",
    "ViolationDetector",
    "ViolationEvent",
]
