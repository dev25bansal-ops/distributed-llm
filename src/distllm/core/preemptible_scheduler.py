"""Preemptible Scheduling — priority-based routing with spot pricing.

Routes inference requests based on priority tiers that balance cost,
reliability, and SLA requirements:

  Tier 1 (Critical):  On-demand only, lowest latency, any carbon
  Tier 2 (Standard):  Spot preferred, moderate latency, carbon < threshold
  Tier 3 (Batch):     Spot only, highest latency tolerance, lowest carbon

Higher-priority requests can preempt lower-priority ones when resources
are constrained, and the scheduler maintains per-tenant priority budgets
to prevent starvation.
"""

from __future__ import annotations

import enum
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


class PriorityTier(enum.IntEnum):
    """Request priority tiers (lower number = higher priority)."""
    CRITICAL = 0    # Mission-critical, latency-sensitive
    HIGH = 1        # Production workloads
    NORMAL = 2      # Standard inference
    LOW = 3         # Batch / async processing
    BACKGROUND = 4  # Non-urgent, best-effort


@dataclass
class TierPolicy:
    """Policy for a priority tier."""
    tier: PriorityTier
    label: str
    prefer_spot: bool = True
    allow_on_demand: bool = True
    max_latency_ms: float = 200.0
    max_price_per_hour: float = float("inf")
    max_carbon_intensity: float = float("inf")
    carbon_weight: float = 0.3
    max_queue_time_s: float = 30.0
    preemption_enabled: bool = False


# Default tier policies
DEFAULT_TIER_POLICIES: dict[PriorityTier, TierPolicy] = {
    PriorityTier.CRITICAL: TierPolicy(
        tier=PriorityTier.CRITICAL,
        label="Critical",
        prefer_spot=False,
        allow_on_demand=True,
        max_latency_ms=50.0,
        max_carbon_intensity=float("inf"),
        carbon_weight=0.0,
        max_queue_time_s=5.0,
        preemption_enabled=True,
    ),
    PriorityTier.HIGH: TierPolicy(
        tier=PriorityTier.HIGH,
        label="High",
        prefer_spot=True,
        allow_on_demand=True,
        max_latency_ms=100.0,
        max_carbon_intensity=500.0,
        carbon_weight=0.15,
        max_queue_time_s=15.0,
        preemption_enabled=True,
    ),
    PriorityTier.NORMAL: TierPolicy(
        tier=PriorityTier.NORMAL,
        label="Normal",
        prefer_spot=True,
        allow_on_demand=False,
        max_latency_ms=200.0,
        max_carbon_intensity=400.0,
        carbon_weight=0.3,
        max_queue_time_s=30.0,
        preemption_enabled=False,
    ),
    PriorityTier.LOW: TierPolicy(
        tier=PriorityTier.LOW,
        label="Batch",
        prefer_spot=True,
        allow_on_demand=False,
        max_latency_ms=5000.0,
        max_carbon_intensity=300.0,
        carbon_weight=0.5,
        max_queue_time_s=300.0,
        preemption_enabled=False,
    ),
    PriorityTier.BACKGROUND: TierPolicy(
        tier=PriorityTier.BACKGROUND,
        label="Background",
        prefer_spot=True,
        allow_on_demand=False,
        max_latency_ms=60000.0,
        max_carbon_intensity=200.0,
        carbon_weight=0.8,
        max_queue_time_s=3600.0,
        preemption_enabled=False,
    ),
}


@dataclass
class ScheduledRequest:
    """A request in the priority queue."""
    request_id: str
    priority: PriorityTier
    gpu_type: str = ""
    min_gpu_memory_gb: float = 0.0
    queued_at: float = field(default_factory=time.time)
    user_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def queue_time_s(self) -> float:
        return time.time() - self.queued_at


@dataclass
class TenantBudget:
    """Per-tenant priority budget to prevent starvation."""
    tenant_id: str
    max_critical_per_hour: int = 10
    max_high_per_hour: int = 100
    max_normal_per_hour: int = 1000
    current_critical: int = 0
    current_high: int = 0
    current_normal: int = 0
    hour_start: float = field(default_factory=time.time)

    def reset_if_needed(self) -> None:
        now = time.time()
        if now - self.hour_start >= 3600:
            self.current_critical = 0
            self.current_high = 0
            self.current_normal = 0
            self.hour_start = now

    def can_use(self, tier: PriorityTier) -> bool:
        self.reset_if_needed()
        if tier == PriorityTier.CRITICAL:
            return self.current_critical < self.max_critical_per_hour
        if tier == PriorityTier.HIGH:
            return self.current_high < self.max_high_per_hour
        if tier == PriorityTier.NORMAL:
            return self.current_normal < self.max_normal_per_hour
        return True

    def record_use(self, tier: PriorityTier) -> None:
        self.reset_if_needed()
        if tier == PriorityTier.CRITICAL:
            self.current_critical += 1
        elif tier == PriorityTier.HIGH:
            self.current_high += 1
        elif tier == PriorityTier.NORMAL:
            self.current_normal += 1


class PreemptibleScheduler:
    """Priority-based scheduler that routes requests through tier policies.

    Integrates with CrossCloudRouter to apply tier-specific routing
    constraints (spot vs on-demand, latency, carbon, pricing).

    Usage::

        scheduler = PreemptibleScheduler()
        policy = scheduler.get_policy(PriorityTier.CRITICAL)
        policy.max_latency_ms  # 50.0
        policy.prefer_spot     # False

        request = scheduler.enqueue("req-1", PriorityTier.HIGH, gpu_type="A100")
        result = scheduler.dequeue()  # Highest priority first
    """

    def __init__(
        self,
        tier_policies: dict[PriorityTier, TierPolicy] | None = None,
    ):
        self._policies = tier_policies or DEFAULT_TIER_POLICIES
        self._queues: dict[PriorityTier, deque[ScheduledRequest]] = {
            tier: deque() for tier in PriorityTier
        }
        self._tenant_budgets: dict[str, TenantBudget] = {}
        self._active_requests: dict[str, ScheduledRequest] = {}
        self._preemption_log: list[dict[str, Any]] = []
        self._stats = {
            "enqueued": 0,
            "dequeued": 0,
            "preempted": 0,
            "budget_rejected": 0,
            "timeout_evicted": 0,
        }
        self._lock = threading.Lock()

    def get_policy(self, tier: PriorityTier) -> TierPolicy:
        """Get the policy for a priority tier."""
        return self._policies.get(tier, DEFAULT_TIER_POLICIES[PriorityTier.NORMAL])

    def set_policy(self, tier: PriorityTier, policy: TierPolicy) -> None:
        """Override the policy for a priority tier."""
        self._policies[tier] = policy

    def set_tenant_budget(self, tenant_id: str, budget: TenantBudget) -> None:
        """Set a per-tenant priority budget."""
        with self._lock:
            self._tenant_budgets[tenant_id] = budget

    def enqueue(
        self,
        request_id: str,
        priority: PriorityTier,
        gpu_type: str = "",
        min_gpu_memory_gb: float = 0.0,
        user_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ScheduledRequest | None:
        """Enqueue a request with the given priority.

        Returns the ScheduledRequest if enqueued, None if budget-rejected.
        """
        with self._lock:
            budget = self._tenant_budgets.get(user_id)
            if budget and not budget.can_use(priority):
                self._stats["budget_rejected"] += 1
                logger.warning(
                    f"Request {request_id} rejected: tenant {user_id} "
                    f"exceeded {priority.name} budget"
                )
                return None
            if budget:
                budget.record_use(priority)
            request = ScheduledRequest(
                request_id=request_id,
                priority=priority,
                gpu_type=gpu_type,
                min_gpu_memory_gb=min_gpu_memory_gb,
                user_id=user_id,
                metadata=metadata or {},
            )
            self._queues[priority].append(request)
            self._stats["enqueued"] += 1
            logger.debug(f"Enqueued {request_id} at priority {priority.name}")
            return request

    def dequeue(self) -> ScheduledRequest | None:
        """Dequeue the highest-priority request.

        Evicts timed-out requests from lower-priority queues when
        higher-priority requests are waiting.
        """
        with self._lock:
            self._evict_timed_out()
            for tier in PriorityTier:
                queue = self._queues[tier]
                if queue:
                    request = queue.popleft()
                    self._active_requests[request.request_id] = request
                    self._stats["dequeued"] += 1
                    logger.debug(f"Dequeued {request.request_id} (priority {tier.name})")
                    return request
            return None

    def complete(self, request_id: str) -> None:
        """Mark a request as completed."""
        with self._lock:
            self._active_requests.pop(request_id, None)

    def preempt_lower(self, incoming_priority: PriorityTier) -> list[str]:
        """Preempt active requests with lower priority to free resources.

        Only works if the incoming priority tier has preemption_enabled.

        Returns:
            List of preempted request IDs.
        """
        policy = self.get_policy(incoming_priority)
        if not policy.preemption_enabled:
            return []
        preempted = []
        with self._lock:
            for req_id, request in list(self._active_requests.items()):
                if request.priority > incoming_priority:
                    self._active_requests.pop(req_id)
                    self._stats["preempted"] += 1
                    preempted.append(req_id)
                    self._preemption_log.append({
                        "preempted_id": req_id,
                        "preempted_priority": request.priority.name,
                        "by_priority": incoming_priority.name,
                        "timestamp": time.time(),
                    })
                    logger.info(
                        f"Preempted {req_id} ({request.priority.name}) "
                        f"for {incoming_priority.name} request"
                    )
        return preempted

    def _evict_timed_out(self) -> None:
        """Evict requests that have been queued too long for their tier."""
        now = time.time()
        for tier in PriorityTier:
            policy = self.get_policy(tier)
            queue = self._queues[tier]
            while queue:
                request = queue[0]
                if now - request.queued_at > policy.max_queue_time_s:
                    queue.popleft()
                    self._stats["timeout_evicted"] += 1
                    logger.warning(
                        f"Evicted {request.request_id} from {tier.name} queue "
                        f"after {request.queue_time_s:.1f}s (limit: {policy.max_queue_time_s}s)"
                    )
                else:
                    break

    def queue_lengths(self) -> dict[str, int]:
        """Get the current queue lengths per tier."""
        with self._lock:
            return {tier.name: len(q) for tier, q in self._queues.items()}

    def is_idle(self) -> bool:
        """Check if all queues are empty and no active requests."""
        with self._lock:
            all_empty = all(len(q) == 0 for q in self._queues.values())
            return all_empty and len(self._active_requests) == 0

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def get_preemption_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent preemption events."""
        with self._lock:
            return list(self._preemption_log[-limit:])
