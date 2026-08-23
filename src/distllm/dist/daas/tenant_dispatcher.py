"""Per-tenant request dispatch with weighted fair queuing and priority levels.

Provides a multi-tenant dispatcher that maintains separate request queues
per tenant and uses weighted fair queuing (WFQ) to select the next request
across tenants. Each request carries a priority level drawn from
``critical > high > normal > low``.

Usage::

    dispatcher = TenantDispatcher()
    dispatcher.set_tenant_weight("tenant-a", weight=2.0)

    # Dispatch a request (queues by tenant).
    dispatcher.dispatch(Request(tenant_id="tenant-a", payload={...}))

    # Dequeue picks the next request using WFQ across tenants.
    while (req := dispatcher.dequeue()) is not None:
        process(req)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from heapq import heappop, heappush
from typing import Any, Generic, TypeVar

from loguru import logger

T = TypeVar("T")


class Priority(IntEnum):
    """Scheduling priority levels.

    Higher numeric value = higher priority.
    """

    CRITICAL = auto()  # 4
    HIGH = auto()      # 3
    NORMAL = auto()    # 2
    LOW = auto()       # 1


@dataclass(order=True)
class _QueueItem(Generic[T]):
    """Internal heap item.  Sorted by (priority, virtual_finish, timestamp)."""

    priority: int
    virtual_finish: float
    timestamp: float
    request: T = field(compare=False)


@dataclass
class Request(Generic[T]):
    """A dispatchable request associated with a tenant.

    Attributes:
        tenant_id:  The tenant this request belongs to.
        payload:    Arbitrary request data.
        priority:   Scheduling priority level.
        submitted_at:  Unix timestamp of submission.
        request_id: Optional unique identifier.
    """

    tenant_id: str
    payload: T
    priority: Priority = Priority.NORMAL
    submitted_at: float = 0.0
    request_id: str = ""


@dataclass
class TenantQueueState:
    """Runtime state for a single tenant's queue."""

    weight: float = 1.0
    virtual_time: float = 0.0  # WFQ virtual finish time
    active_requests: int = 0
    total_queued: int = 0
    total_dispatched: int = 0


class TenantDispatcher:
    """Weighted fair queuing dispatcher with per-tenant queues and priority levels.

    Maintains one heap per tenant.  Each heap item carries a *virtual finish
    time* computed from the tenant's weight and the current round-robin virtual
    time.  ``dequeue()`` pops the smallest ``(priority, virtual_finish)`` item
    across all tenant heaps.

    Thread-safe for concurrent dispatch/dequeue.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[_QueueItem]] = defaultdict(list)
        self._states: dict[str, TenantQueueState] = defaultdict(TenantQueueState)
        self._global_vtime: float = 0.0
        self._lock = threading.Lock()

    # ── Configuration ─────────────────────────────────────────────────

    def set_tenant_weight(self, tenant_id: str, weight: float) -> None:
        """Set the scheduling weight for *tenant_id*.

        A higher weight gives the tenant a larger share of the scheduling
        capacity.  Weight must be positive.

        Raises:
            ValueError: If *weight* is not positive.
        """
        if weight <= 0:
            raise ValueError(f"Weight must be positive, got {weight}")
        with self._lock:
            self._states[tenant_id].weight = weight
            # Ensure a queue exists.
            if tenant_id not in self._queues:
                self._queues[tenant_id] = []

    def get_tenant_state(self, tenant_id: str) -> TenantQueueState | None:
        """Return the current scheduling state for *tenant_id*, or ``None``."""
        return self._states.get(tenant_id)

    # ── Core operations ───────────────────────────────────────────────

    def dispatch(self, request: Request) -> None:
        """Queue *request* in its tenant's queue.

        The request's ``submitted_at`` is set to the current time if it
        was left at ``0.0``.
        """
        if not request.submitted_at:
            request.submitted_at = time.time()

        state = self._states[request.tenant_id]
        weight = state.weight

        # Virtual finish time for WFQ:
        #   v_finish = max(global_vtime, tenant_vtime) + 1 / weight
        with self._lock:
            vtime = max(self._global_vtime, state.virtual_time)
            finish = vtime + 1.0 / weight

            item = _QueueItem(
                priority=request.priority.value,
                virtual_finish=finish,
                timestamp=request.submitted_at,
                request=request,
            )
            heappush(self._queues[request.tenant_id], item)

            state.virtual_time = finish
            state.total_queued += 1

    def dequeue(self) -> Request | None:
        """Pick and return the next request across all tenant queues.

        Uses weighted fair queuing across tenants, with priority as the
        primary sort key within each tenant.  Returns ``None`` when all
        queues are empty.
        """
        with self._lock:
            best_tenant: str | None = None
            best_item: _QueueItem | None = None

            for tid, heap in self._queues.items():
                # Skip empty queues.
                if not heap:
                    continue

                # Peek at the head of each tenant heap.
                candidate = heap[0]
                if best_item is None or candidate < best_item:
                    best_item = candidate
                    best_tenant = tid

            if best_tenant is None or best_item is None:
                return None

            # Pop the winning item.
            heappop(self._queues[best_tenant])

            state = self._states[best_tenant]
            state.active_requests += 1
            state.total_dispatched += 1

            self._global_vtime = best_item.virtual_finish

            request = best_item.request
            return request

    def complete(self, tenant_id: str) -> None:
        """Mark one in-flight request as completed for *tenant_id*.

        Call this after a request dequeued via ``dequeue()`` has finished
        processing so the tenant's active-request count stays accurate.
        """
        with self._lock:
            state = self._states.get(tenant_id)
            if state is not None:
                state.active_requests = max(0, state.active_requests - 1)

    # ── Queue introspection ───────────────────────────────────────────

    def queue_depth(self, tenant_id: str) -> int:
        """Return the number of queued (not yet dequeued) requests for a tenant."""
        with self._lock:
            heap = self._queues.get(tenant_id)
            return len(heap) if heap else 0

    def total_queued(self) -> int:
        """Return the total number of queued requests across all tenants."""
        with self._lock:
            return sum(len(h) for h in self._queues.values())

    def tenant_ids(self) -> list[str]:
        """Return the list of tenant IDs that have ever been seen."""
        return list(self._states.keys())

    def reset(self) -> None:
        """Clear all queues and reset scheduling state."""
        with self._lock:
            self._queues.clear()
            self._states.clear()
            self._global_vtime = 0.0

    # ── Statistics ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return aggregate and per-tenant dispatch statistics."""
        with self._lock:
            per_tenant = {}
            for tid, state in self._states.items():
                per_tenant[tid] = {
                    "weight": state.weight,
                    "virtual_time": round(state.virtual_time, 4),
                    "active_requests": state.active_requests,
                    "total_queued": state.total_queued,
                    "total_dispatched": state.total_dispatched,
                    "queue_depth": len(self._queues.get(tid, [])),
                }
            return {
                "tenants": len(self._states),
                "global_virtual_time": round(self._global_vtime, 4),
                "total_queued_all": sum(s.total_queued for s in self._states.values()),
                "total_dispatched_all": sum(s.total_dispatched for s in self._states.values()),
                "per_tenant": per_tenant,
            }
