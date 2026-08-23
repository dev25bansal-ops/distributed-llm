"""Per-tenant GPU resource isolation — memory budgets and concurrency limits.

Provides a ``ResourceIsolator`` that tracks per-tenant GPU memory allocations
and enforces maximum concurrency limits, enabling safe co-location of multiple
tenants on shared GPU resources.

Usage::

    isolator = ResourceIsolator()

    # Configure tenant budgets.
    isolator.set_memory_budget("tenant-a", memory_mb=4096)
    isolator.set_concurrency_limit("tenant-a", max_concurrent=2)

    # Check and allocate.
    if isolator.can_allocate("tenant-a", memory_mb=1024):
        isolator.allocate("tenant-a", memory_mb=1024)
        # ... run inference ...
        isolator.release("tenant-a", memory_mb=1024)

    remaining = isolator.get_available("tenant-a")
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TenantResourceState:
    """Resource tracking state for a single tenant."""

    memory_budget_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    max_concurrent: int = 0
    current_concurrent: int = 0
    peak_memory_mb: float = 0.0
    peak_concurrent: int = 0


class ResourceIsolator:
    """Per-tenant GPU memory budgeting and concurrency enforcement.

    Thread-safe.  Each tenant has:
    - A *memory budget* (MB) representing the maximum GPU memory they may use.
    - A *max concurrency* limit representing how many requests they may run
      simultaneously.

    ``allocate()`` and ``release()`` are reference-counted on memory so that
    overlapping concurrent requests from the same tenant are tracked
    correctly.
    """

    def __init__(self) -> None:
        self._states: dict[str, TenantResourceState] = {}
        self._lock = threading.Lock()
        self._global_memory_budget_mb: float = 0.0  # 0 = unlimited
        self._global_memory_used_mb: float = 0.0

    # ── Configuration ─────────────────────────────────────────────────

    def set_memory_budget(self, tenant_id: str, memory_mb: float) -> None:
        """Set the GPU memory budget for *tenant_id* (in MB).

        Must be non-negative.  A value of ``0`` means no memory is
        available (effectively blocks the tenant).

        Raises:
            ValueError: If *memory_mb* is negative.
        """
        if memory_mb < 0:
            raise ValueError(f"Memory budget must be non-negative, got {memory_mb}")
        with self._lock:
            state = self._states.setdefault(tenant_id, TenantResourceState())
            state.memory_budget_mb = memory_mb
            logger.debug(f"Memory budget for {tenant_id}: {memory_mb} MB")

    def set_concurrency_limit(self, tenant_id: str, max_concurrent: int) -> None:
        """Set the maximum concurrent requests for *tenant_id*.

        Raises:
            ValueError: If *max_concurrent* is negative.
        """
        if max_concurrent < 0:
            raise ValueError(f"Concurrency limit must be non-negative, got {max_concurrent}")
        with self._lock:
            state = self._states.setdefault(tenant_id, TenantResourceState())
            state.max_concurrent = max_concurrent
            logger.debug(f"Concurrency limit for {tenant_id}: {max_concurrent}")

    def set_global_memory_budget(self, memory_mb: float) -> None:
        """Set an optional global GPU memory budget across all tenants.

        A value of ``0`` means unlimited.
        """
        with self._lock:
            self._global_memory_budget_mb = memory_mb

    # ── Allocation ────────────────────────────────────────────────────

    def can_allocate(self, tenant_id: str, memory_mb: float) -> bool:
        """Check whether *tenant_id* may allocate *memory_mb* MB.

        Returns ``True`` only if both the tenant's budget and the global
        budget (if set) have enough headroom, and the concurrency limit
        is not exceeded.
        """
        if memory_mb < 0:
            return False

        with self._lock:
            state = self._states.get(tenant_id)
            if state is None:
                return False

            # Concurrency check.
            if state.max_concurrent > 0 and state.current_concurrent >= state.max_concurrent:
                return False

            # Tenant memory budget check.
            if state.memory_budget_mb > 0:
                new_tenant_use = state.memory_allocated_mb + memory_mb
                if new_tenant_use > state.memory_budget_mb:
                    return False

            # Global memory budget check.
            if self._global_memory_budget_mb > 0:
                new_global_use = self._global_memory_used_mb + memory_mb
                if new_global_use > self._global_memory_budget_mb:
                    return False

            return True

    def allocate(self, tenant_id: str, memory_mb: float) -> bool:
        """Allocate *memory_mb* MB for *tenant_id*.

        Returns ``True`` on success, ``False`` if the allocation would
        exceed the tenant's or global budget (callers should check
        ``can_allocate()`` first or inspect the return value).
        """
        if memory_mb < 0:
            return False

        with self._lock:
            if not self.can_allocate(tenant_id, memory_mb):
                return False

            state = self._states[tenant_id]
            state.memory_allocated_mb += memory_mb
            state.current_concurrent += 1

            if state.memory_allocated_mb > state.peak_memory_mb:
                state.peak_memory_mb = state.memory_allocated_mb
            if state.current_concurrent > state.peak_concurrent:
                state.peak_concurrent = state.current_concurrent

            self._global_memory_used_mb += memory_mb

            return True

    def release(self, tenant_id: str, memory_mb: float) -> bool:
        """Release *memory_mb* MB previously allocated to *tenant_id*.

        Returns ``True`` on success, ``False`` if the tenant is unknown
        or the released amount would make the allocated count negative.
        """
        if memory_mb < 0:
            return False

        with self._lock:
            state = self._states.get(tenant_id)
            if state is None:
                return False

            state.memory_allocated_mb = max(0.0, state.memory_allocated_mb - memory_mb)
            state.current_concurrent = max(0, state.current_concurrent - 1)
            self._global_memory_used_mb = max(0.0, self._global_memory_used_mb - memory_mb)

            return True

    # ── Querying ──────────────────────────────────────────────────────

    def get_available(self, tenant_id: str) -> float:
        """Return the remaining memory budget (MB) for *tenant_id*.

        Returns ``0.0`` if the tenant is not configured or if the
        budget has been exhausted.  Returns the full budget if the
        tenant has no allocations yet.
        """
        with self._lock:
            state = self._states.get(tenant_id)
            if state is None:
                return 0.0
            if state.memory_budget_mb <= 0:
                return 0.0
            remaining = state.memory_budget_mb - state.memory_allocated_mb
            return max(0.0, remaining)

    def get_usage(self, tenant_id: str) -> dict[str, Any] | None:
        """Return detailed resource usage for *tenant_id*.

        Returns ``None`` if the tenant is unknown.
        """
        with self._lock:
            state = self._states.get(tenant_id)
            if state is None:
                return None
            return {
                "memory_budget_mb": state.memory_budget_mb,
                "memory_allocated_mb": state.memory_allocated_mb,
                "memory_available_mb": max(0.0, state.memory_budget_mb - state.memory_allocated_mb),
                "peak_memory_mb": state.peak_memory_mb,
                "max_concurrent": state.max_concurrent,
                "current_concurrent": state.current_concurrent,
                "peak_concurrent": state.peak_concurrent,
            }

    # ── Administrative ────────────────────────────────────────────────

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove *tenant_id* and release all its tracked resources."""
        with self._lock:
            state = self._states.pop(tenant_id, None)
            if state is not None:
                self._global_memory_used_mb = max(
                    0.0, self._global_memory_used_mb - state.memory_allocated_mb
                )

    def reset(self, tenant_id: str | None = None) -> None:
        """Reset resource tracking.

        If *tenant_id* is provided, only that tenant's state is reset.
        Otherwise all state is cleared.
        """
        with self._lock:
            if tenant_id is not None:
                state = self._states.get(tenant_id)
                if state is not None:
                    self._global_memory_used_mb = max(
                        0.0, self._global_memory_used_mb - state.memory_allocated_mb
                    )
                    self._states[tenant_id] = TenantResourceState()
            else:
                self._states.clear()
                self._global_memory_used_mb = 0.0

    def stats(self) -> dict[str, Any]:
        """Return aggregate and per-tenant resource statistics."""
        with self._lock:
            total_budget = sum(s.memory_budget_mb for s in self._states.values())
            total_allocated = sum(s.memory_allocated_mb for s in self._states.values())
            per_tenant = {
                tid: {
                    "memory_budget_mb": s.memory_budget_mb,
                    "memory_allocated_mb": s.memory_allocated_mb,
                    "memory_available_mb": max(0.0, s.memory_budget_mb - s.memory_allocated_mb),
                    "peak_memory_mb": s.peak_memory_mb,
                    "max_concurrent": s.max_concurrent,
                    "current_concurrent": s.current_concurrent,
                }
                for tid, s in self._states.items()
            }
            return {
                "tenants": len(self._states),
                "global_memory_budget_mb": self._global_memory_budget_mb,
                "global_memory_used_mb": self._global_memory_used_mb,
                "total_budget_mb": total_budget,
                "total_allocated_mb": total_allocated,
                "global_available_mb": max(
                    0.0,
                    (self._global_memory_budget_mb
                     if self._global_memory_budget_mb > 0
                     else float("inf")) - self._global_memory_used_mb,
                ),
                "per_tenant": per_tenant,
            }
