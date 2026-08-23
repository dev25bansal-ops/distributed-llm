"""Deadline-aware scheduling, GPU-memory-aware batch packing,
preemptive stage scheduling, and heterogeneous batch sizing
for distributed LLM inference.

Provides the scheduling logic that sits above the iteration-level
scheduler (``IterationScheduler``) and the latency-aware batcher
(``LatencyAwareBatcher``), adding SLA-deadline awareness, OOM
prevention, prefill/decode/verify stage preemption, and GPU-tier
aware batch sizes.
"""

from __future__ import annotations

import enum
import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GPU tier memory boundaries (in bytes)
TIER_SMALL_MAX_BYTES = 8 * 1024**3       # 8 GB
TIER_MEDIUM_MIN_BYTES = 16 * 1024**3      # 16 GB
TIER_MEDIUM_MAX_BYTES = 24 * 1024**3      # 24 GB
TIER_LARGE_MIN_BYTES = 40 * 1024**3       # 40 GB
TIER_LARGE_MAX_BYTES = 80 * 1024**3       # 80 GB

# Default per-sequence memory estimate when model info is unavailable
DEFAULT_PER_SEQ_BYTES = 512 * 1024        # 512 KB


# ---------------------------------------------------------------------------
# Deadline-Aware Batch Scheduler
# ---------------------------------------------------------------------------

@dataclass(order=True)
class DeadlineRequest:
    """A single request tracked for deadline-aware scheduling.

    The ``order=True`` annotation enables heapq to sort by deadline
    (earliest deadline first).  The tie-breaker guarantees FIFO among
    requests with identical deadlines.
    """

    deadline: float
    tie_breaker: int = field(compare=False, default=0)
    request_id: str = field(compare=False, default="")
    priority: int = field(compare=False, default=0)
    created_at: float = field(compare=False, default=0.0)
    seq_len: int = field(compare=False, default=0)
    estimated_tokens: int = field(compare=False, default=0)


class DeadlineAwareBatchScheduler:
    """Schedules requests by Earliest-Deadline-First (EDF) ordering.

    Responsibilities:
    - Maintain a min-heap of requests keyed by absolute deadline.
    - On ``schedule_batch()``, pop requests in EDF order, skipping any
      that have already missed their deadline.
    - Report missed-deadline stats for SLO monitoring.
    """

    def __init__(self, time_provider: Callable[[], float] | None = None) -> None:
        self._time_provider = time_provider or time.time
        self._heap: list[DeadlineRequest] = []
        self._request_map: dict[str, DeadlineRequest] = {}
        self._tie_counter = 0
        self._lock = threading.Lock()

        # tracking
        self._missed_count = 0
        self._scheduled_count = 0
        self._total_deadline_miss_lag: float = 0.0

    # -- public API ---------------------------------------------------------

    def add_request(
        self,
        request_id: str,
        deadline: float,
        *,
        priority: int = 0,
        seq_len: int = 0,
        estimated_tokens: int = 0,
    ) -> None:
        """Register a request with its absolute deadline (Unix timestamp)."""
        with self._lock:
            self._tie_counter += 1
            req = DeadlineRequest(
                deadline=deadline,
                tie_breaker=self._tie_counter,
                request_id=request_id,
                priority=priority,
                created_at=self._time_provider(),
                seq_len=seq_len,
                estimated_tokens=estimated_tokens,
            )
            heapq.heappush(self._heap, req)
            self._request_map[request_id] = req
            logger.debug("Deadline request added: {} deadline={:.1f}s", request_id, deadline)

    def remove_request(self, request_id: str) -> None:
        """Remove a previously registered request.

        This is a lazy removal — the entry stays in the heap but is
        filtered out when popped from ``schedule_batch()``.
        """
        with self._lock:
            self._request_map.pop(request_id, None)

    def update_deadline(self, request_id: str, new_deadline: float) -> bool:
        """Extend (or shorten) the deadline of an existing request.

        Returns ``True`` if the request was found and updated.
        Because heapq does not support efficient decrease-key, we
        push a new entry and rely on lazy deletion of the old one.
        """
        with self._lock:
            old = self._request_map.get(request_id)
            if old is None:
                return False
            self._tie_counter += 1
            updated = DeadlineRequest(
                deadline=new_deadline,
                tie_breaker=self._tie_counter,
                request_id=request_id,
                priority=old.priority,
                created_at=old.created_at,
                seq_len=old.seq_len,
                estimated_tokens=old.estimated_tokens,
            )
            heapq.heappush(self._heap, updated)
            self._request_map[request_id] = updated
            logger.debug("Deadline updated: {} {:.1f}s -> {:.1f}s", request_id, old.deadline, new_deadline)
            return True

    def schedule_batch(self, max_batch_size: int) -> list[str]:
        """Return up to *max_batch_size* request IDs in EDF order.

        Requests whose deadline has passed are skipped and counted as
        **missed**.  Expired heap entries (from ``remove_request()`` or
        ``update_deadline()``) are also purged.
        """
        batch: list[str] = []
        now = self._time_provider()

        with self._lock:
            while self._heap and len(batch) < max_batch_size:
                req = heapq.heappop(self._heap)

                # Lazy deletion: request was removed or superseded
                active = self._request_map.get(req.request_id)
                if active is None or active.tie_breaker != req.tie_breaker:
                    continue

                # Deadline check — skip and clean up expired requests
                if now > req.deadline:
                    self._missed_count += 1
                    lag = now - req.deadline
                    self._total_deadline_miss_lag += lag
                    self._request_map.pop(req.request_id, None)
                    logger.warning(
                        "Request {} missed deadline by {:.1f}s", req.request_id, lag
                    )
                    continue

                batch.append(req.request_id)
                self._request_map.pop(req.request_id, None)
                self._scheduled_count += 1

            # Re-push remaining heap entries that were popped but not
            # consumed (the loop stops when len(batch) == max_batch_size,
            # so entries already popped are lost).  To avoid losing
            # entries, we use a peek-based approach.  Actually, the loop
            # terminates either by heap exhaustion or by reaching
            # max_batch_size.  In the latter case, un-popped entries
            # remain in the heap — no re-push needed.  Entries already
            # popped (via heapq.heappop) have been either consumed or
            # lazily deleted; no re-push necessary.

        return batch

    def pending_count(self) -> int:
        """Number of active (non-expired) requests waiting to be scheduled."""
        with self._lock:
            return len(self._request_map)

    def missed_deadlines(self) -> int:
        """Cumulative count of requests that missed their deadline."""
        return self._missed_count

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        with self._lock:
            return {
                "pending_requests": len(self._request_map),
                "heap_size": len(self._heap),
                "scheduled_count": self._scheduled_count,
                "missed_count": self._missed_count,
                "avg_miss_lag_ms": round(
                    (self._total_deadline_miss_lag / max(self._missed_count, 1)) * 1000, 1
                ),
            }


# ---------------------------------------------------------------------------
# GPU-Memory-Aware Batch Packer
# ---------------------------------------------------------------------------

@dataclass
class NodeMemoryState:
    """Current memory snapshot for a single GPU node."""

    node_id: str
    total_bytes: int
    available_bytes: int
    last_updated: float = field(default_factory=time.time)


class GPUMemoryBatchPacker:
    """Packs inference batches within per-node GPU memory budgets.

    Responsibilities:
    - Track available GPU memory reported by each worker node.
    - Accept or reject a candidate batch based on estimated memory
      consumption.
    - Pack a list of candidate sequences into the largest batch that
      fits under the memory limit.
    - Provide a heuristic ``heterogeneous_batch_sizes()`` that returns
      per-GPU-tier batch size limits.
    """

    def __init__(
        self,
        per_sequence_bytes: int = DEFAULT_PER_SEQ_BYTES,
        safety_margin: float = 0.9,
    ) -> None:
        self._per_sequence_bytes = per_sequence_bytes
        self._safety_margin = safety_margin
        self._nodes: dict[str, NodeMemoryState] = {}
        self._lock = threading.Lock()
        self._oom_prevented_count = 0

    # -- memory tracking ---------------------------------------------------

    def update_node_memory(
        self,
        node_id: str,
        total_bytes: int,
        available_bytes: int,
    ) -> None:
        """Update the memory snapshot for *node_id*."""
        with self._lock:
            self._nodes[node_id] = NodeMemoryState(
                node_id=node_id,
                total_bytes=total_bytes,
                available_bytes=available_bytes,
            )
            logger.debug(
                "Node {} memory updated: {:.1f}/{:.1f} GB available",
                node_id,
                available_bytes / 1024**3,
                total_bytes / 1024**3,
            )

    def remove_node(self, node_id: str) -> None:
        """Remove a node from tracking (e.g. on disconnect)."""
        with self._lock:
            self._nodes.pop(node_id, None)
            logger.debug("Node {} removed from memory tracking", node_id)

    def get_node_memory(self, node_id: str) -> NodeMemoryState | None:
        """Return the latest known memory state for *node_id*."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_available_bytes(self, node_id: str) -> int:
        """Return available bytes for *node_id*, or 0 if unknown."""
        state = self.get_node_memory(node_id)
        return state.available_bytes if state else 0

    # -- batch admission & packing -----------------------------------------

    def can_fit_batch(
        self,
        node_id: str,
        num_sequences: int,
        per_seq_bytes: int | None = None,
    ) -> bool:
        """Check whether *num_sequences* fits in the memory budget for *node_id*.

        Returns ``False`` when the node is unknown or the batch
        exceeds the available memory (after applying the safety margin).
        """
        state = self.get_node_memory(node_id)
        if state is None:
            return False
        needed = num_sequences * (per_seq_bytes or self._per_sequence_bytes)
        budget = int(state.available_bytes * self._safety_margin)
        fits = needed <= budget
        if not fits:
            logger.debug(
                "Batch of {} seqs needs {:.1f} MB, node {} has {:.1f} MB budget — REJECTED",
                num_sequences,
                needed / 1024**2,
                node_id,
                budget / 1024**2,
            )
        return fits

    def pack_batch(
        self,
        node_id: str,
        candidate_sequences: list[dict],
        per_seq_bytes: int | None = None,
        max_batch_size: int = 64,
    ) -> list[dict]:
        """Greedily pack *candidate_sequences* into a memory-feasible batch.

        ``candidate_sequences``: list of dicts that must contain at least
        a ``"request_id"`` key.

        Returns a sub-list of sequences that fit within the memory
        budget for *node_id*, preserving the input order.  Sequences
        that would cause OOM are left out and logged.
        """
        state = self.get_node_memory(node_id)
        if state is None:
            logger.warning("Node {} unknown — cannot pack batch", node_id)
            return []

        pbs = per_seq_bytes or self._per_sequence_bytes
        budget = int(state.available_bytes * self._safety_margin)
        used = 0
        batch: list[dict] = []

        for seq in candidate_sequences:
            if len(batch) >= max_batch_size:
                break
            seq_cost = pbs
            if seq.get("seq_len"):
                seq_cost = max(seq_cost, seq["seq_len"] * 1024)  # rough estimate
            if used + seq_cost <= budget:
                batch.append(seq)
                used += seq_cost
            else:
                logger.debug(
                    "OOM prevention: seq {} (est {:.1f} MB) excluded from batch on {}",
                    seq.get("request_id", "?"),
                    seq_cost / 1024**2,
                    node_id,
                )
                with self._lock:
                    self._oom_prevented_count += 1

        return batch

    # -- heterogeneous batch sizes -----------------------------------------

    def heterogeneous_batch_sizes(
        self,
        gpu_tiers: dict[str, str] | None = None,
    ) -> dict[str, int]:
        """Return per-node-id batch size limits based on available memory.

        If *gpu_tiers* is provided (``{node_id: tier_label}``), only
        those nodes are considered.  Otherwise every tracked node is
        included.

        Batch size is computed as::

            max_batch = floor(available_bytes * safety / per_seq_bytes)
        """
        pbs = self._per_sequence_bytes
        if pbs <= 0:
            pbs = DEFAULT_PER_SEQ_BYTES

        result: dict[str, int] = {}
        with self._lock:
            nodes_to_scan = {
                nid: state
                for nid, state in self._nodes.items()
                if gpu_tiers is None or nid in gpu_tiers
            }
            for nid, state in nodes_to_scan.items():
                budget = int(state.available_bytes * self._safety_margin)
                bs = max(1, budget // pbs)
                bs = min(bs, 128)  # hard cap
                result[nid] = bs

        return result

    # -- stats -------------------------------------------------------------

    def oom_prevented_count(self) -> int:
        """Number of batch exclusions due to OOM prevention."""
        return self._oom_prevented_count

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        with self._lock:
            return {
                "nodes_tracked": len(self._nodes),
                "oom_prevented_count": self._oom_prevented_count,
                "per_sequence_bytes": self._per_sequence_bytes,
                "safety_margin": self._safety_margin,
                "nodes": {
                    nid: {
                        "total_gb": round(state.total_bytes / 1024**3, 2),
                        "available_gb": round(state.available_bytes / 1024**3, 2),
                        "last_updated": state.last_updated,
                    }
                    for nid, state in self._nodes.items()
                },
            }


# ---------------------------------------------------------------------------
# Preemptive Stage Scheduler
# ---------------------------------------------------------------------------

class StageType(enum.Enum):
    """Processing stage in the LLM inference pipeline."""

    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"

    def priority_rank(self) -> int:
        """Higher rank = higher preemption risk when prefill is backlogged.

        PREFILL (0) must not be preempted — it is boosted.
        VERIFY (1) is fast and short — rarely preempted.
        DECODE (2) is the longest stage and gets preempted first.
        """
        if self == StageType.PREFILL:
            return 0
        elif self == StageType.VERIFY:
            return 1
        return 2  # DECODE


@dataclass(order=True)
class StageRequest:
    """A single request at a specific stage in the pipeline."""

    stage_priority: int = field(compare=False, default=0)
    submitted_at: float = field(compare=False, default=0.0)
    request_id: str = field(compare=False, default="")
    stage: StageType = field(compare=False, default=StageType.PREFILL)
    estimated_duration_ms: float = field(compare=False, default=0.0)
    is_critical_path: bool = field(compare=False, default=False)
    inherited_priority: int = field(compare=False, default=0)
    tenant_id: str = field(compare=False, default="")


class PreemptiveStageScheduler:
    """Three-stage preemptive scheduler for LLM inference pipelines.

    Stages (in order): **prefill** → **decode** → **verify**.

    Scheduling policy:
    - **Prefill priority boost** — prefill requests jump ahead of decode
      to reduce time-to-first-token (TTFT).
    - **Decode preemption** — when the prefill backlog exceeds a
      configurable threshold, in-flight decode requests are preempted
      to free capacity.
    - **Priority inheritance** — critical-path requests inherit the
      highest priority of any request that depends on them, preventing
      priority inversion.
    """

    def __init__(
        self,
        prefill_boost: int = 10,
        preemption_backlog_threshold: int = 4,
        max_prefill_per_batch: int = 4,
        time_provider: Callable[[], float] | None = None,
    ) -> None:
        self._prefill_boost = prefill_boost
        self._preemption_backlog_threshold = preemption_backlog_threshold
        self._max_prefill_per_batch = max_prefill_per_batch
        self._time_provider = time_provider or time.time

        self._lock = threading.Lock()

        # Per-stage heaps (min-heap by priority then submission time)
        self._pending: dict[StageType, list[StageRequest]] = {
            StageType.PREFILL: [],
            StageType.DECODE: [],
            StageType.VERIFY: [],
        }

        # In-flight decode requests (preemptible)
        self._in_flight_decode: dict[str, StageRequest] = {}

        # Critical-path tracking: request_id -> set of dependent request_ids
        self._dependents: dict[str, set[str]] = {}

        # Counters
        self._preempted_count = 0
        self._scheduled_count = 0

    # -- submission ---------------------------------------------------------

    def submit_request(
        self,
        stage: StageType,
        request_id: str,
        *,
        priority: int = 0,
        estimated_duration_ms: float = 0.0,
        is_critical_path: bool = False,
        tenant_id: str = "",
    ) -> None:
        """Submit a request to the given stage queue."""
        req = StageRequest(
            stage_priority=priority,
            submitted_at=self._time_provider(),
            request_id=request_id,
            stage=stage,
            estimated_duration_ms=estimated_duration_ms,
            is_critical_path=is_critical_path,
            tenant_id=tenant_id,
        )

        with self._lock:
            heapq.heappush(self._pending[stage], req)
            logger.debug(
                "Stage request submitted: {} stage={} priority={} critical={}",
                request_id,
                stage.value,
                priority,
                is_critical_path,
            )

    def mark_critical_path(
        self,
        request_id: str,
        dependent_ids: list[str] | None = None,
    ) -> None:
        """Mark *request_id* as critical path and register dependents.

        When *dependent_ids* are provided, *request_id* inherits their
        highest priority (priority inheritance).
        """
        with self._lock:
            if dependent_ids:
                for dep_id in dependent_ids:
                    if dep_id not in self._dependents:
                        self._dependents[dep_id] = set()
                    self._dependents[dep_id].add(request_id)

    # -- scheduling ---------------------------------------------------------

    def schedule(self, max_budget: int = 16) -> list[StageRequest]:
        """Build the next execution batch respecting stage priorities.

        Order of selection:
        1. Up to ``_max_prefill_per_batch`` prefill requests (boosted).
        2. Remaining budget filled by decode (unless preempted) then verify.
        3. If decode preemption is active, prefill may consume decode's
           share of the budget.

        Returns a list of ``StageRequest`` ordered by execution
        priority.
        """
        batch: list[StageRequest] = []
        now = self._time_provider()
        budget_remaining = max_budget

        with self._lock:
            # ---- prefill (boosted) ----
            prefill_heap = self._pending[StageType.PREFILL]
            prefill_taken = 0
            while prefill_heap and budget_remaining > 0 and prefill_taken < self._max_prefill_per_batch:
                req = heapq.heappop(prefill_heap)
                self._apply_priority_inheritance(req)
                # Apply prefill boost
                boosted = StageRequest(
                    stage_priority=req.stage_priority - self._prefill_boost,
                    submitted_at=req.submitted_at,
                    request_id=req.request_id,
                    stage=req.stage,
                    estimated_duration_ms=req.estimated_duration_ms,
                    is_critical_path=req.is_critical_path,
                    inherited_priority=req.inherited_priority,
                    tenant_id=req.tenant_id,
                )
                batch.append(boosted)
                budget_remaining -= 1
                prefill_taken += 1
                self._scheduled_count += 1

            # Check whether decode should be preempted.
            prefill_backlog = len(prefill_heap) + prefill_taken
            should_preempt_decode = prefill_backlog > self._preemption_backlog_threshold

            if should_preempt_decode:
                # Preempt in-flight decode requests
                for rid in list(self._in_flight_decode.keys()):
                    self._in_flight_decode.pop(rid, None)
                    self._preempted_count += 1
                    logger.info("Preempted decode request {} for prefill backlog", rid)

            # ---- decode ----
            decode_heap = self._pending[StageType.DECODE]
            while decode_heap and budget_remaining > 0:
                req = heapq.heappop(decode_heap)
                self._apply_priority_inheritance(req)
                self._in_flight_decode[req.request_id] = req
                batch.append(req)
                budget_remaining -= 1
                self._scheduled_count += 1

            # ---- verify ----
            verify_heap = self._pending[StageType.VERIFY]
            while verify_heap and budget_remaining > 0:
                req = heapq.heappop(verify_heap)
                self._apply_priority_inheritance(req)
                batch.append(req)
                budget_remaining -= 1
                self._scheduled_count += 1

        return batch

    # -- completion ---------------------------------------------------------

    def complete_request(self, request_id: str) -> None:
        """Mark a decode request as completed (remove from in-flight set)."""
        with self._lock:
            self._in_flight_decode.pop(request_id, None)
            self._dependents.pop(request_id, None)

    # -- preemption stats ---------------------------------------------------

    def preempted_count(self) -> int:
        """Number of decode requests that have been preempted."""
        return self._preempted_count

    def pending_counts(self) -> dict[str, int]:
        """Return pending request counts per stage."""
        with self._lock:
            return {
                stage.value: len(heap)
                for stage, heap in self._pending.items()
            }

    # -- internal helpers ---------------------------------------------------

    def _apply_priority_inheritance(self, req: StageRequest) -> None:
        """Apply priority inheritance for critical-path requests.

        If *req* has dependents that are also pending, its inherited
        priority is raised to match the highest priority among those
        dependents, preventing priority inversion.
        """
        dependents = self._dependents.get(req.request_id)
        if not dependents:
            return

        max_dep_priority = req.stage_priority
        for dep_id in dependents:
            for stage_heap in self._pending.values():
                for entry in stage_heap:
                    if entry.request_id == dep_id:
                        if entry.stage_priority < max_dep_priority:
                            max_dep_priority = entry.stage_priority
                        break

        if max_dep_priority < req.stage_priority:
            req.inherited_priority = max_dep_priority
            req.stage_priority = max_dep_priority
            logger.debug(
                "Priority inheritance: {} boosted to {} (dependent critical path)",
                req.request_id,
                max_dep_priority,
            )

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        with self._lock:
            return {
                "preempted_count": self._preempted_count,
                "scheduled_count": self._scheduled_count,
                "prefill_boost": self._prefill_boost,
                "preemption_backlog_threshold": self._preemption_backlog_threshold,
                "in_flight_decode": len(self._in_flight_decode),
                "pending": {
                    stage.value: len(heap)
                    for stage, heap in self._pending.items()
                },
                "dependents_tracked": len(self._dependents),
            }


# ---------------------------------------------------------------------------
# Heterogeneous Batch Sizer
# ---------------------------------------------------------------------------

class GpuTier(enum.Enum):
    """GPU tiers by total device memory."""

    SMALL = "small"        # 4-8 GB
    MEDIUM = "medium"      # 16-24 GB
    LARGE = "large"        # 40-80 GB
    UNKNOWN = "unknown"    # outside known ranges

    @staticmethod
    def from_memory_bytes(memory_bytes: int) -> GpuTier:
        """Classify a GPU by its total memory in bytes."""
        if 4 * 1024**3 <= memory_bytes <= TIER_SMALL_MAX_BYTES:
            return GpuTier.SMALL
        if TIER_MEDIUM_MIN_BYTES <= memory_bytes <= TIER_MEDIUM_MAX_BYTES:
            return GpuTier.MEDIUM
        if TIER_LARGE_MIN_BYTES <= memory_bytes <= TIER_LARGE_MAX_BYTES:
            return GpuTier.LARGE
        return GpuTier.UNKNOWN

    def default_batch_size(self) -> int:
        """Sensible default batch size for this tier."""
        mapping = {
            GpuTier.SMALL: 4,
            GpuTier.MEDIUM: 16,
            GpuTier.LARGE: 64,
            GpuTier.UNKNOWN: 8,
        }
        return mapping[self]

    def memory_ratio(self) -> float:
        """Relative memory capacity vs. SMALL, used for proportional scaling."""
        mapping = {
            GpuTier.SMALL: 1.0,
            GpuTier.MEDIUM: 4.0,
            GpuTier.LARGE: 12.0,
            GpuTier.UNKNOWN: 2.0,
        }
        return mapping[self]


@dataclass
class NodeInfo:
    """Runtime information for a registered GPU node."""

    node_id: str
    memory_bytes: int
    tier: GpuTier
    batch_size: int
    joined_at: float = field(default_factory=time.time)

    def update_batch_size(self, new_batch_size: int) -> None:
        """Update the batch size for this node."""
        self.batch_size = new_batch_size


class HeterogeneousBatchSizer:
    """Manages per-node batch sizes scaled to GPU memory tiers.

    GPU tiers:
    - **Small** (4-8 GB):    default batch size 4
    - **Medium** (16-24 GB): default batch size 16
    - **Large** (40-80 GB):  default batch size 64

    Batch sizes are proportional to available memory and are
    recomputed when nodes join or leave.
    """

    def __init__(
        self,
        base_batch_size: int = 4,
        per_sequence_bytes: int = DEFAULT_PER_SEQ_BYTES,
    ) -> None:
        self._base_batch_size = base_batch_size
        self._per_sequence_bytes = per_sequence_bytes
        self._nodes: dict[str, NodeInfo] = {}
        self._lock = threading.Lock()

    # -- registration -------------------------------------------------------

    def register_node(self, node_id: str, memory_bytes: int) -> NodeInfo:
        """Register a GPU node and compute its initial batch size.

        Returns the ``NodeInfo`` for the registered node.
        """
        tier = GpuTier.from_memory_bytes(memory_bytes)
        bs = self._compute_batch_size(memory_bytes, tier)

        info = NodeInfo(
            node_id=node_id,
            memory_bytes=memory_bytes,
            tier=tier,
            batch_size=bs,
        )

        with self._lock:
            self._nodes[node_id] = info
            logger.info(
                "Node registered: {} tier={} memory={:.1f} GB batch_size={}",
                node_id,
                tier.value,
                memory_bytes / 1024**3,
                bs,
            )
            self._rebalance_all()

        return info

    def unregister_node(self, node_id: str) -> None:
        """Remove a GPU node and rebalance remaining nodes."""
        with self._lock:
            removed = self._nodes.pop(node_id, None)
            if removed:
                logger.info(
                    "Node unregistered: {} tier={}", node_id, removed.tier.value
                )
                self._rebalance_all()

    def handle_node_join(self, node_id: str, memory_bytes: int) -> NodeInfo:
        """Convenience alias for ``register_node()``."""
        return self.register_node(node_id, memory_bytes)

    def handle_node_leave(self, node_id: str) -> None:
        """Convenience alias for ``unregister_node()``."""
        self.unregister_node(node_id)

    # -- querying -----------------------------------------------------------

    def get_node_info(self, node_id: str) -> NodeInfo | None:
        """Return the ``NodeInfo`` for *node_id*, or ``None``."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_batch_size(self, node_id: str) -> int:
        """Return the current batch size for *node_id*.

        Returns the tier default if the node is unknown.
        """
        info = self.get_node_info(node_id)
        if info is not None:
            return info.batch_size
        return GpuTier.UNKNOWN.default_batch_size()

    def get_tier(self, node_id: str) -> GpuTier:
        """Return the GPU tier for *node_id*.

        Returns ``GpuTier.UNKNOWN`` if the node isn't tracked.
        """
        info = self.get_node_info(node_id)
        if info is not None:
            return info.tier
        return GpuTier.UNKNOWN

    def update_batch_size(self, node_id: str, new_batch_size: int) -> bool:
        """Manually override the batch size for *node_id*.

        Returns ``True`` if the node was found and updated.
        """
        with self._lock:
            info = self._nodes.get(node_id)
            if info is None:
                return False
            info.batch_size = new_batch_size
            logger.debug("Batch size updated: {} -> {}", node_id, new_batch_size)
            return True

    def all_batch_sizes(self) -> dict[str, int]:
        """Return ``{node_id: batch_size}`` for all tracked nodes."""
        with self._lock:
            return {nid: info.batch_size for nid, info in self._nodes.items()}

    def tier_summary(self) -> dict[str, list[str]]:
        """Return node IDs grouped by tier."""
        summary: dict[str, list[str]] = {
            tier.value: [] for tier in GpuTier
        }
        with self._lock:
            for nid, info in self._nodes.items():
                summary[info.tier.value].append(nid)
        return summary

    # -- internal -----------------------------------------------------------

    def _compute_batch_size(self, memory_bytes: int, tier: GpuTier) -> int:
        """Compute a batch size proportional to available memory.

        Falls back to the tier default when the per-sequence cost
        is not configured meaningfully.
        """
        pbs = self._per_sequence_bytes
        if pbs <= 0:
            return tier.default_batch_size()

        # Reserve 20% for overhead (KV cache, activations, framework)
        usable = int(memory_bytes * 0.8)
        raw_bs = max(1, usable // pbs)
        capped = min(raw_bs, tier.default_batch_size() * 2)
        return max(capped, tier.default_batch_size() // 2)

    def _rebalance_all(self) -> None:
        """Recompute batch sizes for all nodes.

        Called after a node join or leave to ensure fairness:
        - Each node gets a batch size proportional to its memory
          relative to the smallest node.
        """
        if not self._nodes:
            return

        min_mem = min(info.memory_bytes for info in self._nodes.values())
        if min_mem <= 0:
            return

        for info in self._nodes.values():
            ratio = info.memory_bytes / min_mem
            bs = max(1, int(self._base_batch_size * ratio))
            info.batch_size = min(bs, 128)
            logger.debug(
                "Rebalanced: {} batch_size={} (ratio={:.1f})",
                info.node_id,
                info.batch_size,
                ratio,
            )

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        """Return diagnostic statistics."""
        with self._lock:
            nodes_detail = {
                nid: {
                    "tier": info.tier.value,
                    "memory_gb": round(info.memory_bytes / 1024**3, 2),
                    "batch_size": info.batch_size,
                    "joined_at": info.joined_at,
                }
                for nid, info in self._nodes.items()
            }
            return {
                "nodes": nodes_detail,
                "total_nodes": len(self._nodes),
                "tier_summary": {
                    tier.value: [
                        nid
                        for nid, info in self._nodes.items()
                        if info.tier == tier
                    ]
                    for tier in GpuTier
                },
                "base_batch_size": self._base_batch_size,
            }
