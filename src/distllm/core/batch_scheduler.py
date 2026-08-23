"""Continuous batch scheduler for pipeline-parallel inference."""

import heapq
import math
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import torch

from distllm.core.request_latency import RequestLatencyTracker
from distllm.utils.scheduling import group_by_length

# Import extracted classes from the scheduler package
from distllm.core.scheduler.sequence import (
    SequenceStatus,
    GenerationConfig,
    OpenAICompliance,
    SchedulingHints,
    Sequence,
    ScheduledBatch,
)
from distllm.core.scheduler.pressure import DecodePressureTracker
from distllm.core.scheduler.budget import IterationBudget
from distllm.core.scheduler.chunked_prefill import ChunkedPrefillInfo
from distllm.core.scheduler.kv_cache_manager import KVCacheManager
from distllm.core.scheduler.preemption_manager import PreemptionManager
from distllm.core.scheduler.budget_computer import BudgetComputer

if TYPE_CHECKING:
    from distllm.core.adaptive_batching import AdaptiveBatchingEngine
    from distllm.core.advanced_scheduling import (
        CostAwarePriorityAdjuster,
        EnergyAwareScheduler,
        HeterogeneousBudgetComputer,
        NodeCapabilityInfo,
        WANSchedulingPolicy,
    )

__all__ = [
    "BatchScheduler",
    "Sequence",
    "ScheduledBatch",
]


class BatchScheduler:
    """Continuous batch scheduler for distributed inference.

    Supports iteration-level scheduling with chunked prefill, mixed
    prefill/decode batches, and time-budget-aware scheduling.

    Scheduling policy:
    1. All active (non-complete) sequences stay in the batch
    2. Fill remaining capacity with pending sequences
    3. Respect max_batch_size and max_tokens_per_batch limits
    4. Chunk large prefill loads across iterations to avoid decode starvation

    Supports dynamic token budgets from 32K to 128K tokens per batch,
    auto-scaled based on GPU memory pressure. Integrates with PagedAttention
    for efficient KV cache block allocation.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 32768,
        model_info: dict | None = None,
        paged_attention_mgr: object | None = None,
        enable_chunked_prefill: bool = True,
        max_prefill_tokens: int = 4096,
        prefill_slack_ratio: float = 0.3,
        priority_weights: dict[int, float] | None = None,
        aging_interval_s: float = 30.0,
        aging_max_boost: int = 2,
        aging_enabled: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self._base_batch_size = max_batch_size
        self._base_tokens_per_batch = max_tokens_per_batch
        self._paged_attention_mgr = paged_attention_mgr

        # Extracted sub-managers
        self._kv_cache_mgr = KVCacheManager(paged_attention_mgr)
        self._pressure_tracker = DecodePressureTracker()
        self._budget_computer = BudgetComputer(
            kv_cache_mgr=self._kv_cache_mgr,
            pressure_tracker=self._pressure_tracker,
            adapt_prefill_budget=True,
        )

        self.max_tokens_per_batch = self._budget_computer.compute_dynamic_budget(
            max_tokens_per_batch, self._paged_attention_mgr,
        )
        self._pending_heap: list = []  # Min-heap of (priority, counter, Sequence)
        self._overflow_buffer: list = []  # Overflow from split_overflow
        self._counter: int = 0  # Tiebreaker for FIFO within same priority
        self._max_pending: int = 1000  # Max pending queue size (backpressure)
        self.active: dict[str, Sequence] = {}
        self._total_tokens: int = 0  # Incremental token count for O(1) tracking
        self._model_info = model_info
        self._use_length_grouping = model_info is not None

        # Iteration-level scheduling state
        self._enable_chunked_prefill = enable_chunked_prefill
        self._chunked_prefill: dict[str, ChunkedPrefillInfo] = {}
        self._budget = IterationBudget(
            max_prefill_tokens=max_prefill_tokens,
            max_total_tokens=self.max_tokens_per_batch,
            max_batch_size=max_batch_size,
            prefill_slack_ratio=prefill_slack_ratio,
        )
        self._iteration_count: int = 0
        self._total_prefill_tokens: int = 0
        self._total_decode_tokens: int = 0
        self._avg_decode_time: float = 0.0
        self._avg_prefill_time: float = 0.0

        # Latency-aware scheduling
        self._latency_tracker = RequestLatencyTracker()

        # Starvation prevention: configurable aging parameters
        self._aging_enabled = aging_enabled
        self._aging_interval_s: float = aging_interval_s
        self._aging_max_boost: int = aging_max_boost

        # Starvation watchdog
        self._starvation_threshold_s: float = 120.0
        self._last_starvation_warn: set[str] = set()

        # Priority weights for within-batch token allocation (configurable)
        self._priority_weights: dict[int, float] = priority_weights or {
            0: 1.5,   # critical -- 50% bonus tokens/iteration
            1: 1.25,  # high    -- 25% bonus
            2: 1.0,   # normal  -- baseline
            3: 0.5,   # low     -- half the tokens (slower prefill)
        }

        # Sarathi-Serve style adaptive pressure tracking
        # (shared instance with _budget_computer, created above)
        self._adapt_prefill_budget = True

        # Preemption manager (extracted from inline state)
        self._preemption_mgr = PreemptionManager(
            kv_cache_mgr=self._kv_cache_mgr,
            max_preempted=4,
        )

        # Adaptive batching engine (set externally by coordinator)
        self._adaptive_engine: AdaptiveBatchingEngine | None = None

        self._lock = threading.Lock()

        # Cache manager for radix tree prefix storage (set by coordinator)
        self._cache_mgr = None

        # -- Advanced scheduling integrations (lazy-initialized) --

        # 1. Heterogeneous P2P scheduling
        self._het_budget: HeterogeneousBudgetComputer | None = None

        # 2. Cost-aware scheduling
        self._cost_adjuster: CostAwarePriorityAdjuster | None = None

        # 3. WAN-optimized scheduling
        self._wan_policy: WANSchedulingPolicy | None = None

        # 4. Energy-aware scheduling
        self._energy_scheduler: EnergyAwareScheduler | None = None

        # 5. Pluggable scheduling policy (overrides Sarathi-Serve when set)
        self._scheduling_policy: SchedulingPolicy | None = None

        # 6. Preemption policy (delegated to PreemptionManager)

    def set_model_info(self, model_info) -> None:
        """Attach model metadata (dict or object) used for metric labels.

        Dicts are normalized to an attribute namespace because metric code
        reads ``self._model_info.model_name``.
        """
        if isinstance(model_info, dict):
            model_info = SimpleNamespace(**model_info)
        self._model_info = model_info
        self._use_length_grouping = model_info is not None

    def set_cache_manager(self, cache_mgr) -> None:
        """Set the cache manager for radix tree prefix storage."""
        self._cache_mgr = cache_mgr

    # -- Advanced scheduling configuration ---------------------------------

    def set_node_capabilities(self, nodes: dict[str, "NodeCapabilityInfo"]) -> None:
        """Register cluster node capabilities for heterogeneous scheduling.

        Enables device-aware budget computation: chunk sizes, batch sizes,
        and token budgets are automatically adjusted based on the slowest
        node in the pipeline, memory capacity, and cross-device penalties.

        Also auto-detects WAN mode if any node has high measured latency.

        Args:
            nodes: dict mapping node_id -> NodeCapabilityInfo with GPU specs,
                   memory, compute, cost, and measured latency.
        """
        from distllm.core.advanced_scheduling import HeterogeneousBudgetComputer
        if self._het_budget is None:
            self._het_budget = HeterogeneousBudgetComputer()
        self._het_budget.set_nodes(nodes)

        # Auto-detect WAN mode from node latencies
        if self._wan_policy is not None:
            self._wan_policy.detect_wan_mode(nodes)

        logger.info(f"Heterogeneous scheduling: {len(nodes)} nodes registered")

    def set_cost_awareness(
        self,
        node_costs: dict[str, float] | None = None,
        max_cost_per_request: float = 0.0,
        prefer_cheap_for_low_priority: bool = True,
    ) -> None:
        """Enable cost-aware scheduling for BYO-GPU clusters.

        When enabled, the scheduler:
        - Prefers cheap nodes for low-priority requests
        - Rejects requests that exceed cost limits
        - Tracks total cost across all requests
        - Reports cost in stats()

        Args:
            node_costs: dict mapping node_id -> cost_per_hour (USD).
            max_cost_per_request: Maximum allowed cost per request (0 = unlimited).
            prefer_cheap_for_low_priority: Route low-priority work to cheap nodes.
        """
        from distllm.core.advanced_scheduling import CostAwarePriorityAdjuster
        self._cost_adjuster = CostAwarePriorityAdjuster(
            cost_per_hour_by_node=node_costs,
            max_cost_per_request=max_cost_per_request,
            prefer_cheap_for_low_priority=prefer_cheap_for_low_priority,
        )
        logger.info(f"Cost-aware scheduling enabled: {len(node_costs or {})} nodes priced")

    def set_wan_mode(
        self,
        enabled: bool = True,
        chunk_multiplier: float = 2.0,
        batch_multiplier: float = 1.5,
        rtt_threshold_ms: float = 10.0,
        prefetch_kv: bool = True,
    ) -> None:
        """Enable WAN-optimized scheduling for high-latency links.

        When WAN mode is active:
        - Prefill chunks are multiplied (amortize RTT)
        - Batch sizes are larger (amortize per-request overhead)
        - Sarathi-Serve pressure adaptation is disabled (WAN jitter dominates)
        - KV cache prefetch during pipeline stalls

        WAN mode is auto-detected when any node has measured_latency_ms > rtt_threshold_ms.

        Args:
            enabled: Enable WAN mode.
            chunk_multiplier: Multiply prefill chunk size by this factor.
            batch_multiplier: Multiply max_batch_size by this factor.
            rtt_threshold_ms: RTT above which WAN mode activates.
            prefetch_kv: Prefetch KV cache during pipeline stalls.
        """
        from distllm.core.advanced_scheduling import WANConfig, WANSchedulingPolicy
        self._wan_policy = WANSchedulingPolicy(WANConfig(
            enabled=enabled,
            chunk_multiplier=chunk_multiplier,
            batch_multiplier=batch_multiplier,
            rtt_threshold_ms=rtt_threshold_ms,
            prefetch_kv=prefetch_kv,
        ))
        logger.info(
            f"WAN scheduling mode {'enabled' if enabled else 'disabled'}: "
            f"chunk*{chunk_multiplier}, batch*{batch_multiplier}, "
            f"RTT threshold={rtt_threshold_ms}ms"
        )

    def set_energy_monitor(
        self,
        max_power_watts: float = 0.0,
        energy_cost_per_kwh: float = 0.10,
    ) -> None:
        """Enable energy-aware scheduling.

        When enabled, the scheduler:
        - Monitors GPU power draw via NVML
        - Reduces batch size when power exceeds budget
        - Increases batch size when power is well under budget
        - Tracks energy cost per request for billing

        Args:
            max_power_watts: Maximum total power budget (0 = unlimited).
            energy_cost_per_kwh: Electricity cost ($/kWh) for billing.
        """
        from distllm.core.advanced_scheduling import EnergyAwareScheduler
        self._energy_scheduler = EnergyAwareScheduler(
            max_power_watts=max_power_watts,
            energy_cost_per_kwh=energy_cost_per_kwh,
        )
        logger.info(
            f"Energy-aware scheduling enabled: "
            f"budget={max_power_watts}W, cost=${energy_cost_per_kwh}/kWh"
        )

    def set_scheduling_policy(self, policy: "SchedulingPolicy | None") -> None:
        """Set a pluggable scheduling policy.

        When set, the policy's ``compute_budget`` method is called instead
        of the built-in Sarathi-Serve pressure adaptation.  The policy can
        also modify sequence priorities via ``on_before_schedule``.

        Args:
            policy: A SchedulingPolicy implementation, or None to restore
                    the default Sarathi-Serve behavior.
        """
        from distllm.core.advanced_scheduling import SchedulingPolicy
        if policy is not None and not isinstance(policy, SchedulingPolicy):
            logger.warning(
                f"set_scheduling_policy: {type(policy).__name__} does not "
                f"implement SchedulingPolicy protocol"
            )
        self._scheduling_policy = policy
        if policy is not None:
            logger.info(f"Scheduling policy set to {type(policy).__name__}")

    def set_preemption_policy(self, policy: "PreemptionPolicy | None") -> None:
        """Connect a PreemptionPolicy for SLA-aware preemption decisions.

        When set, ``preempt_if_needed()`` uses the policy's GPU memory
        monitor, SLA tracker, and queue depth checks to decide whether
        to preempt a sequence before scheduling.

        Args:
            policy: A PreemptionPolicy instance from dist/preemption.py,
                    or None to disable policy-driven preemption.
        """
        self._preemption_mgr.set_preemption_policy(policy)

    def preempt_if_needed(self) -> Sequence | None:
        """Check preemption policy and preempt if conditions are met.

        Called before scheduling to free resources when:
        - GPU memory is above threshold
        - A request has exceeded its SLA violation limit
        - Pending queue depth exceeds the configured maximum

        Returns:
            The preempted Sequence, or None if no preemption was needed.
        """
        return self._preemption_mgr.preempt_if_needed(
            pending_count=self.pending_count,
            preempt_fn=self.preempt_lowest,
        )

    # -- Runtime update methods (called by coordinator during inference) ---

    def update_node_power(self, node_id: str, watts: float) -> None:
        """Update power draw for a node (from NVML monitoring).

        Called by the coordinator during health checks or metrics collection.
        Only effective if energy monitor is enabled.
        """
        if self._energy_scheduler is not None:
            self._energy_scheduler.update_power_draw(node_id, watts)

    def update_node_latency(self, node_id: str, latency_ms: float) -> None:
        """Update measured latency for a node.

        Called by the coordinator when RTT measurements are available.
        Feeds into WAN auto-detection and heterogeneous budget computation.
        """
        if self._het_budget is not None and node_id in self._het_budget._nodes:
            self._het_budget._nodes[node_id].measured_latency_ms = latency_ms
            # Re-detect WAN mode with updated latency
            if self._wan_policy is not None:
                self._wan_policy.detect_wan_mode(self._het_budget._nodes)

    def set_iteration_budget(self, budget: IterationBudget) -> None:
        """Override the default iteration budget.

        Also updates ``max_tokens_per_batch`` and ``max_batch_size``
        to stay in sync with the new budget.
        """
        self._budget = budget
        self.max_tokens_per_batch = budget.max_total_tokens
        self.max_batch_size = budget.max_batch_size

    def get_iteration_budget(self) -> IterationBudget:
        """Get current iteration budget, incorporating all active scheduling policies.

        Priority order:
        1. Heterogeneous device-aware budget (if nodes registered)
        2. WAN-adjusted budget (if WAN mode active)
        3. Energy-adjusted budget (if energy monitor active)
        4. Base budget
        """
        return self._budget_computer.get_iteration_budget(
            base_budget=self._budget,
            enable_chunked_prefill=self._enable_chunked_prefill,
            max_tokens_per_batch=self.max_tokens_per_batch,
            max_batch_size=self.max_batch_size,
            het_budget=self._het_budget,
            wan_policy=self._wan_policy,
            energy_scheduler=self._energy_scheduler,
        )

    def _check_starvation(self) -> None:
        """Check pending heap for requests that have been waiting too long.

        Samples the top of the heap (highest-priority items) to avoid O(n)
        copy on every schedule() call.  Logs a warning for any request
        exceeding the starvation threshold.
        """
        now = time.time()
        current_starved: set[str] = set()
        # Sample at most 20 items from the top of the heap (highest priority).
        # Starvation is most likely to affect low-priority items at the bottom,
        # but we can't efficiently inspect those without a full scan.  Instead,
        # we check the items most likely to be scheduled next -- if even those
        # are old, something is very wrong.
        sample_count = min(20, len(self._pending_heap))
        for i in range(sample_count):
            _pri, _cnt, seq = self._pending_heap[i]
            elapsed = now - seq.created_at
            if elapsed > self._starvation_threshold_s:
                current_starved.add(seq.request_id)
                if seq.request_id not in self._last_starvation_warn:
                    logger.warning(
                        f"Request {seq.request_id} pending for {elapsed:.0f}s "
                        f"(priority={seq.priority}) -- possible starvation"
                    )
        self._last_starvation_warn = current_starved

    def _aging_boost(self, seq: Sequence) -> int:
        """Calculate priority boost from aging (starvation prevention).

        The longer a request waits in the pending heap, the more its
        effective priority is boosted, ensuring low-priority requests
        are eventually served even under continuous high-priority load.
        """
        if not self._aging_enabled:
            return 0
        elapsed = time.time() - seq.created_at
        boost = int(elapsed / self._aging_interval_s)
        return min(boost, self._aging_max_boost)

    def _priority_weight(self, priority: int) -> float:
        """Return token allocation weight for a given priority level.

        Higher priority sequences get a larger share of the prefill
        token budget within a batch.
        """
        return self._priority_weights.get(priority, 1.0)

    def set_adaptive_engine(self, engine: 'AdaptiveBatchingEngine') -> None:
        """Connect adaptive batching engine for dynamic batch size tuning."""
        self._adaptive_engine = engine
        # Pull initial batch size from engine
        model = getattr(self._model_info, 'model_name', None) or "default"
        engine_size = engine.get_batch_size(model)
        if engine_size != self.max_batch_size:
            self.max_batch_size = engine_size

    def update_batch_size_from_adaptive(self) -> None:
        """Query adaptive engine and update max_batch_size for next iteration."""
        if self._adaptive_engine is None:
            return
        model = getattr(self._model_info, 'model_name', None) or "default"
        recommended = self._adaptive_engine.get_batch_size(model)
        if recommended != self.max_batch_size and recommended >= 1:
            with self._lock:
                self.max_batch_size = recommended
                self._budget.max_batch_size = recommended

    def adjust_from_gpu_utilization(self, gpu_utilization: float, queue_depth: int = 0) -> None:
        """Dynamically adjust batch size based on GPU utilization feedback.

        Args:
            gpu_utilization: GPU utilization fraction (0.0 to 1.0).
            queue_depth: Number of pending requests in the queue.

        Behavior:
            - GPU util < 70%: increase batch size (up to 2x base)
            - GPU util > 90%: decrease batch size (down to 0.5x base)
            - Queue depth > 2 * base_batch_size: increase prefill budget
        """
        base = self._base_batch_size

        if gpu_utilization < 0.7:
            # GPU is underutilized -- increase batch size
            scale = min(2.0, 1.0 + (0.7 - gpu_utilization) * 3.0)
            new_size = min(int(base * scale), base * 2)
        elif gpu_utilization > 0.9:
            # GPU is overloaded -- decrease batch size
            scale = max(0.5, 1.0 - (gpu_utilization - 0.9) * 5.0)
            new_size = max(int(base * scale), max(1, base // 2))
        else:
            new_size = base

        # Queue pressure: increase prefill budget if queue is deep
        if queue_depth > base * 2:
            self._budget.max_prefill_tokens = min(
                self._budget.max_prefill_tokens * 2,
                self._base_tokens_per_batch // 4,
            )

        if new_size != self.max_batch_size:
            with self._lock:
                self.max_batch_size = new_size
                self._budget.max_batch_size = new_size

    def update_priority_weights(self, weights: dict[int, float]) -> None:
        """Update priority weights at runtime."""
        with self._lock:
            self._priority_weights.update(weights)

    def update_aging_params(self, interval_s: float | None = None, max_boost: int | None = None, enabled: bool | None = None) -> None:
        """Update aging parameters at runtime."""
        with self._lock:
            if interval_s is not None:
                self._aging_interval_s = interval_s
            if max_boost is not None:
                self._aging_max_boost = max_boost
            if enabled is not None:
                self._aging_enabled = enabled

    def set_paged_attention(self, mgr: object) -> None:
        """Connect to PagedAttention manager for KV block-aware scheduling."""
        self._paged_attention_mgr = mgr
        self._kv_cache_mgr.set_paged_attention(mgr)

    def allocate_paged_blocks(self, seq: Sequence) -> list[int] | None:
        """Allocate PagedAttention blocks for a sequence.

        Called when a sequence enters the active batch. If PagedAttention
        is not configured, returns None (caller uses flat KV cache).

        Returns:
            List of allocated block IDs, or None if PagedAttention not active.
        """
        return self._kv_cache_mgr.allocate_paged_blocks(seq)

    def free_paged_blocks(self, request_id: str) -> None:
        """Free PagedAttention blocks for a completed sequence."""
        self._kv_cache_mgr.free_paged_blocks(request_id)

    def swap_evict_to_cpu(self, min_blocks: int = 1) -> int:
        """Evict lowest-priority active sequences to CPU to free GPU blocks.

        Selects sequences with the highest numeric priority (least important:
        3=low > 2=normal > 1=high > 0=critical), breaking ties by oldest first.

        Args:
            min_blocks: Minimum number of blocks to free.

        Returns:
            Number of blocks freed.
        """
        return self._kv_cache_mgr.swap_evict_to_cpu(
            self.active, self._lock, min_blocks,
        )

    def restore_from_cpu(self, request_id: str) -> int:
        """Restore a sequence's blocks from CPU back to GPU."""
        return self._kv_cache_mgr.restore_from_cpu(request_id)

    def copy_on_write(self, source_id: str, dest_id: str) -> None:
        """Copy-on-write for shared prefixes (beam search, speculative decoding)."""
        self._kv_cache_mgr.copy_on_write(source_id, dest_id)

    def _compute_sarathi_budget(self, budget: IterationBudget) -> IterationBudget:
        """Sarathi-Serve style adaptive budget: reserve decode slots first.

        When decode pressure is high (decode latency above target), prefill
        tokens are throttled and more slots are reserved for running decodes.
        When the decode pipeline is idle, more budget is allocated to prefill.

        Returns a modified deep copy of the budget.

        Note: Skipped when WAN mode is active -- WAN latency variance
        dominates the pressure signal, causing oscillation.
        """
        return self._budget_computer.compute_sarathi_budget(
            budget, self.active, self._lock, self._wan_policy,
        )

    def _compute_dynamic_budget(self, base_budget: int) -> int:
        """Auto-scale token budget from 32K-128K based on available GPU memory.

        If PagedAttention is available, uses pool utilization to adjust down
        under memory pressure. Otherwise keeps the base budget.
        """
        return self._budget_computer.compute_dynamic_budget(
            base_budget, self._paged_attention_mgr,
        )

    def adjust_budget(self) -> None:
        """Recompute the dynamic token budget based on current memory state.

        Call this between batches to adapt to changing memory conditions.
        """
        with self._lock:
            self.max_tokens_per_batch = self._budget_computer.compute_dynamic_budget(
                self._base_tokens_per_batch, self._paged_attention_mgr,
            )

    def paged_kv_block_count(self, tokens: int) -> int:
        """Estimate number of PagedAttention blocks needed for this many tokens."""
        return self._kv_cache_mgr.paged_kv_block_count(tokens)

    def add(self, seq: Sequence) -> None:
        """Add a new request to the pending queue (priority-ordered).

        Raises:
            BatchCapacityError: If the pending queue is full (backpressure).
        """
        with self._lock:
            if len(self._pending_heap) >= self._max_pending:
                from distllm.errors.types import BatchCapacityError
                raise BatchCapacityError(
                    current_tokens=len(self._pending_heap),
                    max_tokens=self._max_pending,
                )
            heapq.heappush(self._pending_heap, (seq.priority, self._counter, seq))
            self._counter += 1
            self._pending_index = None  # Invalidate index cache
        self._latency_tracker.register(seq.request_id, sla_ms=seq.max_latency_ms)

    def schedule(self) -> ScheduledBatch | None:
        """Build the next batch from active + pending sequences.

        Increments the iteration counter and applies all active scheduling
        policies (heterogeneous, WAN, energy, Sarathi-Serve pressure).

        Returns None if there are no sequences to process.
        """
        self._iteration_count += 1
        return self._schedule_with_budget(self.get_iteration_budget())

    def schedule_iteration(self, iteration_budget: IterationBudget | None = None) -> ScheduledBatch | None:
        """Schedule with a caller-specified iteration budget.

        Same as ``schedule()`` but allows overriding the budget for this
        iteration.  Increments the iteration counter.

        Args:
            iteration_budget: Budget for this iteration. Uses default if None.

        Returns:
            ScheduledBatch or None if no work.
        """
        self._iteration_count += 1
        return self._schedule_with_budget(iteration_budget or self.get_iteration_budget())

    def _schedule_with_budget(self, budget: IterationBudget) -> ScheduledBatch | None:
        """Build batch respecting iteration-level budget.

        Steps: prefetch → evict + snapshot → build active batch →
        promote pending → group by length → build tensors → return.
        """
        self.update_batch_size_from_adaptive()
        budget = self._budget_computer.apply_budget_policy(
            budget, self._scheduling_policy, self.active, self._lock, self._wan_policy,
        )
        self._check_starvation()
        self.preempt_if_needed()

        # Step 1: Prefetch KV blocks, evict completed, snapshot active
        with self._lock:
            active_ids = list(self.active.keys())
        active_items = self._prefetch_and_snapshot(active_ids)

        # Step 2: Build batch from active decodes + chunked prefill
        batch_seqs, remain_p, remain_t, decode_added = self._build_active_batch(
            budget, active_items)

        # Step 3: Promote pending sequences into the batch
        if batch_seqs is not None:
            result = self._promote_pending(budget, batch_seqs, remain_p, remain_t, decode_added)
            if result is None:
                return None
            batch_seqs, remain_p, remain_t = result

        if not batch_seqs:
            return None

        # Step 4: Length-aware grouping
        batch_seqs = self._apply_length_grouping(batch_seqs)

        # Step 5: Build tensors and return
        return self._build_scheduled_batch(batch_seqs, budget)

    # ── Extracted helper methods ──────────────────────────────────────

    def _prefetch_and_snapshot(self, active_ids: list[str]) -> list[tuple]:
        """Prefetch KV blocks, evict completed sequences, and snapshot active set.

        Single lock acquisition for eviction + snapshot to minimize time
        holding self._lock, which blocks concurrent add() calls.
        """
        with self._lock:
            done_ids = [rid for rid, s in self.active.items() if s.is_complete]
            for rid in done_ids:
                seq = self.active.pop(rid)
                self._total_tokens -= seq.total_len
                self._chunked_prefill.pop(rid, None)
                self._latency_tracker.complete(rid)
                self.free_paged_blocks(rid)
            active_items = list(self.active.items())

        if self._paged_attention_mgr is not None:
            try:
                prefetcher = getattr(self._paged_attention_mgr, '_prefetch_scheduler', None)
                if prefetcher is not None:
                    prefetcher.prefetch_for_stage(active_ids, stage_idx=0)
            except Exception:
                pass
        return active_items

    def _build_active_batch(
        self, budget: IterationBudget, active_items: list[tuple],
    ) -> tuple[list, int, int, int]:
        """Build batch from active decodes + chunked prefill sequences.

        Returns: (batch_seqs, remaining_prefill, remaining_total, decode_added)
        """
        batch_seqs: list[Sequence] = []
        remain_p = budget.max_prefill_tokens
        remain_t = budget.max_total_tokens
        decode_added = 0

        urgency = self._latency_tracker.get_requests_sorted_by_deadline()
        urgent_ids = {rid for rid, _ in urgency}
        active_items.sort(key=lambda item: (0 if item[0] in urgent_ids else 1, item[0]))

        for rid, seq in active_items:
            if decode_added >= budget.decode_slots:
                break
            if seq.status in (SequenceStatus.DECODING, SequenceStatus.PREFILLING) or len(seq.generated_tokens) > 0:
                if self._check_decode_budget(budget, decode_added, remain_t):
                    batch_seqs.append(seq)
                    decode_added += 1
                    remain_t -= 1

        if self._enable_chunked_prefill:
            for _rid, seq in active_items:
                if seq.request_id in self._chunked_prefill:
                    cinfo = self._chunked_prefill[seq.request_id]
                    if cinfo.is_complete:
                        continue
                    chunk = min(cinfo.remaining, budget.max_prefill_tokens)
                    chunk = min(chunk, remain_p)
                    if seq not in batch_seqs:
                        batch_seqs.append(seq)
                    remain_p -= chunk
                    remain_t -= chunk
                    if remain_p <= 0:
                        break

        return batch_seqs, remain_p, remain_t, decode_added

    def _promote_pending(
        self, budget: IterationBudget,
        batch_seqs: list[Sequence],
        remain_p: int, remain_t: int, decode_added: int,
    ) -> tuple[list, int, int] | None:
        """Promote pending heap sequences into the batch under lock."""
        with self._lock:
            remain_slots = budget.max_batch_size - len(batch_seqs)
            max_examine = max(remain_slots * 2, 10)
            batch_avg_remaining = 0
            if self._use_length_grouping and batch_seqs:
                br = [s.max_new_tokens - len(s.generated_tokens) for s in batch_seqs]
                batch_avg_remaining = sum(br) / len(br) if br else 0
            rejected: list[tuple] = []

            while self._pending_heap and len(batch_seqs) + len(rejected) < max_examine:
                pri, cnt, candidate = heapq.heappop(self._pending_heap)
                if candidate.request_id in self.active:
                    continue
                effective_pri = self._latency_tracker.get_latency_boost(candidate.request_id, pri)
                aging = self._aging_boost(candidate)
                if aging > 0:
                    effective_pri = max(0, effective_pri - aging)
                if self._use_length_grouping and batch_avg_remaining > 0:
                    ld = abs((candidate.max_new_tokens - len(candidate.generated_tokens)) - batch_avg_remaining)
                    effective_pri += min((ld / (batch_avg_remaining + 1)) * 0.1, 0.5)
                if self._cost_adjuster is not None:
                    effective_pri, _ = self._cost_adjuster.adjust_priority(
                        base_priority=effective_pri, estimated_tokens=candidate.total_len)

                c_tokens = candidate.total_len
                if remain_slots <= 0 or (remain_p <= 0 and remain_t <= 0) or c_tokens > remain_t:
                    rejected.append((pri, cnt, candidate))
                    continue

                chunk = c_tokens
                if self._enable_chunked_prefill and c_tokens > budget.max_prefill_tokens > 0:
                    chunk = budget.max_prefill_tokens

                if remain_t - chunk < decode_added * budget.prefill_slack_ratio:
                    if c_tokens > budget.max_prefill_tokens:
                        chunk = min(chunk, int(remain_t * (1 - budget.prefill_slack_ratio)))
                    else:
                        rejected.append((pri, cnt, candidate))
                        continue

                if chunk > remain_p and remain_t - chunk < 0:
                    rejected.append((pri, cnt, candidate))
                    continue

                blocks_ok = True
                if self._paged_attention_mgr is not None:
                    needed = self.paged_kv_block_count(self._total_tokens + chunk)
                    pool = getattr(self._paged_attention_mgr, 'pool', None)
                    if pool is not None:
                        total = getattr(pool, 'total_blocks', needed + 1)
                        blocks_ok = needed <= total * 0.9
                if not blocks_ok:
                    rejected.append((pri, cnt, candidate))
                    continue

                candidate.status = SequenceStatus.PREFILLING
                batch_seqs.append(candidate)
                self.active[candidate.request_id] = candidate
                self._total_tokens += c_tokens
                remain_p -= chunk
                remain_t -= chunk
                remain_slots -= 1
                self.allocate_paged_blocks(candidate)
                if self._enable_chunked_prefill and c_tokens > budget.max_prefill_tokens > 0:
                    self._chunked_prefill[candidate.request_id] = ChunkedPrefillInfo(
                        seq_id=candidate.request_id, total_prompt_tokens=c_tokens,
                        chunk_size=budget.max_prefill_tokens,
                        chunks_remaining=math.ceil(c_tokens / budget.max_prefill_tokens))

            for pri, cnt, candidate in rejected:
                heapq.heappush(self._pending_heap, (pri, cnt, candidate))
            self._pending_index = None

        if not batch_seqs:
            return None
        return batch_seqs, remain_p, remain_t

    def _apply_length_grouping(self, batch_seqs: list[Sequence]) -> list[Sequence]:
        """Group sequences by length for efficient ragged attention."""
        if self._use_length_grouping and len(batch_seqs) > 1:
            bucketed = group_by_length(batch_seqs, num_buckets=min(4, len(batch_seqs)))
            result = []
            for bucket_idx in sorted(bucketed.keys()):
                bucket = bucketed[bucket_idx]
                if bucket:
                    bucket.sort(key=lambda s: s.total_len)
                    result.extend(bucket)
            return result
        return batch_seqs

    def _build_scheduled_batch(self, batch_seqs: list[Sequence], budget: IterationBudget) -> ScheduledBatch:
        """Build final ScheduledBatch from sequences."""
        request_ids: list[str] = []
        seq_lengths: list[int] = []
        seq_starts: list[int] = []
        position_offsets: list[int] = []
        is_prefill_list: list[bool] = []
        flat_tokens: list[int] = []

        iter_prefill_tokens, iter_decode_tokens = self._build_batch_tensors(
            batch_seqs, budget, request_ids, seq_starts, seq_lengths,
            position_offsets, is_prefill_list, flat_tokens,
        )

        import torch
        input_ids = torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0)
        batch_tags = self._build_batch_tags(batch_seqs, iter_prefill_tokens, iter_decode_tokens)

        return ScheduledBatch(
            sequences=batch_seqs, input_ids=input_ids,
            seq_starts=seq_starts, seq_lengths=seq_lengths,
            position_offsets=position_offsets, is_prefill=is_prefill_list,
            request_ids=request_ids, batch_tags=batch_tags,
            adapter_ids=[seq.adapter_id for seq in batch_seqs],
        )

    def _build_batch_tensors(
        self,
        batch_seqs: list[Sequence],
        budget: IterationBudget,
        request_ids: list[str],
        seq_starts: list[int],
        seq_lengths: list[int],
        position_offsets: list[int],
        is_prefill_list: list[bool],
        flat_tokens: list[int],
    ) -> tuple[int, int]:
        """Build flat token layout for the batch.

        Populates the output lists in-place and returns
        (iter_prefill_tokens, iter_decode_tokens) counts.
        """
        iter_prefill_tokens = 0
        iter_decode_tokens = 0

        for seq in batch_seqs:
            request_ids.append(seq.request_id)
            seq_starts.append(len(flat_tokens))

            tokens, is_prefill, pos_offset = self._build_seq_tokens(seq, budget)
            flat_tokens.extend(tokens)
            seq_lengths.append(len(tokens))
            position_offsets.append(pos_offset)
            is_prefill_list.append(is_prefill)

            if is_prefill:
                iter_prefill_tokens += len(tokens)
                self._total_prefill_tokens += len(tokens)
            else:
                iter_decode_tokens += 1
                self._total_decode_tokens += 1

        return iter_prefill_tokens, iter_decode_tokens

    def _build_seq_tokens(
        self,
        seq: Sequence,
        budget: IterationBudget,
    ) -> tuple[list[int], bool, int]:
        """Build tokens for a single sequence in the batch.

        Returns:
            (tokens, is_prefill, position_offset) tuple.
        """
        is_chunked = seq.request_id in self._chunked_prefill
        if is_chunked:
            cinfo = self._chunked_prefill[seq.request_id]
            start = seq.prefix_match_len + cinfo.tokens_processed
            weight = self._priority_weight(seq.priority)
            max_chunk = max(1, int(budget.max_prefill_tokens * weight))
            chunk_end = min(start + max_chunk, len(seq.prompt_tokens))
            tokens = seq.prompt_tokens[start:chunk_end]
            pos_offset = seq.prefix_match_len + cinfo.tokens_processed
            cinfo.tokens_processed += len(tokens)
            cinfo.chunks_remaining = max(0, cinfo.chunks_remaining - 1)
            if cinfo.is_complete:
                self._chunked_prefill.pop(seq.request_id, None)
                seq.status = SequenceStatus.DECODING
            else:
                seq.status = SequenceStatus.PREFILLING
            return tokens, True, pos_offset

        if len(seq.generated_tokens) == 0 and seq.prefix_match_len == 0:
            tokens = seq.prompt_tokens[seq.prefix_match_len:]
            seq.status = SequenceStatus.PREFILLING if not seq.is_complete else seq.status
            return tokens, True, seq.prefix_match_len

        # Decode step: single token
        seq.status = SequenceStatus.DECODING
        return [seq.decode_input_token], False, seq.total_len - 1

    def _build_batch_tags(
        self,
        batch_seqs: list[Sequence],
        iter_prefill_tokens: int,
        iter_decode_tokens: int,
    ) -> dict[str, object]:
        """Build batch metadata tags for this iteration."""
        batch_tags: dict[str, object] = {
            "iteration": self._iteration_count,
            "chunked_prefill": len(self._chunked_prefill),
            "prefill_tokens": iter_prefill_tokens,
            "decode_tokens": iter_decode_tokens,
            "total_prefill_tokens": self._total_prefill_tokens,
            "total_decode_tokens": self._total_decode_tokens,
        }
        if self._use_length_grouping:
            lengths = [s.total_len for s in batch_seqs]
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            batch_tags["avg_seq_len"] = avg_len
            batch_tags["length_variance"] = sum((seq_len - avg_len) ** 2 for seq_len in lengths) / len(lengths) if lengths else 0

        remaining = [s.max_new_tokens - len(s.generated_tokens) for s in batch_seqs]
        batch_tags["avg_tokens_remaining"] = sum(remaining) / len(remaining) if remaining else 0
        return batch_tags

    def _check_decode_budget(self, budget: IterationBudget, decode_count: int, remaining_total: int) -> bool:
        """Check if another decode fits in the budget."""
        if decode_count >= budget.decode_slots:
            return False
        if remaining_total < 1:
            return False
        return True

    def _record_step_metrics(self, batch: ScheduledBatch, decode_count: int = 0, decode_elapsed_ms: float = 0.0) -> None:
        """Record decode metrics and feed data to adaptive engine.

        Extracted so subclasses (e.g. IterationScheduler) can call it
        without duplicating token-processing logic.
        """
        # Record decode latency for Sarathi-Serve adaptive pressure tracking
        if decode_count > 0 and decode_elapsed_ms > 0:
            self._pressure_tracker.record_decode_step(decode_count, decode_elapsed_ms)

        # Feed per-sequence latencies to adaptive batching engine
        if self._adaptive_engine is not None:
            model = getattr(self._model_info, 'model_name', None) or "default"
            now = time.time()
            seq_latencies = []
            for seq in batch.sequences:
                if seq.status != SequenceStatus.PENDING:
                    lat = (now - seq.created_at) * 1000
                    seq_latencies.append(lat)
            if seq_latencies:
                self._adaptive_engine.record_batch(
                    model=model,
                    batch_size=len(batch.sequences),
                    latencies=seq_latencies,
                )

    def step(self, batch: ScheduledBatch, next_tokens: "torch.Tensor", kv_caches: dict | None = None, decoded_tokens: list[str] | None = None) -> None:
        """Process sampling output, update sequences, check for completion.

        Args:
            batch: The batch that was just processed.
            next_tokens: [batch_size] tensor of sampled token IDs.
            kv_caches: Optional dict mapping request_id -> KV cache data for radix tree storage.
            decoded_tokens: Optional list of decoded token strings (for constraint updates).
        """
        decode_count = sum(1 for s in batch.sequences if s.status == SequenceStatus.DECODING)
        decode_start = time.monotonic()
        for i, seq in enumerate(batch.sequences):
            token = next_tokens[i].item()
            seq.generated_tokens.append(int(token))
            self._latency_tracker.record_token(seq.request_id)

            # Transition from PREFILLING to DECODING after first generated token
            if seq.status == SequenceStatus.PREFILLING:
                seq.status = SequenceStatus.DECODING
                self._latency_tracker.record_first_token(seq.request_id)

            # Check constraint (structured output)
            if seq.constraint is not None:
                if decoded_tokens is not None and i < len(decoded_tokens):
                    seq.constraint.update(decoded_tokens[i])
                else:
                    seq.constraint.update(str(token))

            if seq.is_complete or token in seq.stop_token_ids:
                seq.status = SequenceStatus.DONE
                self._latency_tracker.complete(seq.request_id)

                # Store completed sequence in radix tree for prefix reuse
                if self._cache_mgr is not None and self._cache_mgr.prefix_cache is not None:
                    all_tokens = seq.prompt_tokens + seq.generated_tokens
                    if len(all_tokens) >= self._cache_mgr.prefix_cache.min_prefix_len:
                        kv_data = None
                        if kv_caches and seq.request_id in kv_caches:
                            kv_data = kv_caches[seq.request_id]
                        if kv_data is not None:
                            self._cache_mgr.store_prefix(all_tokens, kv_data)

        self._record_step_metrics(
            batch,
            decode_count=decode_count,
            decode_elapsed_ms=(time.monotonic() - decode_start) * 1000 if decode_count > 0 else 0.0,
        )

        # Prune completed sequences from active set (under lock -- shared with get_sequence)
        with self._lock:
            done_rids = [s.request_id for s in batch.sequences if s.is_complete]
            for rid in done_rids:
                self.active.pop(rid, None)
                self._chunked_prefill.pop(rid, None)
                # Free the PagedAttention KV blocks for completed sequences
                # (F-041: previously the free path only scanned still-active
                # sequences, so blocks leaked per completed request until the
                # pool was exhausted).
                if self._kv_cache_mgr is not None:
                    try:
                        self._kv_cache_mgr.free_paged_blocks(rid)
                    except Exception as exc:  # noqa: BLE001 - best-effort free
                        logger.debug(f"Failed to free paged blocks for {rid}: {exc}")

        # Record energy usage for this iteration
        if self._energy_scheduler is not None:
            self._energy_scheduler.record_energy_usage(
                duration_seconds=(time.monotonic() - decode_start) if decode_count > 0 else 0.0,
            )

    def get_sequence(self, request_id: str) -> Sequence | None:
        """Get a sequence by request_id (from pending or active)."""
        with self._lock:
            if request_id in self.active:
                return self.active[request_id]
            for _pri, _cnt, seq in self._pending_heap:
                if seq.request_id == request_id:
                    return seq
            return None

    @property
    def has_pending(self) -> bool:
        return len(self._pending_heap) > 0 or any(
            not s.is_complete for s in self.active.values()
        )

    @property
    def active_count(self) -> int:
        return len(self.active)

    @property
    def pending_count(self) -> int:
        return len(self._pending_heap)

    def promote_request(self, request_id: str, new_priority: int) -> bool:
        """Change the priority of a pending request.

        Uses O(log n) indexed heap update instead of O(n) linear scan + heapify.

        Args:
            request_id: The request to promote.
            new_priority: The new priority level.

        Returns:
            True if the request was found and updated.
        """
        with self._lock:
            # Build index if not cached
            if not hasattr(self, '_pending_index') or self._pending_index is None:
                self._pending_index = {
                    seq.request_id: i
                    for i, (_, _, seq) in enumerate(self._pending_heap)
                }

            idx = self._pending_index.get(request_id)
            if idx is None:
                return False

            _pri, _cnt, seq = self._pending_heap[idx]
            seq.priority = new_priority
            self._pending_heap[idx] = (new_priority, _cnt, seq)

            # Bubble up or down -- O(log n) instead of O(n) heapify.
            # For a min-heap, _siftdown moves toward root (smaller index)
            # when priority is lowered (higher importance); _siftup moves
            # toward leaves when priority is raised (lower importance).
            if new_priority < _pri:
                heapq._siftdown(self._pending_heap, 0, idx)
            else:
                heapq._siftup(self._pending_heap, idx)

            # Invalidate the index cache — after sifting, the element's
            # heap position has changed and stored idx is stale.
            self._pending_index = None
            return True

    def preempt_lowest(self, min_priority: int = 3, kv_cache_state: dict | None = None) -> Sequence | None:
        """Preempt the active sequence with the lowest importance and re-queue it.

        Selects the active sequence with the *highest* numeric priority value
        (i.e. the least important: 3=low > 2=normal > 1=high > 0=critical)
        whose priority is >= min_priority, then moves it to the pending queue
        and saves its KV cache state for later restore.

        Args:
            min_priority: Minimum numeric priority to consider for preemption.
                Only sequences with priority >= this value are eligible.
                Default 3 means only "low" priority (3) sequences are preempted.
            kv_cache_state: Optional external KV cache dict for state preservation.

        Returns:
            The preempted Sequence, or None if no eligible candidate exists.
        """
        with self._lock:
            total_ref = [self._total_tokens]
            counter_ref = [self._counter]
            result = self._preemption_mgr.preempt_lowest(
                active=self.active,
                total_tokens_ref=total_ref,
                pending_heap=self._pending_heap,
                counter_ref=counter_ref,
                paged_attention_mgr=self._paged_attention_mgr,
                min_priority=min_priority,
                kv_cache_state=kv_cache_state,
            )
            # Sync mutable refs back to instance state
            self._total_tokens = total_ref[0]
            self._counter = counter_ref[0]
            return result

    def _save_kv_state(self, request_id: str, kv_cache_state: dict | None = None) -> None:
        """Save KV cache state for a preempted sequence.

        Stores the raw KV cache data (any type) for the given request_id
        so it can be restored later via _restore_kv_state().

        Args:
            request_id: The request whose KV state to save.
            kv_cache_state: External dict mapping request_id -> KV cache data.
                If the dict contains request_id, its value is saved.
        """
        self._kv_cache_mgr.save_kv_state(
            request_id, self._preemption_mgr._preempted_kv_state, kv_cache_state,
        )

    def _compress_kv_for_preemption(self, kv_data, method: str = "int4"):
        """Compress KV cache data before storing for preemption.

        Applies int4 quantization (8x reduction) to KV tensors if they
        are torch.Tensor objects.  Non-tensor data is stored as-is.

        Args:
            kv_data: KV cache data (tensor, list of tensors, dict, or any).
            method: Compression method -- "int4" (8x), "int8" (4x), or "none".

        Returns:
            Compressed data with metadata for decompression.
        """
        return self._kv_cache_mgr._compress_kv_for_preemption(kv_data, method)

    @staticmethod
    def _compress_tensor(tensor: "torch.Tensor", method: str) -> dict:
        """Compress a single tensor with int4 or int8 quantization.

        Returns a dict with the quantized tensor and scale factors,
        which can be restored with _decompress_tensor().
        """
        return KVCacheManager._compress_tensor(tensor, method)

    @staticmethod
    def _decompress_tensor(compressed: dict) -> "torch.Tensor":
        """Restore a tensor compressed by _compress_tensor()."""
        return KVCacheManager._decompress_tensor(compressed)

    def decompress_preempted_kv(self, kv_data) -> object:
        """Recursively decompress KV data that was compressed for preemption.

        Args:
            kv_data: Compressed KV data (may contain nested dicts with _compressed flag).

        Returns:
            Decompressed data ready for use.
        """
        return self._kv_cache_mgr.decompress_preempted_kv(kv_data)

    def _restore_kv_state(self, request_id: str, kv_cache_state: dict | None = None) -> bool:
        """Restore KV cache state for a preempted sequence.

        Decompresses the KV data if it was compressed during preemption,
        then writes it back into the external kv_cache_state dict.

        Args:
            request_id: The request whose KV state to restore.
            kv_cache_state: External dict to write the restored KV data into.

        Returns:
            True if KV state was found and restored, False otherwise.
        """
        return self._kv_cache_mgr.restore_kv_state(
            request_id, self._preemption_mgr._preempted_kv_state, kv_cache_state,
        )

    def restore_preempted(self, kv_cache_state: dict | None = None) -> list[Sequence]:
        """Restore all preempted sequences back to active with KV state.

        Removes restored sequences from the pending heap, restores their
        KV cache state, and moves them back to the active set as DECODING.

        Thread safety: The entire operation runs under ``self._lock`` to
        serialize with ``add()``, ``_schedule_with_budget()``, and
        ``get_sequence()`` which also acquire the same lock.  The heap
        rebuild (list comprehension + ``heapify``) is safe because no
        other thread can observe the intermediate state while the lock
        is held.
        """
        with self._lock:
            total_ref = [self._total_tokens]
            restored = self._preemption_mgr.restore_preempted(
                active=self.active,
                total_tokens_ref=total_ref,
                pending_heap=self._pending_heap,
                paged_attention_mgr=self._paged_attention_mgr,
                kv_cache_state=kv_cache_state,
            )
            # Sync mutable ref back to instance state
            self._total_tokens = total_ref[0]
            return restored

    def set_max_preempted(self, max_preempted: int) -> None:
        """Set the maximum number of concurrently preempted sequences."""
        self._preemption_mgr.set_max_preempted(max_preempted)

    def get_preempted_count(self) -> int:
        return self._preemption_mgr.get_preempted_count()

    def stats(self) -> dict:
        stats = {
            "active_requests": self.active_count,
            "pending_requests": self.pending_count,
            "preempted_requests": self._preemption_mgr.get_preempted_count(),
            "max_batch_size": self.max_batch_size,
            "max_tokens_per_batch": self.max_tokens_per_batch,
            "paged_attention": self._paged_attention_mgr is not None,
            "iteration": self._iteration_count,
            "total_prefill_tokens": self._total_prefill_tokens,
            "total_decode_tokens": self._total_decode_tokens,
            "chunked_prefill_active": len(self._chunked_prefill),
            "chunked_prefill_enabled": self._enable_chunked_prefill,
            "adaptive_batching": self._adaptive_engine is not None,
        }
        if self._adaptive_engine is not None:
            model = getattr(self._model_info, 'model_name', None) or "default"
            try:
                astats = self._adaptive_engine.get_stats(model)
                stats["adaptive_avg_latency_ms"] = astats.avg_latency_ms
                stats["adaptive_batch_size"] = self._adaptive_engine.get_current_batch_size(model)
            except Exception as e:
                logger.debug("Adaptive engine get_stats failed: {}", e)

        # Advanced scheduling stats
        if self._het_budget is not None:
            stats["heterogeneous"] = self._het_budget.stats()
        if self._cost_adjuster is not None:
            stats["cost_aware"] = self._cost_adjuster.stats()
        if self._wan_policy is not None:
            stats["wan"] = self._wan_policy.stats()
        if self._energy_scheduler is not None:
            stats["energy"] = self._energy_scheduler.stats()

        return stats

    @property
    def latency_tracker(self) -> RequestLatencyTracker:
        """Return the latency tracker instance."""
        return self._latency_tracker
