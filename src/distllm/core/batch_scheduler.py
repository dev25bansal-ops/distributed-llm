"""Continuous batch scheduler for pipeline-parallel inference."""

import heapq
import math
import threading
import time
from dataclasses import dataclass, field
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
        self.max_tokens_per_batch = self._compute_dynamic_budget(max_tokens_per_batch)
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
            0: 1.5,   # critical — 50% bonus tokens/iteration
            1: 1.25,  # high    — 25% bonus
            2: 1.0,   # normal  — baseline
            3: 0.5,   # low     — half the tokens (slower prefill)
        }

        # Sarathi-Serve style adaptive pressure tracking
        self._pressure_tracker = DecodePressureTracker()
        self._adapt_prefill_budget = True

        # Preemption state
        self._preempted: dict[str, Sequence] = {}  # request_id -> Sequence
        self._preempted_kv_state: dict[str, dict] = {}  # request_id -> KV cache state
        self._max_preempted: int = 4  # Max concurrent preempted sequences

        # Adaptive batching engine (set externally by coordinator)
        self._adaptive_engine: AdaptiveBatchingEngine | None = None

        self._lock = threading.Lock()

        # Cache manager for radix tree prefix storage (set by coordinator)
        self._cache_mgr = None

        # ── Advanced scheduling integrations (lazy-initialized) ──

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

        # 6. Preemption policy (from dist/preemption.py)
        self._preemption_policy: PreemptionPolicy | None = None

    def set_cache_manager(self, cache_mgr) -> None:
        """Set the cache manager for radix tree prefix storage."""
        self._cache_mgr = cache_mgr

    # ── Advanced scheduling configuration ──────────────────────────────────

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
            f"chunk×{chunk_multiplier}, batch×{batch_multiplier}, "
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
        self._preemption_policy = policy
        if policy is not None:
            logger.info("Preemption policy connected")

    def preempt_if_needed(self) -> Sequence | None:
        """Check preemption policy and preempt if conditions are met.

        Called before scheduling to free resources when:
        - GPU memory is above threshold
        - A request has exceeded its SLA violation limit
        - Pending queue depth exceeds the configured maximum

        Returns:
            The preempted Sequence, or None if no preemption was needed.
        """
        if self._preemption_policy is None:
            return None

        if self._preemption_policy.should_preempt(
            pending_count=self.pending_count,
            min_priority=3,
        ):
            preempted = self.preempt_lowest(min_priority=2)
            if preempted is not None:
                logger.info(
                    f"Policy preempted {preempted.request_id} "
                    f"(priority={preempted.priority})"
                )
            return preempted

        return None

    # ── Runtime update methods (called by coordinator during inference) ────

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
        base = self._budget if (self._budget.enable_chunked_prefill and self._enable_chunked_prefill) else IterationBudget(
            max_prefill_tokens=self.max_tokens_per_batch,
            max_decode_tokens=self.max_tokens_per_batch,
            max_batch_size=self.max_batch_size,
            max_total_tokens=self.max_tokens_per_batch,
            enable_chunked_prefill=False,
        )

        # 1. Heterogeneous budget: scale based on cluster device capabilities
        if self._het_budget is not None:
            base = self._het_budget.compute_budget(
                base_prefill_tokens=base.max_prefill_tokens,
                base_decode_tokens=base.max_decode_tokens,
                base_batch_size=base.max_batch_size,
                base_total_tokens=base.max_total_tokens,
            )

        # 2. WAN adjustment: scale chunks and batches for high-latency links
        if self._wan_policy is not None and self._wan_policy.is_wan_active:
            adj_prefill, adj_batch, adj_total = self._wan_policy.adjust_budget_for_wan(
                base_prefill_tokens=base.max_prefill_tokens,
                base_batch_size=base.max_batch_size,
                base_total_tokens=base.max_total_tokens,
            )
            base = IterationBudget(
                max_prefill_tokens=adj_prefill,
                max_decode_tokens=base.max_decode_tokens,
                max_batch_size=adj_batch,
                max_total_tokens=adj_total,
                enable_chunked_prefill=base.enable_chunked_prefill,
                prefill_slack_ratio=base.prefill_slack_ratio,
            )

        # 3. Energy adjustment: reduce batch size if over power budget
        if self._energy_scheduler is not None:
            adj_batch, adj_prefill = self._energy_scheduler.adjust_for_energy(
                base_batch_size=base.max_batch_size,
                base_prefill_tokens=base.max_prefill_tokens,
            )
            if adj_batch != base.max_batch_size or adj_prefill != base.max_prefill_tokens:
                base = IterationBudget(
                    max_prefill_tokens=adj_prefill,
                    max_decode_tokens=base.max_decode_tokens,
                    max_batch_size=adj_batch,
                    max_total_tokens=base.max_total_tokens,
                    enable_chunked_prefill=base.enable_chunked_prefill,
                    prefill_slack_ratio=base.prefill_slack_ratio,
                )

        return base


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
        # we check the items most likely to be scheduled next — if even those
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
                        f"(priority={seq.priority}) — possible starvation"
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
            # GPU is underutilized — increase batch size
            scale = min(2.0, 1.0 + (0.7 - gpu_utilization) * 3.0)
            new_size = min(int(base * scale), base * 2)
        elif gpu_utilization > 0.9:
            # GPU is overloaded — decrease batch size
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
        if interval_s is not None:
            self._aging_interval_s = interval_s
        if max_boost is not None:
            self._aging_max_boost = max_boost
        if enabled is not None:
            self._aging_enabled = enabled

    def set_paged_attention(self, mgr: object) -> None:
        """Connect to PagedAttention manager for KV block-aware scheduling."""
        self._paged_attention_mgr = mgr

    def allocate_paged_blocks(self, seq: Sequence) -> list[int] | None:
        """Allocate PagedAttention blocks for a sequence.

        Called when a sequence enters the active batch. If PagedAttention
        is not configured, returns None (caller uses flat KV cache).

        Returns:
            List of allocated block IDs, or None if PagedAttention not active.
        """
        if self._paged_attention_mgr is None:
            return None
        try:
            num_tokens = len(seq.prompt_tokens) + seq.max_new_tokens
            block_ids = self._paged_attention_mgr.allocate_sequence(
                seq.request_id, num_tokens,
            )
            return block_ids
        except RuntimeError as e:
            logger.warning(f"PagedAttention allocation failed for {seq.request_id}: {e}")
            return None

    def free_paged_blocks(self, request_id: str) -> None:
        """Free PagedAttention blocks for a completed sequence."""
        if self._paged_attention_mgr is not None:
            try:
                self._paged_attention_mgr.free_sequence(request_id)
            except Exception as e:
                logger.warning(f"Failed to free paged blocks for {request_id}: {e}")

    def swap_evict_to_cpu(self, min_blocks: int = 1) -> int:
        """Evict lowest-priority active sequences to CPU to free GPU blocks.

        Selects sequences with the highest numeric priority (least important:
        3=low > 2=normal > 1=high > 0=critical), breaking ties by oldest first.

        Args:
            min_blocks: Minimum number of blocks to free.

        Returns:
            Number of blocks freed.
        """
        if self._paged_attention_mgr is None:
            return 0

        # Find lowest-priority active sequences (highest numeric priority first,
        # then oldest first — evict cheap/old work before expensive/new work)
        with self._lock:
            candidates = sorted(
                self.active.values(),
                key=lambda s: (s.priority, -s.created_at),
            )

        freed = 0
        for seq in candidates:
            if freed >= min_blocks:
                break
            try:
                blocks_freed = self._paged_attention_mgr.swap_blocks_to_cpu(seq.request_id)
                freed += blocks_freed
                logger.debug(f"Swapped {blocks_freed} blocks to CPU for {seq.request_id}")
            except Exception as e:
                logger.debug(f"Failed to swap blocks to CPU for {seq.request_id}: {e}")
                continue
        return freed

    def restore_from_cpu(self, request_id: str) -> int:
        """Restore a sequence's blocks from CPU back to GPU."""
        if self._paged_attention_mgr is None:
            return 0
        try:
            return self._paged_attention_mgr.swap_blocks_to_gpu(request_id)
        except Exception as e:
            logger.warning(f"Failed to restore blocks from CPU for {request_id}: {e}")
            return 0

    def copy_on_write(self, source_id: str, dest_id: str) -> None:
        """Copy-on-write for shared prefixes (beam search, speculative decoding)."""
        if self._paged_attention_mgr is not None:
            try:
                self._paged_attention_mgr.copy_on_write(source_id, dest_id)
            except Exception as e:
                logger.warning(f"Copy-on-write failed from {source_id} to {dest_id}: {e}")

    def _compute_sarathi_budget(self, budget: IterationBudget) -> IterationBudget:
        """Sarathi-Serve style adaptive budget: reserve decode slots first.

        When decode pressure is high (decode latency above target), prefill
        tokens are throttled and more slots are reserved for running decodes.
        When the decode pipeline is idle, more budget is allocated to prefill.

        Returns a modified deep copy of the budget.

        Note: Skipped when WAN mode is active — WAN latency variance
        dominates the pressure signal, causing oscillation.
        """
        # WAN mode: skip pressure adaptation (RTT jitter dominates)
        if self._wan_policy is not None and self._wan_policy.should_disable_pressure_adaptation():
            return budget

        pressure = self._pressure_tracker.pressure
        with self._lock:
            active_snapshot = list(self.active.values())
        active_decode_count = sum(
            1 for s in active_snapshot
            if s.status == SequenceStatus.DECODING
        )
        pending_decode_count = sum(
            1 for s in active_snapshot
            if s.status == SequenceStatus.PREFILLING
            and len(s.generated_tokens) > 0
        )
        total_decode_demand = active_decode_count + pending_decode_count

        # Guarantee decode slots: at least enough for all active decoders.
        # max_decode_tokens here represents *decode slots* (each decode = 1 token).
        base_decode_slots = min(budget.max_batch_size, budget.max_decode_tokens)
        guaranteed_decode = max(base_decode_slots, total_decode_demand)

        if pressure > 0.7:
            # Saturate decode slots up to batch_size under pressure
            adjusted_decode = min(budget.max_batch_size, int(guaranteed_decode * (1.0 + pressure)))
        elif pressure < 0.3:
            # Relax decode slots when idle
            adjusted_decode = max(1, int(base_decode_slots * 0.6))
        else:
            adjusted_decode = base_decode_slots

        adjusted_decode = min(adjusted_decode, budget.max_batch_size)

        # Compute remaining budget for prefill after decode reservation.
        # Each decode consumes position_offsets tracking (≈1 token of budget).
        total_after_decode = max(0, budget.max_total_tokens - adjusted_decode)

        # Adjust prefill budget: throttle under high pressure.
        if pressure > 0.8:
            prefill_scale = max(0.25, 1.0 - pressure)
        elif pressure > 0.5:
            prefill_scale = 0.75
        else:
            prefill_scale = 1.0

        adjusted_prefill = min(
            int(budget.max_prefill_tokens * prefill_scale),
            total_after_decode,
        )

        # Under severe pressure, limit batch size for stability
        adjusted_batch = budget.max_batch_size
        if pressure > 0.9:
            adjusted_batch = max(adjusted_decode, int(budget.max_batch_size * 0.5))

        return IterationBudget(
            max_prefill_tokens=adjusted_prefill,
            max_decode_tokens=adjusted_decode,
            max_batch_size=adjusted_batch,
            max_total_tokens=budget.max_total_tokens,
            enable_chunked_prefill=budget.enable_chunked_prefill,
            prefill_slack_ratio=budget.prefill_slack_ratio,
        )

    def _compute_dynamic_budget(self, base_budget: int) -> int:
        """Auto-scale token budget from 32K-128K based on available GPU memory.

        If PagedAttention is available, uses pool utilization to adjust down
        under memory pressure. Otherwise keeps the base budget.
        """
        budget = min(base_budget, 131072)
        if self._paged_attention_mgr is not None:
            # Try to get precise block-level info from the pool
            pool = getattr(self._paged_attention_mgr, 'pool', None)
            if pool is not None:
                free_blocks = getattr(pool, 'free_count', 0)
                block_size = getattr(pool, 'block_size', 16)
                # Convert free blocks to a token budget: each free block
                # can hold block_size tokens.  Leave 10% headroom so we
                # don't exhaust the pool between budget recalculations.
                block_token_budget = int(free_blocks * block_size * 0.9)
                budget = min(budget, block_token_budget)
            else:
                pool_util = getattr(self._paged_attention_mgr, 'pool_utilization', 0.0)
                if pool_util > 0.85:
                    budget = int(budget * 0.75)
                elif pool_util > 0.70:
                    budget = int(budget * 0.9)
        return budget

    def adjust_budget(self) -> None:
        """Recompute the dynamic token budget based on current memory state.

        Call this between batches to adapt to changing memory conditions.
        """
        with self._lock:
            self.max_tokens_per_batch = self._compute_dynamic_budget(self._base_tokens_per_batch)

    def paged_kv_block_count(self, tokens: int) -> int:
        """Estimate number of PagedAttention blocks needed for this many tokens."""
        if self._paged_attention_mgr is not None:
            block_size = getattr(self._paged_attention_mgr, 'block_size', 16)
        else:
            block_size = 16
        return (tokens + block_size - 1) // block_size

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

        Budget computation priority:
        1. Pluggable SchedulingPolicy (if set via set_scheduling_policy)
        2. Sarathi-Serve adaptive pressure (default when _adapt_prefill_budget=True)
        3. Passthrough (base budget unchanged)

        Then: chunked prefill, pending sequence promotion, batch construction.
        """
        # Update batch size from adaptive engine before scheduling
        self.update_batch_size_from_adaptive()

        # Apply budget policy: pluggable > Sarathi-Serve > passthrough
        if self._scheduling_policy is not None:
            budget = self._scheduling_policy.compute_budget(budget)
        elif self._adapt_prefill_budget:
            budget = self._compute_sarathi_budget(budget)

        # Starvation check: warn if any request has been pending too long
        self._check_starvation()

        # Policy-driven preemption: free resources if conditions are met
        self.preempt_if_needed()

        # 1. Evict completed sequences (under lock — shared with get_sequence, preempt)
        with self._lock:
            done_ids = [rid for rid, s in self.active.items() if s.is_complete]
            for rid in done_ids:
                seq = self.active.pop(rid)
                self._total_tokens -= seq.total_len
                self._chunked_prefill.pop(rid, None)
                self._latency_tracker.complete(rid)
                self.free_paged_blocks(rid)

        # 2. Start with active non-complete sequences (all decode, some prefill)
        batch_seqs: list[Sequence] = []
        remaining_prefill_budget = budget.max_prefill_tokens
        remaining_total_budget = budget.max_total_tokens
        decode_seqs_added = 0

        # Snapshot active set under lock (shared with add, preempt, get_sequence)
        with self._lock:
            active_items = list(self.active.items())
        # 2a. Add active sequences - decode first (always prioritized)
        # Sort by latency urgency so SLO-critical sequences run first
        urgency = self._latency_tracker.get_requests_sorted_by_deadline()
        urgent_ids = {rid for rid, _ in urgency}
        active_items.sort(key=lambda item: (0 if item[0] in urgent_ids else 1, item[0]))

        seen_ids = set()
        for rid, seq in active_items:
            if decode_seqs_added >= budget.decode_slots:
                break
            if seq.status in (SequenceStatus.DECODING, SequenceStatus.PREFILLING) or len(seq.generated_tokens) > 0:
                if self._check_decode_budget(budget, decode_seqs_added, remaining_total_budget):
                    batch_seqs.append(seq)
                    seen_ids.add(rid)
                    decode_seqs_added += 1
                    remaining_total_budget -= 1

        # 2b. Add chunked prefill sequences (already partially prefilled)
        if self._enable_chunked_prefill:
            for _rid, seq in active_items:
                if seq.request_id in self._chunked_prefill:
                    cinfo = self._chunked_prefill[seq.request_id]
                    if cinfo.is_complete:
                        continue
                    chunk = min(cinfo.remaining, budget.max_prefill_tokens)
                    chunk = min(chunk, remaining_prefill_budget)

                    if seq not in batch_seqs:
                        batch_seqs.append(seq)
                    remaining_prefill_budget -= chunk
                    remaining_total_budget -= chunk
                    if remaining_prefill_budget <= 0:
                        break

        # 3. Promote pending sequences respecting budget.
        # Only pop what we need: at most (remaining_batch_slots * 2 + 10) items
        # to avoid O(n log n) on the entire heap.
        with self._lock:
            new_active_ids = set()
            remaining_batch_slots = budget.max_batch_size - len(batch_seqs)
            max_to_examine = max(remaining_batch_slots * 2, 10)

            batch_avg_remaining = 0
            if self._use_length_grouping and batch_seqs:
                batch_remaining = [s.max_new_tokens - len(s.generated_tokens) for s in batch_seqs]
                batch_avg_remaining = sum(batch_remaining) / len(batch_remaining) if batch_remaining else 0

            examined = 0
            accepted = 0
            rejected_original: list[tuple] = []

            while self._pending_heap and examined < max_to_examine:
                pri, cnt, candidate = heapq.heappop(self._pending_heap)
                examined += 1

                # Skip items already active (known race condition)
                if candidate.request_id in self.active:
                    continue

                # Apply latency-based priority boosting + aging
                effective_pri = self._latency_tracker.get_latency_boost(candidate.request_id, pri)
                aging = self._aging_boost(candidate)
                if aging > 0:
                    effective_pri = max(0, effective_pri - aging)
                if self._use_length_grouping and batch_avg_remaining > 0:
                    length_diff = abs((candidate.max_new_tokens - len(candidate.generated_tokens)) - batch_avg_remaining)
                    length_score = length_diff / (batch_avg_remaining + 1)
                    effective_pri += min(length_score * 0.1, 0.5)

                # Cost-aware priority adjustment: prefer cheap nodes for low-priority
                if self._cost_adjuster is not None:
                    est_tokens = candidate.total_len
                    effective_pri, _est_cost = self._cost_adjuster.adjust_priority(
                        base_priority=effective_pri,
                        estimated_tokens=est_tokens,
                    )

                # Check budget constraints
                if remaining_batch_slots <= 0:
                    rejected_original.append((pri, cnt, candidate))
                    continue
                if remaining_prefill_budget <= 0 and remaining_total_budget <= 0:
                    rejected_original.append((pri, cnt, candidate))
                    continue

                c_tokens = candidate.total_len
                if c_tokens > remaining_total_budget:
                    rejected_original.append((pri, cnt, candidate))
                    continue

                if self._enable_chunked_prefill and c_tokens > budget.max_prefill_tokens > 0:
                    chunk = budget.max_prefill_tokens
                else:
                    chunk = c_tokens

                rem_decode_est = decode_seqs_added * budget.prefill_slack_ratio
                if remaining_total_budget - chunk < rem_decode_est:
                    if c_tokens > budget.max_prefill_tokens:
                        chunk = min(chunk, int(remaining_total_budget * (1 - budget.prefill_slack_ratio)))
                    elif chunk > remaining_total_budget:
                        rejected_original.append((pri, cnt, candidate))
                        continue

                if chunk > remaining_prefill_budget and remaining_total_budget - chunk < 0:
                    rejected_original.append((pri, cnt, candidate))
                    continue

                if self._paged_attention_mgr is not None:
                    blocks_needed = self.paged_kv_block_count(self._total_tokens + chunk)
                    pa_pool = getattr(self._paged_attention_mgr, 'pool', None)
                    if pa_pool is not None:
                        total_blocks = getattr(pa_pool, 'total_blocks', blocks_needed + 1)
                        if blocks_needed > total_blocks * 0.9:
                            rejected_original.append((pri, cnt, candidate))
                            continue

                accepted += 1
                candidate.status = SequenceStatus.PREFILLING
                batch_seqs.append(candidate)
                self.active[candidate.request_id] = candidate
                new_active_ids.add(candidate.request_id)
                self._total_tokens += c_tokens
                remaining_prefill_budget -= chunk
                remaining_total_budget -= chunk
                remaining_batch_slots -= 1

                # Allocate PagedAttention blocks for this sequence
                self.allocate_paged_blocks(candidate)

                if self._enable_chunked_prefill and c_tokens > budget.max_prefill_tokens > 0:
                    chunk_size = budget.max_prefill_tokens
                    self._chunked_prefill[candidate.request_id] = ChunkedPrefillInfo(
                        seq_id=candidate.request_id,
                        total_prompt_tokens=c_tokens,
                        chunk_size=chunk_size,
                        chunks_remaining=math.ceil(c_tokens / chunk_size),
                    )

            # Push back rejected items with their original (pre-boost) priority
            for pri, cnt, candidate in rejected_original:
                heapq.heappush(self._pending_heap, (pri, cnt, candidate))
        if not batch_seqs:
            return None

        # 3b. Length-aware grouping: group by total length for efficient attention.
        # Uses log-scale bucketing so sequences of similar length are processed
        # together, reducing ragged attention overhead.
        if self._use_length_grouping and len(batch_seqs) > 1:
            bucketed = group_by_length(batch_seqs, num_buckets=min(4, len(batch_seqs)))
            batch_seqs = []
            for bucket_idx in sorted(bucketed.keys()):
                bucket = bucketed[bucket_idx]
                if bucket:
                    bucket.sort(key=lambda s: s.total_len)
                    batch_seqs.extend(bucket)

        # 4. Build batch tensors with priority-weighted token allocation.
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

        # 5. Build batch tags (per-iteration values, not cumulative)
        batch_tags = self._build_batch_tags(batch_seqs, iter_prefill_tokens, iter_decode_tokens)

        return ScheduledBatch(
            sequences=batch_seqs,
            input_ids=input_ids,
            seq_starts=seq_starts,
            seq_lengths=seq_lengths,
            position_offsets=position_offsets,
            is_prefill=is_prefill_list,
            request_ids=request_ids,
            batch_tags=batch_tags,
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

        # Prune completed sequences from active set (under lock — shared with get_sequence)
        with self._lock:
            done_rids = [s.request_id for s in batch.sequences if s.is_complete]
            for rid in done_rids:
                self.active.pop(rid, None)
                self._chunked_prefill.pop(rid, None)

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

            # Bubble up or down — O(log n) instead of O(n) heapify
            if new_priority < _pri:
                heapq._siftup(self._pending_heap, idx)
            else:
                heapq._siftdown(self._pending_heap, 0, idx)

            self._pending_index[request_id] = idx
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
            if len(self._preempted) >= self._max_preempted:
                return None

            victim_seq = None
            victim_pri = -1
            for _rid, seq in self.active.items():
                if seq.priority >= min_priority and (victim_seq is None or seq.priority > victim_pri):
                    victim_seq = seq
                    victim_pri = seq.priority

            if victim_seq is None:
                return None

            req_id = victim_seq.request_id
            self._save_kv_state(req_id, kv_cache_state)

            if self._paged_attention_mgr is not None:
                try:
                    self._paged_attention_mgr.swap_out_sequence(req_id)
                except Exception as e:
                    logger.debug("PagedAttention swap failed: {}", e)

            del self.active[req_id]
            self._total_tokens -= victim_seq.total_len
            victim_seq.status = SequenceStatus.PENDING
            self._counter += 1
            heapq.heappush(self._pending_heap, (victim_seq.priority, self._counter, victim_seq))
            self._preempted[req_id] = victim_seq
            return victim_seq

    def _save_kv_state(self, request_id: str, kv_cache_state: dict | None = None) -> None:
        """Save KV cache state for a preempted sequence.

        Stores the raw KV cache data (any type) for the given request_id
        so it can be restored later via _restore_kv_state().

        Args:
            request_id: The request whose KV state to save.
            kv_cache_state: External dict mapping request_id -> KV cache data.
                If the dict contains request_id, its value is saved.
        """
        if kv_cache_state is not None and request_id in kv_cache_state:
            data = kv_cache_state[request_id]
            compressed = self._compress_kv_for_preemption(data)
            self._preempted_kv_state[request_id] = compressed

    def _compress_kv_for_preemption(self, kv_data: Any, method: str = "int4") -> Any:
        """Compress KV cache data before storing for preemption.

        Applies int4 quantization (8x reduction) to KV tensors if they
        are torch.Tensor objects.  Non-tensor data is stored as-is.

        Args:
            kv_data: KV cache data (tensor, list of tensors, dict, or any).
            method: Compression method — "int4" (8x), "int8" (4x), or "none".

        Returns:
            Compressed data with metadata for decompression.
        """
        if method == "none" or kv_data is None:
            return kv_data

        import torch

        if isinstance(kv_data, torch.Tensor):
            return self._compress_tensor(kv_data, method)

        if isinstance(kv_data, dict):
            return {k: self._compress_kv_for_preemption(v, method) for k, v in kv_data.items()}

        if isinstance(kv_data, (list, tuple)):
            return [self._compress_kv_for_preemption(item, method) for item in kv_data]

        # Non-tensor data (strings, ints, etc.) — store as-is
        return kv_data

    @staticmethod
    def _compress_tensor(tensor: "torch.Tensor", method: str) -> dict:
        """Compress a single tensor with int4 or int8 quantization.

        Returns a dict with the quantized tensor and scale factors,
        which can be restored with _decompress_tensor().
        """
        import torch

        if method == "int4":
            scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
            quantized = (tensor / scale).clamp(-7, 7).to(torch.int8)
            return {"_compressed": True, "method": "int4", "data": quantized, "scale": scale}

        if method == "int8":
            scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
            quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            return {"_compressed": True, "method": "int8", "data": quantized, "scale": scale}

        return {"_compressed": False, "data": tensor}

    @staticmethod
    def _decompress_tensor(compressed: dict) -> "torch.Tensor":
        """Restore a tensor compressed by _compress_tensor()."""
        import torch

        if not compressed.get("_compressed", False):
            return compressed["data"]

        data = compressed["data"]
        scale = compressed["scale"]
        method = compressed.get("method", "int8")

        if method == "int4":
            return data.to(torch.float16) * scale

        if method == "int8":
            return data.to(torch.float16) * scale

        return data

    def decompress_preempted_kv(self, kv_data: Any) -> Any:
        """Recursively decompress KV data that was compressed for preemption.

        Args:
            kv_data: Compressed KV data (may contain nested dicts with _compressed flag).

        Returns:
            Decompressed data ready for use.
        """

        if isinstance(kv_data, dict) and kv_data.get("_compressed"):
            return self._decompress_tensor(kv_data)

        if isinstance(kv_data, dict):
            return {k: self.decompress_preempted_kv(v) for k, v in kv_data.items()}

        if isinstance(kv_data, list):
            return [self.decompress_preempted_kv(item) for item in kv_data]

        return kv_data

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
        saved = self._preempted_kv_state.pop(request_id, None)
        if saved is not None and kv_cache_state is not None:
            kv_cache_state[request_id] = self.decompress_preempted_kv(saved)
            return True
        return False

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
            if not self._preempted:
                return []

            restored = []
            remove_ids: set[str] = set()

            for req_id, seq in list(self._preempted.items()):
                self._restore_kv_state(req_id, kv_cache_state)
                remove_ids.add(req_id)

                if self._paged_attention_mgr is not None:
                    try:
                        self._paged_attention_mgr.pool.restore_block(
                            self._paged_attention_mgr.get_physical_blocks(req_id)[0]
                        )
                    except Exception as e:
                        logger.debug("PagedAttention restore failed: {}", e)

                seq.status = SequenceStatus.DECODING
                self.active[req_id] = seq
                self._total_tokens += seq.total_len
                restored.append(seq)

            # Single-pass heap rebuild: filter out all restored request IDs
            # in one O(n) pass, then heapify once O(n).  This is correct
            # because the lock prevents concurrent heap modifications.
            if remove_ids:
                self._pending_heap = [
                    item for item in self._pending_heap
                    if item[2].request_id not in remove_ids
                ]
                heapq.heapify(self._pending_heap)

            self._preempted.clear()
            return restored

    def set_max_preempted(self, max_preempted: int) -> None:
        """Set the maximum number of concurrently preempted sequences."""
        self._max_preempted = max(0, max_preempted)

    def get_preempted_count(self) -> int:
        return len(self._preempted)

    def stats(self) -> dict:
        stats = {
            "active_requests": self.active_count,
            "pending_requests": self.pending_count,
            "preempted_requests": len(self._preempted),
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
