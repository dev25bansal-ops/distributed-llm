"""Budget computation for batch scheduling iterations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from distllm.core.scheduler.budget import IterationBudget
from distllm.core.scheduler.kv_cache_manager import KVCacheManager
from distllm.core.scheduler.pressure import DecodePressureTracker

if TYPE_CHECKING:
    from distllm.core.scheduler.sequence import Sequence

__all__ = ["BudgetComputer"]


class BudgetComputer:
    """Computes and adjusts iteration budgets with policy chain logic.

    Encapsulates the multi-layered budget computation previously spread
    across ``BatchScheduler``: dynamic GPU memory scaling, Sarathi-Serve
    adaptive pressure, and the heterogeneous/WAN/energy policy chain.

    Args:
        kv_cache_mgr: KV cache manager for block-level budget queries.
        pressure_tracker: Decode pressure tracker for Sarathi-Serve adaptation.
        adapt_prefill_budget: Whether to apply Sarathi-Serve pressure adaptation.
    """

    def __init__(
        self,
        kv_cache_mgr: KVCacheManager,
        pressure_tracker: DecodePressureTracker,
        adapt_prefill_budget: bool = True,
    ) -> None:
        self._kv_cache_mgr = kv_cache_mgr
        self._pressure_tracker = pressure_tracker
        self._adapt_prefill_budget = adapt_prefill_budget

    # -- Sarathi-Serve adaptive budget -------------------------------------

    def compute_sarathi_budget(
        self,
        budget: IterationBudget,
        active: dict[str, Sequence],
        lock: object,
        wan_policy: object | None = None,
    ) -> IterationBudget:
        """Sarathi-Serve style adaptive budget: reserve decode slots first.

        When decode pressure is high (decode latency above target), prefill
        tokens are throttled and more slots are reserved for running decodes.
        When the decode pipeline is idle, more budget is allocated to prefill.

        Returns a modified deep copy of the budget.

        Note: Skipped when WAN mode is active -- WAN latency variance
        dominates the pressure signal, causing oscillation.

        Args:
            budget: The base iteration budget to adapt.
            active: The active sequences dict (for counting decode demand).
            lock: Threading lock protecting ``active``.
            wan_policy: Optional WAN scheduling policy (checked for skip).
        """
        # WAN mode: skip pressure adaptation (RTT jitter dominates)
        if wan_policy is not None and wan_policy.should_disable_pressure_adaptation():
            return budget

        pressure = self._pressure_tracker.pressure
        with lock:
            active_snapshot = list(active.values())
        active_decode_count = sum(
            1 for s in active_snapshot
            if s.status.value == "decoding"
        )
        pending_decode_count = sum(
            1 for s in active_snapshot
            if s.status.value == "prefilling" and len(s.generated_tokens) > 0
        )
        total_decode_demand = active_decode_count + pending_decode_count
        # Sequences already mid-flight (decoding or prefilling).  Budget
        # relaxation may never shrink slots below this count — doing so
        # silently drops live sequences from the iteration.
        in_flight_count = sum(
            1 for s in active_snapshot
            if s.status.value in ("decoding", "prefilling") or len(s.generated_tokens) > 0
        )

        # Guarantee decode slots: at least enough for all active decoders.
        # max_decode_tokens here represents *decode slots* (each decode = 1 token).
        base_decode_slots = min(budget.max_batch_size, budget.max_decode_tokens)
        guaranteed_decode = max(base_decode_slots, total_decode_demand)

        if pressure > 0.7:
            # Saturate decode slots up to batch_size under pressure
            adjusted_decode = min(
                budget.max_batch_size,
                int(guaranteed_decode * (1.0 + pressure)),
            )
        elif pressure < 0.3:
            # Relax decode slots when idle — but never below the in-flight
            # count, or active sequences get dropped from the iteration.
            adjusted_decode = max(in_flight_count, 1, int(base_decode_slots * 0.6))
        else:
            adjusted_decode = base_decode_slots

        adjusted_decode = min(adjusted_decode, budget.max_batch_size)

        # Compute remaining budget for prefill after decode reservation.
        # Each decode consumes position_offsets tracking (~1 token of budget).
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
            adjusted_batch = max(
                adjusted_decode, int(budget.max_batch_size * 0.5)
            )

        return IterationBudget(
            max_prefill_tokens=adjusted_prefill,
            max_decode_tokens=adjusted_decode,
            max_batch_size=adjusted_batch,
            max_total_tokens=budget.max_total_tokens,
            enable_chunked_prefill=budget.enable_chunked_prefill,
            prefill_slack_ratio=budget.prefill_slack_ratio,
        )

    # -- Dynamic budget scaling --------------------------------------------

    def compute_dynamic_budget(
        self,
        base_budget: int,
        paged_attention_mgr: object | None,
    ) -> int:
        """Auto-scale token budget from 32K-128K based on available GPU memory.

        If PagedAttention is available, uses pool utilization to adjust down
        under memory pressure. Otherwise keeps the base budget.

        Args:
            base_budget: The raw token budget from configuration.
            paged_attention_mgr: Optional PagedAttention manager for pool info.

        Returns:
            Adjusted token budget capped at 128K.
        """
        budget = min(base_budget, 131072)
        if paged_attention_mgr is not None:
            # Try to get precise block-level info from the pool
            pool = getattr(paged_attention_mgr, 'pool', None)
            if pool is not None:
                free_blocks = getattr(pool, 'free_count', 0)
                block_size = getattr(pool, 'block_size', 16)
                # Only cap when both are real numbers — an incomplete/partial
                # pool object (or None fields) must not collapse the budget
                # to a degenerate value that stalls all scheduling.
                if isinstance(free_blocks, (int, float)) and isinstance(
                    block_size, (int, float)
                ):
                    # Each free block holds block_size tokens; leave 10%
                    # headroom so we don't exhaust the pool between
                    # budget recalculations.
                    block_token_budget = int(free_blocks * block_size * 0.9)
                    budget = min(budget, block_token_budget)
            else:
                pool_util = getattr(
                    paged_attention_mgr, 'pool_utilization', 0.0
                )
                if pool_util > 0.85:
                    budget = int(budget * 0.75)
                elif pool_util > 0.70:
                    budget = int(budget * 0.9)
        return budget

    def adjust_budget(
        self,
        base_tokens_per_batch: int,
        paged_attention_mgr: object | None,
    ) -> int:
        """Recompute the dynamic token budget based on current memory state.

        Call this between batches to adapt to changing memory conditions.

        Args:
            base_tokens_per_batch: The configured base token budget.
            paged_attention_mgr: Optional PagedAttention manager for pool info.

        Returns:
            New max_tokens_per_batch value.
        """
        return self.compute_dynamic_budget(base_tokens_per_batch, paged_attention_mgr)

    # -- Full policy chain -------------------------------------------------

    def get_iteration_budget(
        self,
        base_budget: IterationBudget,
        enable_chunked_prefill: bool,
        max_tokens_per_batch: int,
        max_batch_size: int,
        het_budget: object | None = None,
        wan_policy: object | None = None,
        energy_scheduler: object | None = None,
    ) -> IterationBudget:
        """Get current iteration budget, incorporating all active scheduling policies.

        Priority order:
        1. Heterogeneous device-aware budget (if nodes registered)
        2. WAN-adjusted budget (if WAN mode active)
        3. Energy-adjusted budget (if energy monitor active)
        4. Base budget

        Args:
            base_budget: The configured base iteration budget.
            enable_chunked_prefill: Whether chunked prefill is enabled.
            max_tokens_per_batch: Current max tokens per batch.
            max_batch_size: Current max batch size.
            het_budget: Optional heterogeneous budget computer.
            wan_policy: Optional WAN scheduling policy.
            energy_scheduler: Optional energy-aware scheduler.

        Returns:
            Fully resolved iteration budget with all policies applied.
        """
        if base_budget.enable_chunked_prefill and enable_chunked_prefill:
            base = base_budget
        else:
            base = IterationBudget(
                max_prefill_tokens=max_tokens_per_batch,
                max_decode_tokens=max_tokens_per_batch,
                max_batch_size=max_batch_size,
                max_total_tokens=max_tokens_per_batch,
                enable_chunked_prefill=False,
            )

        # 1. Heterogeneous budget: scale based on cluster device capabilities
        if het_budget is not None:
            base = het_budget.compute_budget(
                base_prefill_tokens=base.max_prefill_tokens,
                base_decode_tokens=base.max_decode_tokens,
                base_batch_size=base.max_batch_size,
                base_total_tokens=base.max_total_tokens,
            )

        # 2. WAN adjustment: scale chunks and batches for high-latency links
        if wan_policy is not None and wan_policy.is_wan_active:
            adj_prefill, adj_batch, adj_total = wan_policy.adjust_budget_for_wan(
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
        if energy_scheduler is not None:
            adj_batch, adj_prefill = energy_scheduler.adjust_for_energy(
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

    def apply_budget_policy(
        self,
        budget: IterationBudget,
        scheduling_policy: object | None,
        active: dict[str, Sequence],
        lock: object,
        wan_policy: object | None = None,
    ) -> IterationBudget:
        """Apply budget policy: pluggable > Sarathi-Serve > passthrough.

        Args:
            budget: The base iteration budget.
            scheduling_policy: Optional pluggable scheduling policy.
            active: The active sequences dict.
            lock: Threading lock protecting ``active``.
            wan_policy: Optional WAN scheduling policy.

        Returns:
            The resolved iteration budget.
        """
        if scheduling_policy is not None:
            return scheduling_policy.compute_budget(budget)
        if self._adapt_prefill_budget:
            return self.compute_sarathi_budget(
                budget, active, lock, wan_policy
            )
        return budget
