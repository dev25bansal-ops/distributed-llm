"""Preemption management for batch scheduling."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.scheduler.kv_cache_manager import KVCacheManager
    from distllm.core.scheduler.sequence import Sequence

__all__ = ["PreemptionManager"]


class PreemptionManager:
    """Manages sequence preemption, restoration, and preemption policy.

    Extracts preemption lifecycle from ``BatchScheduler``: preempting the
    lowest-priority active sequence, restoring preempted sequences with
    their KV cache state, and policy-driven preemption decisions.

    Args:
        kv_cache_mgr: KV cache manager for block swap and state save/restore.
        max_preempted: Maximum number of concurrently preempted sequences.
    """

    def __init__(
        self,
        kv_cache_mgr: KVCacheManager,
        max_preempted: int = 4,
    ) -> None:
        self._kv_cache_mgr = kv_cache_mgr
        self._preempted: dict[str, Sequence] = {}
        self._preempted_kv_state: dict[str, dict] = {}
        self._max_preempted: int = max_preempted
        # Preemption policy (from dist/preemption.py) -- set externally
        self._preemption_policy: object | None = None

    # -- Policy binding ----------------------------------------------------

    def set_preemption_policy(self, policy: object | None) -> None:
        """Connect a PreemptionPolicy for SLA-aware preemption decisions.

        When set, ``preempt_if_needed()`` uses the policy's GPU memory
        monitor, SLA tracker, and queue depth checks to decide whether
        to preempt a sequence before scheduling.

        Args:
            policy: A PreemptionPolicy instance, or None to disable
                    policy-driven preemption.
        """
        self._preemption_policy = policy
        if policy is not None:
            logger.info("Preemption policy connected")

    # -- Configuration -----------------------------------------------------

    def set_max_preempted(self, max_preempted: int) -> None:
        """Set the maximum number of concurrently preempted sequences."""
        self._max_preempted = max(0, max_preempted)

    def get_preempted_count(self) -> int:
        """Return the number of currently preempted sequences."""
        return len(self._preempted)

    @property
    def preempted(self) -> dict[str, Sequence]:
        """Direct access to the preempted sequences dict (read-only intent)."""
        return self._preempted

    # -- Policy-driven preemption ------------------------------------------

    def preempt_if_needed(
        self,
        pending_count: int,
        preempt_fn: object | None = None,
    ) -> Sequence | None:
        """Check preemption policy and preempt if conditions are met.

        Called before scheduling to free resources when:
        - GPU memory is above threshold
        - A request has exceeded its SLA violation limit
        - Pending queue depth exceeds the configured maximum

        Args:
            pending_count: Current number of pending requests.
            preempt_fn: Callable ``(min_priority) -> Sequence | None`` that
                performs the actual preemption.  Typically bound to
                ``self.preempt_lowest``.

        Returns:
            The preempted Sequence, or None if no preemption was needed.
        """
        if self._preemption_policy is None:
            return None

        if self._preemption_policy.should_preempt(
            pending_count=pending_count,
            min_priority=3,
        ):
            preempted = None
            if preempt_fn is not None:
                preempted = preempt_fn(min_priority=2)
            if preempted is not None:
                logger.info(
                    f"Policy preempted {preempted.request_id} "
                    f"(priority={preempted.priority})"
                )
            return preempted

        return None

    # -- Core preemption ---------------------------------------------------

    def preempt_lowest(
        self,
        active: dict[str, Sequence],
        total_tokens_ref: list[int],
        pending_heap: list,
        counter_ref: list[int],
        paged_attention_mgr: object | None,
        min_priority: int = 3,
        kv_cache_state: dict | None = None,
    ) -> Sequence | None:
        """Preempt the active sequence with the lowest importance and re-queue it.

        Selects the active sequence with the *highest* numeric priority value
        (i.e. the least important: 3=low > 2=normal > 1=high > 0=critical)
        whose priority is >= ``min_priority``, then moves it to the pending
        queue and saves its KV cache state for later restore.

        Args:
            active: The active sequences dict (modified in-place).
            total_tokens_ref: Single-element list holding the total token count
                (modified in-place for immutability workaround).
            pending_heap: The pending priority heap (modified in-place).
            counter_ref: Single-element list holding the FIFO counter
                (modified in-place).
            paged_attention_mgr: Optional PagedAttention manager for swap-out.
            min_priority: Minimum numeric priority to consider for preemption.
            kv_cache_state: Optional external KV cache dict for state preservation.

        Returns:
            The preempted Sequence, or None if no eligible candidate exists.
        """
        from distllm.core.scheduler.sequence import SequenceStatus

        if len(self._preempted) >= self._max_preempted:
            return None

        victim_seq = None
        victim_pri = -1
        for _rid, seq in active.items():
            if seq.priority >= min_priority and (
                victim_seq is None or seq.priority > victim_pri
            ):
                victim_seq = seq
                victim_pri = seq.priority

        if victim_seq is None:
            return None

        req_id = victim_seq.request_id
        self._kv_cache_mgr.save_kv_state(
            req_id, self._preempted_kv_state, kv_cache_state
        )

        if paged_attention_mgr is not None:
            try:
                paged_attention_mgr.swap_out_sequence(req_id)
            except Exception as e:
                logger.debug("PagedAttention swap failed: {}", e)

        del active[req_id]
        total_tokens_ref[0] -= victim_seq.total_len
        victim_seq.status = SequenceStatus.PENDING
        counter_ref[0] += 1
        heapq.heappush(
            pending_heap, (victim_seq.priority, counter_ref[0], victim_seq)
        )
        self._preempted[req_id] = victim_seq
        return victim_seq

    # -- Restoration -------------------------------------------------------

    def restore_preempted(
        self,
        active: dict[str, Sequence],
        total_tokens_ref: list[int],
        pending_heap: list,
        paged_attention_mgr: object | None,
        kv_cache_state: dict | None = None,
    ) -> list[Sequence]:
        """Restore all preempted sequences back to active with KV state.

        Removes restored sequences from the pending heap, restores their
        KV cache state, and moves them back to the active set as DECODING.

        Args:
            active: The active sequences dict (modified in-place).
            total_tokens_ref: Single-element list holding the total token count.
            pending_heap: The pending priority heap (modified in-place).
            paged_attention_mgr: Optional PagedAttention manager for block restore.
            kv_cache_state: External dict to write restored KV data into.

        Returns:
            List of restored Sequences.
        """
        from distllm.core.scheduler.sequence import SequenceStatus

        if not self._preempted:
            return []

        restored = []
        remove_ids: set[str] = set()

        for req_id, seq in list(self._preempted.items()):
            self._kv_cache_mgr.restore_kv_state(
                req_id, self._preempted_kv_state, kv_cache_state
            )
            remove_ids.add(req_id)

            if paged_attention_mgr is not None:
                try:
                    paged_attention_mgr.pool.restore_block(
                        paged_attention_mgr.get_physical_blocks(req_id)[0]
                    )
                except Exception as e:
                    logger.debug("PagedAttention restore failed: {}", e)

            seq.status = SequenceStatus.DECODING
            active[req_id] = seq
            total_tokens_ref[0] += seq.total_len
            restored.append(seq)

        # Single-pass heap rebuild: filter out all restored request IDs
        # in one O(n) pass, then heapify once O(n).
        if remove_ids:
            pending_heap[:] = [
                item for item in pending_heap
                if item[2].request_id not in remove_ids
            ]
            heapq.heapify(pending_heap)

        self._preempted.clear()
        return restored
