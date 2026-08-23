"""Batch construction pipeline extracted from BatchScheduler.

This module contains the batch-building methods that were originally
defined as private methods on BatchScheduler.  They are exposed as
static methods on BatchBuilder so they can be independently tested
and reasoned about.

No logic changes from the original.  Each method receives the scheduler
instance as the first argument so it can read/write scheduler state
directly, exactly as the original ``self``-based methods did.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.batch_scheduler import BatchScheduler

from distllm.core.scheduler.budget import IterationBudget
from distllm.core.scheduler.chunked_prefill import ChunkedPrefillInfo
from distllm.core.scheduler.sequence import (
    ScheduledBatch,
    Sequence,
    SequenceStatus,
)
from distllm.utils.scheduling import group_by_length


class BatchBuilder:
    """Batch construction pipeline (static methods).

    Each method receives the scheduler instance as the first parameter
    so it can access all scheduler state exactly as the original
    ``self``-based private methods did.
    """

    # ------------------------------------------------------------------
    # Prefetch / snapshot
    # ------------------------------------------------------------------

    @staticmethod
    def prefetch_and_snapshot(
        scheduler: BatchScheduler,
        active_ids: list[str],
    ) -> list[tuple]:
        """Prefetch KV blocks, evict completed, and snapshot active set."""
        with scheduler._lock:
            done_ids = [rid for rid, s in scheduler.active.items() if s.is_complete]
            for rid in done_ids:
                seq = scheduler.active.pop(rid)
                scheduler._total_tokens -= seq.total_len
                scheduler._chunked_prefill.pop(rid, None)
                scheduler._latency_tracker.complete(rid)
                scheduler.free_paged_blocks(rid)
            active_items = list(scheduler.active.items())

        if scheduler._paged_attention_mgr is not None:
            try:
                prefetcher = getattr(
                    scheduler._paged_attention_mgr, "_prefetch_scheduler", None
                )
                if prefetcher is not None:
                    prefetcher.prefetch_for_stage(active_ids, stage_idx=0)
            except Exception:
                logger.warning("Failed to prefetch for scheduling stage")
        return active_items

    # ------------------------------------------------------------------
    # Active batch construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_active_batch(
        scheduler: BatchScheduler,
        budget: IterationBudget,
        active_items: list[tuple],
    ) -> tuple[list, int, int, int]:
        """Build batch from active decodes + chunked prefill sequences.

        Returns:
            (batch_seqs, remaining_prefill, remaining_total, decode_added)
        """
        batch_seqs: list[Sequence] = []
        remain_p = budget.max_prefill_tokens
        remain_t = budget.max_total_tokens
        decode_added = 0

        urgency = scheduler._latency_tracker.get_requests_sorted_by_deadline()
        urgent_ids = {rid for rid, _ in urgency}
        active_items.sort(
            key=lambda item: (0 if item[0] in urgent_ids else 1, item[0])
        )

        for rid, seq in active_items:
            if decode_added >= budget.decode_slots:
                break
            if (
                seq.status
                in (SequenceStatus.DECODING, SequenceStatus.PREFILLING)
                or len(seq.generated_tokens) > 0
            ):
                if BatchBuilder.check_decode_budget(
                    scheduler, budget, decode_added, remain_t
                ):
                    batch_seqs.append(seq)
                    decode_added += 1
                    remain_t -= 1

        if scheduler._enable_chunked_prefill:
            for _rid, seq in active_items:
                if seq.request_id in scheduler._chunked_prefill:
                    cinfo = scheduler._chunked_prefill[seq.request_id]
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

    # ------------------------------------------------------------------
    # Promote pending sequences
    # ------------------------------------------------------------------

    @staticmethod
    def promote_pending(
        scheduler: BatchScheduler,
        budget: IterationBudget,
        batch_seqs: list[Sequence],
        remain_p: int,
        remain_t: int,
        decode_added: int,
    ) -> tuple[list, int, int] | None:
        """Promote pending heap sequences into the batch."""
        remain_slots = budget.max_batch_size - len(batch_seqs)
        max_examine = max(remain_slots * 2, 10)
        batch_avg_remaining = 0
        if scheduler._use_length_grouping and batch_seqs:
            br = [
                s.max_new_tokens - len(s.generated_tokens) for s in batch_seqs
            ]
            batch_avg_remaining = sum(br) / len(br) if br else 0

        candidates: list[tuple[int, int, Sequence]] = []
        rejected: list[tuple] = []
        with scheduler._lock:
            while scheduler._pending_heap and len(candidates) + len(rejected) < max_examine:
                pri, cnt, candidate = heapq.heappop(scheduler._pending_heap)
                if candidate.request_id in scheduler.active:
                    continue
                candidates.append((pri, cnt, candidate))
            scheduler._pending_index = None

        for pri, cnt, candidate in candidates:
            if candidate.request_id in scheduler.active:
                continue
            effective_pri = scheduler._latency_tracker.get_latency_boost(
                candidate.request_id, pri
            )
            aging = scheduler._aging_boost(candidate)
            if aging > 0:
                effective_pri = max(0, effective_pri - aging)
            if scheduler._use_length_grouping and batch_avg_remaining > 0:
                ld = abs(
                    (candidate.max_new_tokens - len(candidate.generated_tokens))
                    - batch_avg_remaining
                )
                effective_pri += min(
                    (ld / (batch_avg_remaining + 1)) * 0.1, 0.5
                )
            if scheduler._cost_adjuster is not None:
                effective_pri, _ = scheduler._cost_adjuster.adjust_priority(
                    base_priority=effective_pri,
                    estimated_tokens=candidate.total_len,
                )

            c_tokens = candidate.total_len
            if (
                remain_slots <= 0
                or (remain_p <= 0 and remain_t <= 0)
                or c_tokens > remain_t
            ):
                rejected.append((pri, cnt, candidate))
                continue

            chunk = c_tokens
            if (
                scheduler._enable_chunked_prefill
                and c_tokens > budget.max_prefill_tokens > 0
            ):
                chunk = budget.max_prefill_tokens

            if remain_t - chunk < decode_added * budget.prefill_slack_ratio:
                if c_tokens > budget.max_prefill_tokens:
                    chunk = min(
                        chunk,
                        int(remain_t * (1 - budget.prefill_slack_ratio)),
                    )
                else:
                    rejected.append((pri, cnt, candidate))
                    continue

            if chunk > remain_p and remain_t - chunk < 0:
                rejected.append((pri, cnt, candidate))
                continue

            blocks_ok = True
            if scheduler._paged_attention_mgr is not None:
                needed = scheduler.paged_kv_block_count(
                    scheduler._total_tokens + chunk
                )
                pool = getattr(scheduler._paged_attention_mgr, "pool", None)
                if pool is not None:
                    total = getattr(pool, "total_blocks", needed + 1)
                    blocks_ok = needed <= total * 0.9
            if not blocks_ok:
                rejected.append((pri, cnt, candidate))
                continue

            with scheduler._lock:
                candidate.status = SequenceStatus.PREFILLING
                batch_seqs.append(candidate)
                scheduler.active[candidate.request_id] = candidate
                scheduler._total_tokens += c_tokens
                remain_p -= chunk
                remain_t -= chunk
                remain_slots -= 1

            scheduler.allocate_paged_blocks(candidate)
            if (
                scheduler._enable_chunked_prefill
                and c_tokens > budget.max_prefill_tokens > 0
            ):
                scheduler._chunked_prefill[candidate.request_id] = ChunkedPrefillInfo(
                    seq_id=candidate.request_id,
                    total_prompt_tokens=c_tokens,
                    chunk_size=budget.max_prefill_tokens,
                    chunks_remaining=math.ceil(
                        c_tokens / budget.max_prefill_tokens
                    ),
                )

        with scheduler._lock:
            for _pri, _cnt, _candidate in rejected:
                heapq.heappush(scheduler._pending_heap, (_pri, _cnt, _candidate))
            scheduler._pending_index = None

        if not batch_seqs:
            return None
        return batch_seqs, remain_p, remain_t

    # ------------------------------------------------------------------
    # Length grouping
    # ------------------------------------------------------------------

    @staticmethod
    def apply_length_grouping(
        scheduler: BatchScheduler,
        batch_seqs: list[Sequence],
    ) -> list[Sequence]:
        """Group sequences by length for efficient ragged attention."""
        if scheduler._use_length_grouping and len(batch_seqs) > 1:
            bucketed = group_by_length(
                batch_seqs, num_buckets=min(4, len(batch_seqs))
            )
            result = []
            for bucket_idx in sorted(bucketed.keys()):
                bucket = bucketed[bucket_idx]
                if bucket:
                    bucket.sort(key=lambda s: s.total_len)
                    result.extend(bucket)
            return result
        return batch_seqs

    # ------------------------------------------------------------------
    # Build ScheduledBatch object
    # ------------------------------------------------------------------

    @staticmethod
    def build_scheduled_batch(
        scheduler: BatchScheduler,
        batch_seqs: list[Sequence],
        budget: IterationBudget,
    ) -> ScheduledBatch:
        """Build final ScheduledBatch from sequences."""
        request_ids: list[str] = []
        seq_lengths: list[int] = []
        seq_starts: list[int] = []
        position_offsets: list[int] = []
        is_prefill_list: list[bool] = []
        flat_tokens: list[int] = []

        iter_prefill_tokens, iter_decode_tokens = BatchBuilder.build_batch_tensors(
            scheduler,
            batch_seqs,
            budget,
            request_ids,
            seq_starts,
            seq_lengths,
            position_offsets,
            is_prefill_list,
            flat_tokens,
        )

        import torch

        input_ids = torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0)
        batch_tags = BatchBuilder.build_batch_tags(
            scheduler, batch_seqs, iter_prefill_tokens, iter_decode_tokens
        )

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

    # ------------------------------------------------------------------
    # Build flat token tensors
    # ------------------------------------------------------------------

    @staticmethod
    def build_batch_tensors(
        scheduler: BatchScheduler,
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

            tokens, is_prefill, pos_offset = BatchBuilder.build_seq_tokens(
                scheduler, seq, budget
            )
            flat_tokens.extend(tokens)
            seq_lengths.append(len(tokens))
            position_offsets.append(pos_offset)
            is_prefill_list.append(is_prefill)

            if is_prefill:
                iter_prefill_tokens += len(tokens)
                scheduler._total_prefill_tokens += len(tokens)
            else:
                iter_decode_tokens += 1
                scheduler._total_decode_tokens += 1

        return iter_prefill_tokens, iter_decode_tokens

    # ------------------------------------------------------------------
    # Build tokens for a single sequence
    # ------------------------------------------------------------------

    @staticmethod
    def build_seq_tokens(
        scheduler: BatchScheduler,
        seq: Sequence,
        budget: IterationBudget,
    ) -> tuple[list[int], bool, int]:
        """Build tokens for a single sequence in the batch.

        Returns:
            (tokens, is_prefill, position_offset) tuple.
        """
        is_chunked = seq.request_id in scheduler._chunked_prefill
        if is_chunked:
            cinfo = scheduler._chunked_prefill[seq.request_id]
            start = seq.prefix_match_len + cinfo.tokens_processed
            weight = scheduler._priority_weight(seq.priority)
            max_chunk = max(1, int(budget.max_prefill_tokens * weight))
            chunk_end = min(start + max_chunk, len(seq.prompt_tokens))
            tokens = seq.prompt_tokens[start:chunk_end]
            pos_offset = seq.prefix_match_len + cinfo.tokens_processed
            cinfo.tokens_processed += len(tokens)
            cinfo.chunks_remaining = max(0, cinfo.chunks_remaining - 1)
            if cinfo.is_complete:
                scheduler._chunked_prefill.pop(seq.request_id, None)
                seq.status = SequenceStatus.DECODING
            else:
                seq.status = SequenceStatus.PREFILLING
            return tokens, True, pos_offset

        if len(seq.generated_tokens) == 0 and seq.prefix_match_len == 0:
            tokens = seq.prompt_tokens[seq.prefix_match_len:]
            seq.status = (
                SequenceStatus.PREFILLING
                if not seq.is_complete
                else seq.status
            )
            return tokens, True, seq.prefix_match_len

        # Decode step: single token
        seq.status = SequenceStatus.DECODING
        return [seq.decode_input_token], False, seq.total_len - 1

    # ------------------------------------------------------------------
    # Build batch tags / metadata
    # ------------------------------------------------------------------

    @staticmethod
    def build_batch_tags(
        scheduler: BatchScheduler,
        batch_seqs: list[Sequence],
        iter_prefill_tokens: int,
        iter_decode_tokens: int,
    ) -> dict[str, object]:
        """Build batch metadata tags for this iteration."""
        batch_tags: dict[str, object] = {
            "iteration": scheduler._iteration_count,
            "chunked_prefill": len(scheduler._chunked_prefill),
            "prefill_tokens": iter_prefill_tokens,
            "decode_tokens": iter_decode_tokens,
            "total_prefill_tokens": scheduler._total_prefill_tokens,
            "total_decode_tokens": scheduler._total_decode_tokens,
        }
        if scheduler._use_length_grouping:
            lengths = [s.total_len for s in batch_seqs]
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            batch_tags["avg_seq_len"] = avg_len
            batch_tags["length_variance"] = (
                sum((seq_len - avg_len) ** 2 for seq_len in lengths)
                / len(lengths)
                if lengths
                else 0
            )

        remaining = [
            s.max_new_tokens - len(s.generated_tokens) for s in batch_seqs
        ]
        batch_tags["avg_tokens_remaining"] = (
            sum(remaining) / len(remaining) if remaining else 0
        )
        return batch_tags

    # ------------------------------------------------------------------
    # Budget check helpers
    # ------------------------------------------------------------------

    @staticmethod
    def check_decode_budget(
        scheduler: BatchScheduler,
        budget: IterationBudget,
        decode_count: int,
        remaining_total: int,
    ) -> bool:
        """Check if another decode fits in the budget."""
        if decode_count >= budget.decode_slots:
            return False
        if remaining_total < 1:
            return False
        return True
