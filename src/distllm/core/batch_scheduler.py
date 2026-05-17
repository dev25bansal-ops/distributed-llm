"""Continuous batch scheduler for pipeline-parallel inference.

Supports iteration-level scheduling with:
- Chunked prefill (interleave prefill with decode)
- Mixed-batch prefill + decode in same iteration
- Time-budget-aware scheduling
- Priority-based preemption with age awareness
"""

import time
import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
import torch

from distllm.core.tensor_pool import TensorPool


class SequenceStatus(Enum):
    """Lifecycle states for a generation sequence."""
    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Sequence:
    """Represents a single generation sequence (one request)."""
    request_id: str
    prompt_tokens: list[int]
    generated_tokens: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.PENDING
    priority: int = 2  # 0=critical, 1=high, 2=normal, 3=low
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: list[int] = field(default_factory=list)
    constraint: object | None = None  # JSONSchemaConstraint from structured_output
    prefix_match_len: int = 0  # Tokens served from prefix cache
    created_at: float = field(default_factory=time.time)
    adapter_id: str | None = None  # LoRA adapter ID for this request (S-LoRA style)

    @property
    def is_complete(self) -> bool:
        if self.status in (SequenceStatus.DONE, SequenceStatus.FAILED):
            return True
        return len(self.generated_tokens) >= self.max_new_tokens

    @property
    def total_len(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def decode_input_token(self) -> int:
        """Token to feed as input for the next decode step."""
        return self.generated_tokens[-1]


@dataclass
class ScheduledBatch:
    """A batch of sequences ready for one forward pass.
    
    Uses ragged/flat token layout (no padding) for zero wasted GPU compute.
    Each sequence's tokens are concatenated into a flat 1D tensor, with
    seq_starts tracking per-sequence boundaries in the flat array.
    """
    sequences: list[Sequence]
    input_ids: torch.Tensor       # [total_tokens] — flattened 1D, all tokens concatenated (no padding)
    seq_starts: list[int]         # Start index in input_ids for each sequence
    seq_lengths: list[int]        # Per-sequence total length
    position_offsets: list[int]   # Cached KV length per sequence
    is_prefill: list[bool]        # Whether each seq is doing prefill vs decode
    request_ids: list[str]
    speculative_enabled: bool = False
    batch_tags: dict[str, object] = field(default_factory=dict)
    adapter_ids: list[str | None] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def max_seq_len(self) -> int:
        return max(self.seq_lengths) if self.seq_lengths else 0

    @property
    def total_tokens(self) -> int:
        """Total tokens in the flat tensor (sum of all seq lengths)."""
        return sum(self.seq_lengths) if self.seq_lengths else 0


@dataclass
class IterationBudget:
    """Budget for a single iteration step.

    Controls how many prefill vs decode tokens to process,
    respecting both batch size and token count limits.
    """
    max_prefill_tokens: int = 4096
    max_decode_tokens: int = 512
    max_batch_size: int = 32
    max_total_tokens: int = 32768
    enable_chunked_prefill: bool = True
    prefill_slack_ratio: float = 0.3  # Reserve % for decode during long prefills

    @property
    def decode_slots(self) -> int:
        return min(self.max_batch_size, self.max_decode_tokens)


@dataclass
class ChunkedPrefillInfo:
    """Tracks chunked prefill state for a sequence."""
    seq_id: str
    total_prompt_tokens: int
    tokens_processed: int = 0
    chunk_size: int = 0
    chunks_remaining: int = 0

    @property
    def is_complete(self) -> bool:
        return self.tokens_processed >= self.total_prompt_tokens

    @property
    def remaining(self) -> int:
        return self.total_prompt_tokens - self.tokens_processed


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
    ):
        self.max_batch_size = max_batch_size
        self._base_tokens_per_batch = max_tokens_per_batch
        self._paged_attention_mgr = paged_attention_mgr
        self.max_tokens_per_batch = self._compute_dynamic_budget(max_tokens_per_batch)
        self._pending_heap: list = []  # Min-heap of (priority, counter, Sequence)
        self._counter: int = 0  # Tiebreaker for FIFO within same priority
        self.active: dict[str, Sequence] = {}
        self._total_tokens: int = 0  # Incremental token count for O(1) tracking
        self._tensor_pool = TensorPool()
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

    def set_iteration_budget(self, budget: IterationBudget) -> None:
        """Override the default iteration budget."""
        self._budget = budget

    def get_iteration_budget(self) -> IterationBudget:
        """Get current iteration budget (configurable per step)."""
        if self._budget.enable_chunked_prefill and self._enable_chunked_prefill:
            return self._budget
        return IterationBudget(
            max_prefill_tokens=self.max_tokens_per_batch,
            max_decode_tokens=self.max_tokens_per_batch,
            max_batch_size=self.max_batch_size,
            max_total_tokens=self.max_tokens_per_batch,
            enable_chunked_prefill=False,
        )

    def set_tensor_pool(self, pool: TensorPool) -> None:
        """Use an external tensor pool (for shared pool across schedulers)."""
        self._tensor_pool = pool

    def set_paged_attention(self, mgr: object) -> None:
        """Connect to PagedAttention manager for KV block-aware scheduling."""
        self._paged_attention_mgr = mgr

    def _compute_dynamic_budget(self, base_budget: int) -> int:
        """Auto-scale token budget from 32K-128K based on available GPU memory.

        If PagedAttention is available, uses pool utilization to adjust down
        under memory pressure. Otherwise keeps the base budget.
        """
        budget = max(32768, min(base_budget, 131072))
        if self._paged_attention_mgr is not None:
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
        self.max_tokens_per_batch = self._compute_dynamic_budget(self._base_tokens_per_batch)

    def paged_kv_block_count(self, tokens: int) -> int:
        """Estimate number of PagedAttention blocks needed for this many tokens."""
        if self._paged_attention_mgr is not None:
            block_size = getattr(self._paged_attention_mgr, 'block_size', 16)
        else:
            block_size = 16
        return (tokens + block_size - 1) // block_size

    def add(self, seq: Sequence) -> None:
        """Add a new request to the pending queue (priority-ordered)."""
        heapq.heappush(self._pending_heap, (seq.priority, self._counter, seq))
        self._counter += 1

    def schedule(self) -> ScheduledBatch | None:
        """Build the next batch from active + pending sequences.

        Returns None if there are no sequences to process.
        """
        return self._schedule_with_budget(self.get_iteration_budget())

    def schedule_iteration(self, iteration_budget: IterationBudget | None = None) -> ScheduledBatch | None:
        """Iteration-level schedule: respects time budget per iteration.

        Supports chunked prefill: if a large prompt exceeds the prefill budget,
        it is split across multiple iterations. Decode steps are always included
        to maintain low latency.

        Args:
            iteration_budget: Budget for this iteration. Uses default if None.

        Returns:
            ScheduledBatch or None if no work.
        """
        budget = iteration_budget or self.get_iteration_budget()
        self._iteration_count += 1
        return self._schedule_with_budget(budget)

    def _schedule_with_budget(self, budget: IterationBudget) -> ScheduledBatch | None:
        """Build batch respecting iteration-level budget."""
        # 1. Evict completed sequences
        done_ids = [rid for rid, s in self.active.items() if s.is_complete]
        for rid in done_ids:
            seq = self.active.pop(rid)
            self._total_tokens -= seq.total_len
            self._chunked_prefill.pop(rid, None)

        # 2. Start with active non-complete sequences (all decode, some prefill)
        batch_seqs: list[Sequence] = []
        remaining_prefill_budget = budget.max_prefill_tokens
        remaining_decode_budget = budget.max_decode_tokens
        remaining_total_budget = budget.max_total_tokens
        decode_seqs_added = 0

        # 2a. Add active sequences - decode first (always prioritized)
        for seq in self.active.values():
            if decode_seqs_added >= budget.decode_slots:
                break
            if seq.status == SequenceStatus.DECODING or len(seq.generated_tokens) > 0:
                if self._check_decode_budget(budget, decode_seqs_added, remaining_total_budget):
                    batch_seqs.append(seq)
                    decode_seqs_added += 1
                    remaining_total_budget -= 1

        # 2b. Add chunked prefill sequences (already partially prefilled)
        if self._enable_chunked_prefill:
            for seq in list(self.active.values()):
                if seq.request_id in self._chunked_prefill:
                    cinfo = self._chunked_prefill[seq.request_id]
                    if cinfo.is_complete:
                        continue
                    chunk = min(cinfo.remaining, max_prefill_tokens := budget.max_prefill_tokens)
                    chunk = min(chunk, remaining_prefill_budget)

                    if seq not in batch_seqs:
                        batch_seqs.append(seq)
                    remaining_prefill_budget -= chunk
                    remaining_total_budget -= chunk
                    if remaining_prefill_budget <= 0:
                        break

        # 3. Promote pending sequences respecting budget
        new_active_ids = set()
        remaining_batch_slots = budget.max_batch_size - len(batch_seqs)

        # Collect pending items, sort by priority then FIFO
        pending_items = []
        while self._pending_heap:
            pending_items.append(heapq.heappop(self._pending_heap))
        pending_items.sort(key=lambda x: (x[0], x[1]))

        for _pri, _cnt, candidate in pending_items:
            if remaining_batch_slots <= 0:
                heapq.heappush(self._pending_heap, (_pri, _cnt, candidate))
                continue
            if remaining_prefill_budget <= 0 and remaining_total_budget <= 0:
                heapq.heappush(self._pending_heap, (_pri, _cnt, candidate))
                continue

            if candidate.request_id in self.active:
                continue

            c_tokens = candidate.total_len
            if self._enable_chunked_prefill and c_tokens > budget.max_prefill_tokens:
                chunk = budget.max_prefill_tokens
            else:
                chunk = c_tokens

            rem_decode_est = decode_seqs_added * budget.prefill_slack_ratio
            if remaining_total_budget - chunk < rem_decode_est:
                if c_tokens > budget.max_prefill_tokens:
                    chunk = min(chunk, int(remaining_total_budget * (1 - budget.prefill_slack_ratio)))
                else:
                    if chunk > remaining_total_budget:
                        heapq.heappush(self._pending_heap, (_pri, _cnt, candidate))
                        continue

            if chunk > remaining_prefill_budget and remaining_total_budget - chunk < 0:
                heapq.heappush(self._pending_heap, (_pri, _cnt, candidate))
                continue

            if self._paged_attention_mgr is not None:
                blocks_needed = self.paged_kv_block_count(self._total_tokens + chunk)
                pa_pool = getattr(self._paged_attention_mgr, 'pool', None)
                if pa_pool is not None:
                    total_blocks = getattr(pa_pool, 'total_blocks', blocks_needed + 1)
                    free_blocks = getattr(pa_pool, 'free_blocks', total_blocks)
                    if blocks_needed > total_blocks * 0.9:
                        heapq.heappush(self._pending_heap, (_pri, _cnt, candidate))
                        continue

            candidate.status = SequenceStatus.PREFILLING
            batch_seqs.append(candidate)
            self.active[candidate.request_id] = candidate
            new_active_ids.add(candidate.request_id)
            self._total_tokens += c_tokens
            remaining_prefill_budget -= chunk
            remaining_total_budget -= chunk
            remaining_batch_slots -= 1

            if self._enable_chunked_prefill and c_tokens > budget.max_prefill_tokens:
                self._chunked_prefill[candidate.request_id] = ChunkedPrefillInfo(
                    seq_id=candidate.request_id,
                    total_prompt_tokens=c_tokens,
                    chunk_size=budget.max_prefill_tokens,
                    chunks_remaining=math.ceil(c_tokens / budget.max_prefill_tokens),
                )

        if not batch_seqs:
            return None

        # 4. Build batch tensors with iteration-level chunking
        request_ids = []
        seq_lengths = []
        seq_starts = []
        position_offsets = []
        is_prefill_list = []
        flat_tokens: list[int] = []

        for seq in batch_seqs:
            request_ids.append(seq.request_id)
            seq_starts.append(len(flat_tokens))

            is_chunked = seq.request_id in self._chunked_prefill
            if is_chunked:
                cinfo = self._chunked_prefill[seq.request_id]
                start = seq.prefix_match_len + cinfo.tokens_processed
                chunk_end = min(start + budget.max_prefill_tokens, len(seq.prompt_tokens))
                tokens = seq.prompt_tokens[start:chunk_end]
                flat_tokens.extend(tokens)
                seq_lengths.append(len(tokens))
                position_offsets.append(seq.prefix_match_len + cinfo.tokens_processed)
                cinfo.tokens_processed += len(tokens)
                cinfo.chunks_remaining = max(0, cinfo.chunks_remaining - 1)
                if cinfo.is_complete:
                    self._chunked_prefill.pop(seq.request_id, None)
                    seq.status = SequenceStatus.DECODING
                else:
                    seq.status = SequenceStatus.PREFILLING
                is_prefill_list.append(True)
                self._total_prefill_tokens += len(tokens)
            elif len(seq.generated_tokens) == 0 and seq.prefix_match_len == 0:
                tokens = seq.prompt_tokens[seq.prefix_match_len:]
                flat_tokens.extend(tokens)
                seq_lengths.append(len(tokens))
                position_offsets.append(seq.prefix_match_len)
                seq.status = SequenceStatus.PREFILLING if not seq.is_complete else seq.status
                is_prefill_list.append(True)
                self._total_prefill_tokens += len(tokens)
            else:
                flat_tokens.append(seq.decode_input_token)
                seq_lengths.append(1)
                position_offsets.append(seq.total_len - 1)
                seq.status = SequenceStatus.DECODING
                is_prefill_list.append(False)
                self._total_decode_tokens += 1

        input_ids = torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0)

        # 5. Build batch tags
        batch_tags: dict[str, object] = {
            "iteration": self._iteration_count,
            "chunked_prefill": len(self._chunked_prefill),
            "prefill_tokens": self._total_prefill_tokens,
            "decode_tokens": self._total_decode_tokens,
        }
        if self._use_length_grouping:
            lengths = [s.total_len for s in batch_seqs]
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            batch_tags["avg_seq_len"] = avg_len
            batch_tags["length_variance"] = sum((l - avg_len) ** 2 for l in lengths) / len(lengths) if lengths else 0

        remaining = [s.max_new_tokens - len(s.generated_tokens) for s in batch_seqs]
        batch_tags["avg_tokens_remaining"] = sum(remaining) / len(remaining) if remaining else 0

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

    def _check_decode_budget(self, budget: IterationBudget, decode_count: int, remaining_total: int) -> bool:
        """Check if another decode fits in the budget."""
        if decode_count >= budget.decode_slots:
            return False
        if remaining_total < 1:
            return False
        return True

    def step(self, batch: ScheduledBatch, next_tokens: torch.Tensor) -> None:
        """Process sampling output, update sequences, check for completion.

        Args:
            batch: The batch that was just processed.
            next_tokens: [batch_size] tensor of sampled token IDs.
        """
        for i, seq in enumerate(batch.sequences):
            token = next_tokens[i].item()
            seq.generated_tokens.append(int(token))

            # Transition from PREFILLING to DECODING after first generated token
            if seq.status == SequenceStatus.PREFILLING:
                seq.status = SequenceStatus.DECODING

            # Check constraint (structured output)
            if seq.constraint is not None:
                seq.constraint.update(next_tokens[i])

            if seq.is_complete or token in seq.stop_token_ids:
                seq.status = SequenceStatus.DONE

    def get_sequence(self, request_id: str) -> Sequence | None:
        """Get a sequence by request_id (from pending or active)."""
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

        Args:
            request_id: The request to promote.
            new_priority: The new priority level.

        Returns:
            True if the request was found and updated.
        """
        for i, (_pri, _cnt, seq) in enumerate(self._pending_heap):
            if seq.request_id == request_id:
                self._pending_heap[i] = (new_priority, _cnt, seq)
                heapq.heapify(self._pending_heap)
                return True
        return False

    def preempt_lowest(self, min_priority: int = 3) -> Sequence | None:
        """Preempt the lowest priority active sequence and re-queue it.

        Args:
            min_priority: Only preempt sequences with priority >= this value.

        Returns:
            The preempted sequence, or None if no candidate found.
        """
        worst_seq = None
        worst_pri = -1
        for rid, seq in self.active.items():
            if seq.priority >= min_priority and (worst_seq is None or seq.priority > worst_pri):
                worst_seq = seq
                worst_pri = seq.priority
        if worst_seq:
            del self.active[worst_seq.request_id]
            self._total_tokens -= worst_seq.total_len
            worst_seq.status = SequenceStatus.PENDING
            heapq.heappush(self._pending_heap, (worst_seq.priority, self._counter, worst_seq))
            self._counter += 1
        return worst_seq

    def stats(self) -> dict:
        return {
            "active_requests": self.active_count,
            "pending_requests": self.pending_count,
            "max_batch_size": self.max_batch_size,
            "max_tokens_per_batch": self.max_tokens_per_batch,
            "paged_attention": self._paged_attention_mgr is not None,
            "iteration": self._iteration_count,
            "total_prefill_tokens": self._total_prefill_tokens,
            "total_decode_tokens": self._total_decode_tokens,
            "chunked_prefill_active": len(self._chunked_prefill),
            "chunked_prefill_enabled": self._enable_chunked_prefill,
        }
