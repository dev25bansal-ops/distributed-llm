"""Continuous batch scheduler for pipeline-parallel inference."""

import time
import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

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
    prompt_tokens: List[int]
    generated_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.PENDING
    priority: int = 2  # 0=critical, 1=high, 2=normal, 3=low
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: List[int] = field(default_factory=list)
    constraint: Optional[object] = None  # JSONSchemaConstraint from structured_output
    prefix_match_len: int = 0  # Tokens served from prefix cache
    created_at: float = field(default_factory=time.time)
    adapter_id: Optional[str] = None  # LoRA adapter ID for this request (S-LoRA style)

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
    sequences: List[Sequence]
    input_ids: torch.Tensor       # [total_tokens] — flattened 1D, all tokens concatenated (no padding)
    seq_starts: List[int]         # Start index in input_ids for each sequence
    seq_lengths: List[int]        # Per-sequence total length
    position_offsets: List[int]   # Cached KV length per sequence
    is_prefill: List[bool]        # Whether each seq is doing prefill vs decode
    request_ids: List[str]
    speculative_enabled: bool = False
    batch_tags: Dict[str, object] = field(default_factory=dict)
    adapter_ids: List[Optional[str]] = field(default_factory=list)

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


class BatchScheduler:
    """Continuous batch scheduler for distributed inference.

    Scheduling policy:
    1. All active (non-complete) sequences stay in the batch
    2. Fill remaining capacity with pending sequences
    3. Respect max_batch_size and max_tokens_per_batch limits

    Supports dynamic token budgets from 32K to 128K tokens per batch,
    auto-scaled based on GPU memory pressure. Integrates with PagedAttention
    for efficient KV cache block allocation.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 32768,
        model_info: Optional[dict] = None,
        paged_attention_mgr: Optional[object] = None,
    ):
        self.max_batch_size = max_batch_size
        self._base_tokens_per_batch = max_tokens_per_batch
        self.max_tokens_per_batch = self._compute_dynamic_budget(max_tokens_per_batch)
        self._paged_attention_mgr = paged_attention_mgr
        self._pending_heap: list = []  # Min-heap of (priority, counter, Sequence)
        self._counter: int = 0  # Tiebreaker for FIFO within same priority
        self.active: Dict[str, Sequence] = {}
        self._total_tokens: int = 0  # Incremental token count for O(1) tracking
        self._tensor_pool = TensorPool()
        self._model_info = model_info
        self._use_length_grouping = model_info is not None

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

    def schedule(self) -> Optional[ScheduledBatch]:
        """Build the next batch from active + pending sequences.

        Returns None if there are no sequences to process.
        """
        # 1. Evict completed sequences from active
        done_ids = [rid for rid, s in self.active.items() if s.is_complete]
        for rid in done_ids:
            seq = self.active.pop(rid)
            self._total_tokens -= seq.total_len

        # 2. Start with all active non-complete sequences
        batch_seqs: List[Sequence] = list(self.active.values())
        current_tokens = self._total_tokens

        # 3. Fill with pending sequences by priority up to limits
        while self._pending_heap:
            if len(batch_seqs) >= self.max_batch_size:
                break
            _pri, _cnt, candidate = self._pending_heap[0]
            candidate_tokens = candidate.total_len
            if current_tokens + candidate_tokens > self.max_tokens_per_batch:
                break
            # Check PagedAttention block capacity if available
            if self._paged_attention_mgr is not None:
                blocks_needed = self.paged_kv_block_count(current_tokens + candidate_tokens)
                pool = getattr(self._paged_attention_mgr, 'pool', None)
                if pool is not None and blocks_needed > getattr(pool, 'total_blocks', blocks_needed + 1):
                    break
            heapq.heappop(self._pending_heap)
            candidate.status = SequenceStatus.PREFILLING
            batch_seqs.append(candidate)
            current_tokens += candidate_tokens

        if not batch_seqs:
            return None

        # 4. Move newly promoted sequences to active and update token count
        for seq in batch_seqs:
            if seq.request_id not in self.active:
                self.active[seq.request_id] = seq
                self._total_tokens += seq.total_len

        # 5. Build batch tensors (ragged/flat layout — no padding)
        request_ids = []
        seq_lengths = []
        seq_starts = []
        position_offsets = []
        is_prefill_list = []
        flat_tokens: List[int] = []

        for seq in batch_seqs:
            request_ids.append(seq.request_id)
            seq_starts.append(len(flat_tokens))
            is_prefill = (len(seq.generated_tokens) == 0 and seq.prefix_match_len == 0)

            if is_prefill:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
                flat_tokens.extend(tokens)
                seq_lengths.append(len(tokens))
                position_offsets.append(seq.prefix_match_len)
                seq.status = SequenceStatus.PREFILLING
            else:
                flat_tokens.append(seq.decode_input_token)
                seq_lengths.append(1)
                position_offsets.append(seq.total_len - 1)
                seq.status = SequenceStatus.DECODING

            is_prefill_list.append(is_prefill)

        # Flat 1D tensor — zero wasted compute vs padded [batch, max_len]
        input_ids = torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0)

        # 6. Build batch metadata tags
        batch_tags: Dict[str, object] = {}
        if self._use_length_grouping:
            lengths = [s.total_len for s in batch_seqs]
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            batch_tags["avg_seq_len"] = avg_len
            batch_tags["length_variance"] = (
                sum((l - avg_len) ** 2 for l in lengths) / len(lengths) if lengths else 0
            )

        # Track avg tokens remaining
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

    def step(self, batch: ScheduledBatch, next_tokens: torch.Tensor) -> None:
        """Process sampling output, update sequences, check for completion.

        Args:
            batch: The batch that was just processed.
            next_tokens: [batch_size] tensor of sampled token IDs.
        """
        for i, seq in enumerate(batch.sequences):
            token = next_tokens[i].item()
            seq.generated_tokens.append(int(token))

            # Check constraint (structured output)
            if seq.constraint is not None:
                seq.constraint.update(next_tokens[i])

            if seq.is_complete or token in seq.stop_token_ids:
                seq.status = SequenceStatus.DONE

    def get_sequence(self, request_id: str) -> Optional[Sequence]:
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

    def preempt_lowest(self, min_priority: int = 3) -> Optional[Sequence]:
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
        }
