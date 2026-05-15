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
    """A batch of sequences ready for one forward pass."""
    sequences: List[Sequence]
    input_ids: torch.Tensor       # [total_tokens] — flattened for prefill, single token per seq for decode
    seq_lengths: List[int]        # Per-sequence total length
    position_offsets: List[int]   # Cached KV length per sequence
    is_prefill: List[bool]        # Whether each seq is doing prefill vs decode
    request_ids: List[str]
    speculative_enabled: bool = False  # Whether speculative decoding is active for this batch
    batch_tags: Dict[str, object] = field(default_factory=dict)  # Metadata: length_bucket, avg_tokens_remaining, etc.

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def max_seq_len(self) -> int:
        return max(self.seq_lengths) if self.seq_lengths else 0


class BatchScheduler:
    """Continuous batch scheduler for distributed inference.

    Scheduling policy:
    1. All active (non-complete) sequences stay in the batch
    2. Fill remaining capacity with pending sequences
    3. Respect max_batch_size and max_tokens_per_batch limits
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 4096,
        model_info: Optional[dict] = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
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
            # Peek at highest priority (lowest number)
            _pri, _cnt, candidate = self._pending_heap[0]
            candidate_tokens = candidate.total_len
            if current_tokens + candidate_tokens > self.max_tokens_per_batch:
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

        # 5. Build batch tensors
        request_ids = []
        seq_lengths = []
        position_offsets = []
        is_prefill_list = []
        input_tokens: List[int] = []

        for seq in batch_seqs:
            request_ids.append(seq.request_id)
            is_prefill = (len(seq.generated_tokens) == 0 and seq.prefix_match_len == 0)

            if is_prefill:
                # Prefill: send all prompt tokens (minus prefix match)
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
                input_tokens.extend(tokens)
                seq_lengths.append(len(tokens))
                position_offsets.append(seq.prefix_match_len)
                seq.status = SequenceStatus.PREFILLING
            else:
                # Decode: send only the last generated token
                input_tokens.append(seq.decode_input_token)
                seq_lengths.append(1)
                position_offsets.append(seq.total_len - 1)
                seq.status = SequenceStatus.DECODING

            is_prefill_list.append(is_prefill)

        # Build padded input tensor for the forward pass
        # For mixed prefill/decode, we need ragged batch handling
        # Simplest: pad all sequences to max length
        max_len = max(seq_lengths) if seq_lengths else 1
        padded = []
        for i, seq in enumerate(batch_seqs):
            if is_prefill_list[i]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
                row = tokens + [0] * (max_len - len(tokens))
            else:
                row = [seq.decode_input_token] + [0] * (max_len - 1)
            padded.append(row)

        input_ids = torch.tensor(padded, dtype=torch.long)

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
            seq_lengths=seq_lengths,
            position_offsets=position_offsets,
            is_prefill=is_prefill_list,
            request_ids=request_ids,
            batch_tags=batch_tags,
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
        }
