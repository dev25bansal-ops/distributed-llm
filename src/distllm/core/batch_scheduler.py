"""Continuous batch scheduler for pipeline-parallel inference."""

from loguru import logger
import heapq
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import torch

from distllm.core.request_latency import RequestLatencyTracker


def group_by_length(
    sequences: list[object],
    num_buckets: int = 4,
) -> dict[int, list[object]]:
    """Group sequences by similar total length into log-scale buckets."""
    buckets: dict[int, list[object]] = {i: [] for i in range(num_buckets)}
    lengths = [s.total_len for s in sequences]
    if not lengths:
        return buckets
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len == max_len:
        buckets[0] = list(sequences)
        return buckets
    log_min = math.log(max(min_len, 1))
    log_max = math.log(max_len)
    log_range = log_max - log_min
    for seq in sequences:
        ln = math.log(max(seq.total_len, 1))
        bucket = min(int((ln - log_min) / log_range * num_buckets), num_buckets - 1)
        buckets[bucket].append(seq)
    return buckets

if TYPE_CHECKING:
    from distllm.core.adaptive_batching import AdaptiveBatchingEngine


class SequenceStatus(Enum):
    """Lifecycle states for a generation sequence."""
    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DONE = "done"
    FAILED = "failed"
    PREEMPTED = "preempted"  # Temporarily removed to free resources


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
    # Logprobs & OpenAI compliance
    include_logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: dict[int, float] = field(default_factory=dict)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    token_counts: dict[int, int] = field(default_factory=dict)
    max_latency_ms: float | None = None  # For frequency penalty

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
    seq_starts: list[int] = field(default_factory=list)       # Start index in input_ids for each sequence
    seq_lengths: list[int] = field(default_factory=list)      # Per-sequence total length
    position_offsets: list[int] = field(default_factory=list) # Cached KV length per sequence
    is_prefill: list[bool] = field(default_factory=list)      # Whether each seq is doing prefill vs decode
    request_ids: list[str] = field(default_factory=list)
    attention_mask: torch.Tensor | None = None  # [1, 1, total_tokens, total_tokens] block-diagonal causal mask
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


class DecodePressureTracker:
    """Tracks decode queue pressure to dynamically adapt prefill/decode split.

    Maintains a sliding window of per-token decode latencies and computes
    a 0.0–1.0 pressure signal used by the Sarathi-Serve style scheduler.
    Higher pressure → more decode slots reserved, prefill tokens throttled.
    """

    def __init__(self, window_size: int = 10, target_ms_per_token: float = 8.0):
        self._decode_latencies: list[float] = []
        self._window_size = window_size
        self._target_ms = target_ms_per_token

    def record_decode_step(self, batch_decode_count: int, elapsed_ms: float) -> None:
        per_token = elapsed_ms / max(batch_decode_count, 1)
        self._decode_latencies.append(per_token)
        if len(self._decode_latencies) > self._window_size:
            self._decode_latencies.pop(0)

    @property
    def pressure(self) -> float:
        if not self._decode_latencies:
            return 0.0
        avg = sum(self._decode_latencies) / len(self._decode_latencies)
        return min(1.0, avg / max(self._target_ms, 0.1))

    @property
    def avg_ms_per_token(self) -> float:
        if not self._decode_latencies:
            return 0.0
        return sum(self._decode_latencies) / len(self._decode_latencies)


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

        # Starvation prevention: aging tracker boosts priority of long-waiting requests
        self._aging_enabled = True
        self._aging_interval_s: float = 30.0  # Boost by 1 level every 30s of waiting
        self._aging_max_boost: int = 2        # Max priority levels an aging request can gain

        # Priority weights for within-batch token allocation.
        # Multiplies the base chunk size (max_prefill_tokens per iteration)
        # so higher-priority sequences finish prefill faster.
        self._priority_weights: dict[int, float] = {
            0: 1.5,   # critical — 50% bonus tokens/iteration
            1: 1.25,  # high    — 25% bonus
            2: 1.0,   # normal  — baseline
            3: 0.5,   # low     — half the tokens (slower prefill)
        }

        # Sarathi-Serve style adaptive pressure tracking
        self._pressure_tracker = DecodePressureTracker()
        self._adapt_prefill_budget = True  # toggle Sarathi-Serve adaptive control

        # Preemption state
        self._preempted: dict[str, Sequence] = {}  # request_id -> Sequence
        self._preempted_kv_state: dict[str, dict] = {}  # request_id -> KV cache state
        self._max_preempted: int = 4  # Max concurrent preempted sequences

        # Adaptive batching engine (set externally by coordinator)
        self._adaptive_engine: 'AdaptiveBatchingEngine | None' = None

        self._lock = threading.Lock()

        # Cache manager for radix tree prefix storage (set by coordinator)
        self._cache_mgr = None

    def set_cache_manager(self, cache_mgr) -> None:
        """Set the cache manager for radix tree prefix storage."""
        self._cache_mgr = cache_mgr

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

    def split_overflow(self, max_size: int | None = None) -> int:
        """Limit the pending heap to at most *max_size* items.

        Excess items are moved to an internal overflow list and re-inserted
        (in priority order) on the next :meth:`schedule` call.  This prevents
        the schedule loop from re-examining thousands of low-priority items
        each iteration.

        Args:
            max_size: Maximum number of items to keep in the pending heap.
                      Defaults to ``self.max_batch_size * 2`` so that the
                      schedule loop still has a moderately-sized pool to
                      choose from.

        Returns:
            Number of items moved to overflow (0 if none).
        """
        max_size = max_size or self.max_batch_size * 2
        with self._lock:
            overflow = len(self._pending_heap) - max_size
            if overflow <= 0:
                return 0
            # Move lowest-priority items to overflow buffer
            kept = self._pending_heap[:max_size]
            overflowed = self._pending_heap[max_size:]
            self._pending_heap = kept + overflowed
            heapq.heapify(self._pending_heap)
            return overflow

    def set_paged_attention(self, mgr: object) -> None:
        """Connect to PagedAttention manager for KV block-aware scheduling."""
        self._paged_attention_mgr = mgr

    def _compute_sarathi_budget(self, budget: IterationBudget) -> IterationBudget:
        """Sarathi-Serve style adaptive budget: reserve decode slots first.

        When decode pressure is high (decode latency above target), prefill
        tokens are throttled and more slots are reserved for running decodes.
        When the decode pipeline is idle, more budget is allocated to prefill.

        Returns a modified deep copy of the budget.
        """
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
        """Add a new request to the pending queue (priority-ordered)."""
        with self._lock:
            heapq.heappush(self._pending_heap, (seq.priority, self._counter, seq))
            self._counter += 1
        self._latency_tracker.register(seq.request_id, sla_ms=seq.max_latency_ms)

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
        """Build batch respecting iteration-level budget.

        Uses Sarathi-Serve style adaptive budget when enabled:
        - Reserve decode slots first based on pressure from DecodePressureTracker
        - Fill remaining capacity with chunked prefill
        - Co-schedule prefill chunks + decode tokens in the same batch
        """
        # Update batch size from adaptive engine before scheduling
        self.update_batch_size_from_adaptive()

        # Apply Sarathi-Serve adaptive budget if enabled
        if self._adapt_prefill_budget:
            budget = self._compute_sarathi_budget(budget)

        # 1. Evict completed sequences (under lock — shared with get_sequence, preempt)
        with self._lock:
            done_ids = [rid for rid, s in self.active.items() if s.is_complete]
            for rid in done_ids:
                seq = self.active.pop(rid)
                self._total_tokens -= seq.total_len
                self._chunked_prefill.pop(rid, None)
                self._latency_tracker.complete(rid)

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
        # Single lock covers drain-process-rebuild atomically so no add() races.
        with self._lock:
            new_active_ids = set()
            remaining_batch_slots = budget.max_batch_size - len(batch_seqs)

            # Collect pending items
            pending_items = [heapq.heappop(self._pending_heap) for _ in range(len(self._pending_heap))]

            # Remove any pending items whose request_ids are already in active set
            # (can happen after restore_preempted when preempted sequences are put
            #  both in active and pending heap — a known race)
            filtered_items = []
            for item in pending_items:
                _pri, _cnt, candidate = item
                if candidate.request_id in self.active:
                    continue
                filtered_items.append(item)
            pending_items = filtered_items

            # Apply latency-based priority boosting with length-aware grouping
            batch_avg_remaining = 0
            if self._use_length_grouping and batch_seqs:
                batch_remaining = [s.max_new_tokens - len(s.generated_tokens) for s in batch_seqs]
                batch_avg_remaining = sum(batch_remaining) / len(batch_remaining) if batch_remaining else 0

            latency_aware_items = []
            for pri, cnt, candidate in pending_items:
                effective_pri = self._latency_tracker.get_latency_boost(candidate.request_id, pri)
                # Starvation prevention: aging reduces effective priority number
                aging = self._aging_boost(candidate)
                if aging > 0:
                    effective_pri = max(0, effective_pri - aging)
                if self._use_length_grouping and batch_avg_remaining > 0:
                    length_diff = abs((candidate.max_new_tokens - len(candidate.generated_tokens)) - batch_avg_remaining)
                    length_score = length_diff / (batch_avg_remaining + 1)
                    effective_pri += min(length_score * 0.1, 0.5)
                latency_aware_items.append((effective_pri, cnt, candidate))
            latency_aware_items.sort(key=lambda x: (x[0], x[1]))

            rejected = []
            for _pri, _cnt, candidate in latency_aware_items:
                if remaining_batch_slots <= 0:
                    rejected.append((_pri, _cnt, candidate))
                    continue
                if remaining_prefill_budget <= 0 and remaining_total_budget <= 0:
                    rejected.append((_pri, _cnt, candidate))
                    continue

                c_tokens = candidate.total_len
                if c_tokens > remaining_total_budget:
                    rejected.append((_pri, _cnt, candidate))
                    continue

                if self._enable_chunked_prefill and c_tokens > budget.max_prefill_tokens > 0:
                    chunk = budget.max_prefill_tokens
                else:
                    chunk = c_tokens

                rem_decode_est = decode_seqs_added * budget.prefill_slack_ratio
                if remaining_total_budget - chunk < rem_decode_est:
                    if c_tokens > budget.max_prefill_tokens:
                        chunk = min(chunk, int(remaining_total_budget * (1 - budget.prefill_slack_ratio)))
                    else:
                        if chunk > remaining_total_budget:
                            rejected.append((_pri, _cnt, candidate))
                            continue

                if chunk > remaining_prefill_budget and remaining_total_budget - chunk < 0:
                    rejected.append((_pri, _cnt, candidate))
                    continue

                if self._paged_attention_mgr is not None:
                    blocks_needed = self.paged_kv_block_count(self._total_tokens + chunk)
                    pa_pool = getattr(self._paged_attention_mgr, 'pool', None)
                    if pa_pool is not None:
                        total_blocks = getattr(pa_pool, 'total_blocks', blocks_needed + 1)
                        if blocks_needed > total_blocks * 0.9:
                            rejected.append((_pri, _cnt, candidate))
                            continue

                candidate.status = SequenceStatus.PREFILLING
                batch_seqs.append(candidate)
                self.active[candidate.request_id] = candidate
                new_active_ids.add(candidate.request_id)
                self._total_tokens += c_tokens
                remaining_prefill_budget -= chunk
                remaining_total_budget -= chunk
                remaining_batch_slots -= 1

                if self._enable_chunked_prefill and c_tokens > budget.max_prefill_tokens > 0:
                    chunk_size = budget.max_prefill_tokens
                    self._chunked_prefill[candidate.request_id] = ChunkedPrefillInfo(
                        seq_id=candidate.request_id,
                        total_prompt_tokens=c_tokens,
                        chunk_size=chunk_size,
                        chunks_remaining=math.ceil(c_tokens / chunk_size),
                    )

            # Rebuild heap from ALL non-promoted items (not just rejected)
            # This ensures no request is silently dropped
            self._pending_heap = rejected
            heapq.heapify(self._pending_heap)
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
        #    Higher-priority sequences get a larger share of the per-iteration
        #    chunked prefill budget, so they finish prefill sooner.
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
                # Priority-weighted chunk size: higher priority gets more tokens per iteration
                weight = self._priority_weight(seq.priority)
                max_chunk = max(1, int(budget.max_prefill_tokens * weight))
                chunk_end = min(start + max_chunk, len(seq.prompt_tokens))
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

        # No attention mask needed — coordinator_model.py builds its own per-sequence
        # padding mask, FlashAttention uses built-in causal=True, and the distributed
        # pipeline_orchestrator constructs its own all-ones mask over the network.
        # Skipping the old [total, total] dense mask avoids O(n²) memory blowup.

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
            batch_tags["length_variance"] = sum((seq_len - avg_len) ** 2 for seq_len in lengths) / len(lengths) if lengths else 0

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

    def step(self, batch: ScheduledBatch, next_tokens: torch.Tensor, kv_caches: dict | None = None, decoded_tokens: list[str] | None = None) -> None:
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

        Args:
            request_id: The request to promote.
            new_priority: The new priority level.

        Returns:
            True if the request was found and updated.
        """
        with self._lock:
            for i, (_pri, _cnt, seq) in enumerate(self._pending_heap):
                if seq.request_id == request_id:
                    seq.priority = new_priority
                    self._pending_heap[i] = (new_priority, _cnt, seq)
                    heapq.heapify(self._pending_heap)
                    return True
            return False

    def preempt_lowest(self, min_priority: int = 3, kv_cache_state: dict | None = None) -> Sequence | None:
        """Preempt the lowest priority active sequence and re-queue it."""
        with self._lock:
            if len(self._preempted) >= self._max_preempted:
                return None

            worst_seq = None
            worst_pri = -1
            for _rid, seq in self.active.items():
                if seq.priority >= min_priority and (worst_seq is None or seq.priority > worst_pri):
                    worst_seq = seq
                    worst_pri = seq.priority

            if worst_seq is None:
                return None

            req_id = worst_seq.request_id
            self._save_kv_state(req_id, kv_cache_state)

            if self._paged_attention_mgr is not None:
                try:
                    self._paged_attention_mgr.swap_out_sequence(req_id)
                except Exception as e:
                    logger.debug("PagedAttention swap failed: {}", e)

            del self.active[req_id]
            self._total_tokens -= worst_seq.total_len
            worst_seq.status = SequenceStatus.PENDING
            self._counter += 1
            heapq.heappush(self._pending_heap, (worst_seq.priority, self._counter, worst_seq))
            self._preempted[req_id] = worst_seq
            return worst_seq

    def _save_kv_state(self, request_id: str, kv_cache_state: dict | None = None) -> None:
        """Save KV cache state for a preempted sequence."""
        if kv_cache_state is not None and request_id in kv_cache_state:
            self._preempted_kv_state[request_id] = {
                "kv_cache": kv_cache_state[request_id],
                "source": "external",
            }

    def _restore_kv_state(self, request_id: str, kv_cache_state: dict | None = None) -> bool:
        """Restore KV cache state for a preempted sequence.

        Returns:
            True if KV state was restored.
        """
        saved = self._preempted_kv_state.pop(request_id, None)
        if saved is not None and kv_cache_state is not None:
            kv_cache_state[request_id] = saved["kv_cache"]
            return True
        return False

    def restore_preempted(self, kv_cache_state: dict | None = None) -> list[Sequence]:
        """Restore all preempted sequences back to active with KV state."""
        with self._lock:
            restored = []
            for req_id, seq in list(self._preempted.items()):
                self._restore_kv_state(req_id, kv_cache_state)

                if self._paged_attention_mgr is not None:
                    try:
                        self._paged_attention_mgr.pool.restore_block(
                            self._paged_attention_mgr.get_physical_blocks(req_id)[0]
                        )
                    except Exception as e:
                        logger.debug("PagedAttention restore failed: {}", e)

                # Remove from pending heap if present (preempt_lowest pushes there)
                self._pending_heap = [
                    item for item in self._pending_heap
                    if not (len(item) >= 3 and item[2].request_id == req_id)
                ]
                heapq.heapify(self._pending_heap)

                seq.status = SequenceStatus.DECODING
                self.active[req_id] = seq
                self._total_tokens += seq.total_len
                restored.append(seq)

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
                stats["adaptive_batch_size"] = astats.avg_latency_ms
            except Exception as e:
                logger.debug("Adaptive engine get_stats failed: {}", e)
        return stats

    @property
    def latency_tracker(self) -> RequestLatencyTracker:
        """Return the latency tracker instance."""
        return self._latency_tracker
