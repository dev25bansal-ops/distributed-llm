"""Sarathi-style iteration-level scheduler.

Separates prefill and decode scheduling at each token step, with:
- Decode-priority: all active decode sequences always included
- Prefill-chunking: pending sequences get chunked prefill fills remaining capacity
- Fair interleaving: decode tokens get priority, prefill fills gaps
- SLA tracking per-tenant with deadline-based priority boosting
- Token-budget fair sharing across tenants
"""

import time
import heapq
from dataclasses import dataclass, field
import torch

from distllm.core.batch_scheduler import (
    Sequence, SequenceStatus, ScheduledBatch, BatchScheduler
)


@dataclass
class TenantSLA:
    """SLA configuration for a tenant."""
    tenant_id: str
    target_ttft_ms: float = 200.0  # Time to first token
    target_tpot_ms: float = 50.0   # Time per output token
    deadline_ms: float = 5000.0    # Absolute deadline from request arrival
    min_throughput_toks_per_s: float = 10.0
    priority_boost_factor: float = 1.5  # Multiplier when SLA is at risk


@dataclass
class TenantBudget:
    """Token budget for fair sharing across tenants."""
    tenant_id: str
    max_tokens_per_minute: float = 1000.0
    tokens_used: float = 0.0
    window_start: float = field(default_factory=time.time)
    _is_throttled: bool = False

    def reset_window(self) -> None:
        """Reset the token usage window."""
        now = time.time()
        if now - self.window_start >= 60.0:
            self.tokens_used = 0.0
            self.window_start = now
            self._is_throttled = False

    def can_spend(self, tokens: int) -> bool:
        """Check if tenant can spend tokens within budget."""
        self.reset_window()
        return not self._is_throttled and (self.tokens_used + tokens <= self.max_tokens_per_minute)

    def spend(self, tokens: int) -> None:
        self.reset_window()
        self.tokens_used += tokens
        if self.tokens_used >= self.max_tokens_per_minute:
            self._is_throttled = True

    @property
    def utilization(self) -> float:
        self.reset_window()
        return self.tokens_used / self.max_tokens_per_minute if self.max_tokens_per_minute > 0 else 0.0


class SLATracker:
    """Tracks SLA compliance per request and triggers priority boosting."""

    def __init__(self):
        self._request_start_times: dict[str, float] = {}
        self._request_first_token_at: dict[str, float] = {}
        self._request_last_token_at: dict[str, float] = {}
        self._request_token_counts: dict[str, int] = {}
        self._tenant_slas: dict[str, TenantSLA] = {}
        self._request_tenants: dict[str, str] = {}  # request_id -> tenant_id

    def register_request(
        self,
        request_id: str,
        tenant_id: str | None = None,
    ) -> None:
        """Register a new request for SLA tracking."""
        now = time.time()
        self._request_start_times[request_id] = now
        self._request_token_counts[request_id] = 0
        if tenant_id:
            self._request_tenants[request_id] = tenant_id

    def record_first_token(self, request_id: str) -> None:
        self._request_first_token_at[request_id] = time.time()

    def record_token(self, request_id: str) -> None:
        self._request_last_token_at[request_id] = time.time()
        self._request_token_counts[request_id] = self._request_token_counts.get(request_id, 0) + 1

    def complete_request(self, request_id: str) -> None:
        self._request_start_times.pop(request_id, None)
        self._request_first_token_at.pop(request_id, None)
        self._request_last_token_at.pop(request_id, None)
        self._request_token_counts.pop(request_id, None)
        self._request_tenants.pop(request_id, None)

    def set_tenant_sla(self, sla: TenantSLA) -> None:
        self._tenant_slas[sla.tenant_id] = sla

    def get_priority_boost(self, request_id: str, base_priority: int) -> int:
        """Calculate boosted priority if SLA is at risk.

        Returns a lower (higher priority) number if the request is at risk
        of missing its SLA deadline.
        """
        tenant_id = self._request_tenants.get(request_id)
        if not tenant_id or tenant_id not in self._tenant_slas:
            return base_priority

        sla = self._tenant_slas[tenant_id]
        start = self._request_start_times.get(request_id)
        if start is None:
            return base_priority

        elapsed_ms = (time.time() - start) * 1000

        # Check TTFT (time to first token)
        first_token_at = self._request_first_token_at.get(request_id)
        if first_token_at is None and elapsed_ms > sla.target_ttft_ms:
            # Haven't generated first token yet, past TTFT target
            return max(0, base_priority - 2)  # Boost by 2 levels

        # Check deadline
        if elapsed_ms > sla.deadline_ms * 0.8:
            # 80% of deadline elapsed
            return max(0, base_priority - 1)

        # Check TPOT (time per output token)
        token_count = self._request_token_counts.get(request_id, 0)
        if token_count > 0 and first_token_at:
            avg_tpot_ms = (time.time() - first_token_at) * 1000 / token_count
            if avg_tpot_ms > sla.target_tpot_ms * sla.priority_boost_factor:
                return max(0, base_priority - 1)

        return base_priority

    def get_request_metrics(self, request_id: str) -> dict:
        """Get SLA metrics for a request."""
        start = self._request_start_times.get(request_id)
        first = self._request_first_token_at.get(request_id)
        last = self._request_last_token_at.get(request_id)
        count = self._request_token_counts.get(request_id, 0)

        ttft_ms = (first - start) * 1000 if start and first else None
        total_ms = (time.time() - start) * 1000 if start else None
        tpot_ms = ((last - first) * 1000 / count) if first and last and count > 0 else None

        return {
            "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
            "total_ms": round(total_ms, 1) if total_ms else None,
            "tpot_ms": round(tpot_ms, 1) if tpot_ms else None,
            "token_count": count,
        }

    def stats(self) -> dict:
        return {
            "active_requests": len(self._request_start_times),
            "tenants_tracked": len(self._tenant_slas),
        }


class IterationScheduler(BatchScheduler):
    """Sarathi-style iteration-level scheduler.

    At each scheduling step:
    1. All active decode sequences are included (decode gets priority)
    2. Remaining capacity is filled with chunked prefill from pending requests
    3. SLA-aware priority boosting adjusts request priorities dynamically
    4. Token budgets enforce fair sharing across tenants
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 4096,
        model_info: dict | None = None,
        prefill_chunk_size: int = 256,
        decode_priority: bool = True,
    ):
        super().__init__(
            max_batch_size=max_batch_size,
            max_tokens_per_batch=max_tokens_per_batch,
            model_info=model_info,
        )
        self.prefill_chunk_size = prefill_chunk_size
        self.decode_priority = decode_priority

        # SLA tracking
        self.sla_tracker = SLATracker()

        # Tenant budgets
        self._tenant_budgets: dict[str, TenantBudget] = {}

        # Per-sequence tenant mapping
        self._seq_tenants: dict[str, str] = {}

    def set_tenant_sla(self, sla: TenantSLA) -> None:
        """Set SLA configuration for a tenant."""
        self.sla_tracker.set_tenant_sla(sla)

    def set_tenant_budget(
        self,
        tenant_id: str,
        max_tokens_per_minute: float = 1000.0,
    ) -> None:
        """Set token budget for a tenant."""
        self._tenant_budgets[tenant_id] = TenantBudget(
            tenant_id=tenant_id,
            max_tokens_per_minute=max_tokens_per_minute,
        )

    def add(self, seq: Sequence, tenant_id: str | None = None) -> None:
        """Add a request with optional tenant tracking."""
        if tenant_id:
            self._seq_tenants[seq.request_id] = tenant_id
            self.sla_tracker.register_request(seq.request_id, tenant_id)
        super().add(seq)

    def schedule(self) -> Optional[ScheduledBatch]:
        """Sarathi-style iteration-level scheduling.

        Policy:
        1. Evict completed sequences
        2. Apply SLA-based priority boosting to pending requests
        3. Include ALL active decode sequences (decode priority)
        4. Fill remaining capacity with chunked prefill from pending
        5. Enforce tenant token budgets
        """
        # 1. Evict completed sequences
        done_ids = [rid for rid, s in self.active.items() if s.is_complete]
        for rid in done_ids:
            seq = self.active.pop(rid)
            self._total_tokens -= seq.total_len
            self.sla_tracker.complete_request(rid)
            self._seq_tenants.pop(rid, None)

        # 2. Apply SLA-based priority boosting
        self._apply_sla_boosts()

        # 3. Include all active decode sequences
        decode_seqs: list[Sequence] = [
            s for s in self.active.values()
            if len(s.generated_tokens) > 0 or s.prefix_match_len > 0
        ]
        decode_tokens = sum(s.total_len for s in decode_seqs)

        # 4. Fill remaining with chunked prefill from pending
        prefill_seqs: list[Sequence] = []
        prefill_tokens = 0
        remaining_slots = self.max_batch_size - len(decode_seqs)
        remaining_token_budget = self.max_tokens_per_batch - decode_tokens

        # Sort pending by priority (SLA-boosted)
        pending_sorted = sorted(self._pending_heap, key=lambda x: (x[0], x[1]))

        for _pri, _cnt, candidate in pending_sorted:
            if remaining_slots <= 0:
                break
            if remaining_token_budget <= 0:
                break

            # Check tenant budget
            tenant_id = self._seq_tenants.get(candidate.request_id)
            if tenant_id and not self._check_tenant_budget(tenant_id):
                continue  # Skip throttled tenant

            # Chunked prefill: only take up to prefill_chunk_size tokens
            full_prompt_len = candidate.total_len
            already_prefilled = candidate.prefix_match_len
            chunk_len = min(
                self.prefill_chunk_size,
                full_prompt_len - already_prefilled,
                remaining_token_budget,
            )

            if chunk_len <= 0:
                continue

            heapq.heappop(self._pending_heap)
            candidate.prefix_match_len += chunk_len
            candidate.status = SequenceStatus.PREFILLING
            prefill_seqs.append(candidate)
            prefill_tokens += chunk_len
            remaining_slots -= 1
            remaining_token_budget -= chunk_len

            # Spend tenant budget
            if tenant_id:
                self._spend_tenant_budget(tenant_id, chunk_len)

        batch_seqs = decode_seqs + prefill_seqs
        if not batch_seqs:
            return None

        # Move newly promoted to active
        for seq in batch_seqs:
            if seq.request_id not in self.active:
                self.active[seq.request_id] = seq
                self._total_tokens += seq.total_len

        self._total_tokens = sum(s.total_len for s in batch_seqs)

        # 5. Build batch tensors
        return self._build_batch(batch_seqs)

    def _build_batch(self, batch_seqs: list[Sequence]) -> ScheduledBatch:
        """Build the ScheduledBatch from selected sequences."""
        request_ids = []
        seq_lengths = []
        position_offsets = []
        is_prefill_list = []
        input_tokens: list[int] = []

        for seq in batch_seqs:
            request_ids.append(seq.request_id)
            # A sequence is doing prefill if it hasn't started generating yet
            is_prefill = (len(seq.generated_tokens) == 0 and seq.prefix_match_len < len(seq.prompt_tokens))

            if is_prefill:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:start + self.prefill_chunk_size]
                if not tokens:
                    tokens = seq.prompt_tokens[start:]
                input_tokens.extend(tokens)
                seq_lengths.append(len(tokens))
                position_offsets.append(seq.prefix_match_len)
                seq.status = SequenceStatus.PREFILLING
            else:
                input_tokens.append(seq.decode_input_token)
                seq_lengths.append(1)
                position_offsets.append(seq.total_len - 1)
                seq.status = SequenceStatus.DECODING
                # Record token for SLA tracking
                self.sla_tracker.record_token(seq.request_id)

            is_prefill_list.append(is_prefill)

        # Pad for ragged batch
        max_len = max(seq_lengths) if seq_lengths else 1
        padded = []
        for i, seq in enumerate(batch_seqs):
            if is_prefill_list[i]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:start + self.prefill_chunk_size]
                if not tokens:
                    tokens = seq.prompt_tokens[start:]
                row = list(tokens) + [0] * (max_len - len(tokens))
            else:
                row = [seq.decode_input_token] + [0] * (max_len - 1)
            padded.append(row)

        input_ids = torch.tensor(padded, dtype=torch.long)

        # Batch tags
        batch_tags: dict[str, object] = {
            "decode_count": len([s for s in batch_seqs if not is_prefill_list[i] for i, _ in enumerate(batch_seqs) if s.request_id == s.request_id]),
            "prefill_count": sum(is_prefill_list),
            "sarathi_iteration": True,
        }

        return ScheduledBatch(
            sequences=batch_seqs,
            input_ids=input_ids,
            seq_lengths=seq_lengths,
            position_offsets=position_offsets,
            is_prefill=is_prefill_list,
            request_ids=request_ids,
            batch_tags=batch_tags,
            adapter_ids=[seq.adapter_id for seq in batch_seqs],
        )

    def step(self, batch: ScheduledBatch, next_tokens: torch.Tensor) -> None:
        """Process step output, update sequences, track SLA."""
        for i, seq in enumerate(batch.sequences):
            token = next_tokens[i].item()
            seq.generated_tokens.append(int(token))

            # Record first token for SLA
            if len(seq.generated_tokens) == 1:
                self.sla_tracker.record_first_token(seq.request_id)
            self.sla_tracker.record_token(seq.request_id)

            if seq.constraint is not None:
                seq.constraint.update(next_tokens[i])

            if seq.is_complete or token in seq.stop_token_ids:
                seq.status = SequenceStatus.DONE

    def _apply_sla_boosts(self) -> None:
        """Apply SLA-based priority boosting to pending requests."""
        boosted = []
        for _pri, _cnt, seq in self._pending_heap:
            new_priority = self.sla_tracker.get_priority_boost(seq.request_id, seq.priority)
            boosted.append((new_priority, _cnt, seq))
        if boosted:
            self._pending_heap = boosted
            heapq.heapify(self._pending_heap)

    def _check_tenant_budget(self, tenant_id: str) -> bool:
        budget = self._tenant_budgets.get(tenant_id)
        if budget is None:
            return True  # No budget = unlimited
        return budget.can_spend(1)

    def _spend_tenant_budget(self, tenant_id: str, tokens: int) -> None:
        budget = self._tenant_budgets.get(tenant_id)
        if budget:
            budget.spend(tokens)

    def get_sla_metrics(self, request_id: str) -> dict:
        return self.sla_tracker.get_request_metrics(request_id)

    def stats(self) -> dict:
        base = super().stats()
        base.update({
            "prefill_chunk_size": self.prefill_chunk_size,
            "decode_priority": self.decode_priority,
            "sla": self.sla_tracker.stats(),
            "tenant_budgets": {
                tid: {
                    "utilization": round(b.utilization, 3),
                    "throttled": b._is_throttled,
                }
                for tid, b in self._tenant_budgets.items()
            },
        })
        return base
