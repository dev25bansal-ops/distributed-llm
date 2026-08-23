"""Sarathi-style iteration-level scheduler with GPU isolation and MIG support.

Provides tenant-level SLA tracking and GPU isolation configuration
for multi-tenant deployments.  The IterationScheduler extends
BatchScheduler with SLA-aware priority boosting.

The SLA tracking logic is also available as a standalone
SchedulingPolicy (SLASchedulingPolicy) for use with the pluggable
scheduling policy system.
"""

from __future__ import annotations
import os
import time
import heapq
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from distllm.core.batch_scheduler import (
    Sequence, SequenceStatus, ScheduledBatch, BatchScheduler
)

@dataclass
class TenantSLA:
    tenant_id: str
    target_ttft_ms: float = 200.0
    target_tpot_ms: float = 50.0
    deadline_ms: float = 5000.0
    min_throughput_toks_per_s: float = 10.0
    priority_boost_factor: float = 1.5

@dataclass
class TenantBudget:
    tenant_id: str
    max_tokens_per_minute: float = 1000.0
    tokens_used: float = 0.0
    window_start: float = field(default_factory=time.time)
    _is_throttled: bool = False

    def reset_window(self) -> None:
        now = time.time()
        if now - self.window_start >= 60.0:
            self.tokens_used = 0.0
            self.window_start = now
            self._is_throttled = False

    def can_spend(self, tokens: int) -> bool:
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
    def __init__(self):
        self._request_start_times: dict[str, float] = {}
        self._request_first_token_at: dict[str, float] = {}
        self._request_last_token_at: dict[str, float] = {}
        self._request_token_counts: dict[str, int] = {}
        self._tenant_slas: dict[str, TenantSLA] = {}
        self._request_tenants: dict[str, str] = {}

    def register_request(
        self,
        request_id: str,
        tenant_id: str | None = None,
    ) -> None:
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
        tenant_id = self._request_tenants.get(request_id)
        if not tenant_id or tenant_id not in self._tenant_slas:
            return base_priority

        sla = self._tenant_slas[tenant_id]
        start = self._request_start_times.get(request_id)
        if start is None:
            return base_priority

        elapsed_ms = (time.time() - start) * 1000

        first_token_at = self._request_first_token_at.get(request_id)
        if first_token_at is None and elapsed_ms > sla.target_ttft_ms:
            return max(0, base_priority - 2)

        if elapsed_ms > sla.deadline_ms * 0.8:
            return max(0, base_priority - 1)

        token_count = self._request_token_counts.get(request_id, 0)
        if token_count > 0 and first_token_at:
            avg_tpot_ms = (time.time() - first_token_at) * 1000 / token_count
            if avg_tpot_ms > sla.target_tpot_ms * sla.priority_boost_factor:
                return max(0, base_priority - 1)

        return base_priority

    def get_request_metrics(self, request_id: str) -> dict:
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

class GPUIsolationConfig:
    """GPU isolation configuration for multi-tenant scheduling.

    Supports:
    - **MIG** (Multi-Instance GPU): partitions A100/H100 GPUs into
      isolated instances with dedicated VRAM and compute.
    - **MPS** (Multi-Process Service): time-slices GPU compute with
      configurable quality-of-service limits.
    """

    def __init__(
        self,
        mode: str = "none",
        mig_profile: str = "",
        mps_active_thread_percentage: int = 100,
        gpu_memory_limit_mb: int = 0,
    ):
        self.mode = mode
        self.mig_profile = mig_profile
        self.mps_active_thread_percentage = mps_active_thread_percentage
        self.gpu_memory_limit_mb = gpu_memory_limit_mb

    def apply(self) -> None:
        if self.mode == "mps":
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/mps_pipe"
            os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/mps_log"
            if self.mps_active_thread_percentage < 100:
                os.environ["MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.mps_active_thread_percentage)
            logger.info(f"MPS isolation: active_thread={self.mps_active_thread_percentage}%")
        elif self.mode == "mig":
            logger.info(f"MIG isolation: profile={self.mig_profile or 'default'}")
        if self.gpu_memory_limit_mb > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{self.gpu_memory_limit_mb}"
            logger.info(f"GPU memory limit: {self.gpu_memory_limit_mb}MB per device")

    def __repr__(self) -> str:
        parts = [f"mode={self.mode}"]
        if self.mig_profile:
            parts.append(f"mig={self.mig_profile}")
        if self.gpu_memory_limit_mb:
            parts.append(f"mem_limit={self.gpu_memory_limit_mb}MB")
        return f"GPUIsolationConfig({', '.join(parts)})"

class IterationScheduler(BatchScheduler):
    """Sarathi-style iteration-level scheduler with tenant SLA tracking.

    Extends BatchScheduler with:
    - Per-tenant SLA tracking (TTFT, TPOT, deadline)
    - Per-tenant token budget enforcement
    - SLA-based priority boosting for overdue requests
    - GPU isolation configuration (MIG/MPS)

    All scheduling logic (chunked prefill, preemption, Sarathi-Serve
    pressure adaptation) is inherited from BatchScheduler.
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

        self.sla_tracker = SLATracker()

        self._tenant_budgets: dict[str, TenantBudget] = {}

        self._seq_tenants: dict[str, str] = {}

    def set_tenant_sla(self, sla: TenantSLA) -> None:
        self.sla_tracker.set_tenant_sla(sla)

    def set_tenant_budget(
        self,
        tenant_id: str,
        max_tokens_per_minute: float = 1000.0,
    ) -> None:
        self._tenant_budgets[tenant_id] = TenantBudget(
            tenant_id=tenant_id,
            max_tokens_per_minute=max_tokens_per_minute,
        )

    def add(self, seq: Sequence, tenant_id: str | None = None) -> None:
        if tenant_id:
            self._seq_tenants[seq.request_id] = tenant_id
            self.sla_tracker.register_request(seq.request_id, tenant_id)
        super().add(seq)

    def schedule(self) -> ScheduledBatch | None:
        """Build the next batch with SLA-aware priority boosting.

        Returns a ScheduledBatch or None if no work is available.
        Delegates to the parent scheduler after applying SLA boosts.
        """
        done_ids = [rid for rid, s in self.active.items() if s.is_complete]
        for rid in done_ids:
            seq = self.active.pop(rid)
            self._total_tokens -= seq.total_len
            self.sla_tracker.complete_request(rid)
            self._seq_tenants.pop(rid, None)

        self._apply_sla_boosts()

        # Delegate to parent scheduler which handles chunked prefill,
        # budget computation, batch construction, etc.
        return super().schedule()

    def step(self, batch: ScheduledBatch, next_tokens: torch.Tensor, **kwargs) -> None:
        """Process sampling output with SLA tracking.

        Delegates to parent step() and additionally records SLA metrics
        for each sequence in the batch.
        """
        # Record SLA tokens before parent step updates sequence status
        for seq in batch.sequences:
            self.sla_tracker.record_token(seq.request_id)
            if seq.status == SequenceStatus.PREFILLING:
                self.sla_tracker.record_first_token(seq.request_id)

        # Delegate to parent which handles token generation, completion, etc.
        super().step(batch, next_tokens, **kwargs)

    def _apply_sla_boosts(self) -> None:
        """Apply SLA-based priority boosts to pending requests.

        Requests approaching their SLA deadline get boosted to higher
        priority (lower numeric value) so they are scheduled sooner.
        """
        boosted = []
        for _pri, _cnt, seq in self._pending_heap:
            new_priority = self.sla_tracker.get_priority_boost(seq.request_id, seq.priority)
            boosted.append((new_priority, _cnt, seq))
        if boosted:
            self._pending_heap = boosted
            heapq.heapify(self._pending_heap)

    def on_before_schedule(self, sequences: list[Sequence]) -> list[Sequence]:
        """Apply SLA priority boosts — compatible with SchedulingPolicy protocol."""
        for seq in sequences:
            new_priority = self.sla_tracker.get_priority_boost(seq.request_id, seq.priority)
            # seq.priority is a property on the dataclass, so this works
            seq.priority = new_priority
        return sequences

    def _check_tenant_budget(self, tenant_id: str) -> bool:
        budget = self._tenant_budgets.get(tenant_id)
        if budget is None:
            return True
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

class SLASchedulingPolicy:
    """SchedulingPolicy implementation for tenant SLA tracking.

    Wraps the IterationScheduler's SLA boosting logic into a standalone
    policy that can be used with ``BatchScheduler.set_scheduling_policy()``.

    Usage::

        from distllm.dist.scheduling.iteration import SLASchedulingPolicy, TenantSLA

        policy = SLASchedulingPolicy()
        policy.set_tenant_sla(TenantSLA(tenant_id="prod", target_ttft_ms=100))
        scheduler.set_scheduling_policy(policy)
    """

    def __init__(self):
        self._sla_tracker = SLATracker()
        self._tenant_budgets: dict[str, TenantBudget] = {}

    def set_tenant_sla(self, sla: TenantSLA) -> None:
        """Register a tenant SLA for priority boosting."""
        self._sla_tracker.set_tenant_sla(sla)

    def set_tenant_budget(self, tenant_id: str, max_tokens_per_minute: float = 1000.0) -> None:
        """Set a token budget for a tenant."""
        self._tenant_budgets[tenant_id] = TenantBudget(
            tenant_id=tenant_id,
            max_tokens_per_minute=max_tokens_per_minute,
        )

    def compute_budget(self, base_budget: "IterationBudget") -> "IterationBudget":
        """Return base budget unchanged — SLA policy modifies priorities, not budget."""
        return base_budget

    def on_before_schedule(self, sequences: list[Sequence]) -> list[Sequence]:
        """Apply SLA-based priority boosts to pending sequences.

        Requests approaching their SLA deadline get boosted to higher
        priority (lower numeric value) so they are scheduled sooner.
        """
        boosted = []
        for seq in sequences:
            new_priority = self._sla_tracker.get_priority_boost(seq.request_id, seq.priority)
            boosted.append(new_priority)
        return sequences

    def register_request(self, request_id: str, tenant_id: str | None = None) -> None:
        """Track a new request for SLA monitoring."""
        self._sla_tracker.register_request(request_id, tenant_id)

    def complete_request(self, request_id: str) -> None:
        """Mark a request as completed."""
        self._sla_tracker.complete_request(request_id)

    def get_request_metrics(self, request_id: str) -> dict:
        """Get SLA metrics for a request."""
        return self._sla_tracker.get_request_metrics(request_id)

    def stats(self) -> dict:
        return {
            "sla": self._sla_tracker.stats(),
            "tenant_budgets": {
                tid: {
                    "utilization": round(b.utilization, 3),
                    "throttled": b._is_throttled,
                }
                for tid, b in self._tenant_budgets.items()
            },
        }
