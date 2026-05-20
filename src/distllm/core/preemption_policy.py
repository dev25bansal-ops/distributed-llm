"""Preemption policy engine for SLA-aware request preemption with checkpoint/resume.

Monitors GPU memory pressure, SLA violations, and queue depth to decide
when to preempt low-priority requests. Saves deep-copied KV state so
preempted sequences can be resumed without data loss.

Inspired by vLLM's preemption policy and Kubernetes eviction policies.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from loguru import logger


class GPUMemoryMonitor:
    """Tracks GPU memory utilization and triggers preemption thresholds."""

    def __init__(self, device: int = 0, warn_threshold: float = 0.85, preempt_threshold: float = 0.92):
        self.device = device
        self.warn_threshold = warn_threshold
        self.preempt_threshold = preempt_threshold
        self._history: list[float] = []

    def get_utilization(self) -> float:
        """Get current GPU memory utilization (0.0 - 1.0)."""
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        util = max(allocated, reserved) / total
        self._history.append(util)
        # Keep last 10 samples for smoothing
        if len(self._history) > 10:
            self._history = self._history[-10:]
        return util

    def get_smoothed_utilization(self) -> float:
        """Get EMA-smoothed utilization."""
        if not self._history:
            return self.get_utilization()
        alpha = 0.3
        ema = self._history[0]
        for val in self._history[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema

    def should_preempt(self) -> bool:
        """Check if GPU memory pressure warrants preemption."""
        return self.get_smoothed_utilization() > self.preempt_threshold


class SLATracker:
    """Tracks per-request SLA compliance and triggers preemption on violations."""

    def __init__(self, max_violations: int = 3, sla_deadline_ms: float = 5000.0):
        self.max_violations = max_violations
        self.sla_deadline_ms = sla_deadline_ms
        self._request_start: dict[str, float] = {}
        self._violations: dict[str, int] = {}
        self._total_violations = 0

    def start_request(self, request_id: str) -> None:
        """Track the start of a request."""
        self._request_start[request_id] = time.monotonic()

    def check_sla(self, request_id: str) -> bool:
        """Check if a request is within SLA deadline.

        Returns:
            True if request is within SLA, False if violated.
        """
        start = self._request_start.get(request_id)
        if start is None:
            return True
        elapsed_ms = (time.monotonic() - start) * 1000
        if elapsed_ms > self.sla_deadline_ms:
            self._violations[request_id] = self._violations.get(request_id, 0) + 1
            self._total_violations += 1
            logger.warning(f"SLA violation for {request_id}: {elapsed_ms:.0f}ms > {self.sla_deadline_ms:.0f}ms")
            return False
        return True

    @property
    def total_violations(self) -> int:
        return self._total_violations

    def get_violation_count(self, request_id: str) -> int:
        return self._violations.get(request_id, 0)

    def complete_request(self, request_id: str) -> None:
        """Mark a request as completed."""
        self._request_start.pop(request_id, None)
        self._violations.pop(request_id, None)


@dataclass
class CheckpointState:
    """Deep-copied state for a preempted sequence."""
    request_id: str
    prompt_tokens: list[int]
    generated_tokens: list[int]
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]]  # Deep-copied KV tensors
    priority: int
    temperature: float
    top_p: float
    top_k: int
    preempted_at: float = field(default_factory=time.time)

    def memory_bytes(self) -> int:
        """Estimate memory usage of the checkpoint."""
        total = 0
        for k, v in self.kv_cache:
            total += k.element_size() * k.numel() + v.element_size() * v.numel()
        return total


class PreemptionPolicy:
    """Decides when to preempt requests based on multiple signals.

    Triggers preemption when:
    - GPU memory > preempt_threshold (default 92%)
    - SLA violations > max_violations
    - Queue depth > max_queue_depth

    Uses deep-copy checkpoint to preserve KV state for resume.
    """

    def __init__(
        self,
        gpu_monitor: GPUMemoryMonitor | None = None,
        sla_tracker: SLATracker | None = None,
        max_queue_depth: int = 100,
        max_checkpoints: int = 10,
        checkpoint_memory_limit_mb: int = 4096,
    ):
        self.gpu_monitor = gpu_monitor or GPUMemoryMonitor()
        self.sla_tracker = sla_tracker or SLATracker()
        self.max_queue_depth = max_queue_depth
        self.max_checkpoints = max_checkpoints
        self.checkpoint_memory_limit_mb = checkpoint_memory_limit_mb
        self._checkpoints: dict[str, CheckpointState] = {}
        self._total_checkpoint_memory = 0

    def should_preempt(
        self,
        pending_count: int,
        min_priority: int = 3,
    ) -> bool:
        """Check if preemption should be triggered.

        Args:
            pending_count: Number of pending requests in the queue.
            min_priority: Minimum priority level to consider for preemption.

        Returns:
            True if preemption should be triggered.
        """
        # GPU memory pressure
        if self.gpu_monitor.should_preempt():
            logger.info(f"Preemption trigger: GPU memory pressure ({self.gpu_monitor.get_smoothed_utilization():.1%})")
            return True

        # SLA violations
        if self.sla_tracker.total_violations > self.sla_tracker.max_violations:
            logger.info(f"Preemption trigger: SLA violations ({self.sla_tracker.total_violations})")
            return True

        # Queue depth
        if pending_count > self.max_queue_depth:
            logger.info(f"Preemption trigger: queue depth ({pending_count} > {self.max_queue_depth})")
            return True

        return False

    def create_checkpoint(self, request_id: str, kv_cache: list, sequence: Any) -> CheckpointState | None:
        """Create a deep-copy checkpoint for a sequence's KV state.

        Args:
            request_id: Request identifier.
            kv_cache: List of (k, v) tensors per layer.
            sequence: Sequence object with metadata.

        Returns:
            CheckpointState or None if checkpoint limit exceeded.
        """
        # Check checkpoint limits
        if len(self._checkpoints) >= self.max_checkpoints:
            logger.warning(f"Preemption: checkpoint limit reached ({self.max_checkpoints})")
            return None

        # Deep-copy KV tensors
        kv_copy = []
        memory = 0
        for k, v in kv_cache:
            k_copy = k.clone()
            v_copy = v.clone()
            kv_copy.append((k_copy, v_copy))
            memory += k_copy.element_size() * k_copy.numel() + v_copy.element_size() * v_copy.numel()

        # Check memory limit
        memory_mb = memory / (1024 * 1024)
        if self._total_checkpoint_memory + memory_mb > self.checkpoint_memory_limit_mb:
            logger.warning(
                f"Preemption: checkpoint memory limit would be exceeded "
                f"({self._total_checkpoint_memory + memory_mb:.0f}MB > {self.checkpoint_memory_limit_mb}MB)"
            )
            return None

        checkpoint = CheckpointState(
            request_id=request_id,
            prompt_tokens=list(sequence.prompt_tokens),
            generated_tokens=list(sequence.generated_tokens),
            kv_cache=kv_copy,
            priority=sequence.priority,
            temperature=sequence.temperature,
            top_p=sequence.top_p,
            top_k=sequence.top_k,
        )

        self._checkpoints[request_id] = checkpoint
        self._total_checkpoint_memory += memory_mb
        logger.info(f"Preemption: checkpointed {request_id} ({memory_mb:.0f}MB)")
        return checkpoint

    def restore_checkpoint(self, request_id: str) -> CheckpointState | None:
        """Restore a checkpoint and remove it from storage.

        Args:
            request_id: Request identifier.

        Returns:
            CheckpointState or None if not found.
        """
        checkpoint = self._checkpoints.pop(request_id, None)
        if checkpoint is not None:
            memory = checkpoint.memory_bytes() / (1024 * 1024)
            self._total_checkpoint_memory -= memory
            logger.info(f"Preemption: restored {request_id} (freed {memory:.0f}MB)")
        return checkpoint

    def get_checkpoint_stats(self) -> dict:
        """Get checkpoint storage statistics."""
        return {
            "checkpoint_count": len(self._checkpoints),
            "total_memory_mb": round(self._total_checkpoint_memory, 1),
            "memory_limit_mb": self.checkpoint_memory_limit_mb,
            "max_checkpoints": self.max_checkpoints,
        }

    def evict_oldest_checkpoint(self) -> str | None:
        """Evict the oldest checkpoint to free memory.

        Returns:
            Evicted request_id or None if no checkpoints.
        """
        if not self._checkpoints:
            return None
        oldest_id = min(self._checkpoints.keys(), key=lambda k: self._checkpoints[k].preempted_at)
        checkpoint = self._checkpoints.pop(oldest_id)
        memory = checkpoint.memory_bytes() / (1024 * 1024)
        self._total_checkpoint_memory -= memory
        logger.info(f"Preemption: evicted oldest checkpoint {oldest_id} ({memory:.0f}MB)")
        return oldest_id
