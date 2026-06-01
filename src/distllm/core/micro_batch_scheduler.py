"""Micro-batching for decode steps to reduce pipeline bubbles.

Instead of processing one token at a time through the pipeline,
micro-batching sends N tokens simultaneously, allowing each pipeline
stage to process multiple tokens in parallel.

This reduces pipeline bubbles from O(stages) to O(stages/N), providing
near-linear throughput improvement for small batch sizes.

Usage::

    micro_batcher = MicroBatchScheduler(
        num_stages=4,
        micro_batch_size=4,
    )
    schedule = micro_batcher.schedule_decode(num_tokens=16)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class MicroBatch:
    """A micro-batch of tokens to process together."""
    batch_id: int
    token_ids: list[int] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)
    stage_id: int = 0
    status: str = "pending"  # pending, in_progress, complete
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class MicroBatchSchedule:
    """Schedule for micro-batched decode."""
    batches: list[MicroBatch] = field(default_factory=list)
    num_stages: int = 0
    total_tokens: int = 0
    estimated_bubble_ratio: float = 0.0


class MicroBatchScheduler:
    """Schedules micro-batches for pipeline-parallel decode.

    Groups consecutive decode tokens into micro-batches that can be
    processed in parallel across pipeline stages. This reduces the
    pipeline bubble ratio from O(S) to O(S/M) where S is the number
    of stages and M is the micro-batch size.

    Args:
        num_stages: Number of pipeline stages.
        micro_batch_size: Number of tokens per micro-batch.
        max_in_flight: Maximum number of in-flight micro-batches.
    """

    def __init__(
        self,
        num_stages: int = 4,
        micro_batch_size: int = 4,
        max_in_flight: int = 8,
    ):
        self._num_stages = num_stages
        self._micro_batch_size = micro_batch_size
        self._max_in_flight = max_in_flight
        self._lock = threading.Lock()
        self._in_flight: list[MicroBatch] = []
        self._completed: list[MicroBatch] = []
        self._stats = {
            "batches_scheduled": 0,
            "tokens_processed": 0,
            "avg_bubble_ratio": 0.0,
        }

    def schedule_decode(self, num_tokens: int) -> MicroBatchSchedule:
        """Schedule micro-batches for a decode phase.

        Args:
            num_tokens: Total number of tokens to decode.

        Returns:
            MicroBatchSchedule with the batch assignments.
        """
        batches = []
        batch_id = 0

        for start in range(0, num_tokens, self._micro_batch_size):
            end = min(start + self._micro_batch_size, num_tokens)
            batch = MicroBatch(
                batch_id=batch_id,
                token_ids=list(range(start, end)),
            )
            batches.append(batch)
            batch_id += 1

        # Estimate bubble ratio
        total_steps = len(batches) + self._num_stages - 1
        useful_steps = len(batches)
        bubble_ratio = 1.0 - (useful_steps / max(total_steps, 1))

        schedule = MicroBatchSchedule(
            batches=batches,
            num_stages=self._num_stages,
            total_tokens=num_tokens,
            estimated_bubble_ratio=bubble_ratio,
        )

        with self._lock:
            self._stats["batches_scheduled"] += len(batches)
            self._stats["tokens_processed"] += num_tokens
            self._stats["avg_bubble_ratio"] = (
                self._stats["avg_bubble_ratio"] * 0.9 + bubble_ratio * 0.1
            )

        return schedule

    def get_next_batch(self) -> MicroBatch | None:
        """Get the next micro-batch to process.

        Returns None if max_in_flight is reached.
        """
        with self._lock:
            if len(self._in_flight) >= self._max_in_flight:
                return None

            # Find next pending batch
            for batch in self._in_flight:
                if batch.status == "pending":
                    batch.status = "in_progress"
                    batch.start_time = time.time()
                    return batch

            return None

    def complete_batch(self, batch_id: int) -> None:
        """Mark a micro-batch as complete."""
        with self._lock:
            for batch in self._in_flight:
                if batch.batch_id == batch_id:
                    batch.status = "complete"
                    batch.end_time = time.time()
                    self._in_flight.remove(batch)
                    self._completed.append(batch)
                    break

    @property
    def micro_batch_size(self) -> int:
        return self._micro_batch_size

    @property
    def in_flight_count(self) -> int:
        with self._lock:
            return len(self._in_flight)

    def stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "in_flight": len(self._in_flight),
                "completed": len(self._completed),
                "micro_batch_size": self._micro_batch_size,
                "num_stages": self._num_stages,
            }
