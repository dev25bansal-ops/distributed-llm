"""Iteration budget for batch scheduling."""

from __future__ import annotations

from dataclasses import dataclass


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
    prefill_slack_ratio: float = 0.3

    @property
    def decode_slots(self) -> int:
        return min(self.max_batch_size, self.max_decode_tokens)
