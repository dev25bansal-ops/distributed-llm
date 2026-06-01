"""Sequence and batch dataclasses for the batch scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import torch


class SequenceStatus(Enum):
    """Lifecycle states for a generation sequence."""

    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DONE = "done"
    FAILED = "failed"
    PREEMPTED = "preempted"


@dataclass
class GenerationConfig:
    """Sampling parameters for text generation."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_new_tokens: int = 256
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class OpenAICompliance:
    """OpenAI-compatible response parameters."""

    include_logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: dict[int, float] = field(default_factory=dict)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    token_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class SchedulingHints:
    """Hints for the batch scheduler."""

    priority: int = 2
    max_latency_ms: float | None = None
    adapter_id: str | None = None


@dataclass
class Sequence:
    """Represents a single generation sequence (one request).

    Fields are grouped by concern into nested config objects:
    - ``generation`` — sampling params (temperature, top_p, top_k, max_new_tokens)
    - ``scheduling`` — priority, max_latency_ms, adapter_id
    - ``openai`` — logprobs, penalties, logit_bias
    """

    request_id: str
    prompt_tokens: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.PENDING
    priority: int = 2
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: list[int] = field(default_factory=list)
    constraint: object | None = None
    prefix_match_len: int = 0
    created_at: float = field(default_factory=time.time)
    adapter_id: str | None = None
    include_logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: dict[int, float] = field(default_factory=dict)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    token_counts: dict[int, int] = field(default_factory=dict)
    max_latency_ms: float | None = None

    # Nested config objects
    generation: GenerationConfig = field(default=None, repr=False)
    scheduling: SchedulingHints = field(default=None, repr=False)
    openai: OpenAICompliance = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.priority, int) or self.priority < 0:
            logger.warning(
                f"Sequence {self.request_id}: priority={self.priority} "
                f"is not a non-negative int, clamping to 0 (critical)"
            )
            self.priority = 0
        elif self.priority > 3:
            logger.debug(
                f"Sequence {self.request_id}: priority={self.priority} "
                f"is outside recommended range 0-3 (will still work)"
            )

        if self.generation is None:
            self.generation = GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_new_tokens=self.max_new_tokens,
                stop_token_ids=list(self.stop_token_ids),
            )
        if self.scheduling is None:
            self.scheduling = SchedulingHints(
                priority=self.priority,
                max_latency_ms=self.max_latency_ms,
                adapter_id=self.adapter_id,
            )
        if self.openai is None:
            self.openai = OpenAICompliance(
                include_logprobs=self.include_logprobs,
                top_logprobs=self.top_logprobs,
                logit_bias=dict(self.logit_bias),
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                token_counts=dict(self.token_counts),
            )

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
    """

    sequences: list[Sequence]
    input_ids: torch.Tensor
    seq_starts: list[int] = field(default_factory=list)
    seq_lengths: list[int] = field(default_factory=list)
    position_offsets: list[int] = field(default_factory=list)
    is_prefill: list[bool] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)
    attention_mask: torch.Tensor | None = None
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
