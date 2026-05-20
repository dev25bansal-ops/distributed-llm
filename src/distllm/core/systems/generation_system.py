"""GenerationSystem: batching, sampling, preemption.

Groups: TokenGenerator, BatchScheduler, RequestTracker, SpeculativeDecoder
"""

from typing import Any, Optional

import torch
from loguru import logger


class GenerationSystem:
    """Manages generation: batching, sampling, speculative decoding, preemption.

    Composes TokenGenerator, BatchScheduler, RequestTracker,
    and SpeculativeDecoder into a single interface.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 32768,
        enable_speculative: bool = False,
        enable_preemption: bool = True,
        max_preempted: int = 4,
    ):
        from distllm.core.token_generator import TokenGenerator
        from distllm.core.batch_scheduler import BatchScheduler
        from distllm.core.coordinator_lifecycle import RequestTracker
        from distllm.core.speculative_decoder import SpeculativeDecoder

        self.tokenizer = tokenizer
        self.token_gen = TokenGenerator()
        self.scheduler = BatchScheduler(
            max_batch_size=max_batch_size,
            max_tokens_per_batch=max_tokens_per_batch,
        )

        if enable_preemption:
            self.scheduler.set_max_preempted(max_preempted)

        self.request_tracker = RequestTracker()

        self.spec_decoder = SpeculativeDecoder() if enable_speculative else None

    def add_request(self, sequence: Any) -> None:
        self.scheduler.add(sequence)

    def schedule(self) -> Any:
        return self.scheduler.schedule()

    def step(self, batch: Any, next_tokens: list[int]) -> None:
        self.scheduler.step(batch, next_tokens)

    def sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        constraint: Any = None,
    ) -> list[int]:
        return self.token_gen.sample_tokens(
            logits, temperature, top_p, top_k, constraint,
        )

    def preempt_lowest(self, min_priority: int = 3, kv_cache_state: dict | None = None) -> Any | None:
        return self.scheduler.preempt_lowest(min_priority, kv_cache_state)

    def restore_preempted(self, kv_cache_state: dict | None = None) -> list:
        return self.scheduler.restore_preempted(kv_cache_state)

    def record_request_latency(self, request_id: str, tokens: int, time_ms: float) -> None:
        self.request_tracker.record_result(request_id, tokens, time_ms)

    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[list[int]]:
        return self.request_tracker.wait_for_result(request_id, timeout)

    def stats(self) -> dict:
        stats = self.scheduler.stats()
        stats["preempted"] = self.scheduler.get_preempted_count()
        return stats
