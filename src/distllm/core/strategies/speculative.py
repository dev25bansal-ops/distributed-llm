"""Strategy module extracted from inference_engine.py (M-05 decomposition).

The InferenceEngine god class previously welded all five execution
strategies to itself via bidirectional references. Each strategy now lives
in its own module and receives the engine through a narrow context seam.
inference_engine.py re-imports these for backward compatibility, so all
existing call sites and tests continue to work unchanged.
"""

from __future__ import annotations

import torch
import uuid
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from distllm.core.inference_engine import InferenceEngine

from loguru import logger

from distllm.core.spec_verify import accept_token


class EngineContext(Protocol):
    """Narrow seam the strategies may depend on.

    Any object providing these attributes can host the strategies,
    which makes them testable without a full InferenceEngine.
    """

    tokenizer: object
    local_partitioner: object | None
    _pipeline: object | None
    _stream_pool: object
    _node_order_property: list | None
    _draft_model_fn: object | None
    _draft_model_fns: list | None
    _draft_fleet: object | None
    _draft_adaptive: bool
    _draft_min_candidates: int
    _draft_max_candidates: int
    _spec_num_candidates: int
    _remote_draft_endpoint: str | None
    _remote_draft_api_key: str
    _remote_draft_model: str
    _remote_draft_transport: str
    _remote_draft_prompt_format: str
    _remote_draft_num_candidates: int
    _draft_min_candidates: int

    def _generate_local(self, prompt, max_new_tokens, temperature, top_p, top_k, **kw): ...
    def _iter_local_tokens(self, input_ids, max_new_tokens, **kw): ...
    def _generate_speculative(self, prompt, max_new_tokens, temperature, top_p, top_k, request_id=None): ...
    def _generate_distributed_speculative(self, prompt, max_new_tokens, temperature, top_p, top_k): ...


class SpeculativeStrategy:
    """Local draft + distributed target speculative decoding strategy."""

    def __init__(self, engine: "InferenceEngine") -> None:
        self._engine = engine

    def generate(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        # Bug H-05 fix: forward the caller's request_id (and any other
        # kwargs) so concurrent speculative requests stop colliding under
        # the shared default id "spec" for pipeline caching, checkpointing,
        # and tracing.
        return self._engine._generate_speculative(
            prompt, max_new_tokens, temperature, top_p, top_k,
            request_id=kwargs.get("request_id"),
        )

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        """Yield tokens from speculative decoding one at a time.

        Speculative decoding produces tokens in batches (draft + verify).
        We run the full speculative loop and yield accepted tokens
        incrementally from the result.
        """
        from distllm.core.speculative_decoder import (
            MultiDraftSpeculativeDecoder,
            SpeculativeDecoder,
        )

        engine = self._engine
        input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")
        num_candidates = engine._spec_num_candidates

        # Bug H-05 fix: unique per-generation id instead of the shared
        # "spec", so concurrent requests stop colliding.
        rid = f"spec-{uuid.uuid4().hex[:12]}"

        def target_fn(tokens, **kw):
            return engine._pipeline.run_pipeline(
                tokens, engine._pipeline.create_node_kv_caches(),
                request_id=kw.get("request_id", rid),
            )

        if engine._draft_model_fns:
            decoder = MultiDraftSpeculativeDecoder(
                target_forward=target_fn,
                draft_forwards=engine._draft_model_fns,
                num_candidates=num_candidates,
                device=input_ids.device,
            )
        else:
            decoder = SpeculativeDecoder(
                target_forward=target_fn,
                draft_forward=engine._draft_model_fn,
                num_candidates=num_candidates,
                device=input_ids.device,
            )

        output_ids = decoder.generate(input_ids, max_new_tokens=max_new_tokens)
        stats = decoder.stats
        if stats["total_proposed"] > 0:
            logger.info(
                f"Speculative decoding: acceptance_rate={stats.get('acceptance_rate', 0):.2f}, "
                f"draft_calls={stats['draft_calls']}, target_calls={stats['target_calls']}"
            )

        new_ids = output_ids[0, input_ids.shape[1]:]
        for token_id in new_ids:
            yield engine.tokenizer.decode([token_id.item()], skip_special_tokens=True)


# Backward-compatible alias (tests import the underscore name)
_SpeculativeStrategy = SpeculativeStrategy
