"""Strategy module extracted from inference_engine.py (M-05 decomposition).

The InferenceEngine god class previously welded all five execution
strategies to itself via bidirectional references. Each strategy now lives
in its own module and receives the engine through a narrow context seam.
inference_engine.py re-imports these for backward compatibility, so all
existing call sites and tests continue to work unchanged.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol

from distllm._optional_deps import optional_import

torch = optional_import("torch")

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


class DistributedSpeculativeStrategy:
    """Remote draft + distributed target speculative decoding strategy."""

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine

    def generate(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        return self._engine._generate_distributed_speculative(
            prompt, max_new_tokens, temperature, top_p, top_k,
        )

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        """Yield tokens from distributed speculative decoding one at a time."""
        from distllm.core.distributed_speculative import (
            DistributedSpeculativeDecoder,
            RemoteDraftConfig,
            RemoteDraftModel,
        )

        engine = self._engine
        input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")

        # Bug H-05 fix: unique per-generation id instead of the shared
        # "spec", so concurrent requests stop colliding.
        rid = f"spec-{uuid.uuid4().hex[:12]}"

        def target_fn(tokens, **kw):
            return engine._pipeline.run_pipeline(
                tokens, engine._pipeline.create_node_kv_caches(),
                request_id=kw.get("request_id", rid),
            )

        fleet = getattr(engine, "_draft_fleet", None)
        num_candidates = getattr(engine, "_remote_draft_num_candidates", 5)

        if fleet is not None:
            decoder = DistributedSpeculativeDecoder(
                target_forward=target_fn,
                draft_fleet=fleet,
                num_candidates=num_candidates,
                adaptive=getattr(engine, "_draft_adaptive", False),
                min_candidates=getattr(engine, "_draft_min_candidates", 2),
                max_candidates=getattr(engine, "_draft_max_candidates", 10),
                temperature=temperature,
                top_k=top_k,
            )
            try:
                output_ids = decoder.generate(input_ids, max_new_tokens=max_new_tokens)
                new_ids = output_ids[0, input_ids.shape[1]:]
                for token_id in new_ids:
                    yield engine.tokenizer.decode([token_id.item()], skip_special_tokens=True)
            finally:
                decoder.close()
            return

        raw_key = getattr(engine, "_remote_draft_api_key", "")
        config = RemoteDraftConfig(
            endpoint_url=engine._remote_draft_endpoint,
            model_name=getattr(engine, "_remote_draft_model", ""),
            api_key=raw_key,
            transport=getattr(engine, "_remote_draft_transport", "http"),
            prompt_format=getattr(engine, "_remote_draft_prompt_format", "auto"),
        )
        draft_model = RemoteDraftModel(config)

        decoder = DistributedSpeculativeDecoder(
            target_forward=target_fn,
            draft_model=draft_model,
            num_candidates=num_candidates,
            adaptive=getattr(engine, "_draft_adaptive", False),
            min_candidates=getattr(engine, "_draft_min_candidates", 2),
            max_candidates=getattr(engine, "_draft_max_candidates", 10),
            temperature=temperature,
            top_k=top_k,
        )

        try:
            output_ids = decoder.generate(input_ids, max_new_tokens=max_new_tokens)
            stats = decoder.stats
            if stats.get("total_proposed", 0) > 0:
                logger.info(
                    f"Distributed speculative decoding: "
                    f"acceptance_rate={stats.get('acceptance_rate', 0):.2f}, "
                    f"draft_calls={stats['draft_calls']}, "
                    f"target_calls={stats['target_calls']}"
                )
            new_ids = output_ids[0, input_ids.shape[1]:]
            for token_id in new_ids:
                yield engine.tokenizer.decode([token_id.item()], skip_special_tokens=True)
        finally:
            draft_model.close()


# Backward-compatible alias (tests import the underscore name)
_DistributedSpeculativeStrategy = DistributedSpeculativeStrategy
