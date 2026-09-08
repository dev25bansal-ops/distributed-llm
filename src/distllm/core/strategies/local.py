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
from typing import Protocol

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


class LocalStrategy:
    """Local single-model generation strategy."""

    def __init__(self, engine: "InferenceEngine") -> None:
        self._engine = engine

    def generate(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        return self._engine._generate_local(
            prompt, max_new_tokens, temperature, top_p, top_k,
            logit_bias=kwargs.get("logit_bias"),
            stop_tokens=kwargs.get("stop_tokens"),
            constraint=kwargs.get("constraint"),
        )

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        """Yield tokens one at a time from local model.

        Prefills once with ``use_cache=True`` and threads ``past_key_values``
        through subsequent single-token forwards (KV-cache reuse), so each
        decode step costs one O(1)-length forward instead of re-forwarding
        the whole sequence every step.
        """
        engine = self._engine
        input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")
        device = next(engine.local_partitioner.full_model.parameters()).device
        input_ids = input_ids.to(device)

        stop_token_ids: set[int] = set()
        stop_tokens = kwargs.get("stop_tokens")
        if stop_tokens:
            stop_token_ids.update(stop_tokens)
        if engine.tokenizer.eos_token_id is not None:
            stop_token_ids.add(engine.tokenizer.eos_token_id)

        for token_id in engine._iter_local_tokens(
            input_ids,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logit_bias=kwargs.get("logit_bias"),
            constraint=kwargs.get("constraint"),
            stop_token_ids=stop_token_ids,
        ):
            yield engine.tokenizer.decode([token_id], skip_special_tokens=True)


# Backward-compatible alias (tests import the underscore name)
_LocalStrategy = LocalStrategy
