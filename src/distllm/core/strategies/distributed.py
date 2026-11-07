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

import torch

if TYPE_CHECKING:
    from distllm.core.inference_engine import InferenceEngine

from distllm.core.spec_verify import accept_token
from distllm.errors import ConfigError


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


class DistributedStrategy:
    """Distributed pipeline generation strategy.

    The stream variant overlaps the next pipeline forward pass with the
    current token sampling using a background thread, improving throughput
    by hiding pipeline latency behind token decoding.
    """

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine

    def generate(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        return self._engine._generate_distributed(
            prompt, max_new_tokens, temperature, top_p, top_k,
            request_id=kwargs.get("request_id"),
            constraint=kwargs.get("constraint"),
        )

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        """Yield tokens one at a time from the distributed pipeline.

        Uses a background thread to overlap the next pipeline execution
        with the current token sampling.  While the caller processes the
        just-yielded token, the pipeline is already computing logits for
        the next step, hiding round-trip latency.
        """
        import concurrent.futures

        engine = self._engine
        if not engine._node_order:
            raise ConfigError(
                "No nodes registered in the pipeline",
                context={"action": "generate_stream"},
            )

        gen_id = kwargs.get("request_id") or str(uuid.uuid4())
        constraint = kwargs.get("constraint")

        input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids.clone()
        node_kv_caches = engine._pipeline.create_node_kv_caches()
        straggler_check_counter = 0
        prompt_len = input_ids.shape[-1]
        total_len = prompt_len + max_new_tokens
        # Pre-allocate buffer to avoid O(n²) torch.cat on each step
        buffer = torch.zeros(1, total_len, dtype=input_ids.dtype, device=input_ids.device)
        buffer[:, :prompt_len] = input_ids
        pos = prompt_len

        with torch.no_grad():
            pool = engine._stream_pool
            # Kick off the first pipeline execution
            future = pool.submit(
                engine._pipeline.run_pipeline, input_ids, node_kv_caches, gen_id,
            )

            for step in range(max_new_tokens):
                # Wait for the pipeline result from the previous submission
                try:
                    logits = future.result()
                    if engine._reputation:
                        for node_id in engine._node_order:
                            engine._reputation.record_success(node_id)
                except Exception:
                    if engine._reputation:
                        for node_id in engine._node_order:
                            engine._reputation.record_failure(node_id)
                    raise

                # Periodic checkpoint
                if engine._recovery_manager is not None and step % 10 == 0:
                    for nid in engine._node_order:
                        engine._recovery_manager.save_checkpoint(
                            request_id=gen_id,
                            kv_cache=node_kv_caches,
                            prompt_tokens=input_ids.flatten().tolist(),
                            generated_tokens=generated_ids[0].tolist(),
                            node_id=nid,
                        )

                straggler_check_counter += 1
                if straggler_check_counter >= 10 and engine._straggler_detector:
                    engine._straggler_detector.check()
                    straggler_check_counter = 0

                logits_slice = logits[:, -1, :]

                if constraint is not None:
                    mask = constraint.get_logits_mask(logits_slice.shape[-1], engine.tokenizer)
                    logits_slice = logits_slice.masked_fill(~mask, float("-inf"))

                next_token = engine._token_gen.sample(
                    logits_slice, temperature=temperature, top_p=top_p, top_k=top_k,
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                token_id = next_token.item()
                buffer[:, pos:pos + 1] = next_token
                generated_ids = buffer[:, :pos + 1]
                pos += 1

                if constraint is not None:
                    token_str = engine.tokenizer.decode([token_id])
                    constraint.update(token_str)

                eos = token_id == engine.tokenizer.eos_token_id

                # Submit NEXT pipeline execution *before* yielding —
                # overlaps pipeline latency with the caller decoding
                # the current token.
                if not eos:
                    step_input = generated_ids[:, -1:]
                    future = pool.submit(
                        engine._pipeline.run_pipeline, step_input, node_kv_caches, gen_id,
                    )

                yield engine.tokenizer.decode([token_id], skip_special_tokens=True)

                if eos:
                    break


# Backward-compatible alias (tests import the underscore name)
_DistributedStrategy = DistributedStrategy
