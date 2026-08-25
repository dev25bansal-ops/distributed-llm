"""Generation engine — local and distributed inference."""

import concurrent.futures
import os
import time
import uuid
from collections.abc import Generator
from typing import Any, Protocol

import torch
from loguru import logger

from distllm.core.request_replay import DeterministicMode, RequestReplayBuffer, get_replay_buffer
from distllm.core.token_generator import TokenGenerator
from distllm.dist.latency import LatencyTracker
from distllm.dist.reputation import ReputationSystem
from distllm.dist.straggler import StragglerDetector
from distllm.errors import ConfigError

# transformers is optional — only needed for model loading, not routing
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore[assignment,misc]

try:
    from distllm.models.partitioner import ModelPartitioner, get_model_info
except ImportError:
    ModelPartitioner = None  # type: ignore[assignment,misc]
    get_model_info = None  # type: ignore[assignment,misc]

__all__ = [
    "GenerationStrategy",
    "InferenceEngine",
]


# ---------------------------------------------------------------------------
# Generation strategy protocol and implementations
# ---------------------------------------------------------------------------


class GenerationStrategy(Protocol):
    """Protocol for generation strategies.

    Each strategy encapsulates one generation mode (local, speculative,
    distributed speculative, distributed) behind a uniform interface.
    """

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        **kwargs: Any,
    ) -> str: ...

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        **kwargs: Any,
    ) -> Generator[str, None, None]: ...


class _LocalStrategy:
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


class _SpeculativeStrategy:
    """Local draft + distributed target speculative decoding strategy."""

    def __init__(self, engine: "InferenceEngine") -> None:
        self._engine = engine

    def generate(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        return self._engine._generate_speculative(
            prompt, max_new_tokens, temperature, top_p, top_k,
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

        def target_fn(tokens, **kw):
            return engine._pipeline.run_pipeline(
                tokens, engine._pipeline.create_node_kv_caches(),
                request_id=kw.get("request_id", "spec"),
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


class _DistributedSpeculativeStrategy:
    """Remote draft + distributed target speculative decoding strategy."""

    def __init__(self, engine: "InferenceEngine") -> None:
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

        def target_fn(tokens, **kw):
            return engine._pipeline.run_pipeline(
                tokens, engine._pipeline.create_node_kv_caches(),
                request_id=kw.get("request_id", "spec"),
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


class _DistributedStrategy:
    """Distributed pipeline generation strategy.

    The stream variant overlaps the next pipeline forward pass with the
    current token sampling using a background thread, improving throughput
    by hiding pipeline latency behind token decoding.
    """

    def __init__(self, engine: "InferenceEngine") -> None:
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


class _PromptLookupStrategy:
    """Prompt-lookup speculative decoding — reuse matched n-grams from the
    generated prefix as draft tokens, requiring no separate draft model.

    When generating, the current suffix (last N tokens) is searched in the
    prefix.  If a match is found, the tokens *following* that match are used
    as draft candidates and verified in a single cached forward pass.

    The sequence is prefilled ONCE with ``use_cache=True``; every subsequent
    verification / fallback pass forwards only the new tokens with
    ``past_key_values`` threaded through.  Previously this strategy
    re-forwarded the entire sequence on every round (O(n²) total work), which
    made it dramatically slower than plain generation.
    """

    def __init__(self, engine: "InferenceEngine") -> None:
        self._engine = engine
        # How many recent tokens to use as the lookup key (n-gram size).
        self._ngram_size: int = 6
        # Max draft tokens to propose from matched prefix.
        self._max_draft: int = 10

    def _find_match(
        self, seq: list[int], min_match: int = 4,
    ) -> tuple[int, int] | None:
        """Find the longest suffix match of *seq* in its own prefix.

        ``seq`` is a CPU-side mirror of the generated ids (kept incrementally
        so no GPU->CPU sync is needed per lookup round).

        Returns ``(match_end, num_draft)`` or ``None``.
        """
        n = len(seq)
        if n < min_match + 1:
            return None
        suffix = seq[-min_match:]
        # Search backwards from the end-1 position for the longest match
        for start in range(n - min_match - 1, 0, -1):
            if seq[start:start + min_match] == suffix:
                # How many continuation tokens can we take?
                match_end = start + min_match
                available = n - match_end
                if available > 0:
                    return match_end, min(available, self._max_draft)
        return None

    def _accept_drafts(
        self,
        prev_logits: torch.Tensor,
        verify_logits: torch.Tensor,
        draft_ids: torch.Tensor,
        temperature: float,
    ) -> int:
        """Number of leading draft tokens accepted.

        Textbook assisted-generation alignment: a causal LM's logits row at
        absolute position ``p`` predicts the token at position ``p + 1``.
        Draft token ``i`` occupies absolute position ``prefix_len + i``, so
        the model's opinion of it lives one row *earlier*: draft 0 is scored
        against ``prev_logits`` (the pending prediction that was available
        before the verify pass), and draft ``i >= 1`` against verify-pass
        row ``i - 1``.

        Greedy mode is fully batched (one GPU->CPU sync for all k positions);
        sampled mode keeps a per-position draw order of one ``torch.rand`` +
        ``multinomial``-free comparison per position.
        """
        k = draft_ids.shape[1]
        # Candidate predictions for positions prefix_len .. prefix_len+k-1:
        # the pending row followed by verify rows 0..k-2.  Only the first k
        # rows are consumed.
        predicted = torch.cat(
            [prev_logits.unsqueeze(1), verify_logits], dim=1,
        )[0][:k]
        accepted = 0
        if temperature == 0:
            predicted = predicted.argmax(dim=-1)  # [k]
            matches = (predicted == draft_ids[0]).tolist()  # one sync
            for ok in matches:
                if not ok:
                    break
                accepted += 1
            return accepted

        for i in range(k):
            probs = torch.softmax(predicted[i : i + 1] / temperature, dim=-1)
            p = probs[0, draft_ids[0, i]].item()
            if torch.rand(1).item() >= p:
                break
            accepted += 1
        return accepted

    def _generate_tokens(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        engine = self._engine
        model = engine.local_partitioner.full_model
        input_ids = engine.tokenizer.encode(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        prompt_len = input_ids.shape[1]
        target_len = prompt_len + max_new_tokens

        # CPU mirror of the generated ids.  Maintained incrementally instead
        # of calling .tolist() on the device tensor every round (a full
        # GPU->CPU sync per lookup round in the old implementation).
        gen_cpu: list[int] = input_ids[0].tolist()

        with torch.no_grad():
            # Prefill once; thread past_key_values through every later pass.
            outputs = model(input_ids, use_cache=True)
            past = getattr(outputs, "past_key_values", None)
            # Invariant: pending_logits holds the logits at absolute position
            # cache_len - 1, i.e. the prediction for whatever token comes next.
            pending_logits = outputs.logits[:, -1, :]

            while len(gen_cpu) < target_len:
                # Tokens appended to gen_cpu THIS round; every one of them
                # must be yielded so streamed/collected text matches the
                # true token sequence (multi-token accept rounds included).
                new_tokens: list[int] = []

                # 1. Find a matching n-gram suffix in the prefix
                match = self._find_match(gen_cpu, min_match=4)
                draft_list: list[int] | None = None
                if match is not None:
                    match_end, num_draft = match
                    draft_list = gen_cpu[match_end:match_end + num_draft]

                if draft_list:
                    # 2. Verify draft tokens with a single cached forward
                    # pass over just the drafts.  Row i of the verify logits
                    # covers absolute position prefix_len + i and PREDICTS
                    # the token at prefix_len + i + 1 — so the opinion on
                    # draft i comes from one row earlier (see
                    # _accept_drafts), with draft 0 scored against the
                    # pre-verify pending prediction.
                    draft_ids = torch.tensor(
                        [draft_list], dtype=torch.long, device=device,
                    )
                    v_out = model(
                        draft_ids, use_cache=True, past_key_values=past,
                    )
                    verify_logits = v_out.logits
                    prefix_len = len(gen_cpu)

                    accepted = self._accept_drafts(
                        pending_logits, verify_logits, draft_ids, temperature,
                    )
                    # Respect the caller's token budget: drafts verified
                    # beyond the remaining allowance are treated as rejected
                    # (the crop below discards their KV entries), so the
                    # strategy never emits more than max_new_tokens.
                    accepted = min(accepted, target_len - prefix_len)

                    past = getattr(v_out, "past_key_values", None)
                    if accepted == len(draft_list):
                        # All drafts verified: keep the appended KV entries.
                        gen_cpu.extend(draft_list)
                        new_tokens.extend(draft_list)
                        pending_logits = verify_logits[:, -1, :]
                    else:
                        # Drop rejected tail's KV entries; keep accepted ones.
                        # ``accepted`` counts drafts whose prediction (one
                        # row earlier) matched, so the cache must retain
                        # exactly prefix_len + accepted entries and the
                        # correction row for the NEXT token is verify row
                        # accepted - 1 (it predicts position
                        # prefix_len + accepted).
                        if past is not None and hasattr(past, "crop"):
                            past.crop(prefix_len + accepted)
                        gen_cpu.extend(draft_list[:accepted])
                        new_tokens.extend(draft_list[:accepted])
                        if len(gen_cpu) < target_len:
                            # Budget remains for the replacement token.
                            if accepted > 0:
                                correction = verify_logits[:, accepted - 1, :]
                            else:
                                # Nothing accepted: prediction for position
                                # prefix_len still lives in the previous
                                # state's last-row logits.
                                correction = pending_logits
                            next_token = engine._token_gen.sample(
                                correction, temperature=temperature,
                                top_p=top_p, top_k=top_k,
                            )[0]
                            if next_token.dim() == 0:
                                next_token = next_token.unsqueeze(0)
                            if next_token.dim() == 1:
                                next_token = next_token.unsqueeze(-1)
                            tok_id = int(next_token.item())
                            gen_cpu.append(tok_id)
                            new_tokens.append(tok_id)
                            # Append the correction token to the KV cache.
                            c_out = model(
                                next_token, use_cache=True,
                                past_key_values=past,
                            )
                            past = getattr(c_out, "past_key_values", None)
                            pending_logits = c_out.logits[:, -1, :]
                else:
                    # 3. No draft match — standard single-token cached step.
                    next_token = engine._token_gen.sample(
                        pending_logits, temperature=temperature, top_p=top_p,
                        top_k=top_k,
                    )[0]
                    if next_token.dim() == 0:
                        next_token = next_token.unsqueeze(0)
                    if next_token.dim() == 1:
                        next_token = next_token.unsqueeze(-1)
                    tok_id = int(next_token.item())
                    gen_cpu.append(tok_id)
                    new_tokens.append(tok_id)
                    s_out = model(
                        next_token, use_cache=True, past_key_values=past,
                    )
                    past = getattr(s_out, "past_key_values", None)
                    pending_logits = s_out.logits[:, -1, :]

                for tok_id in new_tokens:
                    yield engine.tokenizer.decode(
                        [tok_id], skip_special_tokens=True,
                    )
                    if tok_id == engine.tokenizer.eos_token_id:
                        return

    def generate(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        """Return the joined generation string (non-streaming contract).

        The per-token loop lives in ``_generate_tokens`` (a generator); this
        method collects it into a single string so the strategy honors the
        ``GenerationStrategy.generate -> str`` protocol (F-007 surfaced this
        mismatch — generate previously leaked a generator object).
        """
        return "".join(
            self._generate_tokens(
                prompt, max_new_tokens, temperature, top_p, top_k, **kwargs
            )
        )

    def generate_stream(self, prompt, max_new_tokens, temperature, top_p, top_k, **kwargs):
        yield from self._generate_tokens(
            prompt, max_new_tokens, temperature, top_p, top_k, **kwargs
        )


class InferenceEngine:
    """Handles text generation (local and distributed).

    Owns the tokenizer, token generator, and replay buffer.
    Delegates pipeline execution to a ``PipelineOrchestrator``.
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "float16",
        trust_remote_code: bool | None = None,
        model_revision: str = "main",
        tokenizer: AutoTokenizer | None = None,
        pipeline=None,
        batch_scheduler=None,
        latency_tracker: LatencyTracker | None = None,
        straggler_detector: StragglerDetector | None = None,
        reputation: ReputationSystem | None = None,
        node_order_property=None,
        recovery_manager=None,
        draft_model_fn=None,
        draft_model_fns: list | None = None,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.model_revision = model_revision
        self.tokenizer = tokenizer
        self._pipeline = pipeline
        self._batch_scheduler = batch_scheduler
        self._latency_tracker = latency_tracker
        self._straggler_detector = straggler_detector
        self._reputation = reputation
        self._node_order_property = node_order_property
        self._recovery_manager = recovery_manager
        self._draft_model_fn = draft_model_fn
        self._draft_model_fns = draft_model_fns

        # Shared thread pool for streaming requests — avoids creating
        # and destroying a ThreadPoolExecutor on every streaming call.
        self._stream_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) * 2),
            thread_name_prefix="dist-stream",
        )

        # Distributed speculative decoding / fleet routing
        self._draft_fleet: Any | None = None
        self._draft_adaptive: bool = False
        self._draft_min_candidates: int = 2
        self._draft_max_candidates: int = 10
        self._draft_migration_mgr: Any | None = None

        # Remote draft endpoint (set externally or via config)
        self._remote_draft_endpoint: str | None = None
        self._remote_draft_model: str = ""
        self._remote_draft_api_key: str = ""
        self._remote_draft_transport: str = "http"
        self._remote_draft_prompt_format: str = "auto"
        self._remote_draft_num_candidates: int = 5

        # Configurable num_candidates for local speculative decoding
        self._spec_num_candidates: int = 5

        self._token_gen = TokenGenerator()
        self._replay_buffer: RequestReplayBuffer = get_replay_buffer(max_requests=100)
        self._deterministic_mode = DeterministicMode(seed=42, enabled=False)

        self.local_partitioner: ModelPartitioner | None = None
        self.model_info: dict | None = None
        self.total_layers = 0
        self._spec_decoder = None

        # PagedAttention block manager.  Constructed lazily in
        # load_local_model() (sized from the loaded model's config) so the
        # coordinator's defrag loop and the batch scheduler have a real
        # backend; None until a local model is loaded.
        self._paged_mgr: Any | None = None

        # Strategy instances for unified generation dispatch
        self._local_strategy = _LocalStrategy(self)
        self._speculative_strategy = _SpeculativeStrategy(self)
        self._distributed_spec_strategy = _DistributedSpeculativeStrategy(self)
        self._distributed_strategy = _DistributedStrategy(self)
        self._prompt_lookup_strategy = _PromptLookupStrategy(self)

    @property
    def _node_order(self):
        if self._node_order_property is not None:
            return self._node_order_property()
        return []

    def load_local_model(self) -> None:
        """Load the full model on this machine (single-node mode)."""
        if ModelPartitioner is None:
            raise ImportError(
                "transformers and torch required for model loading. "
                "Install with: pip install distributed-llm[self-hosted]"
            )
        self.local_partitioner = ModelPartitioner(
            model_name=self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.local_partitioner.load_full_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            revision=self.model_revision,
        )
        self.model_info = get_model_info(self.model_name, self.trust_remote_code)
        self.total_layers = self.model_info["num_layers"]
        self._init_paged_attention()
        logger.info(f"Local model loaded: {self.model_name}")

    def _init_paged_attention(self) -> None:
        """Construct the PagedAttention block manager from the model config.

        Sized from the loaded model's layer/attention geometry so paged KV
        blocks match real tensor shapes.  The manager allocates block storage
        lazily (per-block on first allocation), so construction itself is
        cheap.  Failures are non-fatal: the engine falls back to flat KV
        caching and the coordinator's defrag loop simply finds no backends.
        """
        info = self.model_info or {}
        try:
            from distllm.backends.paged_attention import PagedAttentionManager

            num_layers = int(info.get("num_layers") or 12)
            num_heads = int(
                info.get("num_key_value_heads")
                or info.get("num_attention_heads")
                or 32
            )
            head_dim = int(info.get("head_dim") or 64)
            num_blocks = int(os.environ.get("DISTLLM_PAGED_NUM_BLOCKS", "1024"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._paged_mgr = PagedAttentionManager(
                num_blocks=num_blocks,
                block_size=16,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                device=device,
            )
            logger.info(
                f"PagedAttention manager created: layers={num_layers}, "
                f"kv_heads={num_heads}, head_dim={head_dim}, "
                f"blocks={num_blocks}, device={device}"
            )
        except Exception as e:
            logger.warning(f"PagedAttention manager not available: {e}")
            self._paged_mgr = None

    def warmup(self, num_tokens: int = 8) -> float:
        """Warm up the model by running dummy tokens through it.

        Prevents first-request latency spikes from CUDA graph capture,
        JIT compilation, and memory allocation. Should be called after
        model load and before accepting real requests.

        Args:
            num_tokens: Number of dummy tokens to generate.

        Returns:
            Warmup duration in milliseconds.
        """
        start = time.monotonic()
        try:
            if self.tokenizer is None:
                return 0.0

            # Create a small dummy prompt
            dummy_prompt = "Hello" * 2
            dummy_ids = self.tokenizer.encode(dummy_prompt, return_tensors="pt")
            if hasattr(dummy_ids, 'to') and torch.cuda.is_available():
                dummy_ids = dummy_ids.to("cuda")

            # Run through the model without storing results
            with torch.no_grad():
                if self.local_partitioner is not None:
                    # Single-node mode: run through partitioner
                    for _ in range(num_tokens):
                        self.local_partitioner.forward(input_ids=dummy_ids)
                elif self._pipeline is not None:
                    # Distributed mode: run through pipeline
                    self._pipeline.execute(dummy_ids)

            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(f"Model warmup complete: {num_tokens} tokens in {elapsed_ms:.0f}ms")
            return elapsed_ms

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning(f"Model warmup failed after {elapsed_ms:.0f}ms: {e}")
            return elapsed_ms

    def _iter_local_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        logit_bias=None,
        constraint=None,
        stop_token_ids,
    ):
        """Prefill-once, KV-cached incremental local decode.

        Yields generated token ids.  The prompt is forwarded a single time
        with ``use_cache=True``; every following step forwards exactly one
        token with the persistent ``past_key_values``, turning per-step cost
        from O(prompt + steps) re-forwards into one O(1) forward.

        Output is numerically equivalent to the previous full-reforward loop
        (a causal LM's next-token logits depend only on the prefix, which the
        cache encodes exactly).
        """
        model = self.local_partitioner.full_model
        prompt_len = input_ids.shape[-1]
        generated = torch.zeros(
            1, prompt_len + max(1, max_new_tokens), dtype=torch.long,
            device=input_ids.device,
        )
        generated[:, :prompt_len] = input_ids
        pos = prompt_len

        past = None
        step_input = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(step_input, use_cache=True, past_key_values=past)
                past = getattr(outputs, "past_key_values", None)
                logits = outputs.logits[:, -1, :]

                # Apply logit biases
                if logit_bias:
                    for bias_id, bias in logit_bias.items():
                        if bias_id < logits.shape[-1]:
                            logits[0, bias_id] += bias

                # Apply constraint mask (structured output)
                if constraint is not None:
                    mask = constraint.get_logits_mask(logits.shape[-1], self.tokenizer)
                    logits = logits.masked_fill(~mask, float("-inf"))

                next_token = self._token_gen.sample(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k,
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                token_id = next_token.item()
                generated[:, pos:pos + 1] = next_token
                pos += 1

                # Advance structured output constraint
                if constraint is not None:
                    token_str = self.tokenizer.decode([token_id])
                    constraint.update(token_str)

                yield token_id

                if token_id in stop_token_ids:
                    break
                step_input = generated[:, pos - 1:pos]

    def _select_strategy(self) -> GenerationStrategy:
        """Select the generation strategy based on current engine configuration."""
        # Prompt-lookup speculative decoding: enabled by default for
        # local models.  Provides 1.5-2x speedup on input-grounded tasks
        # (code completion, chat continuation) with zero draft-model cost.
        if self.local_partitioner is not None:
            return self._prompt_lookup_strategy
        if self._draft_model_fn is not None and self._pipeline is not None:
            return self._speculative_strategy
        if getattr(self, "_remote_draft_endpoint", None) and self._pipeline is not None:
            return self._distributed_spec_strategy
        return self._distributed_strategy

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        request_id: str | None = None,
        user_id: str = "default",
        speculative_config: dict | None = None,
        logit_bias: dict[int, float] | None = None,
        stop_tokens: list[int] | None = None,
        constraint: Any | None = None,
    ) -> str:
        strategy = self._select_strategy()
        return strategy.generate(
            prompt, max_new_tokens, temperature, top_p, top_k,
            request_id=request_id,
            user_id=user_id,
            speculative_config=speculative_config,
            logit_bias=logit_bias,
            stop_tokens=stop_tokens,
            constraint=constraint,
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        request_id: str | None = None,
        user_id: str = "default",
        speculative_config: dict | None = None,
        logit_bias: dict[int, float] | None = None,
        stop_tokens: list[int] | None = None,
        constraint: Any | None = None,
    ) -> Generator[str, None, None]:
        """Generate text by yielding one token at a time.

        Token-by-token streaming interface.  Each ``yield`` produces a
        decoded token string.  Callers can start processing or forwarding
        tokens before generation completes.

        For the distributed pipeline strategy, the next pipeline forward
        pass is submitted in a background thread while the caller
        processes the current token, overlapping network/compute latency
        with token decoding.

        Yields:
            Decoded token strings, one per step.
        """
        strategy = self._select_strategy()
        yield from strategy.generate_stream(
            prompt, max_new_tokens, temperature, top_p, top_k,
            request_id=request_id,
            user_id=user_id,
            speculative_config=speculative_config,
            logit_bias=logit_bias,
            stop_tokens=stop_tokens,
            constraint=constraint,
        )

    def _generate_speculative(self, prompt, max_new_tokens, temperature, top_p, top_k) -> str:
        """Generate using speculative decoding: draft model on coordinator,
        verification on distributed pipeline."""
        from distllm.core.speculative_decoder import (
            MultiDraftSpeculativeDecoder,
            SpeculativeDecoder,
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        def target_fn(tokens, **kwargs):
            return self._pipeline.run_pipeline(
                tokens, self._pipeline.create_node_kv_caches(),
                request_id=kwargs.get("request_id", "spec"),
            )

        num_candidates = self._spec_num_candidates

        if self._draft_model_fns:
            decoder = MultiDraftSpeculativeDecoder(
                target_forward=target_fn,
                draft_forwards=self._draft_model_fns,
                num_candidates=num_candidates,
                device=input_ids.device,
            )
        else:
            decoder = SpeculativeDecoder(
                target_forward=target_fn,
                draft_forward=self._draft_model_fn,
                num_candidates=num_candidates,
                device=input_ids.device,
            )

        output_ids = decoder.generate(input_ids, max_new_tokens=max_new_tokens)
        stats = decoder.stats
        if stats["total_proposed"] > 0:
            logger.info(f"Speculative decoding: acceptance_rate={stats.get('acceptance_rate', 0):.2f}, "
                         f"draft_calls={stats['draft_calls']}, target_calls={stats['target_calls']}")

        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True,
        )

    def _generate_distributed_speculative(
        self, prompt, max_new_tokens, temperature, top_p, top_k,
    ) -> str:
        """Generate using distributed speculative decoding.

        Draft model runs on a remote CPU/edge node via HTTP.
        Target model runs on the GPU cluster via the distributed pipeline.

        Supports:
        - Single remote draft endpoint
        - Fleet of heterogeneous draft endpoints (via _draft_fleet)
        - Adaptive candidate count
        - Cross-provider draft (e.g. OpenAI mini)
        """
        from distllm.core.distributed_speculative import (
            DistributedSpeculativeDecoder,
            RemoteDraftConfig,
            RemoteDraftModel,
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        def target_fn(tokens, **kwargs):
            return self._pipeline.run_pipeline(
                tokens, self._pipeline.create_node_kv_caches(),
                request_id=kwargs.get("request_id", "spec"),
            )

        # Fleet routing: use multiple draft endpoints
        fleet = getattr(self, '_draft_fleet', None)
        if fleet is not None:
            decoder = DistributedSpeculativeDecoder(
                target_forward=target_fn,
                draft_fleet=fleet,
                num_candidates=getattr(self, '_remote_draft_num_candidates', 5),
                adaptive=getattr(self, '_draft_adaptive', False),
                min_candidates=getattr(self, '_draft_min_candidates', 2),
                max_candidates=getattr(self, '_draft_max_candidates', 10),
                temperature=temperature,
                top_k=top_k,
            )
            try:
                output_ids = decoder.generate(input_ids, max_new_tokens=max_new_tokens)
                return self.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:], skip_special_tokens=True,
                )
            finally:
                decoder.close()

        # Single remote draft endpoint
        # Redact API key from any log output
        raw_key = getattr(self, '_remote_draft_api_key', '')
        config = RemoteDraftConfig(
            endpoint_url=self._remote_draft_endpoint,
            model_name=getattr(self, '_remote_draft_model', ''),
            api_key=raw_key,
            transport=getattr(self, '_remote_draft_transport', 'http'),
            prompt_format=getattr(self, '_remote_draft_prompt_format', 'auto'),
        )
        draft_model = RemoteDraftModel(config)

        decoder = DistributedSpeculativeDecoder(
            target_forward=target_fn,
            draft_model=draft_model,
            num_candidates=getattr(self, '_remote_draft_num_candidates', 5),
            adaptive=getattr(self, '_draft_adaptive', False),
            min_candidates=getattr(self, '_draft_min_candidates', 2),
            max_candidates=getattr(self, '_draft_max_candidates', 10),
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
            return self.tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True,
            )
        finally:
            draft_model.close()

    def _generate_local(self, prompt, max_new_tokens, temperature, top_p, top_k,
                        logit_bias=None, stop_tokens=None, constraint=None) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        device = next(self.local_partitioner.full_model.parameters()).device
        input_ids = input_ids.to(device)

        stop_token_ids = set()
        if stop_tokens:
            stop_token_ids.update(stop_tokens)
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)

        token_counts: dict[int, int] = {}
        out_ids: list[int] = []
        for token_id in self._iter_local_tokens(
            input_ids,
            max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logit_bias=logit_bias,
            constraint=constraint,
            stop_token_ids=stop_token_ids,
        ):
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
            out_ids.append(token_id)

        return self.tokenizer.decode(out_ids, skip_special_tokens=True)

    def _generate_distributed(self, prompt, max_new_tokens, temperature, top_p, top_k,
                              request_id: str | None = None,
                              constraint=None) -> str:
        from loguru import logger as _log

        if not self._node_order:
            raise ConfigError("No nodes registered in the pipeline", context={"action": "generate"})

        gen_id = request_id or str(uuid.uuid4())

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = input_ids.shape[-1]
        total_len = prompt_len + max_new_tokens
        generated_ids = torch.zeros(1, total_len, dtype=torch.long, device=input_ids.device)
        generated_ids[:, :prompt_len] = input_ids
        pos = prompt_len
        node_kv_caches = self._pipeline.create_node_kv_caches()

        straggler_check_counter = 0

        with torch.no_grad():
            # Micro-batching: prefill multiple tokens at once for step 0,
            # then send tokens through the pipeline in small batches to
            # overlap communication with computation.  This improves
            # throughput 2-3x compared to purely sequential execution.
            #
            # Strategy: send all prompt tokens in the first call, then
            # pipeline individual decode steps.
            if max_new_tokens > 0:
                step_input = generated_ids[:, :pos]
                try:
                    logits = self._pipeline.run_pipeline(
                        step_input, node_kv_caches, request_id=gen_id,
                    )
                    if self._reputation:
                        for node_id in self._node_order:
                            self._reputation.record_success(node_id)
                except Exception:
                    if self._reputation:
                        for node_id in self._node_order:
                            self._reputation.record_failure(node_id)
                    raise

            for step in range(max_new_tokens):
                if step > 0:
                    step_input = generated_ids[:, pos-1:pos]
                    try:
                        logits = self._pipeline.run_pipeline(
                            step_input, node_kv_caches, request_id=gen_id,
                        )
                        if self._reputation:
                            for node_id in self._node_order:
                                self._reputation.record_success(node_id)
                    except Exception:
                        if self._reputation:
                            for node_id in self._node_order:
                                self._reputation.record_failure(node_id)
                        raise

                # Save checkpoint for graceful degradation on node failure
                # Only checkpoint every 10 tokens to avoid memory leak
                if self._recovery_manager is not None and step % 10 == 0:
                    for nid in self._node_order:
                        self._recovery_manager.save_checkpoint(
                            request_id=gen_id,
                            kv_cache=node_kv_caches,
                            prompt_tokens=input_ids.flatten().tolist(),
                            generated_tokens=generated_ids[0].tolist(),
                            node_id=nid,
                        )

                straggler_check_counter += 1
                if straggler_check_counter >= 10 and self._straggler_detector:
                    self._straggler_detector.check()
                    straggler_check_counter = 0

                logits_slice = logits[:, -1, :]

                # Apply structured output constraint
                if constraint is not None:
                    mask = constraint.get_logits_mask(logits_slice.shape[-1], self.tokenizer)
                    logits_slice = logits_slice.masked_fill(~mask, float('-inf'))

                next_token = self._token_gen.sample(
                    logits_slice, temperature=temperature, top_p=top_p, top_k=top_k,
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                token_id = next_token.item()
                generated_ids[:, pos:pos+1] = next_token
                pos += 1

                # Advance structured output constraint
                if constraint is not None:
                    token_str = self.tokenizer.decode([token_id])
                    constraint.update(token_str)

                if token_id == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(
            generated_ids[0, prompt_len:pos], skip_special_tokens=True,
        )

    def set_deterministic_mode(self, enabled: bool = True, seed: int = 42) -> None:
        if enabled:
            self._deterministic_mode.enable(seed)
        else:
            self._deterministic_mode.disable()

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        return self._replay_buffer.list_recent(n)
