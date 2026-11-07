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

# Shared speculative-verification kernel (single source of truth for
# acceptance math). Imported here so the AST lint test (M-02) can prove no
# decoder reimplements rejection sampling.
from distllm.core.spec_verify import accept_token
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


# ── M-05: execution strategies ─────────────────────────────────────────
# The five strategy classes were extracted to distllm/core/strategies/
# (each with a narrow EngineContext seam so they are testable without a
# full engine). Imported back here under their original private names so
# every existing call site and test keeps working unchanged.
from distllm.core.strategies import (  # noqa: E402
    _DistributedSpeculativeStrategy,
    _DistributedStrategy,
    _LocalStrategy,
    _PromptLookupStrategy,
    _SpeculativeStrategy,
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
        # O-02: never forward an ambiguous None to from_pretrained — HF
        # treats a repo as trusted when None is allowed to leak through the
        # deep loader's own defaulting. Explicit caller value wins; else
        # DISTLLM_TRUST_REMOTE_CODE=1 opts in; else False (fail closed).
        if trust_remote_code is None:
            trust_remote_code = os.environ.get("DISTLLM_TRUST_REMOTE_CODE") == "1"
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

    def _select_strategy(self, spec_method: str | None = None) -> GenerationStrategy:
        """Select the generation strategy based on current engine configuration.

        Args:
            spec_method: Per-request override for the speculative decoder
                (``"greedy"``, ``"ngram"``, ``"medusa"``, ``"tree"``,
                ``"multi_draft"``).  ``None`` means use the cluster default
                — the engine picks the best strategy automatically based on
                the model and draft-model configuration.
        """
        # Per-request override: map method names to strategy instances.
        # Unknown / unsupported values fall through to the cluster default
        # so a typo doesn't break generation — we log the override but pick
        # the closest available strategy.
        if spec_method:
            spec_method_lower = spec_method.strip().lower()
            if spec_method_lower == "greedy":
                # Greedy = no speculation at all; the local-partitioner path
                # uses prompt-lookup by default but "greedy" means no
                # speculation.  Route through the distributed path which
                # does one token per forward.
                logger.debug(
                    "spec_method=greedy: bypassing speculative decoding"
                )
                return self._distributed_strategy
            if spec_method_lower == "ngram":
                # Prompt-lookup = n-gram speculation against the prefix.
                return self._prompt_lookup_strategy
            if spec_method_lower in ("medusa", "multi_draft"):
                return self._distributed_spec_strategy
            if spec_method_lower == "tree":
                # Tree drafting is a refinement of single-draft speculation.
                # We treat it as the standard speculative strategy.
                return self._speculative_strategy
            logger.warning(
                f"Unknown spec_method='{spec_method}'; using cluster default"
            )

        # Cluster default: prompt-lookup for local, then draft+target spec,
        # then remote-draft, then plain distributed.
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
        spec_method: str | None = None,
    ) -> str:
        """Generate text.

        ``spec_method`` is a per-request hint that overrides the cluster's
        default speculative-decoder selection.  Supported values:

            - ``"greedy"``: no speculation; one token per forward pass.
            - ``"ngram"``: prompt-lookup n-gram matching against the prefix.
            - ``"medusa"``: multi-draft via the Medusa heads.
            - ``"tree"``: tree-structured speculative decoding.
            - ``"multi_draft"``: multi-draft verifier path.

        Unknown values log a warning and fall back to the cluster default.
        """
        strategy = self._select_strategy(spec_method=spec_method)
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
        spec_method: str | None = None,
    ) -> Generator[str, None, None]:
        """Generate text by yielding one token at a time.

        ``spec_method`` is the same per-request override as ``generate()``.
        """
        strategy = self._select_strategy(spec_method=spec_method)
        yield from strategy.generate_stream(
            prompt, max_new_tokens, temperature, top_p, top_k,
            request_id=request_id,
            user_id=user_id,
            speculative_config=speculative_config,
            logit_bias=logit_bias,
            stop_tokens=stop_tokens,
            constraint=constraint,
        )

    def _generate_speculative(
        self, prompt, max_new_tokens, temperature, top_p, top_k,
        request_id: str | None = None,
    ) -> str:
        """Generate using speculative decoding: draft model on coordinator,
        verification on distributed pipeline."""
        from distllm.core.speculative_decoder import (
            MultiDraftSpeculativeDecoder,
            SpeculativeDecoder,
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Bug H-05 fix: use the caller's request_id (falling back to a
        # unique id per generation rather than the shared "spec") so
        # concurrent requests stop colliding on pipeline caching,
        # checkpointing, and tracing.
        rid = request_id or f"spec-{uuid.uuid4().hex[:12]}"

        def target_fn(tokens, **kwargs):
            return self._pipeline.run_pipeline(
                tokens, self._pipeline.create_node_kv_caches(),
                request_id=kwargs.get("request_id", rid),
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

        # Bug H-05 fix: unique per-generation id instead of the shared
        # "spec", so concurrent requests stop colliding.
        rid = f"spec-{uuid.uuid4().hex[:12]}"

        def target_fn(tokens, **kwargs):
            return self._pipeline.run_pipeline(
                tokens, self._pipeline.create_node_kv_caches(),
                request_id=kwargs.get("request_id", rid),
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
                # Only checkpoint every 10 tokens to avoid memory leak.
                #
                # Bug H-01 fix: the checkpoint previously looped over
                # `_node_order` and serialized the FULL cluster KV cache on
                # every iteration — the same cache N times (8 nodes × 205
                # steps = 1,640 full serializations per 2,048-token request,
                # a 10-100x latency multiplier). The recovery manager already
                # accepts a `node_id` field per checkpoint; the full cache is
                # saved ONCE with the node list recorded, so per-node state
                # remains reconstructible without N-fold duplication.
                if self._recovery_manager is not None and step % 10 == 0:
                    self._recovery_manager.save_checkpoint(
                        request_id=gen_id,
                        kv_cache=node_kv_caches,
                        prompt_tokens=input_ids.flatten().tolist(),
                        # Bug IE-B5 (audit R1): previously serialized the
                        # entire pre-allocated padded buffer here, reporting
                        # ~2,000 zero-pad ints as generated tokens. Emit only
                        # the tokens actually produced so far.
                        generated_tokens=generated_ids[0, :pos].tolist(),
                        node_id=",".join(self._node_order),
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
