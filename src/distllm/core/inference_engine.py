"""Generation engine — local and distributed inference."""

import time
import uuid
from typing import Any

import torch
from loguru import logger
import torch
from transformers import AutoTokenizer

from distllm.errors import ConfigError
from distllm.core.token_generator import TokenGenerator
from distllm.core.request_replay import get_replay_buffer, RequestReplayBuffer, DeterministicMode
from distllm.models.partitioner import ModelPartitioner, get_model_info
from distllm.dist.latency import LatencyTracker
from distllm.dist.reputation import ReputationSystem
from distllm.dist.straggler import StragglerDetector


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

        self._token_gen = TokenGenerator()
        self._replay_buffer: RequestReplayBuffer = get_replay_buffer(max_requests=100)
        self._deterministic_mode = DeterministicMode(seed=42, enabled=False)

        self.local_partitioner: ModelPartitioner | None = None
        self.model_info: dict | None = None
        self.total_layers = 0
        self._spec_decoder = None

    @property
    def _node_order(self):
        if self._node_order_property is not None:
            return self._node_order_property()
        return []

    def load_local_model(self) -> None:
        """Load the full model on this machine (single-node mode)."""
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
        logger.info(f"Local model loaded: {self.model_name}")

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
        gen_id = request_id or str(uuid.uuid4())
        if self.local_partitioner is not None:
            return self._generate_local(
                prompt, max_new_tokens, temperature, top_p, top_k,
                logit_bias=logit_bias, stop_tokens=stop_tokens,
                constraint=constraint,
            )
        if self._draft_model_fn is not None and self._pipeline is not None:
            return self._generate_speculative(prompt, max_new_tokens, temperature, top_p, top_k)
        if getattr(self, '_remote_draft_endpoint', None) and self._pipeline is not None:
            return self._generate_distributed_speculative(prompt, max_new_tokens, temperature, top_p, top_k)
        return self._generate_distributed(
            prompt, max_new_tokens, temperature, top_p, top_k, gen_id,
            constraint=constraint,
        )

    def _generate_speculative(self, prompt, max_new_tokens, temperature, top_p, top_k) -> str:
        """Generate using speculative decoding: draft model on coordinator,
        verification on distributed pipeline."""
        from distllm.core.speculative_decoder import SpeculativeDecoder, MultiDraftSpeculativeDecoder

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        def target_fn(tokens, **kwargs):
            return self._pipeline.run_pipeline(
                tokens, self._pipeline.create_node_kv_caches(),
                request_id=kwargs.get("request_id", "spec"),
            )

        if self._draft_model_fns:
            decoder = MultiDraftSpeculativeDecoder(
                target_forward=target_fn,
                draft_forwards=self._draft_model_fns,
                num_candidates=5,
                device=input_ids.device,
            )
        else:
            decoder = SpeculativeDecoder(
                target_forward=target_fn,
                draft_forward=self._draft_model_fn,
                num_candidates=5,
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
            DistributedSpeculativeDecoder, RemoteDraftModel, RemoteDraftConfig,
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
            return

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
        generated = input_ids

        stop_token_ids = set()
        if stop_tokens:
            stop_token_ids.update(stop_tokens)
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.add(self.tokenizer.eos_token_id)

        token_counts: dict[int, int] = {}

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.local_partitioner.full_model(generated)
                logits = outputs.logits[:, -1, :]

                # Apply logit biases
                if logit_bias:
                    for token_id, bias in logit_bias.items():
                        if token_id < logits.shape[-1]:
                            logits[0, token_id] += bias

                # Apply constraint mask (structured output)
                if constraint is not None:
                    mask = constraint.get_logits_mask(logits.shape[-1], self.tokenizer)
                    logits = logits.masked_fill(~mask, float('-inf'))

                next_token = self._token_gen.sample(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k,
                )[0]
                if next_token.dim() == 0:
                    next_token = next_token.unsqueeze(0)
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(-1)
                token_id = next_token.item()
                token_counts[token_id] = token_counts.get(token_id, 0) + 1
                generated = torch.cat([generated, next_token], dim=-1)

                # Advance structured output constraint
                if constraint is not None:
                    token_str = self.tokenizer.decode([token_id])
                    constraint.update(token_str)

                if token_id in stop_token_ids:
                    break

        return self.tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)

    def _generate_distributed(self, prompt, max_new_tokens, temperature, top_p, top_k,
                              request_id: str | None = None,
                              constraint=None) -> str:
        from loguru import logger as _log

        if not self._node_order:
            raise ConfigError("No nodes registered in the pipeline", context={"action": "generate"})

        gen_id = request_id or str(uuid.uuid4())

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids.clone()
        node_kv_caches = self._pipeline.create_node_kv_caches()

        straggler_check_counter = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                step_t0 = time.monotonic()
                step_input = generated_ids if step == 0 else generated_ids[:, -1:]

                try:
                    logits = self._pipeline.run_pipeline(
                        step_input, node_kv_caches, request_id=gen_id,
                    )
                    if self._reputation:
                        for node_id in self._node_order:
                            self._reputation.record_success(node_id)
                except Exception as e:
                    if self._reputation:
                        for node_id in self._node_order:
                            self._reputation.record_failure(node_id)
                    raise

                # Save checkpoint for graceful degradation on node failure
                if self._recovery_manager is not None:
                    self._recovery_manager.save_checkpoint(
                        request_id=f"req_{step}",
                        kv_cache=node_kv_caches,
                        prompt_tokens=input_ids.flatten().tolist(),
                        generated_tokens=generated_ids[0].tolist(),
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
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Advance structured output constraint
                if constraint is not None:
                    token_str = self.tokenizer.decode([token_id])
                    constraint.update(token_str)

                if token_id == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(
            generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True,
        )

    def set_deterministic_mode(self, enabled: bool = True, seed: int = 42) -> None:
        if enabled:
            self._deterministic_mode.enable(seed)
        else:
            self._deterministic_mode.disable()

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        return self._replay_buffer.list_recent(n)
