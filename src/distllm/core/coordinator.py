"""Coordinator facade for distributed LLM inference.

This is a backward-compatible facade that composes 4 specialized components:
- ResourceManager: node lifecycle, health, circuit breaking
- CacheManager: prefix cache, KV cache, chunked prefill
- TokenGenerator: token sampling and generation
- PipelineOrchestrator: node topology and layer routing

All original constructor parameters and public methods are preserved.
"""

import argparse
import torch
from transformers import AutoTokenizer
from loguru import logger
import uuid
import time
import asyncio
import concurrent.futures
import threading
from typing import List, Dict, Optional, Tuple, Callable
from contextvars import ContextVar

# Context variable for per-request isolation of generation parameters
_current_request_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "current_request_id", default=None
)

from distllm.models.partitioner import ModelPartitioner, partition_model_across_nodes, get_model_info
from distllm.communication.grpc import CoordinatorService, GRPCServer, NodeClient
from distllm.core.kv_cache import KVCache
from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus, ScheduledBatch
from distllm.core.structured_output import JSONSchemaConstraint
from distllm.core.chunked_prefill import ChunkState, maybe_chunk
from distllm.config.loader import NodeRole
from distllm.config.settings import DistLLMSettings
from distllm.communication.grpc import set_debug_mode
from distllm.errors.types import (
    ConfigValidationError,
    NodeError,
    NodeUnreachableError,
    BatchError,
)

from distllm.core.resource_manager import ResourceManager, CircuitBreakerConfig
from distllm.core.cache_manager import CacheManager
from distllm.core.token_generator import TokenGenerator
from distllm.core.pipeline_orchestrator import PipelineOrchestrator
from distllm.core.speculative_decoder import SpeculativeDecoder
from distllm.core.model_registry import ModelRegistry
from distllm.core.latency_tracker import LatencyTracker
from distllm.core.rebalancer import Rebalancer
from distllm.core.cache_persistence import CachePersistenceManager
from distllm.core.cache_warming import CacheWarmer


class Coordinator:
    """Orchestrates distributed inference across worker nodes.

    This is a backward-compatible facade that delegates to 4 specialized components.
    """

    def __init__(
        self,
        model_name: str,
        port: int = 50050,
        dtype: str = "float16",
        trust_remote_code: Optional[bool] = None,
        max_batch_size: int = 1,
        max_tokens_per_batch: int = 4096,
        prefix_cache_enabled: bool = False,
        prefix_cache_max_entries: int = 1024,
        prefix_cache_min_prefix_len: int = 16,
        chunked_prefill_enabled: bool = True,
        chunked_prefill_chunk_size: int = 512,
        quantization_config=None,
        speculative_config=None,
        lora_config=None,
        metrics_exporter=None,
        multi_model_config=None,
        rebalancer_config=None,
        cache_persistence_config=None,
        gossip_config=None,
        moe_config=None,
        discovery_mode: str = "static",
    ):
        self.model_name = model_name
        self.port = port
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.quantization_config = quantization_config
        self.metrics_exporter = metrics_exporter
        self.discovery_mode = discovery_mode

        # Component: ResourceManager (health, circuit breaker, lifecycle)
        self._resource_mgr = ResourceManager()

        # Component: CacheManager (prefix cache, chunked prefill)
        self._cache_mgr = CacheManager(
            prefix_cache_enabled=prefix_cache_enabled,
            prefix_cache_max_entries=prefix_cache_max_entries,
            prefix_cache_min_prefix_len=prefix_cache_min_prefix_len,
            chunked_prefill_enabled=chunked_prefill_enabled,
            chunked_prefill_chunk_size=chunked_prefill_chunk_size,
        )

        # Component: PipelineOrchestrator (node topology, distributed pipeline)
        self._pipeline = PipelineOrchestrator(
            resource_mgr=self._resource_mgr,
        )

        # Component: TokenGenerator (sampling)
        self._token_gen = TokenGenerator()

        # Shared state
        self.tokenizer = None
        self.server = None
        self.model_info = None
        self.total_layers = 0
        self.local_partitioner = None

        # Adapter manager for LoRA
        self.adapter_manager = None
        if lora_config and lora_config.enabled:
            from distllm.models.adapter import AdapterManager
            self.adapter_manager = AdapterManager()
            self._lora_adapters_config = lora_config.adapters

        # Speculative decoding
        self.draft_model = None
        self.num_assistant_tokens = 5
        self._spec_decoder: Optional[SpeculativeDecoder] = None
        if speculative_config and speculative_config.draft_model:
            self._draft_model_name = speculative_config.draft_model
            self.num_assistant_tokens = speculative_config.num_assistant_tokens
            self._spec_decoder = SpeculativeDecoder(
                num_assistant_tokens=self.num_assistant_tokens,
            )
        else:
            self._draft_model_name = None

        # Batch scheduler
        self.scheduler: Optional[BatchScheduler] = None
        if max_batch_size > 1:
            # Pass model_info if available for model-aware batch sizing
            model_info = getattr(self, "model_info", None)
            self.scheduler = BatchScheduler(
                max_batch_size=max_batch_size,
                max_tokens_per_batch=max_tokens_per_batch,
                model_info=model_info,
            )

        # Chunked prefill config (alias for backward compat)
        self.chunked_prefill_enabled = self._cache_mgr.chunked_prefill_enabled
        self.chunked_prefill_chunk_size = self._cache_mgr.chunked_prefill_chunk_size

        # Request completion tracking
        self._request_results: Dict[str, str] = {}
        self._request_events: Dict[str, threading.Event] = {}
        self._request_lock = threading.Lock()
        self._shutting_down = False  # Graceful shutdown flag

        # Metrics tracking
        self._metrics: Dict[str, float] = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "total_generation_time": 0.0,
            "errors": 0,
            "node_failures": 0,
        }
        self._metrics_lock = threading.Lock()

        # Multi-model registry
        self._model_registry: Optional[ModelRegistry] = None
        if multi_model_config and multi_model_config.enabled:
            self._model_registry = ModelRegistry(max_models=multi_model_config.max_models)
            self._model_registry._default_model = (
                multi_model_config.default_model or model_name
            )
            self._model_registry.register(model_name, model_name, 0)
            for name, path in multi_model_config.models.items():
                self._model_registry.register(name, path, 0)

        # Dynamic rebalancing
        self._latency_tracker: Optional[LatencyTracker] = None
        self._rebalancer: Optional[Rebalancer] = None
        self._rebalancer_task: Optional[threading.Thread] = None
        if rebalancer_config and rebalancer_config.enabled:
            self._latency_tracker = LatencyTracker()
            self._pipeline.set_latency_tracker(self._latency_tracker)
            self._rebalancer = Rebalancer(self._latency_tracker, rebalancer_config)

        # Cache persistence
        self._cache_persistence: Optional[CachePersistenceManager] = None
        if cache_persistence_config and cache_persistence_config.enabled:
            self._cache_persistence = CachePersistenceManager(cache_persistence_config)
            self._cache_mgr.persistence_manager = self._cache_persistence

        # P2P KV cache gossip
        self._cache_index = None
        self._gossip_protocol = None
        self._gossip_client = None
        self._gossip_loop_task: Optional[threading.Thread] = None
        if gossip_config and gossip_config.enabled:
            from distllm.core.cache_index import CacheIndex
            from distllm.core.gossip_protocol import GossipProtocol
            self._cache_index = CacheIndex()
            self._gossip_protocol = GossipProtocol(
                node_id=f"coordinator-{self.model_name}",
                max_peers=gossip_config.max_peers,
                cache_ttl=gossip_config.cache_ttl,
            )
            # Wire gossip to cache manager
            self._cache_mgr.cache_index = self._cache_index
            self._cache_mgr.gossip_protocol = self._gossip_protocol

        # Distributed MoE
        self._expert_registry = None
        self._moe_orchestrator = None
        if moe_config and moe_config.enabled:
            from distllm.core.expert_registry import ExpertRegistry
            from distllm.core.moe_orchestrator import MoEOrchestrator
            self._expert_registry = ExpertRegistry()
            self._moe_orchestrator = MoEOrchestrator(expert_registry=self._expert_registry)

        # Streaming parameter updates
        from distllm.core.param_update_channel import ParamUpdateChannel
        self._param_update_channel = ParamUpdateChannel()

        # Cross-cluster federation
        from distllm.core.cluster_topology import FederationManager, CrossClusterLatencyMonitor
        self._federation_manager = FederationManager()
        self._latency_monitor = CrossClusterLatencyMonitor(self._federation_manager)

    # -- Property aliases for backward compat --

    @property
    def nodes(self) -> Dict[str, object]:
        return self._pipeline.nodes

    @nodes.setter
    def nodes(self, value: Dict[str, object]):
        self._pipeline.nodes = value

    @property
    def node_order(self) -> List[str]:
        return self._pipeline.node_order

    @node_order.setter
    def node_order(self, value: List[str]):
        self._pipeline.node_order = value

    @property
    def prefill_nodes(self) -> Dict[str, object]:
        return self._pipeline.prefill_nodes

    @prefill_nodes.setter
    def prefill_nodes(self, value: Dict[str, object]):
        self._pipeline.prefill_nodes = value

    @property
    def decode_nodes(self) -> Dict[str, object]:
        return self._pipeline.decode_nodes

    @decode_nodes.setter
    def decode_nodes(self, value: Dict[str, object]):
        self._pipeline.decode_nodes = value

    @property
    def prefix_cache(self):
        return self._cache_mgr.prefix_cache

    @prefix_cache.setter
    def prefix_cache(self, value):
        self._cache_mgr.prefix_cache = value

    # -- Backward-compat for circuit breaker internals (used by tests) --

    @property
    def _node_failure_counts(self) -> Dict[str, int]:
        return self._resource_mgr._node_failure_counts

    @_node_failure_counts.setter
    def _node_failure_counts(self, value: Dict[str, int]):
        self._resource_mgr._node_failure_counts = value

    @property
    def _node_recovery_time(self) -> Dict[str, float]:
        return self._resource_mgr._node_recovery_time

    @_node_recovery_time.setter
    def _node_recovery_time(self, value: Dict[str, float]):
        self._resource_mgr._node_recovery_time = value

    @property
    def _node_circuit_breaker_threshold(self) -> int:
        return self._resource_mgr.cb_config.threshold

    @_node_circuit_breaker_threshold.setter
    def _node_circuit_breaker_threshold(self, value: int):
        self._resource_mgr.cb_config.threshold = value

    @property
    def _node_base_retry_delay(self) -> float:
        return self._resource_mgr.cb_config.base_delay

    @_node_base_retry_delay.setter
    def _node_base_retry_delay(self, value: float):
        self._resource_mgr.cb_config.base_delay = value

    @property
    def _node_max_retry_delay(self) -> float:
        return self._resource_mgr.cb_config.max_delay

    @_node_max_retry_delay.setter
    def _node_max_retry_delay(self, value: float):
        self._resource_mgr.cb_config.max_delay = value

    @property
    def metrics(self) -> Dict[str, float]:
        return self._metrics

    # -- Multi-Model Serving --

    def register_model(self, name: str, path: str, total_layers: int) -> ModelEntry:
        """Register an additional model."""
        from distllm.core.model_registry import ModelEntry
        if self._model_registry is None:
            self._model_registry = ModelRegistry()
        return self._model_registry.register(name, path, total_layers)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        if self._model_registry is None:
            return [self.model_name]
        return [m.name for m in self._model_registry.list_models()]

    def get_model_name(self, requested: Optional[str] = None) -> str:
        """Resolve model name: requested > registry default > self.model_name."""
        if requested and self._model_registry and self._model_registry.is_registered(requested):
            return requested
        if self._model_registry and self._model_registry.default_model:
            return self._model_registry.default_model
        return self.model_name

    def warm_cache(self, prompts: List[str]) -> int:
        """Warm caches by running prompts through the pipeline."""
        warmer = CacheWarmer()
        return warmer.warm(prompts, self)

    def _gossip_loop(self):
        """Background daemon that runs periodic gossip rounds."""
        import time
        interval = 10.0  # default, could be read from gossip_config
        while True:
            try:
                time.sleep(interval)
                if self._cache_mgr is not None:
                    discovered = self._cache_mgr.sync_with_peers()
                    if discovered > 0:
                        logger.debug(f"Gossip round: discovered {discovered} new cache entries")
            except Exception:
                logger.debug("Gossip round error (non-fatal)")

    def register_expert_on_node(self, node_id: str, expert_ids: List[int], layer_idx: int = 0):
        """Register experts on a node in the expert registry.

        Args:
            node_id: Node identifier.
            expert_ids: List of expert IDs hosted by this node.
            layer_idx: Layer index the experts belong to.
        """
        if self._expert_registry is None:
            return
        for eid in expert_ids:
            self._expert_registry.register_expert(eid, node_id, layer_idx)
        logger.info(f"Registered experts {expert_ids} on {node_id}")

    def moe_forward(self, hidden_states: torch.Tensor, moe_router) -> torch.Tensor:
        """Execute MoE forward pass via distributed expert orchestration.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].
            moe_router: MoERouter instance.

        Returns:
            Aggregated expert output tensor.
        """
        if self._moe_orchestrator is None:
            raise RuntimeError("MoE orchestrator not initialized")
        # Build node_clients from registered nodes
        node_clients = {}
        for node_id in self._pipeline.nodes:
            node_clients[node_id] = self._pipeline.nodes[node_id]
        return self._moe_orchestrator.forward(hidden_states, moe_router, node_clients)

    # -- Metrics --

    def record_metric(self, metric_name: str, value: float):
        """Record a metric value (thread-safe)."""
        with self._metrics_lock:
            if metric_name in self._metrics:
                if isinstance(self._metrics[metric_name], (int, float)):
                    self._metrics[metric_name] += value
                else:
                    self._metrics[metric_name] = value
            else:
                self._metrics[metric_name] = value

    def get_metrics(self) -> dict:
        """Get current metrics in Prometheus-compatible format (thread-safe)."""
        with self._metrics_lock:
            avg_tokens_per_sec = 0.0
            if self._metrics["total_generation_time"] > 0:
                avg_tokens_per_sec = self._metrics["total_tokens_generated"] / self._metrics["total_generation_time"]

            return {
                "distllm_requests_total": self._metrics["total_requests"],
                "distllm_tokens_generated_total": self._metrics["total_tokens_generated"],
                "distllm_generation_time_seconds_total": round(self._metrics["total_generation_time"], 3),
                "distllm_errors_total": self._metrics["errors"],
                "distllm_node_failures_total": self._metrics["node_failures"],
                "distllm_avg_tokens_per_second": round(avg_tokens_per_sec, 2),
            }

    # -- Delegation to ResourceManager --

    def _check_circuit_breaker(self, node_id: str) -> bool:
        return self._resource_mgr.check_circuit_breaker(node_id)

    def _record_node_success(self, node_id: str):
        self._resource_mgr.record_success(node_id)

    def _record_node_failure(self, node_id: str):
        self._resource_mgr.record_failure(node_id)
        # Sync to coordinator metrics for backward compat
        with self._metrics_lock:
            self._metrics["node_failures"] += 1
            self._metrics["errors"] += 1

    # -- Node Registration (delegates to PipelineOrchestrator) --

    def auto_setup(self, nodes_config: List[Dict]) -> None:
        """Automatically partition model and assign layers to nodes."""
        logger.info(f"Auto-setup: partitioning {self.model_name} across {len(nodes_config)} nodes")

        self.model_info = get_model_info(self.model_name, self.trust_remote_code)
        self.total_layers = self.model_info["num_layers"]
        self._pipeline.total_layers = self.total_layers
        logger.info(f"Model has {self.total_layers} layers")

        assignments = partition_model_across_nodes(self.model_name, len(nodes_config), self.trust_remote_code)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

        for i, config in enumerate(nodes_config):
            start, end = assignments[i]
            node_id = config.get("node_id", f"node_{i}")

            self._pipeline.register_node(
                node_id=node_id,
                host=config.get("host", "localhost"),
                port=config.get("port", 50051 + i),
                start_layer=start,
                end_layer=end,
            )

            logger.info(f"Assigned {node_id}: layers {start}-{end}")

    def manual_register(self, node_id: str, host: str, port: int, start_layer: int, end_layer: int, total_layers: Optional[int] = None, role: NodeRole = NodeRole.AUTO, expert_ids: Optional[List[int]] = None, cluster_id: str = "default"):
        """Manually register a node."""
        if total_layers:
            self.total_layers = total_layers
            self._pipeline.total_layers = total_layers

        if self.total_layers is None:
            if self.model_info is None:
                self.model_info = get_model_info(self.model_name, self.trust_remote_code)
            self.total_layers = self.model_info["num_layers"]
            self._pipeline.total_layers = self.total_layers

        self._pipeline.register_node(
            node_id=node_id,
            host=host,
            port=port,
            start_layer=start_layer,
            end_layer=end_layer,
            role=role,
            expert_ids=expert_ids,
            cluster_id=cluster_id,
        )

        # Register with federation manager
        if self._federation_manager is not None:
            self._federation_manager.register_node(node_id, cluster_id)

        # Register experts on this node if MoE is enabled
        if expert_ids and self._expert_registry is not None:
            for eid in expert_ids:
                self._expert_registry.register_expert(eid, node_id)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

        if self.model_info is None:
            self.model_info = get_model_info(self.model_name, self.trust_remote_code)
            if total_layers is None:
                self.total_layers = self.model_info["num_layers"]

        logger.info(f"Registered {node_id}: layers {start_layer}-{end_layer}")

    def _validate_layer_assignment(self, node_id: str, start_layer: int, end_layer: int):
        self._pipeline.validate_layer_assignment(node_id, start_layer, end_layer)

    # -- Token Sampling (delegates to TokenGenerator) --

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
        self._token_gen.tokenizer = self.tokenizer
        # Check for dynamic param updates using context variable
        request_id = _current_request_id_ctx.get()
        if request_id is not None:
            params = self._param_update_channel.get(request_id)
            if params is not None:
                temperature = params.temperature
                top_p = params.top_p
                top_k = params.top_k
        return self._token_gen.sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)

    def _sample_batch(self, logits: torch.Tensor, batch: ScheduledBatch) -> torch.Tensor:
        self._token_gen.tokenizer = self.tokenizer
        # Apply param updates per sequence
        for seq in batch.sequences:
            params = self._param_update_channel.get(seq.request_id)
            if params is not None:
                seq.temperature = params.temperature
                seq.top_p = params.top_p
                seq.top_k = params.top_k
        return self._token_gen.sample_batch(logits, batch.sequences, tokenizer=self.tokenizer)

    # -- Generation --

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0, request_id: Optional[str] = None) -> str:
        """Generate text using distributed pipeline parallelism.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling. 0 means disabled.
            request_id: Optional request ID for param updates. Auto-generated if None.

        Returns:
            Generated text.
        """
        if not self.node_order and self.local_partitioner is None:
            raise NodeError("No nodes registered and no local model loaded")

        request_id = request_id or str(uuid.uuid4())
        self._param_update_channel.register(request_id)
        token = _current_request_id_ctx.set(request_id)
        req_log = logger.bind(request_id=request_id, mode="distributed" if self.node_order else "local")

        self.record_metric("total_requests", 1)
        start_time = time.time()

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

            if self.node_order:
                req_log.info(f"Starting distributed generation: {max_new_tokens} tokens max")
                input_ids = input_ids.to("cpu")
                generated_ids = input_ids.clone()

                node_kv_caches: Dict[str, Optional[List]] = {
                    nid: None for nid in self.node_order
                }

                # Check if speculative decoding is available and enabled
                use_speculative = (
                    self.draft_model is not None
                    and self._spec_decoder is not None
                    and self._spec_decoder.is_enabled
                )
                if use_speculative:
                    req_log.info(f"Speculative decoding enabled: {self.num_assistant_tokens} draft tokens")

                step = 0
                while step < max_new_tokens:
                    if step == 0:
                        step_input = generated_ids
                    else:
                        step_input = generated_ids[:, -1:]

                    draft_tokens = None
                    if use_speculative:
                        # Generate draft tokens using the draft model
                        draft_tokens, _ = self._spec_decoder.generate_draft_tokens(
                            self.draft_model, step_input
                        )
                        if is_debug_mode():
                            req_log.debug(f"Draft tokens: {draft_tokens}")

                    logits = self._pipeline.run_pipeline(
                        step_input, node_kv_caches, request_id=request_id,
                        draft_tokens=draft_tokens if use_speculative else None,
                    )

                    if use_speculative and draft_tokens:
                        # Verify draft tokens against target model logits
                        accepted_count, accepted_tokens, next_token = self._spec_decoder.verify_and_accept(
                            draft_tokens, logits, self.tokenizer
                        )
                        # Append accepted tokens
                        for token_id in accepted_tokens:
                            generated_ids = torch.cat([generated_ids, torch.tensor([[token_id]])], dim=1)
                            if next_token == self.tokenizer.eos_token_id:
                                break
                        # If we got a valid next_token after accepted prefix, append it
                        if next_token > 0 and next_token != self.tokenizer.eos_token_id:
                            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]])], dim=1)
                        elif next_token == self.tokenizer.eos_token_id:
                            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]])], dim=1)
                            break
                        # Count tokens generated this step
                        step += accepted_count + (1 if next_token > 0 else 0)
                    else:
                        next_token = self._sample(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
                        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                        step += 1
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                result = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                req_log.info(f"Starting local generation: {max_new_tokens} tokens max")
                model_device = next(self.local_partitioner.full_model.parameters()).device
                input_ids = input_ids.to(model_device)

                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                if self.draft_model is not None:
                    gen_kwargs["assistant_model"] = self.draft_model
                    gen_kwargs["num_assistant_tokens"] = self.num_assistant_tokens
                    req_log.info(f"Speculative decoding enabled with {self.num_assistant_tokens} assistant tokens")

                with torch.no_grad():
                    output = self.local_partitioner.full_model.generate(input_ids, **gen_kwargs)
                result = self.tokenizer.decode(output[0], skip_special_tokens=True)
                tokens_generated = output.shape[1] - input_ids.shape[1]

            elapsed = time.time() - start_time
            self.record_metric("total_tokens_generated", tokens_generated)
            self.record_metric("total_generation_time", elapsed)

            if self.metrics_exporter:
                self.metrics_exporter.tokens_generated.inc(tokens_generated)
                self.metrics_exporter.token_latency.observe(elapsed)
                if elapsed > 0:
                    self.metrics_exporter.tokens_per_second.set(tokens_generated / elapsed)

            req_log.info(f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tok/s)")

            return result

        except Exception as e:
            self.record_metric("errors", 1)
            if self.metrics_exporter:
                self.metrics_exporter.errors_total.labels(type=type(e).__name__).inc()
            req_log.error(f"Generation failed: {e}")
            raise

        finally:
            self._param_update_channel.unregister(request_id)
            _current_request_id_ctx.reset(token)

    def generate_async(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        schema: Optional[dict] = None,
        priority: int = 2,
    ) -> str:
        """Add a request to the batch scheduler (non-blocking)."""
        if self.scheduler is None:
            raise BatchError("Batch scheduler not configured. Use generate() instead.")

        if request_id is None:
            request_id = str(uuid.uuid4())

        self._param_update_channel.register(request_id)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).tolist()

        prefix_match_len = 0
        if self.prefix_cache:
            prefix_match_len, _ = self._cache_mgr.lookup_prefix(input_ids)

        constraint = None
        if schema:
            constraint = JSONSchemaConstraint(schema=schema)

        chunk_state = self._cache_mgr.maybe_chunk(input_ids)

        seq = Sequence(
            request_id=request_id,
            prompt_tokens=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            constraint=constraint,
            prefix_match_len=prefix_match_len,
            priority=priority,
        )
        if chunk_state:
            seq.chunk_state = chunk_state

        if self.tokenizer.eos_token_id is not None:
            seq.stop_token_ids = [self.tokenizer.eos_token_id]

        self.scheduler.add(seq)

        event = threading.Event()
        with self._request_lock:
            self._request_events[request_id] = event

        self.record_metric("total_requests", 1)
        return request_id

    def wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        """Wait for a batched request to complete and return the result."""
        with self._request_lock:
            event = self._request_events.get(request_id)
            if event is None:
                return self._request_results.pop(request_id, "")

        if event.wait(timeout=timeout):
            with self._request_lock:
                return self._request_results.pop(request_id, "")

        with self._request_lock:
            self._request_events.pop(request_id, None)
            self._request_results.pop(request_id, None)
        return ""

    def generate_batch(self, timeout: float = 120.0, max_steps: int = 0) -> None:
        """Run the batch generation loop until all pending requests are complete."""
        if self.scheduler is None:
            raise BatchError("Batch scheduler not configured. Use generate() instead.")

        step = 0
        idle_time = 0.0

        while self.scheduler.has_pending:
            batch = self.scheduler.schedule()
            if batch is None:
                time.sleep(0.01)
                idle_time += 0.01
                if idle_time > timeout:
                    break
                continue

            idle_time = 0.0

            if self.local_partitioner is not None:
                self._generate_local_batch(batch)
            else:
                self._run_distributed_pipeline_batch(batch)

            step += 1
            if max_steps > 0 and step >= max_steps:
                break

        if self.scheduler is not None:
            with self._request_lock:
                for rid, seq in self.scheduler.active.items():
                    if seq.is_complete:
                        result = self.tokenizer.decode(seq.prompt_tokens + seq.generated_tokens, skip_special_tokens=True)
                        self._request_results[rid] = result
                        event = self._request_events.pop(rid, None)
                        if event:
                            event.set()

                for seq in list(self.scheduler.pending_queue):
                    if seq.is_complete:
                        result = self.tokenizer.decode(seq.prompt_tokens + seq.generated_tokens, skip_special_tokens=True)
                        self._request_results[seq.request_id] = result
                        event = self._request_events.pop(seq.request_id, None)
                        if event:
                            event.set()

    def _generate_local_batch(self, batch: ScheduledBatch) -> None:
        """Run a batch through the local model."""
        batch_size = batch.batch_size
        device = next(self.local_partitioner.full_model.parameters()).device

        max_len = batch.max_seq_len
        input_ids_list = []
        for i, seq in enumerate(batch.sequences):
            if batch.is_prefill[i]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
            else:
                tokens = [seq.decode_input_token]
            padded = tokens + [0] * (max_len - len(tokens))
            input_ids_list.append(padded)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        attention_mask = (input_ids != 0).long()

        # Speculative batch decoding: generate draft tokens for all sequences,
        # verify against target model logits in one pass
        if self._spec_decoder and self._spec_decoder.is_enabled and self.draft_model is not None:
            draft_tokens_per_seq = []
            for i, seq in enumerate(batch.sequences):
                seq_input = input_ids[i:i+1]  # [1, seq_len]
                drafts, _ = self._spec_decoder.generate_draft_tokens(
                    self.draft_model, seq_input
                )
                draft_tokens_per_seq.append(drafts)

            with torch.no_grad():
                outputs = self.local_partitioner.full_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # [batch, seq_len, vocab]

            # Verify each sequence's draft tokens
            all_next_tokens = []
            for i, seq in enumerate(batch.sequences):
                drafts = draft_tokens_per_seq[i] if i < len(draft_tokens_per_seq) else []
                _, accepted, next_token = self._spec_decoder.verify_and_accept(
                    drafts, logits[i:i+1], self.tokenizer
                )
                all_next_tokens.append(next_token)

            next_tokens = torch.tensor(all_next_tokens, device=device)
        else:
            with torch.no_grad():
                outputs = self.local_partitioner.full_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[:, -1, :]

            next_tokens = self._sample_batch(logits, batch)

        self.scheduler.step(batch, next_tokens)

    def _run_distributed_pipeline_batch(self, batch: ScheduledBatch) -> None:
        """Run a batch through the distributed pipeline."""
        next_tokens: List[torch.Tensor] = []

        for seq_idx, seq in enumerate(batch.sequences):
            if batch.is_prefill[seq_idx]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
            else:
                tokens = [seq.decode_input_token]

            input_ids = torch.tensor([tokens], dtype=torch.long)

            node_kv_caches: Dict[str, Optional[List]] = {
                nid: None for nid in self.node_order
            }

            logits = self._pipeline.run_pipeline(
                input_ids,
                node_kv_caches,
                request_id=seq.request_id,
            )

            seq_logits = logits[:, -1, :]
            if seq.constraint is not None:
                mask = seq.constraint.get_logits_mask(seq_logits.shape[-1], self.tokenizer)
                seq_logits = seq_logits.masked_fill(~mask, float('-inf'))

            token = self._sample(seq_logits, temperature=seq.temperature, top_p=seq.top_p, top_k=seq.top_k)
            next_tokens.append(token)

        next_tokens_tensor = torch.stack(next_tokens).squeeze(-1)
        self.scheduler.step(batch, next_tokens_tensor)

    # -- Model Loading --

    def load_local_model(self):
        """Load the full model locally (for single-node testing)."""
        logger.info(f"Loading full model locally: {self.model_name}")
        self.local_partitioner = ModelPartitioner(
            model_name=self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
        )
        self.local_partitioner.load_full_model()
        self.tokenizer = self.local_partitioner.tokenizer

        if self.adapter_manager is not None:
            self.adapter_manager.set_base_model(self.local_partitioner.full_model, self.tokenizer)
            if hasattr(self, '_lora_adapters_config') and self._lora_adapters_config:
                for adapter_id, adapter_path in self._lora_adapters_config.items():
                    self.adapter_manager.load_adapter(adapter_id, adapter_path)

        if self._draft_model_name:
            self._load_draft_model()

        logger.info("Full model loaded locally")

    def _load_draft_model(self):
        """Load a smaller draft model for speculative decoding."""
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading draft model: {self._draft_model_name}")
        trust = self.trust_remote_code
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self._draft_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust,
            low_cpu_mem_usage=True,
        )
        self.draft_model.eval()
        logger.info(f"Draft model loaded: {self._draft_model_name}")

    # -- Health Checks --

    def health_check(self) -> dict:
        """Check health of all registered nodes."""
        results = self._resource_mgr.health_check_all(self.nodes)

        for node_id, result in results.items():
            if self.metrics_exporter:
                node = self.nodes[node_id]
                layer_range = f"{node.start_layer}-{node.end_layer}"
                self.metrics_exporter.node_health.labels(node_id, layer_range).set(
                    1 if result.get("healthy") else 0
                )
                if "memory_used" in result:
                    self.metrics_exporter.node_gpu_memory_bytes.labels(node_id).set(result["memory_used"])

        if self.metrics_exporter:
            for node_id in self.node_order:
                is_open = self._check_circuit_breaker(node_id)
                self.metrics_exporter.circuit_breaker_state.labels(target_node=node_id).set(
                    1 if is_open else 0
                )

        return results

    async def health_check_async(self) -> dict:
        """Check health of all registered nodes (async)."""
        results = await self._resource_mgr.health_check_all_async(self.nodes)

        for node_id, result in results.items():
            if self.metrics_exporter:
                node = self.nodes[node_id]
                layer_range = f"{node.start_layer}-{node.end_layer}"
                self.metrics_exporter.node_health.labels(node_id, layer_range).set(
                    1 if result.get("healthy") else 0
                )
                if "memory_used" in result:
                    self.metrics_exporter.node_gpu_memory_bytes.labels(node_id).set(result["memory_used"])

        if self.metrics_exporter:
            for node_id in self.node_order:
                is_open = self._check_circuit_breaker(node_id)
                self.metrics_exporter.circuit_breaker_state.labels(target_node=node_id).set(
                    1 if is_open else 0
                )

        return results

    # -- Server Lifecycle --

    def start(self, blocking: bool = True, on_stop: Optional[Callable] = None):
        """Start the coordinator gRPC server."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

        # If model_info became available, update the scheduler for model-aware batching
        if self.model_info is not None and self.scheduler is not None:
            self.scheduler._model_info = self.model_info
            self.scheduler._use_length_grouping = True

        # Launch gossip loop if enabled
        if self._gossip_protocol is not None:
            self._gossip_loop_task = threading.Thread(
                target=self._gossip_loop, daemon=True, name="gossip-loop"
            )
            self._gossip_loop_task.start()

        servicer = CoordinatorService()
        self.server = GRPCServer(port=self.port, servicer=servicer)
        self.server.start()

        logger.info(f"Coordinator started on port {self.port}")

        if blocking:
            try:
                self.server.wait_for_termination()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()
        else:
            def _wait_and_callback():
                try:
                    self.server.wait_for_termination()
                except KeyboardInterrupt:
                    pass
                finally:
                    if on_stop:
                        on_stop()

            thread = threading.Thread(target=_wait_and_callback, daemon=True)
            thread.start()

        # Start rebalancer loop if enabled
        if self._rebalancer and self._rebalancer._settings.enabled:
            self._rebalancer_task = threading.Thread(target=self._rebalancer_loop, daemon=True)
            self._rebalancer_task.start()

    def _rebalancer_loop(self) -> None:
        """Background loop that checks for stragglers periodically."""
        while True:
            time.sleep(self._rebalancer._settings.check_interval)
            if not self._rebalancer._settings.enabled:
                continue
            should, reason = self._rebalancer.should_rebalance()
            if should:
                stragglers = self._rebalancer.detect_stragglers()
                logger.warning(f"Stragglers detected: {stragglers}")
                all_avg = self._latency_tracker.get_all_avg()
                partition = self._rebalancer.compute_new_partition(self.total_layers, all_avg)
                logger.info(f"Recommended partition: {[(p.node_id, p.start_layer, p.end_layer) for p in partition]}")
                logger.info("NOTE: Partition recommendation is logged for manual approval (v1)")
                self._rebalancer.record_rebalance()

    def wait_for_termination(self):
        """Block until the coordinator server terminates."""
        if self.server:
            try:
                self.server.wait_for_termination()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()

    def stop(self):
        """Stop the coordinator with graceful shutdown."""
        logger.info("Initiating graceful shutdown...")

        # Phase 1: Stop accepting new requests
        self._shutting_down = True
        logger.info("Phase 1: Stopped accepting new requests")

        # Phase 2: Wait for in-flight requests to complete (up to 30s)
        if self._request_events:
            logger.info(f"Phase 2: Waiting for {len(self._request_events)} in-flight requests...")
            for event in self._request_events.values():
                event.wait(timeout=30.0)

        # Phase 3: Persist cache if enabled
        if self._cache_persistence and self._cache_persistence._settings.enabled:
            logger.info("Phase 3: Persisting cache to disk...")
            self._cache_persistence.enforce_disk_limit()

        # Phase 4: Stop gRPC server
        if self.server:
            logger.info("Phase 4: Stopping gRPC server...")
            self.server.stop(grace=10)

        # Phase 5: Close node connections
        logger.info("Phase 5: Closing node connections...")
        self._resource_mgr.close_all(self.nodes)

        # Phase 6: Cleanup request state
        self._request_results.clear()
        self._request_events.clear()

        # Phase 7: Shutdown plugins if loaded
        if hasattr(self, '_plugin_manager') and self._plugin_manager:
            logger.info("Phase 7: Shutting down plugins...")
            self._plugin_manager.shutdown_all()

        logger.info("Graceful shutdown complete")

    async def stop_async(self):
        """Stop the coordinator with graceful shutdown (async)."""
        logger.info("Initiating graceful shutdown (async)...")

        # Phase 1: Stop accepting new requests
        self._shutting_down = True
        logger.info("Phase 1: Stopped accepting new requests")

        # Phase 2: Wait for in-flight requests (up to 30s)
        if self._request_events:
            logger.info(f"Phase 2: Waiting for {len(self._request_events)} in-flight requests...")
            for event in self._request_events.values():
                event.wait(timeout=30.0)

        # Phase 3: Persist cache
        if self._cache_persistence and self._cache_persistence._settings.enabled:
            logger.info("Phase 3: Persisting cache to disk...")
            self._cache_persistence.enforce_disk_limit()

        # Phase 4: Stop gRPC server
        if self.server:
            logger.info("Phase 4: Stopping gRPC server...")
            self.server.stop(grace=10)

        # Phase 5: Close node connections
        logger.info("Phase 5: Closing node connections...")
        await self._resource_mgr.close_all_async(self.nodes)

        # Phase 6: Cleanup request state
        self._request_results.clear()
        self._request_events.clear()

        # Phase 7: Shutdown plugins
        if hasattr(self, '_plugin_manager') and self._plugin_manager:
            logger.info("Phase 7: Shutting down plugins...")
            self._plugin_manager.shutdown_all()

        logger.info("Graceful shutdown complete (async)")

    # -- Async Generation (Phase 5) --

    async def generate_async_v2(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        request_id: Optional[str] = None,
    ) -> str:
        """Properly async text generation.

        Uses asyncio.to_thread() for blocking operations (tokenizer, model inference).
        """
        if not self.node_order and self.local_partitioner is None:
            raise NodeError("No nodes registered and no local model loaded")

        request_id = request_id or str(uuid.uuid4())
        token = _current_request_id_ctx.set(request_id)
        self._param_update_channel.register(request_id)

        self.record_metric("total_requests", 1)
        start_time = time.time()

        try:
            # Tokenize in thread pool
            input_ids = await asyncio.to_thread(
                self.tokenizer.encode, prompt, return_tensors="pt"
            )

            if self.node_order:
                input_ids = input_ids.to("cpu")
                generated_ids = input_ids.clone()

                node_kv_caches: Dict[str, Optional[List]] = {
                    nid: None for nid in self.node_order
                }

                use_speculative = (
                    self.draft_model is not None
                    and self._spec_decoder is not None
                    and self._spec_decoder.is_enabled
                )

                step = 0
                while step < max_new_tokens:
                    step_input = generated_ids if step == 0 else generated_ids[:, -1:]
                    request_id = str(uuid.uuid4())

                    draft_tokens = None
                    if use_speculative:
                        draft_tokens, _ = await asyncio.to_thread(
                            self._spec_decoder.generate_draft_tokens,
                            self.draft_model, step_input,
                        )

                    logits = await self._pipeline.run_pipeline_async(
                        step_input, node_kv_caches, request_id,
                        draft_tokens=draft_tokens if use_speculative else None,
                    )

                    if use_speculative and draft_tokens:
                        accepted_count, accepted_tokens, next_token = await asyncio.to_thread(
                            self._spec_decoder.verify_and_accept,
                            draft_tokens, logits, self.tokenizer,
                        )
                        for token_id in accepted_tokens:
                            generated_ids = torch.cat([generated_ids, torch.tensor([[token_id]])], dim=1)
                        if next_token > 0 and next_token != self.tokenizer.eos_token_id:
                            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]])], dim=1)
                        elif next_token == self.tokenizer.eos_token_id:
                            generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]])], dim=1)
                            break
                        step += accepted_count + (1 if next_token > 0 else 0)
                    else:
                        next_token = self._sample(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
                        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                        step += 1
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                result = await asyncio.to_thread(
                    self.tokenizer.decode, generated_ids[0], skip_special_tokens=True
                )
            else:
                result = await asyncio.to_thread(
                    self._generate_local_sync, prompt, max_new_tokens, temperature, top_p
                )

            elapsed = time.time() - start_time
            tokens_generated = len(self.tokenizer.encode(result)) - len(self.tokenizer.encode(prompt))
            self.record_metric("total_tokens_generated", tokens_generated)
            self.record_metric("total_generation_time", elapsed)

            return result

        except Exception as e:
            self.record_metric("errors", 1)
            raise

        finally:
            self._param_update_channel.unregister(request_id)
            _current_request_id_ctx.reset(token)

    def _generate_local_sync(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ) -> str:
        """Synchronous local generation helper."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        model_device = next(self.local_partitioner.full_model.parameters()).device
        input_ids = input_ids.to(model_device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.draft_model is not None:
            gen_kwargs["assistant_model"] = self.draft_model
            gen_kwargs["num_assistant_tokens"] = self.num_assistant_tokens

        with torch.no_grad():
            output = self.local_partitioner.full_model.generate(input_ids, **gen_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Distributed LLM Coordinator")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=50050)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--nodes", type=str, nargs="+", help="host:port:start:end per node")
    parser.add_argument("--total-layers", type=int, help="Total layers in model")
    parser.add_argument("--local", action="store_true", help="Run full model locally (single-node mode)")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode (requires --local)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code from HuggingFace models")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with tensor shape logging")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration at startup and exit")

    args = parser.parse_args()

    # Optional: validate config and exit
    if args.validate_config:
        DistLLMSettings.validate_startup()
        print("✅ Config validation passed")
        return

    if args.debug:
        set_debug_mode(True)
        logger.info("Debug mode enabled: tensor shape logging active")

    coordinator = Coordinator(
        model_name=args.model,
        port=args.port,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code or None,
    )

    if args.local:
        coordinator.load_local_model()
        if args.chat:
            print(f"Model loaded: {args.model}")
            while True:
                prompt = input("\nPrompt (or 'quit' to exit): ")
                if prompt.lower() in ('quit', 'exit'):
                    break
                result = coordinator.generate(prompt, max_new_tokens=128)
                print(f"\nResult: {result}")
        else:
            coordinator.start()
    else:
        if args.nodes:
            for i, node_str in enumerate(args.nodes):
                parts = node_str.split(":")
                host = parts[0]
                port = int(parts[1])
                start = int(parts[2])
                end = int(parts[3])
                coordinator.manual_register(
                    node_id=f"node_{i}",
                    host=host,
                    port=port,
                    start_layer=start,
                    end_layer=end,
                    total_layers=args.total_layers,
                )

        coordinator.start()


if __name__ == "__main__":
    main()
