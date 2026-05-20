"""Coordinator facade for distributed LLM inference.

Thin facade that delegates to 5 focused services:
- OrchestratorService: pipeline management, node topology, forward execution
- SchedulerService: batch scheduling, priority queue, preemption
- ModelService: model loading, quantization, adapters, hot-swap
- HealthService: health checks, circuit breakers, node recovery
- MetricsService: metrics collection, observability

All original constructor parameters and public methods are preserved.
"""

import argparse
import torch
from transformers import AutoTokenizer
from loguru import logger
import time
import threading
from typing import Any, Callable
from distllm.models.partitioner import get_model_info
from distllm.communication.grpc import CoordinatorService, GRPCServer
from distllm.core.batch_scheduler import BatchScheduler, ScheduledBatch
from distllm.config.loader import NodeRole
from distllm.config.settings import DistLLMSettings
from distllm.communication.grpc import set_debug_mode
from distllm.core.moe_router import MoERouter
from distllm.core.prefix_cache import PrefixCache

from distllm.core.resource_manager import ResourceManager
from distllm.core.cache_manager import CacheManager
from distllm.core.token_generator import TokenGenerator
from distllm.core.pipeline_orchestrator import PipelineOrchestrator
from distllm.core.di import Container
from distllm.core.subsystem_manager import SubsystemManager
from distllm.core.speculative_trainer import ContinuousTrainConfig
from distllm.core.request_replay import RequestReplayBuffer, DeterministicMode, get_replay_buffer
from distllm.core.latency_tracker import LatencyTracker
from distllm.core.rebalancer import Rebalancer
from distllm.core.cache_persistence import CachePersistenceManager
from distllm.core.cache_warming import CacheWarmer
from distllm.core.coordinator_metrics import MetricsManager
from distllm.core.coordinator_model import ModelManager
from distllm.core.coordinator_health import HealthChecker
from distllm.core.coordinator_lifecycle import RequestTracker
from distllm.core.coordinator_nodes import NodeRegistrar
from distllm.core.coordinator_config import CoordinatorConfig
from distllm.core.request_pipeline import RequestPipeline
from distllm.core.request_auditor import RequestAuditor
from distllm.core.prompt_caching_service import PromptCachingService
from distllm.core.graceful_degradation import GracefulDegradation
from distllm.core.adaptive_batching import AdaptiveBatchingEngine
from distllm.core.model_comparator import ModelVersionComparator
from distllm.core.token_streaming_buffer import TokenStreamingBuffer
from distllm.core.request_fingerprinting import RequestFingerprinter
from distllm.core.leaky_bucket_limiter import LeakyBucketRateLimiter
from distllm.core.node_recovery import NodeRecoveryManager, NodeRecoveryPlan

from distllm.core.services import (
    OrchestratorService, SchedulerService, ModelService,
    HealthService, MetricsService,
)


class Coordinator:
    """Orchestrates distributed inference across worker nodes.

    Thin facade that delegates to 5 focused services. All 130+ public methods
    and 44 constructor parameters are preserved for backward compatibility.
    """

    def __init__(
        self,
        model_name: str | None = None,
        port: int = 50050,
        dtype: str = "float16",
        trust_remote_code: bool | None = None,
        max_batch_size: int = 1,
        max_tokens_per_batch: int = 4096,
        prefix_cache_enabled: bool = False,
        prefix_cache_max_entries: int = 1024,
        prefix_cache_min_prefix_len: int = 16,
        radix_tree_cache_enabled: bool = True,
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
        embedding_config=None,
        version_config=None,
        hybrid_parallel_config=None,
        zero_copy_config=None,
        adaptive_precision_config=None,
        predictive_cache_config=None,
        pipeline_schedule_config=None,
        self_optimizing_config=None,
        cuda_graph_config=None,
        compile_config=None,
        slora_config=None,
        rag_config=None,
        agent_config=None,
        disagg_config=None,
        request_auditor_config=None,
        prompt_cache_config=None,
        graceful_degradation_config=None,
        adaptive_batching_config=None,
        request_fingerprinting_config=None,
        leaky_bucket_config=None,
        config: CoordinatorConfig | None = None,
    ):
        if config is not None:
            cfg = config
        else:
            cfg = CoordinatorConfig(
                model_name=model_name or "",
                port=port,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                max_batch_size=max_batch_size,
                max_tokens_per_batch=max_tokens_per_batch,
                max_context_length=8192,
                prefix_cache_enabled=prefix_cache_enabled,
                prefix_cache_max_entries=prefix_cache_max_entries,
                prefix_cache_min_prefix_len=prefix_cache_min_prefix_len,
                radix_tree_cache_enabled=radix_tree_cache_enabled,
                chunked_prefill_enabled=chunked_prefill_enabled,
                chunked_prefill_chunk_size=chunked_prefill_chunk_size,
                quantization_config=quantization_config,
                speculative_config=speculative_config,
                lora_config=lora_config,
                metrics_exporter=metrics_exporter,
                multi_model_config=multi_model_config,
                rebalancer_config=rebalancer_config,
                cache_persistence_config=cache_persistence_config,
                gossip_config=gossip_config,
                moe_config=moe_config,
                discovery_mode=discovery_mode,
                embedding_config=embedding_config,
                version_config=version_config,
                hybrid_parallel_config=hybrid_parallel_config,
                zero_copy_config=zero_copy_config,
                adaptive_precision_config=adaptive_precision_config,
                predictive_cache_config=predictive_cache_config,
                pipeline_schedule_config=pipeline_schedule_config,
                self_optimizing_config=self_optimizing_config,
                cuda_graph_config=cuda_graph_config,
                compile_config=compile_config,
                slora_config=slora_config,
                rag_config=rag_config,
                agent_config=agent_config,
                disagg_config=disagg_config,
                request_auditor_config=request_auditor_config,
                prompt_cache_config=prompt_cache_config,
                graceful_degradation_config=graceful_degradation_config,
                adaptive_batching_config=adaptive_batching_config,
                request_fingerprinting_config=request_fingerprinting_config,
                leaky_bucket_config=leaky_bucket_config,
            )
        self.config = cfg
        self.model_name = cfg.model_name
        self.port = cfg.port
        self.dtype = cfg.dtype
        self.trust_remote_code = cfg.trust_remote_code
        self.quantization_config = cfg.quantization_config
        self.metrics_exporter = cfg.metrics_exporter
        self.discovery_mode = cfg.discovery_mode

        # --- Core Infrastructure ---
        self._container = Container()
        self._subsystems = SubsystemManager()
        self._resource_mgr = ResourceManager()
        self._container.register(ResourceManager, self._resource_mgr)
        self._cache_mgr = CacheManager(
            prefix_cache_enabled=cfg.prefix_cache_enabled,
            prefix_cache_max_entries=cfg.prefix_cache_max_entries,
            prefix_cache_min_prefix_len=cfg.prefix_cache_min_prefix_len,
            radix_tree_cache_enabled=cfg.radix_tree_cache_enabled,
            chunked_prefill_enabled=cfg.chunked_prefill_enabled,
            chunked_prefill_chunk_size=cfg.chunked_prefill_chunk_size,
        )
        self._pipeline = PipelineOrchestrator(resource_mgr=self._resource_mgr)
        self._container.register(PipelineOrchestrator, self._pipeline)
        self._container.register(Container, self._container)
        self._health_checker = HealthChecker(self._resource_mgr, self.metrics_exporter)
        self._node_registrar = NodeRegistrar(
            pipeline=self._pipeline, model_name=self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._token_gen = TokenGenerator()
        self._model_mgr = ModelManager(
            model_name=self.model_name, dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
        )
        self._request_tracker = RequestTracker()
        self._metrics_mgr = MetricsManager()

        # --- Initialize Services ---
        self._orchestrator = OrchestratorService(
            pipeline=self._pipeline, resource_mgr=self._resource_mgr,
            cache_mgr=self._cache_mgr, container=self._container,
            node_registrar=self._node_registrar,
            model_name=self.model_name, trust_remote_code=self.trust_remote_code,
        )
        self._scheduler_svc = SchedulerService(cache_mgr=self._cache_mgr)
        self._model_svc = ModelService(
            model_name=self.model_name, dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
            model_mgr=self._model_mgr, pipeline=self._pipeline,
        )
        self._health_svc = HealthService(
            resource_mgr=self._resource_mgr, health_checker=self._health_checker,
            recovery=None, metrics_exporter=self.metrics_exporter,
        )
        self._metrics_svc = MetricsService(metrics_mgr=self._metrics_mgr)
        self._metrics_svc.set_exporter(self.metrics_exporter)

        # --- Init Pipeline Schedule ---
        self._orchestrator.init_pipeline_schedule(
            cfg.pipeline_schedule_config if config is None else pipeline_schedule_config
        )

        # --- Init Scheduler ---
        self._scheduler_svc.init_scheduler(
            cfg.max_batch_size if config is None else max_batch_size,
            cfg.max_tokens_per_batch if config is None else max_tokens_per_batch,
        )

        # --- Init Model Extensions ---
        self._model_svc.init_adapter(lora_config if config is None else cfg.lora_config)
        self._model_svc.init_speculative(speculative_config if config is None else cfg.speculative_config)
        self._model_svc.init_multi_model(cfg.multi_model_config)
        self._model_svc.init_moe(cfg.moe_config)

        # --- Shared State ---
        self.tokenizer = None
        self.server = None
        self.model_info = None
        self.total_layers = 0
        self.local_partitioner = None
        self.nodes_info = {}
        self._batch_kv_caches: dict[str, dict[str, list | None]] = {}
        self._batch_kv_caches_lock = threading.Lock()
        self._batch_event = threading.Event()
        self._fallback_models: dict[str, list[str]] = {}
        self._replay_buffer: RequestReplayBuffer = get_replay_buffer(max_requests=100)
        self._deterministic_mode = DeterministicMode(seed=42, enabled=False)
        self._running = threading.Event()
        self.chunked_prefill_enabled = self._cache_mgr.chunked_prefill_enabled
        self.chunked_prefill_chunk_size = self._cache_mgr.chunked_prefill_chunk_size

        # --- Extension Configs ---
        self._embedding_config = cfg.embedding_config
        self._version_config = cfg.version_config
        self._init_embedding_config()
        self._init_version_config()
        self._model_svc.init_flash_attention(causal=True, enable_fa2=True)
        self._model_svc.init_plugin_manager(self)
        self._model_svc.init_hybrid_parallel(
            cfg.hybrid_parallel_config, self.total_layers,
            self._model_svc._expert_registry, self._model_svc._moe_orchestrator,
            self._pipeline,
        )
        self._model_svc.init_zero_copy(cfg.zero_copy_config)
        self._model_svc.init_adaptive_precision(cfg.adaptive_precision_config)
        self._model_svc.init_predictive_cache(cfg.predictive_cache_config, self._cache_mgr)
        self._model_svc.init_self_optimizing(cfg.self_optimizing_config, self._apply_tunable_params)
        self._model_svc.init_cuda_graph(cfg.cuda_graph_config)
        self._model_svc.init_compile_support(cfg.compile_config)
        self._model_svc.init_slora(cfg.slora_config)
        self._model_svc.init_rag(cfg.rag_config)
        self._model_svc.init_agent(cfg.agent_config, self.generate)
        self._model_svc.init_disagg(cfg.disagg_config, self)

        # --- Extra Modules ---
        self._model_comparator = ModelVersionComparator()
        self._token_streaming_buffer: TokenStreamingBuffer | None = None
        self._pipeline_composer = None
        self._chat_router = None

        # --- Rebalancer ---
        self._latency_tracker = None
        self._rebalancer = None
        self._rebalancer_task = None
        if rebalancer_config and getattr(rebalancer_config, "enabled", False):
            self._latency_tracker = LatencyTracker()
            self._pipeline.set_latency_tracker(self._latency_tracker)
            self._orchestrator.set_latency_tracker(self._latency_tracker)
            self._rebalancer = Rebalancer(self._latency_tracker, rebalancer_config)
            self._orchestrator.set_rebalancer(self._rebalancer)

        # --- Cache Persistence ---
        self._cache_persistence = None
        if cache_persistence_config and getattr(cache_persistence_config, "enabled", False):
            self._cache_persistence = CachePersistenceManager(cache_persistence_config)
            self._cache_mgr.persistence_manager = self._cache_persistence

        # --- Gossip Protocol ---
        self._cache_index = None
        self._gossip_protocol = None
        self._gossip_client = None
        self._gossip_loop_task = None
        if gossip_config and getattr(gossip_config, "enabled", False):
            from distllm.core.cache_index import CacheIndex
            from distllm.core.gossip_protocol import GossipProtocol, GossipClient
            self._cache_index = CacheIndex()
            self._gossip_protocol = GossipProtocol(
                node_id=f"coordinator-{self.model_name}",
                max_peers=gossip_config.max_peers, cache_ttl=gossip_config.cache_ttl,
            )
            def resolve_peer(peer_id: str):
                nodes_snapshot = dict(self.nodes)
                for node_id, reg in nodes_snapshot.items():
                    if node_id == peer_id or peer_id in node_id:
                        return reg.host, reg.port
                return None
            self._gossip_client = GossipClient(
                node_id=self._gossip_protocol.state.node_id,
                peer_resolver=resolve_peer,
            )
            self._cache_mgr.cache_index = self._cache_index
            self._cache_mgr.gossip_protocol = self._gossip_protocol
            self._cache_mgr.gossip_client = self._gossip_client

        # --- Federation ---
        from distllm.core.param_update_channel import ParamUpdateChannel
        self._param_update_channel = ParamUpdateChannel()
        from distllm.core.federation_router import FederationRouter
        self._federation_router = FederationRouter()
        self._federation_manager = self._federation_router.federation_manager
        self._latency_monitor = self._federation_router.latency_monitor
        self._geo_router = self._federation_router.geo_router
        self._federation_router.attach_registrar(self._node_registrar, expert_registry=self._model_svc._expert_registry)

        # --- Recovery ---
        self._recovery = NodeRecoveryManager()
        self._recovery.set_drain_callback(self._on_recovery_drain)
        self._recovery.set_redistribute_layers_callback(self._on_recovery_redistribute)
        self._recovery.set_recover_sequences_callback(self._on_recovery_recover)
        self._recovery.set_mark_dead_callback(self._on_recovery_mark_dead)
        self._resource_mgr.set_node_failure_callback(self._on_resource_mgr_failure)
        self._health_svc = HealthService(
            resource_mgr=self._resource_mgr, health_checker=self._health_checker,
            recovery=self._recovery, metrics_exporter=self.metrics_exporter,
        )

        # --- New Modules ---
        self._request_auditor = None
        self._prompt_cache_service = None
        self._graceful_degradation = None
        self._adaptive_batching = None
        self._request_fingerprinter = None
        self._rate_limiter = None
        self._pipeline_runner = RequestPipeline(self)
        if cfg.request_auditor_config:
            aud_cfg = cfg.request_auditor_config if not isinstance(cfg.request_auditor_config, bool) else {}
            self._request_auditor = RequestAuditor(
                max_entries=getattr(aud_cfg, 'max_entries', 10000) if not isinstance(aud_cfg, dict) else aud_cfg.get('max_entries', 10000),
                log_dir=getattr(aud_cfg, 'log_dir', None) if not isinstance(aud_cfg, dict) else aud_cfg.get('log_dir', None),
                enable_pii_detection=getattr(aud_cfg, 'enable_pii_detection', True) if not isinstance(aud_cfg, dict) else aud_cfg.get('enable_pii_detection', True),
            )
        if cfg.prompt_cache_config:
            pc_cfg = cfg.prompt_cache_config if not isinstance(cfg.prompt_cache_config, bool) else {}
            self._prompt_cache_service = PromptCachingService(
                redis_url=getattr(pc_cfg, 'redis_url', '') if not isinstance(pc_cfg, dict) else pc_cfg.get('redis_url', ''),
                memory_cache_size=getattr(pc_cfg, 'memory_cache_size', 256) if not isinstance(pc_cfg, dict) else pc_cfg.get('memory_cache_size', 256),
                default_ttl_s=getattr(pc_cfg, 'default_ttl_s', 3600.0) if not isinstance(pc_cfg, dict) else pc_cfg.get('default_ttl_s', 3600.0),
            )
        if cfg.graceful_degradation_config:
            gd_cfg = cfg.graceful_degradation_config if not isinstance(cfg.graceful_degradation_config, bool) else {}
            self._graceful_degradation = GracefulDegradation(
                enabled=True,
                light_threshold=getattr(gd_cfg, 'light_threshold', 0.3) if not isinstance(gd_cfg, dict) else gd_cfg.get('light_threshold', 0.3),
                moderate_threshold=getattr(gd_cfg, 'moderate_threshold', 0.5) if not isinstance(gd_cfg, dict) else gd_cfg.get('moderate_threshold', 0.5),
                severe_threshold=getattr(gd_cfg, 'severe_threshold', 0.7) if not isinstance(gd_cfg, dict) else gd_cfg.get('severe_threshold', 0.7),
                critical_threshold=getattr(gd_cfg, 'critical_threshold', 0.85) if not isinstance(gd_cfg, dict) else gd_cfg.get('critical_threshold', 0.85),
                fallback_model=getattr(gd_cfg, 'fallback_model', None) if not isinstance(gd_cfg, dict) else gd_cfg.get('fallback_model', None),
            )
        if cfg.adaptive_batching_config:
            ab_cfg = cfg.adaptive_batching_config if not isinstance(cfg.adaptive_batching_config, bool) else {}
            slo = SLOConfig(
                p50_latency_ms=getattr(ab_cfg, 'p50_latency_ms', 500) if not isinstance(ab_cfg, dict) else ab_cfg.get('p50_latency_ms', 500),
                p99_latency_ms=getattr(ab_cfg, 'p99_latency_ms', 2000) if not isinstance(ab_cfg, dict) else ab_cfg.get('p99_latency_ms', 2000),
                max_batch_size=getattr(ab_cfg, 'max_batch_size', 64) if not isinstance(ab_cfg, dict) else ab_cfg.get('max_batch_size', 64),
                min_batch_size=getattr(ab_cfg, 'min_batch_size', 1) if not isinstance(ab_cfg, dict) else ab_cfg.get('min_batch_size', 1),
            )
            from distllm.core.adaptive_batching import AdaptiveBatchingEngine
            self._adaptive_batching = AdaptiveBatchingEngine(default_config=slo)
            self._adaptive_batching.set_slo(self.model_name)
            if self.scheduler is not None and self._adaptive_batching is not None:
                self.scheduler.max_batch_size = self._adaptive_batching.get_batch_size(self.model_name)
        if cfg.request_fingerprinting_config:
            rf_cfg = cfg.request_fingerprinting_config if not isinstance(cfg.request_fingerprinting_config, bool) else {}
            self._request_fingerprinter = RequestFingerprinter(
                cache_size=getattr(rf_cfg, 'cache_size', 10000) if not isinstance(rf_cfg, dict) else rf_cfg.get('cache_size', 10000),
                cache_ttl_s=getattr(rf_cfg, 'cache_ttl_s', 3600.0) if not isinstance(rf_cfg, dict) else rf_cfg.get('cache_ttl_s', 3600.0),
                enable_dedup=getattr(rf_cfg, 'enable_dedup', True) if not isinstance(rf_cfg, dict) else rf_cfg.get('enable_dedup', True),
            )
        if cfg.leaky_bucket_config:
            lb_cfg = cfg.leaky_bucket_config if not isinstance(cfg.leaky_bucket_config, bool) else {}
            self._rate_limiter = LeakyBucketRateLimiter(
                default_rate=getattr(lb_cfg, 'default_rate', 10.0) if not isinstance(lb_cfg, dict) else lb_cfg.get('default_rate', 10.0),
                default_burst=getattr(lb_cfg, 'default_burst', 20) if not isinstance(lb_cfg, dict) else lb_cfg.get('default_burst', 20),
                enable_backoff=getattr(lb_cfg, 'enable_backoff', True) if not isinstance(lb_cfg, dict) else lb_cfg.get('enable_backoff', True),
            )

    # -- Property Aliases (delegated to OrchestratorService) --

    @property
    def nodes(self) -> dict:
        return self._orchestrator.nodes

    @nodes.setter
    def nodes(self, value: dict):
        self._orchestrator.nodes = value

    @property
    def node_order(self) -> list[str]:
        return self._orchestrator.node_order

    @node_order.setter
    def node_order(self, value: list[str]):
        self._orchestrator.node_order = value

    @property
    def prefill_nodes(self) -> dict:
        return self._orchestrator.prefill_nodes

    @prefill_nodes.setter
    def prefill_nodes(self, value: dict):
        self._orchestrator.prefill_nodes = value

    @property
    def decode_nodes(self) -> dict:
        return self._orchestrator.decode_nodes

    @decode_nodes.setter
    def decode_nodes(self, value: dict):
        self._orchestrator.decode_nodes = value

    @property
    def prefix_cache(self) -> "PrefixCache | None":
        return self._orchestrator.prefix_cache

    @prefix_cache.setter
    def prefix_cache(self, value: "PrefixCache | None"):
        self._orchestrator.prefix_cache = value

    # -- Property Aliases (delegated to SchedulerService) --

    @property
    def scheduler(self) -> BatchScheduler | None:
        return self._scheduler_svc.scheduler

    # -- Property Aliases (delegated to MetricsService) --

    @property
    def metrics(self) -> dict[str, float]:
        return self._metrics_svc._metrics_mgr.get()

    @property
    def _shutting_down(self) -> bool:
        return self._request_tracker.shutting_down

    @_shutting_down.setter
    def _shutting_down(self, value: bool):
        self._request_tracker.shutting_down = value

    @property
    def _request_results(self) -> dict[str, str]:
        return self._request_tracker._results

    @property
    def _request_events(self) -> dict[str, threading.Event]:
        return self._request_tracker._events

    @property
    def _request_lock(self) -> threading.Lock:
        return self._request_tracker._lock

    @property
    def _model_registry(self):
        if self._model_svc._multi_model is None:
            return None
        return self._model_svc._multi_model.model_registry

    # -- Circuit Breaker (delegated to HealthService) --

    def get_circuit_breaker_status(self, node_id: str | None = None) -> dict:
        return self._health_svc.get_circuit_breaker_status(node_id)

    # -- Pipeline / Node Registration (delegated to OrchestratorService) --

    def auto_setup(self, nodes_config: list[dict]) -> None:
        model_info, total_layers = self._orchestrator.auto_setup(nodes_config)
        self.model_info = model_info
        self.total_layers = total_layers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

    def manual_register(self, node_id: str, host: str, port: int,
                        start_layer: int, end_layer: int,
                        total_layers: int | None = None,
                        role: NodeRole = NodeRole.AUTO,
                        expert_ids: list[int] | None = None,
                        cluster_id: str = "default"):
        self._orchestrator.manual_register(
            node_id, host, port, start_layer, end_layer,
            total_layers=total_layers, role=role,
            expert_ids=expert_ids, cluster_id=cluster_id,
        )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        if self.model_info is None:
            self.model_info = get_model_info(self.model_name, self.trust_remote_code)
            if total_layers is None:
                self.total_layers = self.model_info["num_layers"]

    # -- Model / Registration (delegated to ModelService) --

    def register_model(self, name: str, path: str, total_layers: int):
        return self._model_svc.register_model(name, path, total_layers)

    def list_models(self) -> list[str]:
        return self._model_svc.list_models(self._chat_router)

    def get_model_name(self, requested: str | None = None) -> str:
        return self._model_svc.get_model_name(requested)

    def register_expert_on_node(self, node_id: str, expert_ids: list[int], layer_idx: int = 0):
        self._orchestrator.register_expert_on_node(node_id, expert_ids, layer_idx)

    def moe_forward(self, hidden_states: torch.Tensor, moe_router: "MoERouter") -> torch.Tensor:
        return self._model_svc.moe_forward(hidden_states, moe_router)

    # -- Sampling (delegated to SchedulerService) --

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0,
                top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
        return self._scheduler_svc.sample(logits, temperature, top_p, top_k)

    def _sample_batch(self, logits: torch.Tensor, batch: ScheduledBatch) -> torch.Tensor:
        return self._scheduler_svc.sample_batch(logits, batch)

    def _speculative_tokens_to_append(self, draft_tokens, target_logits,
                                       accepted_count, accepted_tokens, next_token) -> list[int]:
        return self._pipeline_runner._speculative_tokens_to_append(
            draft_tokens, target_logits, accepted_count, accepted_tokens, next_token
        )

    # -- Generation (delegated to RequestPipeline) --

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7,
                 top_p: float = 0.9, top_k: int = 0, request_id: str | None = None,
                 user_id: str = "default", speculative_config: dict | None = None) -> str:
        return self._pipeline_runner.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, request_id=request_id, user_id=user_id,
            speculative_config=speculative_config,
        )

    def generate_async(self, prompt: str, request_id: str | None = None,
                       max_new_tokens: int = 128, temperature: float = 0.7,
                       top_p: float = 0.9, top_k: int = 0,
                       schema: dict | None = None, response_format: dict | None = None,
                       priority: int = 2, adapter_id: str | None = None,
                       include_logprobs: bool = False, top_logprobs: int = 0,
                       logit_bias: dict[int, float] | None = None,
                       presence_penalty: float = 0.0, frequency_penalty: float = 0.0,
                       max_latency_ms: float | None = None, user_id: str = "default") -> str:
        return self._pipeline_runner.generate_async(
            prompt, request_id=request_id, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k,
            schema=schema, response_format=response_format, priority=priority,
            adapter_id=adapter_id, include_logprobs=include_logprobs,
            top_logprobs=top_logprobs, logit_bias=logit_bias,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
            max_latency_ms=max_latency_ms, user_id=user_id,
        )

    def wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        return self._pipeline_runner.wait_for_result(request_id, timeout)

    def get_logprobs(self, request_id: str) -> list[dict] | None:
        return self._pipeline_runner.get_logprobs(request_id)

    def generate_batch(self, timeout: float = 120.0, max_steps: int = 0) -> None:
        self._scheduler_svc.generate_batch(self._pipeline_runner, timeout=timeout, max_steps=max_steps)

    def _generate_local_batch(self, batch: ScheduledBatch) -> None:
        self._pipeline_runner._generate_local_batch(batch)

    def _run_distributed_pipeline_batch(self, batch: ScheduledBatch) -> None:
        self._orchestrator.run_distributed_pipeline_batch(batch)

    def _run_async_pipeline_batch(self, batch: ScheduledBatch) -> None:
        self._orchestrator.run_async_pipeline_batch(
            batch, self._model_svc._spec_decoder, self._model_svc._continuous_trainer,
            self._scheduler_svc.scheduler, self.tokenizer,
            self._batch_kv_caches, self._batch_kv_caches_lock,
        )

    # -- Cache / Misc --

    def warm_cache(self, prompts: list[str]) -> int:
        warmer = CacheWarmer()
        return warmer.warm(prompts, self)

    # -- Health (delegated to HealthService) --

    def health_check(self) -> dict:
        return self._health_svc.health_check(
            self.nodes, self.node_order, self._health_svc.check_circuit_breaker,
        )

    async def health_check_async(self) -> dict:
        return await self._health_svc.health_check_async(
            self.nodes, self.node_order, self._health_svc.check_circuit_breaker,
        )

    # -- Metrics (delegated to MetricsService) --

    def record_metric(self, metric_name: str, value: float):
        self._metrics_svc.record_metric(metric_name, value)

    def get_metrics(self) -> dict:
        return self._metrics_svc.get_with_recovery(self._recovery.get_metrics())

    # -- New Module Public API --

    @property
    def request_auditor(self) -> RequestAuditor | None:
        return self._request_auditor

    @property
    def prompt_cache_service(self) -> PromptCachingService | None:
        return self._prompt_cache_service

    @property
    def graceful_degradation(self) -> GracefulDegradation | None:
        return self._graceful_degradation

    @property
    def adaptive_batching(self) -> AdaptiveBatchingEngine | None:
        return self._adaptive_batching

    @property
    def model_comparator(self) -> ModelVersionComparator:
        return self._model_comparator

    @property
    def token_streaming_buffer(self) -> TokenStreamingBuffer | None:
        return self._token_streaming_buffer

    @token_streaming_buffer.setter
    def token_streaming_buffer(self, value: TokenStreamingBuffer | None):
        self._token_streaming_buffer = value

    @property
    def request_fingerprinter(self) -> RequestFingerprinter | None:
        return self._request_fingerprinter

    @property
    def rate_limiter(self) -> LeakyBucketRateLimiter | None:
        return self._rate_limiter

    def set_streaming_buffer(self, flush_handler=None, max_batch_size: int = 5,
                             flush_interval_ms: float = 50.0) -> TokenStreamingBuffer:
        buf = TokenStreamingBuffer(
            flush_handler=flush_handler,
            max_batch_size=max_batch_size,
            flush_interval_ms=flush_interval_ms,
        )
        self._token_streaming_buffer = buf
        return buf

    def configure_adaptive_batching(self, model: str, p50: float | None = None,
                                     p99: float | None = None,
                                     max_batch: int | None = None) -> None:
        if self._adaptive_batching is not None:
            self._adaptive_batching.set_slo(model, p50=p50, p99=p99, max_batch=max_batch)

    def rate_limit_key(self, key: str, rate: float | None = None,
                       burst: int | None = None) -> None:
        if self._rate_limiter is not None:
            if rate is not None and burst is not None:
                self._rate_limiter.set_limit(key, rate, burst)

    def get_new_module_stats(self) -> dict:
        modules = {
            "auditor": self._request_auditor,
            "prompt_cache": self._prompt_cache_service,
            "degradation": self._graceful_degradation,
            "adaptive_batching": self._adaptive_batching,
            "rate_limiter": self._rate_limiter,
            "fingerprinter": self._request_fingerprinter,
        }
        return self._metrics_svc.get_new_module_stats(modules)

    # -- Lifecycle: Start --

    def start(self, blocking: bool = True, on_stop: Callable | None = None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

        if self._model_svc._draft_model_name and self._model_svc.draft_model is None:
            self._model_mgr.load_draft_model_early(self)

        if self.model_info is not None and self._scheduler_svc.scheduler is not None:
            self._scheduler_svc.set_model_info(self.model_info)

        self._running.set()

        if self._gossip_protocol is not None:
            self._gossip_loop_task = threading.Thread(
                target=self._gossip_loop, daemon=True, name="gossip-loop"
            )
            self._gossip_loop_task.start()

        if self.model_info is not None:
            max_pos = self.model_info.get("max_position_embeddings", 4096)
            target_ctx = 131072
            if target_ctx > max_pos and self.nodes_info:
                rope_config = {
                    "type": "yarn",
                    "factor": target_ctx / max_pos,
                    "original_max_position_embeddings": max_pos,
                }
                self.nodes_info["rope_scaling"] = rope_config
                logger.info(f"Distributed RoPE scaling configured: {max_pos} -> {target_ctx}")

        servicer = CoordinatorService()
        self.server = GRPCServer(port=self.port, servicer=servicer)
        self.server.start()
        logger.info(f"Coordinator started on port {self.port}")

        if self._model_svc._self_optimizing:
            self._model_svc._self_optimizing.start()

        if self._rebalancer and self._rebalancer._settings.enabled:
            self._rebalancer_task = threading.Thread(
                target=self._rebalancer_loop, daemon=True,
            )
            self._rebalancer_task.start()

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

    def _gossip_loop(self):
        interval = 10.0
        while self._running.is_set():
            try:
                time.sleep(interval)
                if self._cache_mgr is not None:
                    discovered = self._cache_mgr.sync_with_peers()
                    if discovered > 0:
                        logger.debug(f"Gossip round: discovered {discovered} new cache entries")
                    if self._cache_mgr.prefix_cache and hasattr(self._cache_mgr.prefix_cache, "_root"):
                        self._cache_mgr.prefix_cache._root.evict_lru(self._cache_mgr.prefix_cache.max_entries)
            except Exception:
                logger.debug("Gossip round error (non-fatal)", exc_info=True)

    def _rebalancer_loop(self) -> None:
        while self._running.is_set():
            time.sleep(self._rebalancer._settings.check_interval)
            if not self._running.is_set() or not self._rebalancer._settings.enabled:
                continue
            should, reason = self._rebalancer.should_rebalance()
            if should:
                stragglers = self._rebalancer.detect_stragglers()
                logger.warning(f"Stragglers detected: {stragglers}")
                all_avg = self._latency_tracker.get_all_avg()
                partition = self._rebalancer.compute_new_partition(self.total_layers, all_avg)
                logger.info(f"Recommended partition: {[(p.node_id, p.start_layer, p.end_layer) for p in partition]}")
                self._rebalancer.record_rebalance()

    # -- Self-healing: Recovery Callbacks (delegated to HealthService) --

    def _on_resource_mgr_failure(self, node_id: str) -> None:
        self._health_svc.on_node_failure(node_id)

    def _on_recovery_drain(self, node_id: str) -> None:
        self._health_svc.on_drain(node_id, self._pipeline)

    def _on_recovery_redistribute(self, failed_node_id: str, plan: NodeRecoveryPlan) -> None:
        self._health_svc.on_redistribute(failed_node_id, plan, self._pipeline)

    def _on_recovery_recover(self, failed_node_id: str, request_ids: list[str]) -> list[dict]:
        return self._health_svc.on_recover(failed_node_id, request_ids)

    def _on_recovery_mark_dead(self, node_id: str) -> None:
        self._health_svc.on_mark_dead(node_id)

    def check_recovery(self, request_id: str) -> bool:
        return self._health_svc.check_recovery(request_id)

    def get_recovery_metrics(self) -> dict:
        return self._health_svc.get_recovery_metrics()

    # -- SLA Helpers --

    def register_fallback_models(self, model: str, fallbacks: list[str]) -> None:
        self._fallback_models[model] = fallbacks

    def wait_for_termination(self):
        if self.server:
            try:
                self.server.wait_for_termination()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()

    # -- Model Loading (delegated to ModelService) --

    def load_local_model(self):
        self._model_svc.load_local_model(self)
        self._apply_flash_attention()
        self._apply_rope_scaling()
        self._wire_paged_attention()
        self._apply_adaptive_precision()
        self._model_svc.cuda_graph_capture(self.local_partitioner)
        self._model_svc.compile_model(self.local_partitioner)
        self._model_svc.setup_slora(self.local_partitioner)

    def _load_draft_model(self):
        self._model_svc.load_draft_model(self)

    def _apply_flash_attention(self):
        self._model_svc.apply_flash_attention(self.local_partitioner)

    def _apply_rope_scaling(self):
        self._model_svc.apply_rope_scaling(self.local_partitioner)

    def _wire_paged_attention(self):
        self._model_svc.wire_paged_attention(self.local_partitioner)

    def _apply_adaptive_precision(self):
        self._model_svc.apply_adaptive_precision(self.local_partitioner)

    def enable_continuous_training(self, base_model: torch.nn.Module,
                                    draft_head: torch.nn.Module | None = None,
                                    config: ContinuousTrainConfig | None = None) -> None:
        self._model_svc.enable_continuous_training(base_model, draft_head, config)

    # -- Init extension configs (preserved for backward compat) --

    def _init_embedding_config(self):
        self._model_svc.init_embedding_loader(getattr(self, "_embedding_config", None))

    def _init_version_config(self):
        self._model_svc.init_version_manager(getattr(self, "_version_config", None))

    # -- Apply tunable params --

    def _apply_tunable_params(self, params) -> None:
        if self._scheduler_svc.scheduler is not None and hasattr(params, 'batch_size'):
            if params.batch_size != self._scheduler_svc.scheduler.max_batch_size:
                self._scheduler_svc.scheduler.max_batch_size = params.batch_size
        spec_decoder = self._model_svc.get_spec_decoder()
        if spec_decoder is not None and hasattr(params, 'speculative_decoding_enabled') and params.speculative_decoding_enabled is not None:
            spec_decoder._enabled = params.speculative_decoding_enabled

    def set_deterministic_mode(self, enabled: bool = True, seed: int = 42) -> None:
        if enabled:
            self._deterministic_mode.enable(seed)
        else:
            self._deterministic_mode.disable()

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        return self._replay_buffer.list_recent(n)

    def replay_request(self, request_id: str) -> str | None:
        entry = self._replay_buffer.get(request_id)
        if entry is None:
            return None
        if self._scheduler_svc.scheduler is not None:
            rid = self.generate_async(
                prompt=entry.prompt,
                max_new_tokens=entry.params.get("max_new_tokens", 128),
                temperature=entry.params.get("temperature", 0.7),
                top_p=entry.params.get("top_p", 0.9),
                top_k=entry.params.get("top_k", 0),
                priority=entry.params.get("priority", 2),
            )
            return self.wait_for_result(rid)
        return self.generate(
            entry.prompt,
            entry.params.get("max_new_tokens", 128),
            entry.params.get("temperature", 0.7),
            entry.params.get("top_p", 0.9),
        )

    # -- Lifecycle: Stop --

    def stop(self):
        logger.info("Initiating graceful shutdown...")
        self._running.clear()
        if self._model_svc._continuous_trainer is not None:
            self._model_svc._continuous_trainer.stop()
            logger.info("Continuous speculative training stopped")
        self._shutting_down = True
        logger.info("Phase 1: Stopped accepting new requests")
        with self._request_tracker._lock:
            events = list(self._request_tracker._events.values())
        if events:
            logger.info(f"Phase 2: Waiting for {len(events)} in-flight requests...")
            deadline = time.monotonic() + 30.0
            for event in events:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                event.wait(timeout=remaining)
        if self._cache_persistence and self._cache_persistence._settings.enabled:
            logger.info("Phase 3: Persisting cache to disk...")
            self._cache_persistence.enforce_disk_limit()
        if self.server:
            logger.info("Phase 4: Stopping gRPC server...")
            self.server.stop(grace=10)
        logger.info("Phase 5: Closing node connections...")
        self._health_svc.close_all(self.nodes)
        self._request_tracker.clear()
        if hasattr(self, '_plugin_manager') and self._plugin_manager:
            logger.info("Phase 7: Shutting down plugins...")
            self._plugin_manager.shutdown_all()
        if hasattr(self, '_gossip_loop_task') and self._gossip_loop_task:
            self._gossip_loop_task.join(timeout=5.0)
        if hasattr(self, '_rebalancer_task') and self._rebalancer_task:
            self._rebalancer_task.join(timeout=5.0)
        if self._gossip_client:
            self._gossip_client.close()
        self._orchestrator.shutdown()
        self._subsystems.stop_all()
        if self._model_svc._self_optimizing:
            self._model_svc._self_optimizing.stop()
        if self._model_svc._slora_manager:
            self._model_svc._slora_manager.shutdown()
        if self._model_svc._agent_loop:
            self._model_svc._agent_loop.stop()
        rec_metrics = self._recovery.get_metrics()
        if rec_metrics.get("recoveries", 0) > 0:
            logger.info(
                f"Recovery stats: {rec_metrics['recoveries']} events, "
                f"{rec_metrics['sequences_recovered']} seqs recovered, "
                f"{rec_metrics['sequences_lost']} seqs lost"
            )
        logger.info("Graceful shutdown complete")

    async def stop_async(self):
        logger.info("Initiating graceful shutdown (async)...")
        self._running.clear()
        self._shutting_down = True
        logger.info("Phase 1: Stopped accepting new requests")
        with self._request_tracker._lock:
            events = list(self._request_tracker._events.values())
        if events:
            logger.info(f"Phase 2: Waiting for {len(events)} in-flight requests...")
            for event in events:
                event.wait(timeout=30.0)
        if self._cache_persistence and self._cache_persistence._settings.enabled:
            logger.info("Phase 3: Persisting cache to disk...")
            self._cache_persistence.enforce_disk_limit()
        if self.server:
            logger.info("Phase 4: Stopping gRPC server...")
            self.server.stop(grace=10)
        logger.info("Phase 5: Closing node connections...")
        await self._health_svc.close_all_async(self.nodes)
        self._request_tracker.clear()
        if hasattr(self, '_plugin_manager') and self._plugin_manager:
            logger.info("Phase 7: Shutting down plugins...")
            self._plugin_manager.shutdown_all()
        if hasattr(self, '_gossip_loop_task') and self._gossip_loop_task:
            self._gossip_loop_task.join(timeout=5.0)
        if hasattr(self, '_rebalancer_task') and self._rebalancer_task:
            self._rebalancer_task.join(timeout=5.0)
        if self._gossip_client:
            self._gossip_client.close()
        self._orchestrator.shutdown()
        self._subsystems.stop_all()
        if self._model_svc._self_optimizing:
            self._model_svc._self_optimizing.stop()
        if self._model_svc._slora_manager:
            self._model_svc._slora_manager.shutdown()
        if self._model_svc._agent_loop:
            self._model_svc._agent_loop.stop()
        logger.info("Graceful shutdown complete (async)")

    def _generate_local_sync(self, prompt: str, max_new_tokens: int,
                              temperature: float, top_p: float) -> str:
        return self._pipeline_runner._generate_local_sync(prompt, max_new_tokens, temperature, top_p)

    def _generate_local_eagle_sync(self, input_ids: torch.Tensor, *,
                                    max_new_tokens: int, temperature: float,
                                    top_p: float, top_k: int = 0) -> str:
        return self._pipeline_runner._generate_local_eagle_sync(
            input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
        )


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
