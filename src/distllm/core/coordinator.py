"""Coordinator facade for distributed LLM inference.

This is a backward-compatible facade that composes 4 specialized components:
- ResourceManager: node lifecycle, health, circuit breaking
- CacheManager: prefix cache, KV cache, chunked prefill
- TokenGenerator: token sampling and generation
- PipelineOrchestrator: node topology and layer routing

All original constructor parameters and public methods are preserved.
"""

import argparse
import gc
import torch
from transformers import AutoTokenizer
from loguru import logger
import uuid
import time
import asyncio
import threading
from typing import Any, Callable
from distllm.models.partitioner import ModelPartitioner, get_model_info
from distllm.communication.grpc import CoordinatorService, GRPCServer
from distllm.core.batch_scheduler import BatchScheduler, Sequence, ScheduledBatch
from distllm.core.structured_output import JSONSchemaConstraint
from distllm.core.chunked_prefill import maybe_chunk
from distllm.config.loader import NodeRole
from distllm.config.settings import DistLLMSettings, MultiModelSettings, MoESettings
from distllm.communication.grpc import set_debug_mode, is_debug_mode
from distllm.core.moe_router import MoERouter
from distllm.core.prefix_cache import PrefixCache
from distllm.errors.types import (
    NodeError,
    NodeUnreachableError,
    OOMError,
    GRPCTimeoutError,
    BatchError,
)

from distllm.core.resource_manager import ResourceManager
from distllm.core.cache_manager import CacheManager
from distllm.core.token_generator import TokenGenerator
from distllm.core.pipeline_orchestrator import PipelineOrchestrator
from distllm.core.di import Container
from distllm.core.subsystem_manager import SubsystemManager
from distllm.core.speculative_decoder import SpeculativeDecoder
from distllm.core.speculative_trainer import ContinuousSpeculativeTrainer, ContinuousTrainConfig
from distllm.core.request_replay import RequestReplayBuffer, DeterministicMode, get_replay_buffer
from distllm.core.model_registry import ModelRegistry
from distllm.core.latency_tracker import LatencyTracker
from distllm.core.rebalancer import Rebalancer
from distllm.core.cache_persistence import CachePersistenceManager
from distllm.core.cache_warming import CacheWarmer
from distllm.core.coordinator_metrics import MetricsManager
from distllm.core.coordinator_model import ModelManager
from distllm.core.coordinator_health import HealthChecker
from distllm.core.coordinator_lifecycle import RequestTracker
from distllm.core.coordinator_nodes import NodeRegistrar
from distllm.core.coordinator_multi_model import MultiModelManager
from distllm.core.coordinator_config import CoordinatorConfig
from distllm.core.request_pipeline import RequestPipeline, _current_request_id_ctx
from distllm.core.preemption_policy import PreemptionPolicy, GPUMemoryMonitor, SLATracker
from distllm.core.request_auditor import RequestAuditor
from distllm.core.prompt_caching_service import PromptCachingService
from distllm.core.graceful_degradation import GracefulDegradation, LoadSnapshot
from distllm.core.adaptive_batching import AdaptiveBatchingEngine
from distllm.core.model_comparator import ModelVersionComparator
from distllm.core.token_streaming_buffer import TokenStreamingBuffer
from distllm.core.request_fingerprinting import RequestFingerprinter
from distllm.core.leaky_bucket_limiter import LeakyBucketRateLimiter
from distllm.core.node_recovery import NodeRecoveryManager, NodeRecoveryPlan, LayerRedistribution


class Coordinator:
    """Orchestrates distributed inference across worker nodes.

    This is a backward-compatible facade that composes specialized components.
    New features should be built as separate components and composed here.

    Architecture — Component Map:
    ┌─────────────────────────────────────────────────────────┐
    │  Core Components (delegated)                            │
    │  _resource_mgr     → node lifecycle, circuit breaker    │
    │  _cache_mgr        → prefix cache, chunked prefill      │
    │  _pipeline         → node topology, layer routing       │
    │  _token_gen        → token sampling                     │
    │  _model_mgr        → model/draft model loading          │
    │  _health_checker   → health checks                      │
    │  _node_registrar   → node registration, layer assign    │
    │  _metrics_mgr      → metrics tracking                   │
    │  _request_tracker  → request completion tracking        │
    ├─────────────────────────────────────────────────────────┤
    │  Extension Components (initialized on demand)           │
    │  _multi_model      → multi-model serving                │
    │  _moe_orchestrator → mixture of experts                 │
    │  _spec_decoder     → speculative decoding               │
    │  _hybrid_parallel  → TP/PP/EP parallelism              │
    │  _zero_copy_engine → GPU-direct tensor transfer         │
    │  _adaptive_precision → precision calibration            │
    │  _predictive_cache → KV cache prediction                │
    │  _cache_persistence → disk cache persistence            │
    │  _rebalancer       → dynamic pipeline rebalancing       │
    │  _gossip_protocol  → P2P cache gossip                   │
    │  _version_manager  → shadow/A-B/blue-green              │
    │  _plugin_manager   → plugin system                      │
    │  _embedding_loader → embedding/reranking models         │
    │  _vlm_pipeline     → vision-language models             │
    │  _paged_attention  → paged KV cache                     │
    │  _flash_attention  → FlashAttention wrapper             │
    │  _federation_manager → cross-cluster federation         │
    │  _param_update_channel → streaming param updates        │
    │  _model_hotswap    → model hot-swap                     │
    ├─────────────────────────────────────────────────────────┤
    │  Public facade properties:                              │
    │  nodes, node_order, prefill_nodes, decode_nodes,       │
    │  prefix_cache, metrics, scheduler, tokenizer           │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        model_name: str,
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
    ):
        self.model_name = model_name
        self.port = port
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.quantization_config = quantization_config
        self.metrics_exporter = metrics_exporter
        self.discovery_mode = discovery_mode

        self.config = CoordinatorConfig(
            model_name=model_name,
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

        # Dependency injection container and subsystem lifecycle manager
        self._container = Container()
        self._container.register(Container, self._container)
        self._subsystems = SubsystemManager()

        # Component: ResourceManager (health, circuit breaker, lifecycle)
        self._resource_mgr = ResourceManager()
        self._container.register(ResourceManager, self._resource_mgr)

        # Component: CacheManager (prefix cache, chunked prefill)
        self._cache_mgr = CacheManager(
            prefix_cache_enabled=prefix_cache_enabled,
            prefix_cache_max_entries=prefix_cache_max_entries,
            prefix_cache_min_prefix_len=prefix_cache_min_prefix_len,
            radix_tree_cache_enabled=radix_tree_cache_enabled,
            chunked_prefill_enabled=chunked_prefill_enabled,
            chunked_prefill_chunk_size=chunked_prefill_chunk_size,
        )

        # Component: PipelineOrchestrator (node topology, distributed pipeline)
        self._pipeline = PipelineOrchestrator(
            resource_mgr=self._resource_mgr,
        )
        self._container.register(PipelineOrchestrator, self._pipeline)

        # 1F1B / Interleaved pipeline scheduling (async micro-batch pipelining)
        self._async_pipeline = None
        self._pipeline_schedule_type = "sequential"  # sequential, overlap, 1f1b, interleaved
        if pipeline_schedule_config:
            schedule_type = getattr(pipeline_schedule_config, "schedule", "sequential")
            if schedule_type in ("1f1b", "interleaved"):
                from distllm.core.async_pipeline import (
                    AsyncPipelineEngine,
                    AsyncPipelineConfig,
                    ScheduleType,
                )
                num_micro = getattr(pipeline_schedule_config, "num_micro_batches", 4)
                num_stages = getattr(pipeline_schedule_config, "num_stages", 1)
                async_config = AsyncPipelineConfig(
                    schedule=ScheduleType.ONE_F_ONE_B if schedule_type == "1f1b" else ScheduleType.INTERLEAVED,
                    num_micro_batches=num_micro,
                    num_stages=num_stages,
                    overlap_allreduce=getattr(pipeline_schedule_config, "overlap_allreduce", True),
                    prefetch_next_batch=getattr(pipeline_schedule_config, "prefetch_next_batch", True),
                )
                self._async_pipeline = AsyncPipelineEngine(config=async_config)
                self._pipeline_schedule_type = schedule_type
                logger.info(f"Async pipeline scheduling enabled: {schedule_type} ({num_micro} micro-batches, {num_stages} stages)")

        # Health checker (delegates to resource_mgr + metrics_exporter)
        self._health_checker = HealthChecker(self._resource_mgr, self.metrics_exporter)

        # Node registrar (node registration, layer assignment, expert registration)
        self._node_registrar = NodeRegistrar(
            pipeline=self._pipeline,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )

        # Component: TokenGenerator (sampling)
        self._token_gen = TokenGenerator()

        # Model manager (model loading, draft model, local generation)
        self._model_mgr = ModelManager(
            model_name=model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
        )

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
        self._spec_decoder: SpeculativeDecoder | None = None
        self._spec_method = "draft_model"
        if speculative_config:
            self._draft_model_name = speculative_config.get("draft_model") or None
            self.num_assistant_tokens = speculative_config.get("num_assistant_tokens", 5)
            self._spec_method = speculative_config.get("method", "draft_model")
            self._spec_decoder = SpeculativeDecoder(
                num_assistant_tokens=self.num_assistant_tokens,
                min_acceptance_rate=speculative_config.get("min_acceptance_rate", 0.3),
                warmup_steps=speculative_config.get("warmup_steps", 10),
                method=self._spec_method,
                medusa_num_heads=speculative_config.get("medusa_num_heads", 4),
                medusa_num_tokens_per_head=speculative_config.get("medusa_num_tokens_per_head", 3),
                eagle_hidden_size=speculative_config.get("eagle_hidden_size", 4096),
                eagle_vocab_size=speculative_config.get("eagle_vocab_size", 32000),
                ngram_min_match=speculative_config.get("ngram_min_match", 4),
            )
            eagle_checkpoint = speculative_config.get("eagle_checkpoint")
            if eagle_checkpoint:
                self._spec_decoder.load_eagle_checkpoint(
                    eagle_checkpoint,
                    variant=speculative_config.get("eagle_variant", "eagle"),
                    hidden_size=speculative_config.get("eagle_hidden_size"),
                    vocab_size=speculative_config.get("eagle_vocab_size"),
                    num_layers=speculative_config.get("eagle_num_layers", 2),
                )
        else:
            self._draft_model_name = None

        # Continuous speculative fine-tuning
        self._continuous_trainer: ContinuousSpeculativeTrainer | None = None
        self._continuous_trainer_config = None

        # Request replay buffer and deterministic debug mode
        self._replay_buffer: RequestReplayBuffer = get_replay_buffer(max_requests=100)
        self._deterministic_mode = DeterministicMode(seed=42, enabled=False)

        # Batch scheduler
        self.scheduler: BatchScheduler | None = None
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

        # Wire cache manager to scheduler for radix tree storage
        if self.scheduler is not None:
            self.scheduler.set_cache_manager(self._cache_mgr)

        # Preemption policy: SLA-aware request preemption with checkpoint/resume
        self._preemption_policy: PreemptionPolicy | None = None
        if max_batch_size > 1:
            self._preemption_policy = PreemptionPolicy(
                gpu_monitor=GPUMemoryMonitor(),
                sla_tracker=SLATracker(max_violations=3, sla_deadline_ms=5000.0),
                max_queue_depth=100,
                max_checkpoints=10,
                checkpoint_memory_limit_mb=4096,
            )

        # Request completion tracking (delegated to RequestTracker)
        self._request_tracker = RequestTracker()

        # Metrics tracking (delegated to MetricsManager)
        self._metrics_mgr = MetricsManager()

        # Node info for distributed state
        self.nodes_info = {}

        # Extension points initialized by the micro-orchestrator setup below.
        self._embedding_config = embedding_config
        self._version_config = version_config
        self._embedding_loader = None
        self._embedding_max_length = 512
        self._embedding_normalize = True
        self._embedding_batch_size = 32
        self._version_manager = None
        self._model_hotswap = None
        self._paged_attention = None
        self._paged_kv_backend = None
        self._vlm_pipeline = None
        self._flash_attention = None
        self._plugin_manager: 'PluginManager' | None = None
        self._hybrid_parallel_planner = None
        self._hybrid_parallel_executor = None
        self._zero_copy_engine = None
        self._adaptive_precision = None
        self._predictive_cache = None

        # Multi-model registry and MoE (delegated to managers)
        self._multi_model: MultiModelManager | None = None
        self._expert_registry = None
        self._moe_orchestrator = None
        self._init_multi_model(multi_model_config)
        self._init_moe(moe_config)
        self._init_embedding_config()
        self._init_version_config()
        self._init_flash_attention(causal=True, enable_fa2=True)
        self._init_plugin_manager()
        self._init_hybrid_parallel(hybrid_parallel_config)
        self._init_zero_copy(zero_copy_config)
        self._init_adaptive_precision(adaptive_precision_config)
        self._init_predictive_cache(predictive_cache_config)

        # Self-optimizing engine, CUDA graphs, compile support, SLoRA, RAG, agent, disagg
        self._self_optimizing = None
        self._cuda_graph_pool = None
        self._slora_manager = None
        self._rag_pipeline = None
        self._agent_loop = None
        self._disagg_orchestrator = None
        self._init_self_optimizing(self_optimizing_config)
        self._init_cuda_graph(cuda_graph_config)
        self._init_compile_support(compile_config)
        self._init_slora(slora_config)
        self._init_rag(rag_config)
        self._init_agent(agent_config)
        self._init_disagg(disagg_config)

        # Dynamic rebalancing
        self._latency_tracker: LatencyTracker | None = None
        self._rebalancer: Rebalancer | None = None
        self._rebalancer_task: threading.Thread | None = None
        if rebalancer_config and rebalancer_config.enabled:
            self._latency_tracker = LatencyTracker()
            self._pipeline.set_latency_tracker(self._latency_tracker)
            self._rebalancer = Rebalancer(self._latency_tracker, rebalancer_config)

        # Cache persistence
        self._cache_persistence: CachePersistenceManager | None = None
        if cache_persistence_config and cache_persistence_config.enabled:
            self._cache_persistence = CachePersistenceManager(cache_persistence_config)
            self._cache_mgr.persistence_manager = self._cache_persistence

        # Persistent KV cache state for batch pipeline (keyed by request_id)
        self._batch_kv_caches: dict[str, dict[str, list | None]] = {}
        self._batch_kv_caches_lock = threading.Lock()
        self._batch_event = threading.Event()

        # Fallback model registry: primary_model -> [fallback_model_names]
        self._fallback_models: dict[str, list[str]] = {}

        # Pipeline composition runtime
        self._pipeline_composer = None

        # P2P KV cache gossip
        self._cache_index = None
        self._gossip_protocol = None
        self._gossip_client = None
        self._gossip_loop_task: threading.Thread | None = None
        if gossip_config and gossip_config.enabled:
            from distllm.core.cache_index import CacheIndex
            from distllm.core.gossip_protocol import GossipProtocol, GossipClient
            self._cache_index = CacheIndex()
            self._gossip_protocol = GossipProtocol(
                node_id=f"coordinator-{self.model_name}",
                max_peers=gossip_config.max_peers,
                cache_ttl=gossip_config.cache_ttl,
            )
            # Create gossip client with peer resolver from node registrar
            def resolve_peer(peer_id: str):
                """Resolve peer_id to (host, port) from registered nodes."""
                nodes_snapshot = dict(self.nodes)
                for node_id, reg in nodes_snapshot.items():
                    if node_id == peer_id or peer_id in node_id:
                        return reg.host, reg.port
                return None

            self._gossip_client = GossipClient(
                node_id=self._gossip_protocol.state.node_id,
                peer_resolver=resolve_peer,
            )
            # Wire gossip to cache manager
            self._cache_mgr.cache_index = self._cache_index
            self._cache_mgr.gossip_protocol = self._gossip_protocol
            self._cache_mgr.gossip_client = self._gossip_client

        # --- New Modules (Request Auditor, Prompt Cache, Graceful Degradation, etc.) ---

        # Request Auditor: compliance logging, PII detection
        self._request_auditor: RequestAuditor | None = None
        if request_auditor_config:
            aud_cfg = request_auditor_config if not isinstance(request_auditor_config, bool) else {}
            self._request_auditor = RequestAuditor(
                max_entries=getattr(aud_cfg, 'max_entries', 10000) if not isinstance(aud_cfg, dict) else aud_cfg.get('max_entries', 10000),
                log_dir=getattr(aud_cfg, 'log_dir', None) if not isinstance(aud_cfg, dict) else aud_cfg.get('log_dir', None),
                enable_pii_detection=getattr(aud_cfg, 'enable_pii_detection', True) if not isinstance(aud_cfg, dict) else aud_cfg.get('enable_pii_detection', True),
            )

        # Prompt Caching Service: Redis-backed shared prompt cache
        self._prompt_cache_service: PromptCachingService | None = None
        if prompt_cache_config:
            pc_cfg = prompt_cache_config if not isinstance(prompt_cache_config, bool) else {}
            self._prompt_cache_service = PromptCachingService(
                redis_url=getattr(pc_cfg, 'redis_url', '') if not isinstance(pc_cfg, dict) else pc_cfg.get('redis_url', ''),
                memory_cache_size=getattr(pc_cfg, 'memory_cache_size', 256) if not isinstance(pc_cfg, dict) else pc_cfg.get('memory_cache_size', 256),
                default_ttl_s=getattr(pc_cfg, 'default_ttl_s', 3600.0) if not isinstance(pc_cfg, dict) else pc_cfg.get('default_ttl_s', 3600.0),
            )

        # Graceful Degradation: partial responses instead of 503
        self._graceful_degradation: GracefulDegradation | None = None
        if graceful_degradation_config:
            gd_cfg = graceful_degradation_config if not isinstance(graceful_degradation_config, bool) else {}
            self._graceful_degradation = GracefulDegradation(
                enabled=True,
                light_threshold=getattr(gd_cfg, 'light_threshold', 0.3) if not isinstance(gd_cfg, dict) else gd_cfg.get('light_threshold', 0.3),
                moderate_threshold=getattr(gd_cfg, 'moderate_threshold', 0.5) if not isinstance(gd_cfg, dict) else gd_cfg.get('moderate_threshold', 0.5),
                severe_threshold=getattr(gd_cfg, 'severe_threshold', 0.7) if not isinstance(gd_cfg, dict) else gd_cfg.get('severe_threshold', 0.7),
                critical_threshold=getattr(gd_cfg, 'critical_threshold', 0.85) if not isinstance(gd_cfg, dict) else gd_cfg.get('critical_threshold', 0.85),
                fallback_model=getattr(gd_cfg, 'fallback_model', None) if not isinstance(gd_cfg, dict) else gd_cfg.get('fallback_model', None),
            )

        # Adaptive Batching Engine: dynamic batch size from latency SLOs
        self._adaptive_batching: AdaptiveBatchingEngine | None = None
        if adaptive_batching_config:
            ab_cfg = adaptive_batching_config if not isinstance(adaptive_batching_config, bool) else {}
            from distllm.core.adaptive_batching import SLOConfig
            slo = SLOConfig(
                p50_latency_ms=getattr(ab_cfg, 'p50_latency_ms', 500) if not isinstance(ab_cfg, dict) else ab_cfg.get('p50_latency_ms', 500),
                p99_latency_ms=getattr(ab_cfg, 'p99_latency_ms', 2000) if not isinstance(ab_cfg, dict) else ab_cfg.get('p99_latency_ms', 2000),
                max_batch_size=getattr(ab_cfg, 'max_batch_size', 64) if not isinstance(ab_cfg, dict) else ab_cfg.get('max_batch_size', 64),
                min_batch_size=getattr(ab_cfg, 'min_batch_size', 1) if not isinstance(ab_cfg, dict) else ab_cfg.get('min_batch_size', 1),
            )
            self._adaptive_batching = AdaptiveBatchingEngine(default_config=slo)
            self._adaptive_batching.set_slo(self.model_name)
            if self.scheduler is not None and self._adaptive_batching is not None:
                self.scheduler.max_batch_size = self._adaptive_batching.get_batch_size(self.model_name)

        # Model Version Comparator: statistical comparison (used externally)
        self._model_comparator = ModelVersionComparator()

        # Token Streaming Buffer: batch tokens to reduce SSE overhead
        self._token_streaming_buffer: TokenStreamingBuffer | None = None

        # Request Fingerprinting: deduplication
        self._request_fingerprinter: RequestFingerprinter | None = None
        if request_fingerprinting_config:
            rf_cfg = request_fingerprinting_config if not isinstance(request_fingerprinting_config, bool) else {}
            self._request_fingerprinter = RequestFingerprinter(
                cache_size=getattr(rf_cfg, 'cache_size', 10000) if not isinstance(rf_cfg, dict) else rf_cfg.get('cache_size', 10000),
                cache_ttl_s=getattr(rf_cfg, 'cache_ttl_s', 3600.0) if not isinstance(rf_cfg, dict) else rf_cfg.get('cache_ttl_s', 3600.0),
                enable_dedup=getattr(rf_cfg, 'enable_dedup', True) if not isinstance(rf_cfg, dict) else rf_cfg.get('enable_dedup', True),
            )

        # Rate Limiter with Leaky Bucket
        self._rate_limiter: LeakyBucketRateLimiter | None = None
        if leaky_bucket_config:
            lb_cfg = leaky_bucket_config if not isinstance(leaky_bucket_config, bool) else {}
            self._rate_limiter = LeakyBucketRateLimiter(
                default_rate=getattr(lb_cfg, 'default_rate', 10.0) if not isinstance(lb_cfg, dict) else lb_cfg.get('default_rate', 10.0),
                default_burst=getattr(lb_cfg, 'default_burst', 20) if not isinstance(lb_cfg, dict) else lb_cfg.get('default_burst', 20),
                enable_backoff=getattr(lb_cfg, 'enable_backoff', True) if not isinstance(lb_cfg, dict) else lb_cfg.get('enable_backoff', True),
            )

        # Lifecycle flag for background threads (rebalancer, gossip)
        self._running = threading.Event()

        # Streaming parameter updates
        from distllm.core.param_update_channel import ParamUpdateChannel
        self._param_update_channel = ParamUpdateChannel()

        # Cross-cluster federation and geo-aware routing
        from distllm.core.federation_router import FederationRouter
        self._federation_router = FederationRouter()
        self._federation_manager = self._federation_router.federation_manager
        self._latency_monitor = self._federation_router.latency_monitor
        self._geo_router = self._federation_router.geo_router

        # Wire federation and expert resources to node registrar
        self._federation_router.attach_registrar(
            self._node_registrar,
            expert_registry=self._expert_registry,
        )

        # Node recovery manager: self-healing cluster
        self._recovery = NodeRecoveryManager()
        self._recovery.set_drain_callback(self._on_recovery_drain)
        self._recovery.set_redistribute_layers_callback(self._on_recovery_redistribute)
        self._recovery.set_recover_sequences_callback(self._on_recovery_recover)
        self._recovery.set_mark_dead_callback(self._on_recovery_mark_dead)

        # Wire resource manager circuit breaker to node recovery
        self._resource_mgr.set_node_failure_callback(self._on_resource_mgr_failure)

        # Request pipeline: generates tokens through local or distributed execution
        self._pipeline_runner = RequestPipeline(self)

        # Multi-model chat router: content-based model selection
        self._chat_router = None

    # -- Property aliases for backward compat --

    @property
    def nodes(self) -> dict[str, object]:
        return self._pipeline.nodes

    @nodes.setter
    def nodes(self, value: dict[str, object]):
        self._pipeline.nodes = value

    @property
    def node_order(self) -> list[str]:
        return self._pipeline.node_order

    @node_order.setter
    def node_order(self, value: list[str]):
        self._pipeline.node_order = value

    @property
    def prefill_nodes(self) -> dict[str, object]:
        return self._pipeline.prefill_nodes

    @prefill_nodes.setter
    def prefill_nodes(self, value: dict[str, object]):
        self._pipeline.prefill_nodes = value

    @property
    def decode_nodes(self) -> dict[str, object]:
        return self._pipeline.decode_nodes

    @decode_nodes.setter
    def decode_nodes(self, value: dict[str, object]):
        self._pipeline.decode_nodes = value

    @property
    def prefix_cache(self) -> "PrefixCache | None":
        return self._cache_mgr.prefix_cache

    @prefix_cache.setter
    def prefix_cache(self, value: "PrefixCache | None") -> None:
        self._cache_mgr.prefix_cache = value

    # -- Circuit breaker status (consolidated public API) --

    def get_circuit_breaker_status(self, node_id: str | None = None) -> dict:
        """Get circuit breaker status for one or all nodes.

        Replaces the old test-only proxy properties:
        _node_failure_counts, _node_recovery_time, _node_circuit_breaker_threshold,
        _node_base_retry_delay, _node_max_retry_delay.
        """
        with self._resource_mgr._lock:
            if node_id:
                return {
                    "failures": self._resource_mgr._node_failure_counts.get(node_id, 0),
                    "recovery_time": self._resource_mgr._node_recovery_time.get(node_id, 0.0),
                    "threshold": self._resource_mgr.cb_config.threshold,
                    "base_delay": self._resource_mgr.cb_config.base_delay,
                    "max_delay": self._resource_mgr.cb_config.max_delay,
                }
            return {
                "nodes": {
                    nid: {
                        "failures": self._resource_mgr._node_failure_counts.get(nid, 0),
                        "recovery_time": self._resource_mgr._node_recovery_time.get(nid, 0.0),
                    }
                    for nid in self._resource_mgr._node_failure_counts
                },
                "threshold": self._resource_mgr.cb_config.threshold,
                "base_delay": self._resource_mgr.cb_config.base_delay,
                "max_delay": self._resource_mgr.cb_config.max_delay,
            }

    @property
    def metrics(self) -> dict[str, float]:
        return self._metrics_mgr.get()

    @property
    def _shutting_down(self) -> bool:
        return self._request_tracker.shutting_down

    @_shutting_down.setter
    def _shutting_down(self, value: bool):
        self._request_tracker.shutting_down = value

    @property
    def _request_results(self) -> dict[str, str]:
        """Backward compat: access to request results dict."""
        return self._request_tracker._results

    @property
    def _request_events(self) -> dict[str, threading.Event]:
        """Backward compat: access to request events dict."""
        return self._request_tracker._events

    @property
    def _request_lock(self) -> threading.Lock:
        """Backward compat: access to request lock."""
        return self._request_tracker._lock

    @property
    def _model_registry(self):
        """Backward compat: access to model registry via MultiModelManager."""
        if self._multi_model is None:
            return None
        return self._multi_model.model_registry

    # -- Multi-Model Serving (delegated to MultiModelManager) --

    def _init_multi_model(self, multi_model_config: MultiModelSettings | None) -> None:
        """Initialize multi-model and MoE subsystems."""
        if multi_model_config and multi_model_config.enabled:
            model_registry = ModelRegistry(max_models=multi_model_config.max_models)
            model_registry._default_model = multi_model_config.default_model or self.model_name
            model_registry.register(self.model_name, self.model_name, 0)
            for name, path in multi_model_config.models.items():
                model_registry.register(name, path, 0)
            self._multi_model = MultiModelManager(
                model_name=self.model_name,
                model_registry=model_registry,
                pipeline=self._pipeline,
            )

    def _init_embedding_config(self) -> None:
        """Initialize embedding loader from constructor config."""
        # embedding_config is stored as self._embedding_config by constructor param
        config = getattr(self, "_embedding_config", None)
        self._init_embedding_loader(config)

    def _init_version_config(self) -> None:
        """Initialize version manager from constructor config."""
        config = getattr(self, "_version_config", None)
        self._init_version_manager(config)

    def _init_embedding_loader(self, embedding_config=None) -> None:
        """Initialize embedding and reranking model loader."""
        if not embedding_config:
            return
        embed_model = getattr(embedding_config, "embedding_model", "") or ""
        rerank_model = getattr(embedding_config, "rerank_model", "") or ""
        if not embed_model and not rerank_model:
            return

        from distllm.core.embedding_loader import EmbeddingModelLoader
        self._embedding_loader = EmbeddingModelLoader(
            embedding_model=embed_model or None,
            rerank_model=rerank_model or None,
            device="auto",
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if embed_model:
            self._embedding_loader.load_embedding_model()
            self._embedding_max_length = getattr(embedding_config, "max_length", 512)
            self._embedding_normalize = getattr(embedding_config, "normalize", True)
            self._embedding_batch_size = getattr(embedding_config, "batch_size", 32)
        if rerank_model:
            self._embedding_loader.load_rerank_model()

    def _init_version_manager(self, version_config=None) -> None:
        """Initialize version manager for shadow mode, A/B testing, blue-green."""
        if not version_config or not getattr(version_config, "enabled", False):
            return

        from distllm.deploy.version_manager import VersionManager
        self._version_manager = VersionManager(
            max_versions=getattr(version_config, "max_versions", 4),
            shadow_enabled=getattr(version_config, "shadow_enabled", False),
            shadow_pct=getattr(version_config, "shadow_pct", 0.0),
            blue_green_enabled=getattr(version_config, "blue_green_enabled", False),
            ab_testing_enabled=getattr(version_config, "ab_testing_enabled", False),
            ab_test_split=getattr(version_config, "ab_test_split", 50.0),
            auto_promote_enabled=getattr(version_config, "auto_promote_enabled", False),
            min_samples=getattr(version_config, "min_samples", 100),
            significance_level=getattr(version_config, "significance_level", 0.05),
        )

    def _init_model_hotswap(self, max_models: int = 4, total_gpu_memory_gb: float = 0.0) -> None:
        """Initialize multi-model hot-swap manager."""
        from distllm.core.multi_model_serving import ModelHotSwapManager
        from distllm.core.model_registry import ModelRegistry

        registry = ModelRegistry(max_models=max_models)
        registry.register(self.model_name, self.model_name, 0)

        def _load_model_callback(name: str, path: str):
            """Load a model and return (model, tokenizer, memory_gb)."""
            from distllm.models.partitioner import ModelPartitioner
            partitioner = ModelPartitioner(
                model_name=path,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                quantization_config=self.quantization_config,
            )
            partitioner.load_full_model()
            # Estimate memory
            mem_gb = 0.0
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            return partitioner.full_model, partitioner.tokenizer, mem_gb

        def _unload_model_callback(name: str, model, tokenizer):
            """Unload a model and free GPU memory."""
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._model_hotswap = ModelHotSwapManager(
            model_registry=registry,
            total_gpu_memory_gb=total_gpu_memory_gb,
            max_models=max_models,
            on_load_model=_load_model_callback,
            on_unload_model=_unload_model_callback,
        )

    def _init_paged_attention(
        self,
        num_blocks: int = 256,
        block_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        swap_to_cpu: bool = False,
        max_swap_blocks: int = 0,
    ) -> None:
        """Initialize PagedAttention block manager."""
        from distllm.core.paged_attention import PagedAttentionManager

        dtype = getattr(torch, self.dtype, torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._paged_attention = PagedAttentionManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            swap_to_cpu=swap_to_cpu,
            max_swap_blocks=max_swap_blocks,
        )

    def _init_vlm_pipeline(
        self,
        vision_model: str | None = None,
        llm_hidden_size: int = 4096,
    ) -> None:
        """Initialize VLM pipeline for multi-modal (image + text) support."""
        if not vision_model:
            return

        from distllm.core.vlm_pipeline import VLMPipeline
        self._vlm_pipeline = VLMPipeline(
            vision_model_name=vision_model,
            llm_hidden_size=llm_hidden_size,
            device="auto",
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self._vlm_pipeline.load_vision_tower()

    def _init_flash_attention(self, causal: bool = True, enable_fa2: bool = True) -> None:
        """Initialize FlashAttention wrapper.

        Attempts to load flash-attn-2 (or flash-attn v1 as fallback).
        Falls back to PyTorch SDPA when no CUDA kernel is available.

        Args:
            causal: Whether to use causal masking.
            enable_fa2: If False, skip initialization entirely.
        """
        if not enable_fa2:
            self._flash_attention = None
            return
        try:
            from distllm.core.flash_attention import FlashAttentionWrapper
            self._flash_attention = FlashAttentionWrapper(causal=causal)
            logger.info("FlashAttention-2 wrapper initialized")
        except ImportError:
            self._flash_attention = None
            logger.warning("FlashAttention module not available, using default attention")
        if self._flash_attention is not None:
            logger.info(f"FlashAttention initialized (available={self._flash_attention.is_available})")

    def _init_plugin_manager(self) -> None:
        """Initialize the plugin manager with coordinator context."""
        from distllm.core.plugin import PluginManager
        context = {
            "coordinator": self,
            "model_name": self.model_name,
            "dtype": str(self.dtype),
            "trust_remote_code": self.trust_remote_code,
        }
        self._plugin_manager = PluginManager(context=context)

    def _init_hybrid_parallel(self, config: Any = None) -> None:
        """Initialize hybrid parallelism engine (TP + PP + EP)."""
        self._hybrid_parallel_planner = None
        self._hybrid_parallel_executor = None
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.hybrid_parallel import (
            HardwareProber,
            HybridParallelPlanner,
            HybridParallelExecutor,
        )
        topology = HardwareProber.probe()
        self._hybrid_parallel_planner = HybridParallelPlanner(topology)
        plan = self._hybrid_parallel_planner.plan(
            total_layers=self.total_layers,
            num_experts=getattr(self._expert_registry, 'num_experts', 0) if self._expert_registry else 0,
            use_moe=self._moe_orchestrator is not None,
            pp_overlap=getattr(config, "pp_overlap", True),
            tp_enabled=getattr(config, "tp_enabled", True),
            ep_enabled=getattr(config, "ep_enabled", True),
        )
        force_tp = getattr(config, "force_tp_world_size", 0)
        if force_tp and force_tp > 1:
            from distllm.core.hybrid_parallel import ParallelStrategy
            plan.tp_world_size = force_tp
            if plan.strategy == ParallelStrategy.PP:
                plan.strategy = ParallelStrategy.TP_PP if plan.pp_num_stages > 1 else ParallelStrategy.TP
        self._hybrid_parallel_executor = HybridParallelExecutor(plan, coordinator=self)
        self._hybrid_parallel_executor.configure_pp(self._pipeline)
        self._hybrid_parallel_executor.launch_tp(
            model_name=self.model_name,
            dtype=self.dtype,
        )
        if hasattr(self, '_pipeline') and plan.pp_num_stages > 1:
            self._pipeline.enable_overlap = True
        logger.info(f"Hybrid parallel plan: {plan.explanation}")

    def _init_zero_copy(self, config: Any = None) -> None:
        """Initialize zero-copy GPU tensor transfer engine."""
        self._zero_copy_engine = None
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.zero_copy_transfer import ZeroCopyTransferEngine
        self._zero_copy_engine = ZeroCopyTransferEngine()
        logger.info("Zero-copy transfer engine initialized")

    def _init_adaptive_precision(self, config: Any = None) -> None:
        """Initialize adaptive precision pipeline."""
        self._adaptive_precision = None
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        cal_samples = getattr(config, 'calibration_samples', 64) if not isinstance(config, bool) else 64
        from distllm.core.adaptive_precision import AdaptivePrecisionEngine
        self._adaptive_precision = AdaptivePrecisionEngine(calibration_samples=cal_samples)
        logger.info("Adaptive precision engine initialized")

    def _init_predictive_cache(self, config: Any = None) -> None:
        """Initialize predictive KV cache management."""
        self._predictive_cache = None
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        gpu_mb = getattr(config, 'gpu_cache_mb', 512) if not isinstance(config, bool) else 512
        cpu_mb = getattr(config, 'cpu_cache_mb', 4096) if not isinstance(config, bool) else 4096
        compress_int = getattr(config, 'background_compress_interval_s', 300) if not isinstance(config, bool) else 300
        from distllm.core.predictive_cache import PredictiveCacheManager
        from distllm.core.prefix_cache import PrefixCache
        gpu_cache = PrefixCache(
            max_entries=0,
            memory_budget_bytes=gpu_mb * 1024 * 1024,
        ) if self._cache_mgr else None
        self._predictive_cache = PredictiveCacheManager(
            gpu_cache=gpu_cache,
            gpu_memory_bytes=gpu_mb * 1024 * 1024,
            cpu_memory_bytes=cpu_mb * 1024 * 1024,
        )
        self._predictive_cache.start_background_compression(compress_int)
        logger.info(f"Predictive cache initialized (GPU={gpu_mb}MB, CPU={cpu_mb}MB)")

    def _init_self_optimizing(self, config: Any = None) -> None:
        """Initialize self-optimizing engine."""
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.self_optimizing_engine import SelfOptimizingEngine
        self._self_optimizing = SelfOptimizingEngine(
            model_name=self.model_name,
            profile_dir=getattr(config, 'profile_dir', None),
            tune_interval_seconds=getattr(config, 'tune_interval_seconds', 60.0),
            warmup_seconds=getattr(config, 'warmup_seconds', 30.0),
            apply_params=self._apply_tunable_params,
        )
        logger.info("Self-optimizing engine initialized")

    def _apply_tunable_params(self, params) -> None:
        """Apply auto-tuned parameters to the live system."""
        if self.scheduler is not None and hasattr(params, 'batch_size'):
            if params.batch_size != self.scheduler.max_batch_size:
                self.scheduler.max_batch_size = params.batch_size
        if self._spec_decoder is not None and hasattr(params, 'speculative_decoding_enabled') and params.speculative_decoding_enabled is not None:
            self._spec_decoder._enabled = params.speculative_decoding_enabled

    def _init_cuda_graph(self, config: Any = None) -> None:
        """Initialize CUDA graph pool (deferred to model load time)."""
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        self._cuda_graph_batch_sizes = getattr(config, 'batch_sizes', [1, 2, 4, 8, 16, 32])
        logger.info("CUDA graph capture enabled (will initialize after model load)")

    def _init_compile_support(self, config: Any = None) -> None:
        """Initialize torch.compile support (deferred to model load time)."""
        self._compile_enabled = False
        if config is None:
            return
        self._compile_enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if self._compile_enabled:
            self._compile_mode = getattr(config, 'mode', 'reduce-overhead')
            self._compile_fullgraph = getattr(config, 'fullgraph', False)
            logger.info("torch.compile enabled (will apply after model load)")

    def _init_slora(self, config: Any = None) -> None:
        """Initialize SLoRA manager (deferred to model load time)."""
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        self._slora_max_adapters = getattr(config, 'max_adapters', 64)
        logger.info("SLoRA multi-adapter serving enabled (will initialize after model load)")

    def _init_rag(self, config: Any = None) -> None:
        """Initialize RAG pipeline."""
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.rag_pipeline import RAGPipeline
        embedding_fn = getattr(self._embedding_loader, 'encode', None) if hasattr(self, '_embedding_loader') and self._embedding_loader else None
        if embedding_fn is None:
            logger.warning("RAG enabled but embedding_loader not available — skipping")
            return
        self._rag_pipeline = RAGPipeline(
            embedding_fn=embedding_fn,
            dimension=getattr(config, 'dimension', 768),
            chunk_size=getattr(config, 'chunk_size', 512),
            chunk_overlap=getattr(config, 'chunk_overlap', 50),
            index_path=getattr(config, 'index_path', None),
        )
        logger.info("RAG pipeline initialized")

    def _init_agent(self, config: Any = None) -> None:
        """Initialize agent loop."""
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.agent_loop import AgentLoop
        def llm_fn(prompt: str) -> str:
            return self.generate(prompt, max_new_tokens=512, temperature=0.7)
        self._agent_loop = AgentLoop(
            llm_fn=llm_fn,
            max_iterations=getattr(config, 'max_iterations', 10),
            reflection_enabled=getattr(config, 'reflection_enabled', True),
        )
        logger.info("Agent loop initialized")

    def _init_disagg(self, config: Any = None) -> None:
        """Initialize disaggregated serving."""
        if config is None:
            return
        enabled = getattr(config, 'enabled', False) if not isinstance(config, bool) else config
        if not enabled:
            return
        from distllm.core.disagg_serving import DisaggRouter, DisaggOrchestrator
        router = DisaggRouter(local_coordinator=self, local_model_name=self.model_name)
        for node in getattr(config, 'prefill_nodes', []):
            router.add_prefill_node(**node)
        for node in getattr(config, 'decode_nodes', []):
            router.add_decode_node(**node)
        self._disagg_orchestrator = DisaggOrchestrator(router=router)
        logger.info("Disaggregated orchestrator initialized")

    def _init_moe(self, moe_config: MoESettings | None) -> None:
        """Initialize MoE subsystem."""
        if moe_config and moe_config.enabled:
            from distllm.core.expert_registry import ExpertRegistry
            from distllm.core.moe_orchestrator import MoEOrchestrator
            self._expert_registry = ExpertRegistry()
            self._moe_orchestrator = MoEOrchestrator(expert_registry=self._expert_registry)
            if self._multi_model is None:
                self._multi_model = MultiModelManager(
                    model_name=self.model_name,
                    pipeline=self._pipeline,
                    moe_orchestrator=self._moe_orchestrator,
                )
            else:
                self._multi_model.moe_orchestrator = self._moe_orchestrator

    def register_model(self, name: str, path: str, total_layers: int) -> ModelEntry:
        """Register an additional model."""
        from distllm.core.model_registry import ModelEntry
        if self._multi_model is None:
            self._multi_model = MultiModelManager(
                model_name=self.model_name,
                pipeline=self._pipeline,
            )
        return self._multi_model.register_model(name, path, total_layers)

    def list_models(self) -> list[str]:
        """List all registered model names, including hybrid router models."""
        if self._multi_model is None:
            models = [self.model_name]
        else:
            models = self._multi_model.list_models()
        # Append hybrid router model names so clients can discover them
        if self._chat_router is not None:
            for hname in self._chat_router.list_hybrid_models():
                if hname and hname not in models:
                    models.append(hname)
        return models

    def get_model_name(self, requested: str | None = None) -> str:
        """Resolve model name: requested > registry default > self.model_name."""
        if self._multi_model is None:
            return self.model_name
        return self._multi_model.get_model_name(requested)

    def warm_cache(self, prompts: list[str]) -> int:
        """Warm caches by running prompts through the pipeline."""
        warmer = CacheWarmer()
        return warmer.warm(prompts, self)

    def _gossip_loop(self):
        """Background daemon that runs periodic gossip rounds."""
        interval = 10.0  # default, could be read from gossip_config
        while self._running.is_set():
            try:
                time.sleep(interval)
                if self._cache_mgr is not None:
                    discovered = self._cache_mgr.sync_with_peers()
                    if discovered > 0:
                        logger.debug(f"Gossip round: discovered {discovered} new cache entries")

                    # Evict old entries from the prefix cache
                    if self._cache_mgr.prefix_cache and hasattr(self._cache_mgr.prefix_cache, "_root"):
                        # RadixTreeCache: evict LRU entries
                        self._cache_mgr.prefix_cache._root.evict_lru(self._cache_mgr.prefix_cache.max_entries)
            except Exception:
                logger.debug("Gossip round error (non-fatal)", exc_info=True)

    def register_expert_on_node(self, node_id: str, expert_ids: list[int], layer_idx: int = 0):
        """Register experts on a node in the expert registry."""
        if self._expert_registry is None:
            return
        for eid in expert_ids:
            self._expert_registry.register_expert(eid, node_id, layer_idx)
        logger.info(f"Registered experts {expert_ids} on {node_id}")

    def moe_forward(self, hidden_states: torch.Tensor, moe_router: "MoERouter") -> torch.Tensor:
        """Execute MoE forward pass via distributed expert orchestration."""
        if self._multi_model is None or self._multi_model.moe_orchestrator is None:
            raise RuntimeError("MoE orchestrator not initialized")
        return self._multi_model.moe_forward(hidden_states, moe_router)

    # -- Metrics --

    def record_metric(self, metric_name: str, value: float):
        """Record a metric value (thread-safe)."""
        self._metrics_mgr.record(metric_name, value)

    def get_metrics(self) -> dict:
        """Get current metrics in Prometheus-compatible format (thread-safe)."""
        base = self._metrics_mgr.get_prometheus()
        base["recovery"] = self._recovery.get_metrics()
        return base

    # -- Delegation to ResourceManager --

    def _check_circuit_breaker(self, node_id: str) -> bool:
        return self._resource_mgr.check_circuit_breaker(node_id)

    def _record_node_success(self, node_id: str):
        self._resource_mgr.record_success(node_id)

    def _record_node_failure(self, node_id: str):
        self._resource_mgr.record_failure(node_id)
        # Sync to coordinator metrics for backward compat
        self._metrics_mgr.increment("node_failures")
        self._metrics_mgr.increment("errors")

    # -- Node Registration (delegated to NodeRegistrar) --

    def auto_setup(self, nodes_config: list[dict]) -> None:
        """Automatically partition model and assign layers to nodes."""
        model_info, total_layers = self._node_registrar.auto_setup(nodes_config)
        self.model_info = model_info
        self.total_layers = total_layers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

    def manual_register(self, node_id: str, host: str, port: int, start_layer: int, end_layer: int, total_layers: int | None = None, role: NodeRole = NodeRole.AUTO, expert_ids: list[int] | None = None, cluster_id: str = "default"):
        """Manually register a node."""
        self._node_registrar.manual_register(
            node_id=node_id,
            host=host,
            port=port,
            start_layer=start_layer,
            end_layer=end_layer,
            total_layers=total_layers,
            role=role,
            expert_ids=expert_ids,
            cluster_id=cluster_id,
        )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

        if self.model_info is None:
            self.model_info = get_model_info(self.model_name, self.trust_remote_code)
            if total_layers is None:
                self.total_layers = self.model_info["num_layers"]

    def _validate_layer_assignment(self, node_id: str, start_layer: int, end_layer: int):
        self._pipeline.validate_layer_assignment(node_id, start_layer, end_layer)

    # -- Token Sampling (delegates to TokenGenerator) --

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
        return self._pipeline_runner._sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)

    def _sample_batch(self, logits: torch.Tensor, batch: ScheduledBatch) -> torch.Tensor:
        return self._pipeline_runner._sample_batch(logits, batch)

    def _speculative_tokens_to_append(
        self,
        draft_tokens: list[int] | torch.Tensor,
        target_logits: torch.Tensor,
        accepted_count: int,
        accepted_tokens: list[int],
        next_token: int,
    ) -> list[int]:
        return self._pipeline_runner._speculative_tokens_to_append(
            draft_tokens, target_logits, accepted_count, accepted_tokens, next_token
        )

    # -- Generation --

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0, request_id: str | None = None, user_id: str = "default", speculative_config: dict | None = None) -> str:
        return self._pipeline_runner.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
            request_id=request_id, user_id=user_id, speculative_config=speculative_config,
        )

    def generate_async(
        self,
        prompt: str,
        request_id: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        schema: dict | None = None,
        response_format: dict | None = None,
        priority: int = 2,
        adapter_id: str | None = None,
        include_logprobs: bool = False,
        top_logprobs: int = 0,
        logit_bias: dict[int, float] | None = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_latency_ms: float | None = None,
        user_id: str = "default",
    ) -> str:
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
        self._pipeline_runner.generate_batch(timeout=timeout, max_steps=max_steps)

    def _generate_local_batch(self, batch: ScheduledBatch) -> None:
        self._pipeline_runner._generate_local_batch(batch)

    def _run_distributed_pipeline_batch(self, batch: ScheduledBatch) -> None:
        self._pipeline_runner._run_distributed_pipeline_batch(batch)

    def _run_async_pipeline_batch(self, batch: ScheduledBatch) -> None:
        self._pipeline_runner._run_async_pipeline_batch(batch)
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        # Build input tensors for all sequences in the batch
        input_tensors = []
        for seq_idx, seq in enumerate(batch.sequences):
            if batch.is_prefill[seq_idx]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
            else:
                tokens = [seq.decode_input_token]
            input_tensors.append(torch.tensor([tokens], dtype=torch.long))

        # Stack into a single batch tensor [batch_size, seq_len]
        max_len = max(t.shape[1] for t in input_tensors)
        padded_tensors = []
        for t in input_tensors:
            if t.shape[1] < max_len:
                padding = torch.zeros((1, max_len - t.shape[1]), dtype=torch.long)
                t = torch.cat([t, padding], dim=1)
            padded_tensors.append(t)

        batch_input = torch.cat(padded_tensors, dim=0)  # [batch_size, max_seq_len]

        # Speculative decoding: generate drafts before pipeline run
        use_spec = (
            self._spec_decoder is not None
            and self._spec_decoder.is_enabled
            and self.draft_model is not None
            and not all(batch.is_prefill)
        )

        draft_tokens_list = None
        if use_spec:
            draft_tokens_list, _ = self._spec_decoder.generate_batch_draft_tokens(
                self.draft_model, input_tensors
            )

        # Wrap existing pipeline run_pipeline as a stage forward function
        # The async engine splits the batch into micro-batches and runs them
        # through stages with 1F1B/interleaved scheduling
        def stage_forward(micro_batch: torch.Tensor) -> torch.Tensor:
            """Forward pass for a micro-batch through the full pipeline.

            This delegates to the existing distributed pipeline execution
            but with micro-batch granularity for async scheduling.
            """
            # Initialize KV caches for this micro-batch
            micro_kv_caches = self._pipeline.create_node_kv_caches()

            if self._pipeline.enable_overlap:
                logits = self._pipeline.run_pipeline_overlap(
                    micro_batch, micro_kv_caches, request_id="async_micro"
                )
            else:
                logits = self._pipeline.run_pipeline(
                    micro_batch, micro_kv_caches, request_id="async_micro"
                )
            return logits

        # Create a single stage wrapping the full pipeline (for now)
        # In a full implementation, each node would be a separate stage
        if not self._async_pipeline._stages:
            from distllm.core.async_pipeline import AsyncPipelineStage
            stage = AsyncPipelineStage(
                stage_id=0,
                forward_fn=stage_forward,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self._async_pipeline.add_stage(stage)

        # Run through async pipeline engine
        logits = self._async_pipeline.forward(batch_input)

        # Sample next tokens for each sequence
        if use_spec and draft_tokens_list is not None:
            target_logits_list = [logits[i:i+1] for i in range(len(batch.sequences))]
            results = self._spec_decoder.verify_batch(
                draft_tokens_list=draft_tokens_list,
                target_logits_list=target_logits_list,
                tokenizer=self.tokenizer,
            )
            if self._continuous_trainer is not None:
                for idx, (_, accepted, _) in enumerate(results):
                    if accepted and idx < len(draft_tokens_list):
                        dt = draft_tokens_list[idx]
                        draft_ids = dt.tolist() if hasattr(dt, 'tolist') else list(dt) if dt else []
                        if draft_ids:
                            self._continuous_trainer.record(draft_ids, list(accepted))
            for i, seq in enumerate(batch.sequences):
                next_token = results[i][2]
                seq._async_next_token = torch.tensor([next_token], dtype=torch.long)
        else:
            for i, seq in enumerate(batch.sequences):
                seq_logits = logits[i:i+1, -1, :]
                if seq.constraint is not None:
                    mask = seq.constraint.get_logits_mask(seq_logits.shape[-1], self.tokenizer)
                    seq_logits = seq_logits.masked_fill(~mask, float('-inf'))

                token = self._sample(seq_logits, temperature=seq.temperature, top_p=seq.top_p, top_k=seq.top_k)
                # Store in batch.sequences for scheduler.step
                if not hasattr(seq, '_async_next_token'):
                    seq._async_next_token = token
                else:
                    seq._async_next_token = token

        # Collect tokens into tensor for scheduler
        next_tokens_list = [seq._async_next_token for seq in batch.sequences]
        next_tokens_tensor = torch.stack(next_tokens_list).squeeze(-1)
        with self._batch_kv_caches_lock:
            kv_copy = dict(self._batch_kv_caches)
        decoded = [self.tokenizer.decode([int(next_tokens_tensor[i])]) if batch.sequences[i].constraint is not None else None for i in range(len(batch.sequences))]
        self.scheduler.step(batch, next_tokens_tensor, kv_caches=kv_copy, decoded_tokens=decoded)

        # Log async pipeline stats
        if is_debug_mode():
            logger.debug(f"Async pipeline stats: {self._async_pipeline.summary()}")

    # -- Model Loading (delegated to ModelManager) --

    def load_local_model(self):
        """Load the full model locally (for single-node testing)."""
        self._model_mgr.load_local_model(self)
        self._apply_flash_attention()
        self._apply_rope_scaling()
        self._wire_paged_attention()
        self._apply_adaptive_precision()

        # CUDA graph capture
        if getattr(self, '_cuda_graph_batch_sizes', None) and self.local_partitioner is not None:
            model = self.local_partitioner.full_model
            config = model.config
            from distllm.core.cuda_graph import CUDAGraphPool
            self._cuda_graph_pool = CUDAGraphPool(
                model=model,
                batch_sizes=self._cuda_graph_batch_sizes,
                num_layers=getattr(config, 'num_hidden_layers', 0),
                num_heads=getattr(config, 'num_attention_heads', 0),
                head_dim=getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32),
            )
            self._cuda_graph_pool.capture_all()

        # torch.compile
        if getattr(self, '_compile_enabled', False) and self.local_partitioner is not None:
            from distllm.core.compile_support import compile_model
            self.local_partitioner.full_model = compile_model(
                self.local_partitioner.full_model,
                mode=getattr(self, '_compile_mode', 'reduce-overhead'),
                fullgraph=getattr(self, '_compile_fullgraph', False),
            )

        # SLoRA manager
        if getattr(self, '_slora_max_adapters', None) and self.local_partitioner is not None:
            from distllm.core.slora_manager import SLoRAManager
            self._slora_manager = SLoRAManager(
                base_model=self.local_partitioner.full_model,
                max_adapters=self._slora_max_adapters,
            )

    def _apply_flash_attention(self):
        """Patch the loaded model's attention layers to use FlashAttention-2."""
        if self.local_partitioner is not None and self.local_partitioner.full_model is not None:
            model = self.local_partitioner.full_model
            try:
                from distllm.core.flash_attention import apply_flash_attention_to_model
                patched = apply_flash_attention_to_model(model)
                if patched > 0:
                    logger.info(f"FlashAttention-2: patched {patched} attention modules in model")
            except ImportError:
                logger.debug("FlashAttention module not available, skipping patch")

    def _apply_rope_scaling(self):
        """Apply RoPE scaling for long context support (128K+).

        Uses YaRN scaling by default, with NTK-aware fallback.
        Only applies if target context > model's native max position.
        """
        if self.local_partitioner is not None and self.local_partitioner.full_model is not None:
            model = self.local_partitioner.full_model
            config = getattr(model, "config", None)
            if config is not None:
                max_pos = getattr(config, "max_position_embeddings", 4096)
                target_ctx = 131072
                if target_ctx > max_pos:
                    from distllm.models.partitioner import apply_rope_scaling
                    apply_rope_scaling(model, target_context_len=target_ctx, scaling_type="yarn")
                    logger.info(f"RoPE scaling applied: {max_pos} -> {target_ctx}")

    def _wire_paged_attention(self):
        """Initialize PagedAttention and wire it into the KV cache flow."""
        if self.local_partitioner is not None and self.local_partitioner.full_model is not None:
            model = self.local_partitioner.full_model
            config = getattr(model, "config", None)
            if config is not None:
                num_layers = getattr(config, "num_hidden_layers", 32)
                num_heads = getattr(config, "num_attention_heads", 32)
                head_dim = getattr(config, "hidden_size", 4096) // num_heads
                self._init_paged_attention(
                    num_blocks=512,
                    block_size=16,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
                if self._paged_attention is not None:
                    from distllm.core.kv_cache import PagedKVCacheBackend
                    self._paged_kv_backend = PagedKVCacheBackend(self._paged_attention)

    def _load_draft_model(self):
        """Load a smaller draft model for speculative decoding."""
        self._model_mgr.load_draft_model(self)

    def _apply_adaptive_precision(self):
        """Profile and apply per-layer adaptive precision after model load."""
        engine = getattr(self, '_adaptive_precision', None)
        if engine is None or self.local_partitioner is None:
            return
        model = self.local_partitioner.full_model
        if model is None:
            return
        try:
            sample_input = torch.randint(0, 100, (1, 64), device=next(model.parameters()).device)
            engine.profile_model(model, sample_input)
            converted = engine.apply_precision(model)
            logger.info(f"Adaptive precision: profiled & converted {converted} layers")
        except Exception as e:
            logger.warning(f"Adaptive precision profiling failed: {e}")

    # -- Health Checks (delegated to HealthChecker) --

    def health_check(self) -> dict:
        """Check health of all registered nodes."""
        self._health_checker.metrics_exporter = self.metrics_exporter
        return self._health_checker.check_all(self.nodes, self.node_order, self._check_circuit_breaker)

    async def health_check_async(self) -> dict:
        """Check health of all registered nodes (async)."""
        self._health_checker.metrics_exporter = self.metrics_exporter
        return await self._health_checker.check_all_async(self.nodes, self.node_order, self._check_circuit_breaker)

    # -- Server Lifecycle --

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
    def token_streaming_buffer(self, value: TokenStreamingBuffer | None) -> None:
        self._token_streaming_buffer = value

    @property
    def request_fingerprinter(self) -> RequestFingerprinter | None:
        return self._request_fingerprinter

    @property
    def rate_limiter(self) -> LeakyBucketRateLimiter | None:
        return self._rate_limiter

    def set_streaming_buffer(self, flush_handler=None, max_batch_size: int = 5, flush_interval_ms: float = 50.0) -> TokenStreamingBuffer:
        """Create and set a token streaming buffer with the given flush handler."""
        buf = TokenStreamingBuffer(
            flush_handler=flush_handler,
            max_batch_size=max_batch_size,
            flush_interval_ms=flush_interval_ms,
        )
        self._token_streaming_buffer = buf
        return buf

    def configure_adaptive_batching(self, model: str, p50: float | None = None, p99: float | None = None, max_batch: int | None = None) -> None:
        """Configure per-model SLO for adaptive batching."""
        if self._adaptive_batching is not None:
            self._adaptive_batching.set_slo(model, p50=p50, p99=p99, max_batch=max_batch)

    def rate_limit_key(self, key: str, rate: float | None = None, burst: int | None = None) -> None:
        """Configure rate limit for a specific key (user, IP, endpoint)."""
        if self._rate_limiter is not None:
            if rate is not None and burst is not None:
                self._rate_limiter.set_limit(key, rate, burst)

    def get_new_module_stats(self) -> dict:
        """Get stats from all new modules."""
        stats = {}
        if self._request_auditor:
            stats["auditor"] = self._request_auditor.stats()
        if self._prompt_cache_service:
            stats["prompt_cache"] = self._prompt_cache_service.stats()
        if self._graceful_degradation:
            stats["degradation"] = self._graceful_degradation.stats()
        if self._adaptive_batching:
            stats["adaptive_batching"] = self._adaptive_batching.all_stats()
        if self._rate_limiter:
            stats["rate_limiter"] = self._rate_limiter.stats()
        if self._request_fingerprinter:
            stats["fingerprinter"] = self._request_fingerprinter.stats()
        return stats

    def start(self, blocking: bool = True, on_stop: Callable | None = None):
        """Start the coordinator gRPC server."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

        # Load draft model early if configured (for distributed speculative decoding)
        if self._draft_model_name and self.draft_model is None:
            self._model_mgr.load_draft_model_early(self)

        # If model_info became available, update the scheduler for model-aware batching
        if self.model_info is not None and self.scheduler is not None:
            self.scheduler._model_info = self.model_info
            self.scheduler._use_length_grouping = True

        # Signal background threads that coordinator is running
        self._running.set()

        # Launch gossip loop if enabled
        if self._gossip_protocol is not None:
            self._gossip_loop_task = threading.Thread(
                target=self._gossip_loop, daemon=True, name="gossip-loop"
            )
            self._gossip_loop_task.start()

        # Apply RoPE scaling for long context if model info is available
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

        # Start self-optimizing engine
        if self._self_optimizing:
            self._self_optimizing.start()

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
        while self._running.is_set():
            time.sleep(self._rebalancer._settings.check_interval)
            if not self._running.is_set():
                break
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

    # ------------------------------------------------------------------
    # Self-healing: node recovery callbacks
    # ------------------------------------------------------------------

    def _on_resource_mgr_failure(self, node_id: str) -> None:
        """Called by ``ResourceManager`` when circuit breaker opens for a node.

        Delegates to ``NodeRecoveryManager`` which orchestrates drain,
        layer redistribution, and sequence recovery.
        """
        logger.warning(f"Resource manager reported failure for {node_id}")
        plan = self._recovery.on_node_failure(node_id)
        if plan.recovered_sequences:
            logger.info(
                f"Recovered {len(plan.recovered_sequences)} sequences "
                f"from failed node {node_id}"
            )

    def _on_recovery_drain(self, node_id: str) -> None:
        """Callback: drain a failed node — remove from active topology."""
        logger.info(f"Draining node {node_id}")
        self._pipeline.unregister_node(node_id)

    def _on_recovery_redistribute(
        self, failed_node_id: str, plan: NodeRecoveryPlan
    ) -> None:
        """Callback: redistribute the failed node's layers to survivors.

        Computes a new partition by dividing the total layers evenly across
        surviving nodes (since the failed node is already removed from topology
        by the drain callback). Updates ``NodeRegistration`` records.
        """
        with self._pipeline._topology_lock:
            survivors = sorted(
                self._pipeline.nodes.keys(),
                key=lambda nid: self._pipeline.nodes[nid].start_layer,
            )
            num_survivors = len(survivors)
            if num_survivors == 0:
                logger.error(f"No survivors to redistribute layers from {failed_node_id}")
                return

            total_layers = self._pipeline.total_layers
            layers_per_node = total_layers // num_survivors
            remainder = total_layers % num_survivors

            for i, nid in enumerate(survivors):
                start = i * layers_per_node + min(i, remainder)
                end = start + layers_per_node - 1
                if i < remainder:
                    end += 1
                node_reg = self._pipeline.nodes.get(nid)
                if node_reg:
                    redist = LayerRedistribution(
                        surviving_node_id=nid,
                        added_start_layer=node_reg.end_layer + 1 if i > 0 else 0,
                        added_end_layer=end,
                        new_start_layer=start,
                        new_end_layer=end,
                    )
                    node_reg.start_layer = start
                    node_reg.end_layer = end
                    plan.redistributions.append(redist)

            self._pipeline.node_order = survivors
            logger.info(
                f"Redistributed layers for {failed_node_id}: "
                f"{len(plan.redistributions)} survivors updated"
            )

    def _on_recovery_recover(
        self, failed_node_id: str, request_ids: list[str]
    ) -> list[dict]:
        """Callback: recover in-flight sequences from checkpoints.

        Returns empty snapshots — the surviving nodes will regenerate
        the KV cache from the prompt tokens stored in the checkpoint.
        """
        recovered = []
        for rid in request_ids:
            ckpt = self._recovery.get_checkpoint(rid)
            if ckpt is not None:
                recovered.append({
                    "request_id": rid,
                    "prompt_tokens": ckpt.prompt_tokens,
                    "generated_tokens": ckpt.generated_tokens,
                    "kv_cache": ckpt.kv_cache,
                })
        logger.info(
            f"Recovering {len(recovered)} sequences from {failed_node_id}"
        )
        return recovered

    def _on_recovery_mark_dead(self, node_id: str) -> None:
        """Callback: final cleanup after a failed node."""
        logger.warning(f"Node {node_id} marked as dead — pipeline reconfigured")

    def check_recovery(self, request_id: str) -> bool:
        """Check if a request was recovered from a node failure.
        
        Used to inject ``x-distllm-recovered`` into responses.
        Returns True once per request, then clears the flag.
        """
        return self._recovery.consume_recovered_flag(request_id)

    def get_recovery_metrics(self) -> dict:
        """Get self-healing recovery metrics."""
        return self._recovery.get_metrics()

    # ------------------------------------------------------------------
    # Request-level SLA: latency-aware routing & model fallback
    # ------------------------------------------------------------------

    def register_fallback_models(self, model: str, fallbacks: list[str]) -> None:
        """Register fallback models for SLA-aware model downgrade."""
        self._fallback_models[model] = fallbacks

    def _get_nodes_within_sla(self, max_latency_ms: float) -> set[str]:
        """Return node IDs whose average latency is below the threshold."""
        if self._latency_tracker is None:
            return set(self._pipeline.nodes.keys())
        sla_nodes: set[str] = set()
        for node_id in self._pipeline.nodes:
            avg = self._latency_tracker.get_avg(node_id)
            if avg is None or avg < max_latency_ms:
                sla_nodes.add(node_id)
        return sla_nodes

    def _get_fallback_model(self, model: str) -> str | None:
        """Return the fastest fallback model whose latency meets SLA."""
        fallbacks = self._fallback_models.get(model, [])
        if not fallbacks:
            return None
        return fallbacks[-1]  # smallest / fastest model as final fallback

    def wait_for_termination(self):
        """Block until the coordinator server terminates."""
        if self.server:
            try:
                self.server.wait_for_termination()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()

    def enable_continuous_training(
        self,
        base_model: torch.nn.Module,
        draft_head: torch.nn.Module | None = None,
        config: ContinuousTrainConfig | None = None,
    ) -> None:
        """Enable continuous speculative fine-tuning during serving.

        Args:
            base_model: The target base model for hidden state extraction.
            draft_head: Optional draft head module. If None, uses spec decoder's heads.
            config: Training configuration.
        """
        head = draft_head or (
            self._spec_decoder._eagle_heads if self._spec_decoder and self._spec_decoder.has_eagle_heads else None
        )
        if head is None:
            logger.warning("Continuous training requires a draft head module")
            return

        device = next(base_model.parameters()).device
        self._continuous_trainer = ContinuousSpeculativeTrainer(
            base_model=base_model,
            draft_head=head,
            config=config or ContinuousTrainConfig(),
            device=str(device),
        )
        self._continuous_trainer.start_background()
        logger.info("Continuous speculative training enabled")

    def set_deterministic_mode(self, enabled: bool = True, seed: int = 42) -> None:
        """Enable or disable deterministic debug mode."""
        if enabled:
            self._deterministic_mode.enable(seed)
        else:
            self._deterministic_mode.disable()

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        """Get recent requests from the replay buffer."""
        return self._replay_buffer.list_recent(n)

    def replay_request(self, request_id: str) -> str | None:
        """Replay a stored request through the generate path."""
        entry = self._replay_buffer.get(request_id)
        if entry is None:
            return None

        if self.scheduler is not None:
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

    def stop(self):
        """Stop the coordinator with graceful shutdown."""
        logger.info("Initiating graceful shutdown...")

        # Phase 0: Signal background threads to stop
        self._running.clear()

        # Phase 0a: Stop continuous speculative training
        if self._continuous_trainer is not None:
            self._continuous_trainer.stop()
            logger.info("Continuous speculative training stopped")

        # Phase 1: Stop accepting new requests
        self._shutting_down = True
        logger.info("Phase 1: Stopped accepting new requests")

        # Phase 2: Wait for in-flight requests to complete (up to 30s)
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

        # Phase 6: Cleanup request state (thread-safe)
        self._request_tracker.clear()

        # Phase 7: Shutdown plugins if loaded
        if hasattr(self, '_plugin_manager') and self._plugin_manager:
            logger.info("Phase 7: Shutting down plugins...")
            self._plugin_manager.shutdown_all()

        # Phase 8: Join background threads
        if hasattr(self, '_gossip_loop_task') and self._gossip_loop_task:
            self._gossip_loop_task.join(timeout=5.0)
        if hasattr(self, '_rebalancer_task') and self._rebalancer_task:
            self._rebalancer_task.join(timeout=5.0)

        # Phase 9: Close gossip client
        if self._gossip_client:
            self._gossip_client.close()

        # Phase 10: Shutdown pipeline (NCCL transport, thread pool)
        self._pipeline.shutdown()

        # Phase 11: Stop subsystems
        self._subsystems.stop_all()
        if self._self_optimizing:
            self._self_optimizing.stop()
        if self._slora_manager:
            self._slora_manager.shutdown()
        if self._agent_loop:
            self._agent_loop.stop()

        # Phase 12: Log recovery metrics
        rec_metrics = self._recovery.get_metrics()
        if rec_metrics.get("recoveries", 0) > 0:
            logger.info(
                f"Recovery stats: {rec_metrics['recoveries']} events, "
                f"{rec_metrics['sequences_recovered']} seqs recovered, "
                f"{rec_metrics['sequences_lost']} seqs lost"
            )

        logger.info("Graceful shutdown complete")

    async def stop_async(self):
        """Stop the coordinator with graceful shutdown (async)."""
        logger.info("Initiating graceful shutdown (async)...")

        # Phase 0: Signal background threads to stop
        self._running.clear()

        # Phase 1: Stop accepting new requests
        self._shutting_down = True
        logger.info("Phase 1: Stopped accepting new requests")

        # Phase 2: Wait for in-flight requests (up to 30s)
        with self._request_tracker._lock:
            events = list(self._request_tracker._events.values())
        if events:
            logger.info(f"Phase 2: Waiting for {len(events)} in-flight requests...")
            for event in events:
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

        # Phase 6: Cleanup request state (thread-safe)
        self._request_tracker.clear()

        # Phase 7: Shutdown plugins
        if hasattr(self, '_plugin_manager') and self._plugin_manager:
            logger.info("Phase 7: Shutting down plugins...")
            self._plugin_manager.shutdown_all()

        # Phase 8: Join background threads
        if hasattr(self, '_gossip_loop_task') and self._gossip_loop_task:
            self._gossip_loop_task.join(timeout=5.0)
        if hasattr(self, '_rebalancer_task') and self._rebalancer_task:
            self._rebalancer_task.join(timeout=5.0)

        # Phase 9: Close gossip client
        if self._gossip_client:
            self._gossip_client.close()

        # Phase 10: Shutdown pipeline (NCCL transport, thread pool)
        self._pipeline.shutdown()

        # Phase 11: Stop subsystems
        self._subsystems.stop_all()
        if self._self_optimizing:
            self._self_optimizing.stop()
        if self._slora_manager:
            self._slora_manager.shutdown()
        if self._agent_loop:
            self._agent_loop.stop()

        logger.info("Graceful shutdown complete (async)")

    # -- Async Generation (Phase 5) --

    async def generate_async_v2(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        request_id: str | None = None,
    ) -> str:
        """Properly async text generation.

        Uses asyncio.to_thread() for blocking operations (tokenizer, model inference).
        """
        if not self.node_order and self.local_partitioner is None:
            raise NodeError("No nodes registered and no local model loaded")

        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_local_model() or register nodes first.")

        request_id = request_id or str(uuid.uuid4())
        token = _current_request_id_ctx.set(request_id)
        self._param_update_channel.register(request_id)

        self.record_metric("total_requests", 1)
        start_time = time.time()

        prompt_len = len(self.tokenizer.encode(prompt)) if self.tokenizer else 0

        try:
            # Tokenize in thread pool
            input_ids = await asyncio.to_thread(
                self.tokenizer.encode, prompt, return_tensors="pt"
            )

            if self.node_order:
                input_ids = input_ids.to("cpu")
                prompt_len = input_ids.shape[1]
                total_capacity = min(prompt_len + max_new_tokens, self.config.max_context_length)
                generated_ids = torch.zeros(1, total_capacity, dtype=torch.long)
                generated_ids[:, :prompt_len] = input_ids
                gen_pos = prompt_len

                node_kv_caches = self._pipeline.create_node_kv_caches()

                active_method = self._spec_decoder.get_active_method(self.draft_model) if self._spec_decoder else None
                use_speculative = (
                    self._spec_decoder is not None
                    and self._spec_decoder.is_enabled
                    and active_method in ("draft_model", "medusa", "ngram")
                )

                # Predictive cache: observe request patterns & pre-warm
                predictive_cache = self._predictive_cache
                if predictive_cache is not None:
                    token_ids = input_ids[0].tolist()
                    predictions = predictive_cache.observe_request(token_ids)
                    for pred in predictions:
                        if pred.should_prefetch and pred.confidence > 0.3:
                            self._cache_mgr.lookup_prefix(list(pred.prefix_tokens))

                step = 0
                while step < max_new_tokens:
                    step_input = generated_ids[:, :gen_pos] if gen_pos == prompt_len else generated_ids[:, gen_pos-1:gen_pos]
                    step_request_id = str(uuid.uuid4())

                    draft_tokens = None
                    if use_speculative:
                        if active_method == "draft_model" and self.draft_model is not None:
                            draft_tokens, _ = await asyncio.to_thread(
                                self._spec_decoder.generate_draft_tokens,
                                self.draft_model, step_input,
                            )
                        elif active_method == "ngram":
                            generated_list = generated_ids[0].tolist() if generated_ids.dim() == 2 else generated_ids.tolist()
                            draft_tokens, _ = await asyncio.to_thread(
                                self._spec_decoder.generate_draft_tokens,
                                None, step_input, generated_ids=generated_list,
                            )

                    hybrid_executor = self._hybrid_parallel_executor
                    if hybrid_executor is not None:
                        logits = await asyncio.to_thread(
                            hybrid_executor.execute,
                            step_input, node_kv_caches, step_request_id,
                            draft_tokens if use_speculative else None,
                        )
                    else:
                        logits = await self._pipeline.run_pipeline_async(
                            step_input, node_kv_caches, step_request_id,
                            draft_tokens=draft_tokens if use_speculative else None,
                        )

                    if use_speculative and active_method == "medusa":
                        draft_tokens, _ = await asyncio.to_thread(
                            self._spec_decoder.generate_draft_tokens,
                            None, step_input, target_logits=logits,
                        )

                    if use_speculative and draft_tokens:
                        accepted_count, accepted_tokens, next_token = await asyncio.to_thread(
                            self._spec_decoder.verify_and_accept,
                            draft_tokens, logits, self.tokenizer,
                        )
                        if self._continuous_trainer is not None and accepted_tokens:
                            draft_ids = draft_tokens.tolist() if hasattr(draft_tokens, 'tolist') else list(draft_tokens) if draft_tokens else []
                            self._continuous_trainer.record(draft_ids, list(accepted_tokens))
                        tokens_to_append = self._speculative_tokens_to_append(
                            draft_tokens, logits, accepted_count, accepted_tokens, next_token
                        )
                        tokens_to_append = tokens_to_append[: max_new_tokens - step]
                        hit_eos = False
                        for token_id in tokens_to_append:
                            generated_ids[:, gen_pos] = token_id
                            gen_pos += 1
                            if token_id == self.tokenizer.eos_token_id:
                                hit_eos = True
                                break
                        if hit_eos:
                            break
                        step += len(tokens_to_append)
                    else:
                        next_token = self._sample(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
                        generated_ids[:, gen_pos] = next_token.item()
                        gen_pos += 1
                        step += 1
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                result = await asyncio.to_thread(
                    self.tokenizer.decode, generated_ids[0, :gen_pos], skip_special_tokens=True
                )
                tokens_generated = gen_pos - prompt_len
            else:
                result = await asyncio.to_thread(
                    self._generate_local_sync, prompt, max_new_tokens, temperature, top_p
                )
                tokens_generated = len(self.tokenizer.encode(result)) - len(self.tokenizer.encode(prompt))

            elapsed = time.time() - start_time
            self.record_metric("total_tokens_generated", tokens_generated)
            self.record_metric("total_generation_time", elapsed)

            # Prometheus exporter updates (was missing)
            if self.metrics_exporter:
                self.metrics_exporter.tokens_generated.inc(tokens_generated)
                self.metrics_exporter.token_latency.observe(elapsed)
                if elapsed > 0:
                    self.metrics_exporter.tokens_per_second.set(tokens_generated / elapsed)

            return result

        except (NodeUnreachableError, OOMError, GRPCTimeoutError, NodeError) as e:
            self.record_metric("errors", 1)
            if self.metrics_exporter:
                self.metrics_exporter.errors_total.labels(type=type(e).__name__).inc()
            raise
        except Exception as e:
            self.record_metric("errors", 1)
            if self.metrics_exporter:
                self.metrics_exporter.errors_total.labels(type=type(e).__name__).inc()
            raise

        finally:
            self._param_update_channel.unregister(request_id)
            _current_request_id_ctx.reset(token)

    def _generate_local_sync(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ) -> str:
        return self._pipeline_runner._generate_local_sync(prompt, max_new_tokens, temperature, top_p)

    def _generate_local_eagle_sync(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 0,
    ) -> str:
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
