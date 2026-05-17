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
from distllm.config.settings import DistLLMSettings, MultiModelSettings, MoESettings
from distllm.communication.grpc import set_debug_mode
from distllm.core.moe_router import MoERouter
from distllm.core.prefix_cache import PrefixCache
from distllm.errors.types import (
    ConfigValidationError,
    NodeError,
    NodeUnreachableError,
    OOMError,
    GRPCTimeoutError,
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
from distllm.core.coordinator_metrics import MetricsManager
from distllm.core.coordinator_model import ModelManager
from distllm.core.coordinator_health import HealthChecker
from distllm.core.coordinator_lifecycle import ServerLifecycle, RequestTracker
from distllm.core.coordinator_nodes import NodeRegistrar
from distllm.core.coordinator_multi_model import MultiModelManager


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
        radix_tree_cache_enabled: bool = False,
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
            radix_tree_cache_enabled=radix_tree_cache_enabled,
            chunked_prefill_enabled=chunked_prefill_enabled,
            chunked_prefill_chunk_size=chunked_prefill_chunk_size,
        )

        # Component: PipelineOrchestrator (node topology, distributed pipeline)
        self._pipeline = PipelineOrchestrator(
            resource_mgr=self._resource_mgr,
        )

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
        self._spec_decoder: Optional[SpeculativeDecoder] = None
        self._spec_method = "draft_model"
        if speculative_config:
            self._draft_model_name = speculative_config.get("draft_model") or None
            self.num_assistant_tokens = speculative_config.get("num_assistant_tokens", 5)
            self._spec_method = speculative_config.get("method", "draft_model")
            self._spec_decoder = SpeculativeDecoder(
                num_assistant_tokens=self.num_assistant_tokens,
                method=self._spec_method,
                medusa_num_heads=speculative_config.get("medusa_num_heads", 4),
                medusa_num_tokens_per_head=speculative_config.get("medusa_num_tokens_per_head", 3),
                ngram_min_match=speculative_config.get("ngram_min_match", 4),
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

        # Request completion tracking (delegated to RequestTracker)
        self._request_tracker = RequestTracker()

        # Metrics tracking (delegated to MetricsManager)
        self._metrics_mgr = MetricsManager()

        # Node info for distributed state
        self.nodes_info = {}

        # Multi-model registry and MoE (delegated to managers)
        self._multi_model: Optional[MultiModelManager] = None
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

        # Persistent KV cache state for batch pipeline (keyed by request_id)
        self._batch_kv_caches: Dict[str, Dict[str, Optional[List]]] = {}
        self._batch_kv_caches_lock = threading.Lock()
        self._batch_event = threading.Event()

        # P2P KV cache gossip
        self._cache_index = None
        self._gossip_protocol = None
        self._gossip_client = None
        self._gossip_loop_task: Optional[threading.Thread] = None
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
                with self._nodes_lock:
                    nodes_snapshot = dict(self.nodes)
                for node_id, reg in nodes_snapshot.items():
                    if node_id == peer_id or peer_id in node_id:
                        return reg.host, reg.port
                return None

            self._gossip_client = GossipClient(peer_resolver=resolve_peer)
            # Wire gossip to cache manager
            self._cache_mgr.cache_index = self._cache_index
            self._cache_mgr.gossip_protocol = self._gossip_protocol
            self._cache_mgr.gossip_client = self._gossip_client

        # Streaming parameter updates
        from distllm.core.param_update_channel import ParamUpdateChannel
        self._param_update_channel = ParamUpdateChannel()

        # Cross-cluster federation
        from distllm.core.cluster_topology import FederationManager, CrossClusterLatencyMonitor
        self._federation_manager = FederationManager()
        self._latency_monitor = CrossClusterLatencyMonitor(self._federation_manager)

        # Wire federation and expert resources to node registrar
        self._node_registrar.federation_manager = self._federation_manager
        self._node_registrar.expert_registry = self._expert_registry

        # Embedding and reranking model loader
        self._embedding_config = embedding_config
        self._version_config = version_config
        self._embedding_loader = None
        self._embedding_max_length = 512
        self._embedding_normalize = True
        self._embedding_batch_size = 32

        # Version manager (shadow mode, A/B testing, blue-green)
        self._version_manager = None

        # Multi-model hot-swap manager
        self._model_hotswap = None

        # PagedAttention block manager
        self._paged_attention = None
        self._paged_kv_backend = None

        # VLM (Vision-Language Model) pipeline
        self._vlm_pipeline = None

        # FlashAttention wrapper
        self._flash_attention = None

        # Plugin manager
        self._plugin_manager: Optional['PluginManager'] = None

        # Hybrid parallelism (TP + PP + EP)
        self._hybrid_parallel_planner = None
        self._hybrid_parallel_executor = None

        # Zero-copy GPU tensor transfer
        self._zero_copy_engine = None

        # Adaptive precision pipeline
        self._adaptive_precision = None

        # Predictive KV cache management
        self._predictive_cache = None

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
    def prefix_cache(self) -> "Optional[PrefixCache]":
        return self._cache_mgr.prefix_cache

    @prefix_cache.setter
    def prefix_cache(self, value: "Optional[PrefixCache]") -> None:
        self._cache_mgr.prefix_cache = value

    # -- Backward-compat for circuit breaker internals (used by tests) --

    @property
    def _node_failure_counts(self) -> Dict[str, int]:
        with self._resource_mgr._lock:
            return dict(self._resource_mgr._node_failure_counts)

    @_node_failure_counts.setter
    def _node_failure_counts(self, value: Dict[str, int]):
        with self._resource_mgr._lock:
            self._resource_mgr._node_failure_counts = dict(value)

    @property
    def _node_recovery_time(self) -> Dict[str, float]:
        with self._resource_mgr._lock:
            return dict(self._resource_mgr._node_recovery_time)

    @_node_recovery_time.setter
    def _node_recovery_time(self, value: Dict[str, float]):
        with self._resource_mgr._lock:
            self._resource_mgr._node_recovery_time = dict(value)

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
        return self._metrics_mgr.get()

    @property
    def _shutting_down(self) -> bool:
        return self._request_tracker.shutting_down

    @_shutting_down.setter
    def _shutting_down(self, value: bool):
        self._request_tracker.shutting_down = value

    @property
    def _request_results(self) -> Dict[str, str]:
        """Backward compat: access to request results dict."""
        return self._request_tracker._results

    @property
    def _request_events(self) -> Dict[str, threading.Event]:
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

    def _init_multi_model(self, multi_model_config: Optional[MultiModelSettings]) -> None:
        """Initialize multi-model and MoE subsystems."""
        self._multi_model: Optional[MultiModelManager] = None
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
            import torch
            mem_gb = 0.0
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            return partitioner.full_model, partitioner.tokenizer, mem_gb

        def _unload_model_callback(name: str, model, tokenizer):
            """Unload a model and free GPU memory."""
            import gc
            import torch
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
        import torch
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
        vision_model: Optional[str] = None,
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

    def _init_hybrid_parallel(self, config: Optional[Any] = None) -> None:
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
        )
        self._hybrid_parallel_executor = HybridParallelExecutor(plan, coordinator=self)
        self._hybrid_parallel_executor.configure_pp(self._pipeline)
        if hasattr(self, '_pipeline') and plan.pp_num_stages > 1:
            self._pipeline.enable_overlap = True
        logger.info(f"Hybrid parallel plan: {plan.explanation}")

    def _init_zero_copy(self, config: Optional[Any] = None) -> None:
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

    def _init_adaptive_precision(self, config: Optional[Any] = None) -> None:
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

    def _init_predictive_cache(self, config: Optional[Any] = None) -> None:
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

    def _init_moe(self, moe_config: Optional[MoESettings]) -> None:
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

    def list_models(self) -> List[str]:
        """List all registered model names."""
        if self._multi_model is None:
            return [self.model_name]
        return self._multi_model.list_models()

    def get_model_name(self, requested: Optional[str] = None) -> str:
        """Resolve model name: requested > registry default > self.model_name."""
        if self._multi_model is None:
            return self.model_name
        return self._multi_model.get_model_name(requested)

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

                    # Evict old entries from the prefix cache
                    if self._cache_mgr.prefix_cache and hasattr(self._cache_mgr.prefix_cache, "_root"):
                        # RadixTreeCache: evict LRU entries
                        self._cache_mgr.prefix_cache._root.evict_lru(self._cache_mgr.prefix_cache.max_entries)
            except Exception:
                logger.debug("Gossip round error (non-fatal)", exc_info=True)

    def register_expert_on_node(self, node_id: str, expert_ids: List[int], layer_idx: int = 0):
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
        return self._metrics_mgr.get_prometheus()

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

    def auto_setup(self, nodes_config: List[Dict]) -> None:
        """Automatically partition model and assign layers to nodes."""
        model_info, total_layers = self._node_registrar.auto_setup(nodes_config)
        self.model_info = model_info
        self.total_layers = total_layers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)

    def manual_register(self, node_id: str, host: str, port: int, start_layer: int, end_layer: int, total_layers: Optional[int] = None, role: NodeRole = NodeRole.AUTO, expert_ids: Optional[List[int]] = None, cluster_id: str = "default"):
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

        # Generation span
        from distllm.observability.spans import span_prefill, record_ttft
        prompt_len = len(self.tokenizer.encode(prompt)) if self.tokenizer else 0
        tenant = "default"

        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not loaded")
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

            if self.node_order:
                req_log.info(f"Starting distributed generation: {max_new_tokens} tokens max")
                input_ids = input_ids.to("cpu")
                prompt_len = input_ids.shape[1]
                total_capacity = prompt_len + max_new_tokens
                generated_ids = torch.zeros(1, total_capacity, dtype=torch.long)
                generated_ids[:, :prompt_len] = input_ids
                gen_pos = prompt_len

                node_kv_caches: Dict[str, Optional[List]] = {
                    nid: None for nid in self.node_order
                }

                prefill_start = time.monotonic()

                # Check if speculative decoding is available and enabled
                active_method = self._spec_decoder.get_active_method(self.draft_model) if self._spec_decoder else None
                use_speculative = (
                    self._spec_decoder is not None
                    and self._spec_decoder.is_enabled
                    and active_method in ("draft_model", "medusa", "ngram")
                )
                if use_speculative:
                    draft_tokens_count = self.num_assistant_tokens
                    req_log.info(f"Speculative decoding enabled ({active_method}): {draft_tokens_count} draft tokens")

                step = 0
                while step < max_new_tokens:
                    if gen_pos == prompt_len:
                        step_input = generated_ids[:, :gen_pos]
                    else:
                        step_input = generated_ids[:, gen_pos-1:gen_pos]

                    draft_tokens = None
                    if use_speculative:
                        # Generate draft tokens using the active speculation method
                        active_method = self._spec_decoder.get_active_method(self.draft_model)
                        if active_method == "draft_model" and self.draft_model is not None:
                            draft_tokens, _ = self._spec_decoder.generate_draft_tokens(
                                self.draft_model, step_input
                            )
                        elif active_method == "ngram":
                            # N-gram matching uses generated_ids as context
                            generated_list = generated_ids[0].tolist() if generated_ids.dim() == 2 else generated_ids.tolist()
                            draft_tokens, _ = self._spec_decoder.generate_draft_tokens(
                                None, step_input, generated_ids=generated_list
                            )
                        else:
                            # Medusa: generate drafts after we get target logits
                            pass

                        if draft_tokens and is_debug_mode():
                            req_log.debug(f"Draft tokens: {draft_tokens}")

                    logits = self._pipeline.run_pipeline(
                        step_input, node_kv_caches, request_id=request_id,
                        draft_tokens=draft_tokens if use_speculative else None,
                    )

                    if use_speculative and active_method == "medusa":
                        # Medusa: generate draft tokens from target logits
                        draft_tokens, _ = self._spec_decoder.generate_draft_tokens(
                            None, step_input, target_logits=logits
                        )
                        if draft_tokens and is_debug_mode():
                            req_log.debug(f"Medusa draft tokens: {draft_tokens}")

                    if use_speculative and draft_tokens:
                        # Verify draft tokens against target model logits
                        accepted_count, accepted_tokens, next_token = self._spec_decoder.verify_and_accept(
                            draft_tokens, logits, self.tokenizer
                        )
                        # Append accepted tokens
                        for token_id in accepted_tokens:
                            generated_ids[:, gen_pos] = token_id
                            gen_pos += 1
                            if next_token == self.tokenizer.eos_token_id:
                                break
                        # If we got a valid next_token after accepted prefix, append it
                        if next_token > 0 and next_token != self.tokenizer.eos_token_id:
                            generated_ids[:, gen_pos] = next_token
                            gen_pos += 1
                        elif next_token == self.tokenizer.eos_token_id:
                            generated_ids[:, gen_pos] = next_token
                            gen_pos += 1
                            break
                        # Count tokens generated this step
                        step += accepted_count + (1 if next_token > 0 else 0)

                        # Record generated tokens for n-gram indexing
                        self._spec_decoder.record_generated_tokens(generated_ids[0, :gen_pos].tolist())
                    else:
                        next_token = self._sample(logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k)
                        generated_ids[:, gen_pos] = next_token.item()
                        gen_pos += 1
                        step += 1
                        if next_token.item() == self.tokenizer.eos_token_id:
                            break

                        # Record tokens for n-gram indexing even without speculation
                        if self._spec_decoder:
                            self._spec_decoder.record_generated_tokens(generated_ids[0, :gen_pos].tolist())

                result = self.tokenizer.decode(generated_ids[0, :gen_pos], skip_special_tokens=True)
                tokens_generated = gen_pos - prompt_len
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

        except (NodeUnreachableError, OOMError, GRPCTimeoutError, NodeError) as e:
            self.record_metric("errors", 1)
            if self.metrics_exporter:
                self.metrics_exporter.errors_total.labels(type=type(e).__name__).inc()
            req_log.error(f"Generation failed: {e}")
            raise
        except Exception as e:
            self.record_metric("errors", 1)
            if self.metrics_exporter:
                self.metrics_exporter.errors_total.labels(type=type(e).__name__).inc()
            req_log.error(f"Generation failed: {e}")
            raise

        finally:
            paged_bt = locals().get('paged_block_tables', {})
            if self._paged_kv_backend is not None and paged_bt:
                for nid, block_table in paged_bt.items():
                    self._paged_kv_backend.free(block_table)
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
        adapter_id: Optional[str] = None,
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
            adapter_id=adapter_id,
        )
        if chunk_state:
            seq.chunk_state = chunk_state

        if self.tokenizer.eos_token_id is not None:
            seq.stop_token_ids = [self.tokenizer.eos_token_id]

        self.scheduler.add(seq)
        self._batch_event.set()

        event = self._request_tracker.register_request(request_id)

        self.record_metric("total_requests", 1)
        return request_id

    def wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        """Wait for a batched request to complete and return the result."""
        return self._request_tracker.wait_for_result(request_id, timeout)

    def generate_batch(self, timeout: float = 120.0, max_steps: int = 0) -> None:
        """Run the batch generation loop until all pending requests are complete."""
        if self.scheduler is None:
            raise BatchError("Batch scheduler not configured. Use generate() instead.")

        step = 0
        idle_time = 0.0

        while self.scheduler.has_pending:
            batch = self.scheduler.schedule()
            if batch is None:
                self._batch_event.wait(timeout=0.01)
                self._batch_event.clear()
                idle_time += 0.01
                if idle_time > timeout:
                    break
                continue

            idle_time = 0.0

            batch_request_ids = [seq.request_id for seq in batch.sequences]
            try:
                if self.local_partitioner is not None:
                    self._generate_local_batch(batch)
                else:
                    self._run_distributed_pipeline_batch(batch)
            except Exception:
                with self._batch_kv_caches_lock:
                    for rid in batch_request_ids:
                        self._batch_kv_caches.pop(rid, None)
                raise

            step += 1
            if max_steps > 0 and step >= max_steps:
                break

        if self.scheduler is not None:
            self._request_tracker.complete_batch_requests(
                self.scheduler.active, list(self.scheduler.pending_queue), self.tokenizer
            )
            # Clean up KV cache state for completed requests
            with self._batch_kv_caches_lock:
                for rid in list(self._batch_kv_caches.keys()):
                    if rid not in self.scheduler.active:
                        self._batch_kv_caches.pop(rid, None)

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
        """Run a batch through the distributed pipeline with persistent KV caches."""
        next_tokens: List[torch.Tensor] = []

        for seq_idx, seq in enumerate(batch.sequences):
            if batch.is_prefill[seq_idx]:
                start = seq.prefix_match_len
                tokens = seq.prompt_tokens[start:]
            else:
                tokens = [seq.decode_input_token]

            input_ids = torch.tensor([tokens], dtype=torch.long)

            # Reuse KV cache state if it exists from a prior step
            with self._batch_kv_caches_lock:
                if seq.request_id in self._batch_kv_caches:
                    node_kv_caches = self._batch_kv_caches[seq.request_id]
                else:
                    node_kv_caches: Dict[str, Optional[List]] = {
                        nid: None for nid in self.node_order
                    }

                    if self._paged_kv_backend is not None:
                        logger.debug("PagedAttention backend available for distributed pipeline")
                    self._batch_kv_caches[seq.request_id] = node_kv_caches

            if self._pipeline.enable_overlap:
                logits = self._pipeline.run_pipeline_overlap(
                    input_ids,
                    node_kv_caches,
                    request_id=seq.request_id,
                )
            else:
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

    # -- Model Loading (delegated to ModelManager) --

    def load_local_model(self):
        """Load the full model locally (for single-node testing)."""
        self._model_mgr.load_local_model(self)
        self._apply_flash_attention()
        self._apply_rope_scaling()
        self._wire_paged_attention()

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

    def start(self, blocking: bool = True, on_stop: Optional[Callable] = None):
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

        logger.info("Graceful shutdown complete")

    async def stop_async(self):
        """Stop the coordinator with graceful shutdown (async)."""
        logger.info("Initiating graceful shutdown (async)...")

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

        # Generation span
        from distllm.observability.spans import async_span_generation, record_ttft
        prompt_len = len(self.tokenizer.encode(prompt)) if self.tokenizer else 0
        tenant = "default"

        try:
            # Tokenize in thread pool
            input_ids = await asyncio.to_thread(
                self.tokenizer.encode, prompt, return_tensors="pt"
            )

            if self.node_order:
                input_ids = input_ids.to("cpu")
                prompt_len = input_ids.shape[1]
                total_capacity = prompt_len + max_new_tokens
                generated_ids = torch.zeros(1, total_capacity, dtype=torch.long)
                generated_ids[:, :prompt_len] = input_ids
                gen_pos = prompt_len

                node_kv_caches: Dict[str, Optional[List]] = {
                    nid: None for nid in self.node_order
                }

                active_method = self._spec_decoder.get_active_method(self.draft_model) if self._spec_decoder else None
                use_speculative = (
                    self._spec_decoder is not None
                    and self._spec_decoder.is_enabled
                    and active_method in ("draft_model", "medusa", "ngram")
                )

                prefill_start = time.monotonic()
                ttft_recorded = None

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
                        for token_id in accepted_tokens:
                            generated_ids[:, gen_pos] = token_id
                            gen_pos += 1
                        if next_token > 0 and next_token != self.tokenizer.eos_token_id:
                            generated_ids[:, gen_pos] = next_token
                            gen_pos += 1
                        elif next_token == self.tokenizer.eos_token_id:
                            generated_ids[:, gen_pos] = next_token
                            gen_pos += 1
                            break
                        step += accepted_count + (1 if next_token > 0 else 0)
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
