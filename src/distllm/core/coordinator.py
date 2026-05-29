"""Coordinator for distributed LLM inference across multiple devices.

Splits responsibilities across four specialized components:
  - ``InferenceEngine`` — local/distributed text generation
  - ``ClusterManager`` — node registration, topology, weight distribution
  - ``HealthManager`` — periodic health probes, recovery, straggler detection
  - ``MetricsCollector`` — aggregated metrics from all subsystems
"""

import argparse
import asyncio
import os
import threading
import time
import uuid
from typing import Any, Callable

import torch
from loguru import logger
from transformers import AutoTokenizer

from distllm.config.settings import DistLLMSettings, NodeRole
from distllm.core.batch_scheduler import BatchScheduler
from distllm.core.cache_manager import CacheManager
from distllm.core.cluster_manager import ClusterManager
from distllm.core.debug import set_debug_mode
from distllm.core.health_manager import HealthManager
from distllm.core.inference_engine import InferenceEngine
from distllm.core.memory_defragmenter import (
    DefragConfig,
    DefragPolicy,
    MemoryDefragmenter,
    TieredCompactionLevel,
)
from distllm.core.metrics_collector import MetricsCollector
from distllm.core.model_router import ModelRouter
from distllm.core.resource_manager import ResourceManager
from distllm.dist.federation import FederationConfig
from distllm.dist.latency import LatencyTracker
from distllm.dist.pipeline import PipelineOrchestrator
from distllm.dist.recovery import NodeRecoveryManager
from distllm.dist.reputation import ReputationSystem
from distllm.dist.straggler import DetectionMethod, StragglerDetector
from distllm.models.partitioner import ModelPartitioner, get_model_info
from distllm.security import hf_revision


class _ParamUpdateChannel:
    """Channel for mid-stream parameter updates (temperature, top_p, top_k)."""

    def __init__(self):
        self._channels: dict[str, dict] = {}

    def register(self, request_id: str) -> None:
        self._channels[request_id] = {}

    def update(self, request_id: str, **params) -> None:
        if request_id in self._channels:
            self._channels[request_id].update(params)

    def get(self, request_id: str) -> dict | None:
        return self._channels.get(request_id)

    def unregister(self, request_id: str) -> None:
        self._channels.pop(request_id, None)


class _RequestTracker:
    """Tracks async request results and completion events."""

    def __init__(self):
        self._results: dict[str, str] = {}
        self._events: dict[str, threading.Event] = {}
        self._logprobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def register_request(self, request_id: str) -> None:
        with self._lock:
            self._events[request_id] = threading.Event()

    def set_result(self, request_id: str, result: str) -> None:
        with self._lock:
            self._results[request_id] = result
            event = self._events.get(request_id)
            if event:
                event.set()

    def wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        event = self._events.get(request_id)
        if event is None:
            raise ValueError(f"Unknown request_id: {request_id}")
        event.wait(timeout=timeout)
        with self._lock:
            result = self._results.pop(request_id, None)
            self._events.pop(request_id, None)
        if result is None:
            raise TimeoutError(f"Request {request_id} timed out after {timeout}s")
        return result

    def get_logprobs(self, request_id: str) -> dict | None:
        return self._logprobs.get(request_id)

    def complete_batch_requests(self, active_seqs, pending_seqs, tokenizer) -> None:
        """Complete all active/pending requests in the batch.

        Called when the batch scheduler is shutting down or timing out.
        Finishes any active sequences by decoding their generated tokens,
        and marks pending sequences with an error.
        """
        with self._lock:
            # Complete active sequences that have generated tokens
            for seq_id, seq in (active_seqs.items() if isinstance(active_seqs, dict) else []):
                try:
                    if hasattr(seq, 'generated_tokens') and seq.generated_tokens:
                        if tokenizer is not None:
                            result = tokenizer.decode(seq.generated_tokens, skip_special_tokens=True)
                        else:
                            result = str(seq.generated_tokens)
                        self._results[seq_id] = result
                    else:
                        self._results[seq_id] = "[Error: Sequence completed without output]"
                except Exception as e:
                    self._results[seq_id] = f"[Error decoding output: {e}]"
                event = self._events.pop(seq_id, None)
                if event:
                    event.set()
                self._logprobs.pop(seq_id, None)

            # Mark pending sequences as timed out
            for seq in pending_seqs:
                sid = getattr(seq, 'request_id', str(seq))
                self._results[sid] = "[Error: Request timed out waiting in scheduler queue]"
                event = self._events.pop(sid, None)
                if event:
                    event.set()


class CoordinatorConfig:
    """Configuration for the distributed coordinator.

    Use :meth:`from_settings` to create from a :class:`DistLLMSettings`
    instance, which extracts all values from the Pydantic settings model.
    """

    def __init__(
        self,
        model_name: str = "",
        port: int = 50050,
        dtype: str = "float16",
        trust_remote_code: bool | None = None,
        max_batch_size: int = 4,
        max_tokens_per_batch: int = 1024,
        pipeline_timeout: float = 30.0,
        cluster_key: str | None = None,
        model_cache_dir: str | None = None,
        metrics_exporter=None,
        discovery_mode: str | None = None,
        wide_area_config=None,
        redundancy: int = 1,
        federation_config=None,
        plugin_system=None,
    ):
        self.model_name = model_name
        self.port = port
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.metrics_exporter = metrics_exporter
        self.discovery_mode = discovery_mode
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pipeline_timeout = pipeline_timeout
        self.cluster_key = cluster_key
        self.model_cache_dir = model_cache_dir
        self.wide_area_config = wide_area_config
        self.redundancy = redundancy
        self.min_reputation = 0.0
        self.federation_config = federation_config
        self.prefix_cache_enabled = False
        self.prefix_cache_max_entries = 256
        self.prefix_cache_min_prefix_len = 4
        self.radix_tree_cache_enabled = False
        self.chunked_prefill_enabled = False
        self.chunked_prefill_chunk_size = 512
        self.enable_pipeline_overlap = False
        self.plugin_system = plugin_system

    @classmethod
    def from_settings(cls, settings: "DistLLMSettings", **overrides) -> "CoordinatorConfig":
        wa = settings.wide_area
        wide_area_config = None
        if wa.enabled:
            from distllm.dist.config import WideAreaConfig
            wide_area_config = WideAreaConfig(
                enabled=wa.enabled,
                p2p_forwarding=wa.p2p_forwarding,
                tokens_before_forward=wa.tokens_before_forward,
                wan_timeout_seconds=wa.wan_timeout_seconds,
                max_retries=wa.max_retries,
                backoff_base_seconds=wa.backoff_base_seconds,
            )

        config = cls(
            model_name=settings.model.name,
            port=settings.coordinator.port,
            dtype=settings.model.dtype,
            trust_remote_code=settings.model.trust_remote_code or None,
            max_batch_size=settings.batching.max_batch_size,
            max_tokens_per_batch=settings.batching.max_tokens_per_batch,
            pipeline_timeout=settings.network.grpc_timeout,
            model_cache_dir=settings.model_hub.cache_dir,
            wide_area_config=wide_area_config,
        )

        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


class Coordinator:
    """Orchestrates distributed inference across multiple worker nodes.

    Splits model layers across connected devices and runs pipeline-parallel
    inference. Supports dynamic node registration, failure recovery,
    straggler detection, and latency tracking.

    Internally composes:
        - :class:`ClusterManager` — node lifecycle
        - :class:`InferenceEngine` — text generation
        - :class:`HealthManager` — health probes + recovery
        - :class:`MetricsCollector` — aggregated metrics
    """

    def __init__(self, config: CoordinatorConfig | None = None, **kwargs: Any) -> None:
        if config is None and kwargs:
            config = CoordinatorConfig(
                model_name=kwargs.get("model_name", ""),
                port=kwargs.get("port", 50050),
                dtype=kwargs.get("dtype", "float16"),
                trust_remote_code=kwargs.get("trust_remote_code"),
                max_batch_size=kwargs.get("max_batch_size", 4),
                max_tokens_per_batch=kwargs.get("max_tokens_per_batch", 1024),
                pipeline_timeout=kwargs.get("pipeline_timeout", 30.0),
                cluster_key=kwargs.get("cluster_key"),
                model_cache_dir=kwargs.get("model_cache_dir"),
                metrics_exporter=kwargs.get("metrics_exporter"),
                discovery_mode=kwargs.get("discovery_mode"),
                wide_area_config=kwargs.get("wide_area_config"),
                plugin_system=kwargs.get("plugin_system"),
            )
        self.config = config or CoordinatorConfig(model_name="")
        self.model_name = self.config.model_name
        self.model_revision = hf_revision()
        self.port = self.config.port
        self.dtype = self.config.dtype
        self.trust_remote_code = self.config.trust_remote_code
        self.total_layers = 0

        self._resource_mgr = ResourceManager()
        self._cache_mgr = CacheManager()
        self._pipeline = PipelineOrchestrator(
            resource_mgr=self._resource_mgr,
            pipeline_timeout=self.config.pipeline_timeout,
            redundancy=self.config.redundancy,
        )
        self._batch_scheduler = BatchScheduler(
            max_batch_size=self.config.max_batch_size,
            max_tokens_per_batch=self.config.max_tokens_per_batch,
        )
        self._init_adaptive_batching()

        self._latency_tracker = LatencyTracker()
        self._straggler_detector = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            on_straggler_cb=self._on_straggler_detected,
        )
        self._recovery_manager = NodeRecoveryManager()
        self._reputation = ReputationSystem(
            min_reputation=getattr(self.config, 'min_reputation', 0.0),
        )
        self._federation: Any = None

        self._pipeline.set_latency_tracker(self._latency_tracker)
        self._pipeline.set_straggler_detector(self._straggler_detector)

        if self.config.wide_area_config:
            wa = self.config.wide_area_config
            self._pipeline.pipeline_timeout = wa.wan_timeout_seconds
            if hasattr(self._pipeline, 'wan'):
                self._pipeline.wan = wa
            logger.info(f"WAN mode: timeout={wa.wan_timeout_seconds}s, "
                         f"token_accumulation={wa.accumulation_window}")

        self.tokenizer: AutoTokenizer | None = None

        # ── Decomposed components ──

        self._cluster_mgr = ClusterManager(
            pipeline=self._pipeline,
            model_name=self.model_name,
            trust_remote_code=self.trust_remote_code,
            cluster_key=self.config.cluster_key,
        )

        self._inference_engine = InferenceEngine(
            model_name=self.model_name,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            model_revision=self.model_revision,
            tokenizer=None,
            pipeline=self._pipeline,
            batch_scheduler=self._batch_scheduler,
            latency_tracker=self._latency_tracker,
            straggler_detector=self._straggler_detector,
            reputation=self._reputation,
            node_order_property=lambda: self.node_order,
            recovery_manager=self._recovery_manager,
        )

        self._health_mgr = HealthManager(
            pipeline=self._pipeline,
            resource_mgr=self._resource_mgr,
            reputation=self._reputation,
            recovery_manager=self._recovery_manager,
            straggler_detector=self._straggler_detector,
        )

        self._metrics_collector = MetricsCollector(
            latency_tracker=self._latency_tracker,
            straggler_detector=self._straggler_detector,
            recovery_manager=self._recovery_manager,
        )

        self._running = threading.Event()
        self._async_shutdown = asyncio.Event()
        self._request_results: dict[str, str] = {}
        self._request_events: dict[str, threading.Event] = {}
        self._health_check_interval_s: float = 10.0
        self._straggler_check_counter: int = 0
        self._health_thread: threading.Thread | None = None
        self._health_event = threading.Event()
        self._distribute_weights: bool = True
        self._hot_swap_mgr = None

        # Async batch scheduler support (used by RequestPipeline)
        self._batch_event = threading.Event()
        self._param_update_channel = _ParamUpdateChannel()
        self._request_tracker = _RequestTracker()
        self._rate_limiter = None  # Set externally if needed
        self._request_fingerprinter = None
        self._request_auditor = None
        self._graceful_degradation = None
        self._preemption_policy = None
        self.model_info = None
        self._shutting_down = False

        # Per-request scheduling hints (populated by API layer before generate())
        self._pending_scheduling_hints: dict[str, dict] = {}

        # Plugin system hooks (optional)
        self._plugin_system = getattr(self.config, "plugin_system", None)

        # Model router for query-based model switching
        self._model_router: ModelRouter | None = None

        # Memory defragmentation
        self._defragmenter: MemoryDefragmenter | None = None
        self._defrag_task: asyncio.Task | None = None

        logger.info(f"Coordinator initialized for model: {self.model_name}")

    def _init_adaptive_batching(self) -> None:
        """Connect the adaptive batching engine if the module is available."""
        try:
            from distllm.core.adaptive_batching import AdaptiveBatchingEngine
            engine = AdaptiveBatchingEngine()
            self._batch_scheduler.set_adaptive_engine(engine)
            logger.debug("Adaptive batching engine initialized")
        except ImportError:
            logger.debug("Adaptive batching engine not available")

    def init_model_router(self, settings: Any | None = None) -> ModelRouter:
        """Initialize the model router from ChatRouterSettings.

        Args:
            settings: A ChatRouterSettings instance. If None, router is
                created with the coordinator's default model as fallback.

        Returns:
            The initialized ModelRouter instance.
        """
        from distllm.config.settings import ChatRouterSettings
        if settings is None:
            settings = ChatRouterSettings(
                enabled=True,
                default_model=self.model_name,
            )
        self._model_router = ModelRouter(settings)
        # Register the router name from settings as a hybrid name
        router_name = getattr(settings, "name", "")
        if router_name:
            self._model_router.register_hybrid_name(router_name)
        logger.info(
            f"Model router initialized: default={self._model_router._default_model}, "
            f"rules={len(self._model_router._rules)}, "
            f"hybrid_names={self._model_router.list_hybrid_models()}"
        )
        return self._model_router

    # ── Callbacks ──

    def _on_straggler_detected(self, report) -> None:
        logger.warning(
            f"Straggler {report.node_id}: {report.slowdown_factor}x slower "
            f"(action: {report.recommended_action})"
        )
        if report.recommended_action == "reassign_layers" and self._recovery_manager is not None:
            self._recovery_manager.on_node_failure(report.node_id)

    # ── Properties (delegated to ClusterManager) ──

    @property
    def nodes(self) -> dict:
        return self._cluster_mgr.nodes

    @nodes.setter
    def nodes(self, value: dict):
        self._cluster_mgr.nodes = value

    @property
    def node_order(self) -> list[str]:
        return self._cluster_mgr.node_order

    @node_order.setter
    def node_order(self, value: list[str]):
        self._cluster_mgr.node_order = value

    @property
    def scheduler(self) -> BatchScheduler | None:
        return self._batch_scheduler

    # ── Node lifecycle (delegated to ClusterManager) ──

    def auto_setup(self, nodes_config: list[dict]) -> tuple[dict, int] | None:
        self._cluster_mgr.model_revision = self.model_revision
        result = self._cluster_mgr.auto_setup(nodes_config)
        self.tokenizer = self._cluster_mgr.tokenizer
        self._inference_engine.tokenizer = self.tokenizer
        return result

    def manual_register(self, node_id: str, host: str, port: int,
                        start_layer: int, end_layer: int,
                        total_layers: int | None = None,
                        role: NodeRole = NodeRole.AUTO,
                        expert_ids: list[int] | None = None,
                        cluster_id: str = "default",
                        cluster_key: str | None = None) -> None:
        self._cluster_mgr.manual_register(
            node_id, host, port, start_layer, end_layer,
            total_layers=total_layers, role=role,
            expert_ids=expert_ids, cluster_id=cluster_id,
            cluster_key=cluster_key,
        )
        self.tokenizer = self._cluster_mgr.tokenizer

    # ── Generation (delegated to InferenceEngine) ──

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 0, request_id: str | None = None,
                 user_id: str = "default",
                 speculative_config: dict | None = None,
                 response_format: dict | None = None) -> str:
        self._inference_engine.tokenizer = self.tokenizer

        constraint = None
        if response_format:
            from distllm.core.structured_output import JSONSchemaConstraint
            constraint = JSONSchemaConstraint.from_response_format(
                response_format, tokenizer=self.tokenizer,
            )

        try:
            return self._inference_engine.generate(
                prompt, max_new_tokens, temperature, top_p, top_k,
                request_id, user_id, speculative_config,
                constraint=constraint,
            )
        except Exception as exc:
            if self._plugin_system:
                self._plugin_system.dispatch("on_error", {
                    "prompt": prompt[:128],
                    "request_id": request_id or "",
                    "user_id": user_id,
                }, exc)
            raise

    def load_local_model(self) -> None:
        self._inference_engine.tokenizer = self.tokenizer
        self._inference_engine.load_local_model()
        self.tokenizer = self._inference_engine.tokenizer
        self._inference_engine.tokenizer = self.tokenizer
        if self._plugin_system:
            self._plugin_system.dispatch("on_model_load", self.model_name, {"local": True})

    def set_deterministic_mode(self, enabled: bool = True, seed: int = 42) -> None:
        self._inference_engine.set_deterministic_mode(enabled, seed)

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        return self._inference_engine.get_recent_requests(n)

    # ── Hot-swap model management ──

    def init_hot_swap(self, total_gpu_memory_gb: float = 0.0, max_models: int = 4) -> None:
        """Initialize the hot-swap model manager for dynamic model loading."""
        from distllm.core.multi_model_serving import ModelHotSwapManager
        self._hot_swap_mgr = ModelHotSwapManager(
            total_gpu_memory_gb=total_gpu_memory_gb,
            max_models=max_models,
            on_load_model=self._load_model_callback,
            on_unload_model=self._unload_model_callback,
        )
        logger.info(f"Hot-swap manager initialized (max {max_models} models)")

    # ── Adaptive compression ──

    def init_adaptive_compression(
        self,
        settings: Any | None = None,
        utilization_fn: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the adaptive compression manager.

        Args:
            settings: An ``AdaptiveCompressionSettings`` instance or None to
                use defaults (disabled).
            utilization_fn: Optional callable returning cluster utilization
                as a fraction (0.0–1.0). Defaults to a function that reads
                request load from the batch scheduler.
        """
        from distllm.core.adaptive_compression import (
            AdaptiveCompressionConfig,
            AdaptiveCompressionManager,
            SimpleCompressor,
        )

        if settings is None:
            self._adaptive_compression_mgr = None
            return

        config = AdaptiveCompressionConfig(
            enabled=settings.enabled,
            idle_threshold_pct=settings.idle_threshold_pct,
            idle_duration_s=settings.idle_duration_s,
            check_interval_s=settings.check_interval_s,
            compression_method=settings.compression_method,
            calibration_samples=settings.calibration_samples,
            output_dir=settings.output_dir,
        )

        if utilization_fn is None:
            utilization_fn = self._default_utilization_fn

        compressor = SimpleCompressor(
            output_base=settings.output_dir,
            method=settings.compression_method,
            calibration_samples=settings.calibration_samples,
        )

        self._adaptive_compression_mgr = AdaptiveCompressionManager(
            config=config,
            utilization_fn=utilization_fn,
            hot_swap_mgr=getattr(self, "_hot_swap_mgr", None),
            compressor=compressor,
        )

    def init_defragmentation(self, settings: Any | None = None) -> None:
        """Initialize the GPU memory defragmenter.

        Args:
            settings: A DefragmentationSettings instance or None to disable.
        """
        if settings is None or not settings.enabled:
            self._defragmenter = None
            return

        policy_map = {
            "lazy": DefragPolicy.LAZY,
            "balanced": DefragPolicy.BALANCED,
            "aggressive": DefragPolicy.AGGRESSIVE,
        }
        policy = policy_map.get(settings.policy, DefragPolicy.BALANCED)

        threshold = settings.threshold if settings.threshold > 0.0 else policy.threshold

        config = DefragConfig(
            enabled=settings.enabled,
            policy=policy,
            interval_seconds=settings.interval_seconds,
            max_blocks_per_pass=settings.max_blocks_per_pass,
            tiered_compaction=settings.tiered_compaction,
            l2_cpu_swap_threshold=settings.l2_cpu_swap_threshold,
            l3_nvme_swap_threshold=settings.l3_nvme_swap_threshold,
            cuda_stream_priority=settings.cuda_stream_priority,
            enable_predictive=settings.enable_predictive,
            enable_prometheus=settings.enable_prometheus,
        )
        self._defragmenter = MemoryDefragmenter(
            config=config,
            metrics_collector=self._metrics_collector,
        )
        logger.info(
            f"Defragmenter initialized: policy={settings.policy}, "
            f"threshold={threshold:.0%}, interval={settings.interval_seconds}s"
        )

    async def _defrag_loop(self) -> None:
        """Background loop that periodically checks and runs defragmentation."""
        if self._defragmenter is None:
            return

        interval = self._defragmenter.config.interval_seconds
        logger.debug(f"Defrag background loop started (interval={interval}s)")

        while not self._shutting_down:
            try:
                await asyncio.sleep(interval)

                if self._shutting_down:
                    break

                # Find PagedAttentionManager instances to defrag
                for backend in self._get_paged_backends():
                    if self._defragmenter.should_defragment(backend._blocks):
                        # Determine tier
                        ratio = self._defragmenter._compute_fragmentation_ratio(backend._blocks)
                        tier = TieredCompactionLevel.L1_HOT
                        if self._defragmenter.config.tiered_compaction:
                            if ratio > self._defragmenter.config.l3_nvme_swap_threshold:
                                tier = TieredCompactionLevel.L3_COLD
                            elif ratio > self._defragmenter.config.l2_cpu_swap_threshold:
                                tier = TieredCompactionLevel.L2_WARM

                        result = await self._defragmenter.defragment_with_tier_async(backend, tier)
                        self.record_metric("defrag_blocks_moved", result.blocks_moved)
                        self.record_metric("defrag_duration_ms", result.time_ms)

                        if result.fragmentation_after < result.fragmentation_before:
                            logger.info(
                                f"Defrag improved fragmentation: "
                                f"{result.fragmentation_before:.1%} -> {result.fragmentation_after:.1%}"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Defrag loop error: {e}")

    def _get_paged_backends(self) -> list[Any]:
        """Collect PagedAttentionManager instances from all backends."""
        backends = []
        if hasattr(self, "_inference_engine") and self._inference_engine is not None:
            engine = self._inference_engine
            if hasattr(engine, "_paged_mgr") and engine._paged_mgr is not None:
                backends.append(engine._paged_mgr)
            if hasattr(engine, "backends"):
                for be in engine.backends:
                    if hasattr(be, "_paged_mgr") and be._paged_mgr is not None:
                        backends.append(be._paged_mgr)
        return backends

    def defrag_status(self) -> dict:
        """Get current defragmentation status.

        Returns:
            Dict with fragmentation ratio, stats, and configuration.
        """
        if self._defragmenter is None:
            return {"enabled": False}

        # Sample first available backend
        frag_ratio = 0.0
        for backend in self._get_paged_backends():
            frag_ratio = self._defragmenter._compute_fragmentation_ratio(backend._blocks)
            break

        return {
            "enabled": True,
            "policy": self._defragmenter.config.policy.value,
            "fragmentation_ratio": round(frag_ratio, 4),
            "predictive_fragmentation": round(self._defragmenter.predict_fragmentation(), 4),
            "stats": self._defragmenter.stats,
            "config": {
                "interval_seconds": self._defragmenter.config.interval_seconds,
                "max_blocks_per_pass": self._defragmenter.config.max_blocks_per_pass,
                "tiered_compaction": self._defragmenter.config.tiered_compaction,
                "enable_predictive": self._defragmenter.config.enable_predictive,
            },
        }

    def defrag_stats(self) -> dict:
        """Get historical defragmentation statistics."""
        if self._defragmenter is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "stats": self._defragmenter.stats,
            "fragmentation_history": self._defragmenter.fragmentation_history[-50:],
        }

    def defrag_run_now(self) -> dict:
        """Trigger an immediate defragmentation pass on all backends.

        Returns:
            Dict with per-backend results.
        """
        results = {}
        if self._defragmenter is None:
            return {"error": "Defragmenter not initialized"}

        for i, backend in enumerate(self._get_paged_backends()):
            result = self._defragmenter.defragment(backend)
            results[f"backend_{i}"] = result.to_dict()

        return results

    def _default_utilization_fn(self) -> float:
        """Compute cluster utilization fraction (0.0 idle, 1.0 max)."""
        try:
            if hasattr(self, "_batch_scheduler") and self._batch_scheduler is not None:
                stats = self._batch_scheduler.stats()
                active = stats.get("active_requests", 0)
                pending = stats.get("pending_requests", 0)
                max_batch = stats.get("max_batch_size", 4)
                total = float(active + pending)
                return min(total / max(max_batch, 1), 1.0)
        except Exception:
            pass
        return 0.0

    def _load_model_callback(self, name: str, path: str):
        """Load a model for hot-swap with a configurable timeout (default: 5 min)."""
        import concurrent.futures

        from distllm.models.partitioner import ModelPartitioner

        timeout = int(os.environ.get("DISTLLM_HOT_SWAP_TIMEOUT", "300"))

        def _load():
            partitioner = ModelPartitioner(model_name=path, dtype=self.dtype)
            partitioner.load_full_model()
            import torch
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            return partitioner, partitioner.tokenizer, mem_gb

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_load)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Hot-swap model loading timed out after {timeout}s for {path}")
                raise

    def _unload_model_callback(self, name: str, model, tokenizer) -> None:
        """Callback to unload a model from GPU."""
        import torch
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def hot_swap_list_models(self) -> list[dict]:
        if self._hot_swap_mgr is None:
            return []
        return self._hot_swap_mgr.list_loaded_models()

    def hot_swap_register(self, name: str, path: str, total_layers: int,
                           memory_budget_gb: float = 0.0) -> bool:
        if self._hot_swap_mgr is None:
            return False
        self._hot_swap_mgr.register_model(name, path, total_layers, memory_budget_gb)
        return True

    def hot_swap_load(self, name: str) -> bool:
        if self._hot_swap_mgr is None:
            return False
        return self._hot_swap_mgr.load_model(name)

    def hot_swap_unload(self, name: str) -> bool:
        if self._hot_swap_mgr is None:
            return False
        return self._hot_swap_mgr.unload_model(name)

    # ── Health (delegated to HealthManager) ──

    def health_check(self) -> dict:
        nodes_status = self._health_mgr.get_node_status()
        return {
            "status": "ok" if self.nodes else "no_nodes",
            "num_nodes": len(self.nodes),
            "total_layers": self._pipeline.total_layers,
            "nodes": nodes_status,
            "reputation": self._reputation.get_scores(),
        }

    # ── Metrics (delegated to MetricsCollector) ──

    def get_metrics(self) -> dict:
        return self._metrics_collector.collect()

    def list_models(self) -> list[str]:
        """Return available model names including hybrid router names."""
        models: list[str] = []
        if self.model_name:
            models.append(self.model_name)
        # Include hybrid model names registered with the router
        if self._model_router is not None:
            models.extend(self._model_router.list_hybrid_models())
        return models

    def generate_async(self, prompt: str, **kwargs) -> str:
        """Schedule async generation via the batch scheduler.

        If the batch scheduler is configured, adds a Sequence directly
        to the scheduler for true continuous batching. Otherwise falls
        back to a background thread.

        Returns a request_id immediately. Call ``wait_for_result()``
        to get the result when ready.
        """
        request_id = kwargs.pop("request_id", None) or str(uuid.uuid4())

        # Try the real batch scheduler path first
        if self._batch_scheduler is not None and self.tokenizer is not None:
            try:
                from distllm.core.request_pipeline import RequestPipeline
                pipeline = RequestPipeline(self)
                return pipeline.generate_async(
                    prompt=prompt,
                    request_id=request_id,
                    max_new_tokens=kwargs.get("max_new_tokens", 128),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 0),
                    user_id=kwargs.get("user_id", "default"),
                )
            except Exception as e:
                logger.warning(f"Batch scheduler path failed, falling back to thread: {e}")

        # Fallback: background thread
        max_new_tokens = kwargs.get("max_new_tokens", 128)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.9)
        top_k = kwargs.get("top_k", 0)
        user_id = kwargs.get("user_id", "default")

        event = threading.Event()
        self._request_events[request_id] = event

        def _run():
            try:
                result = self.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    user_id=user_id,
                )
                self._request_results[request_id] = result
            except Exception as e:
                self._request_results[request_id] = f"[Error: {e}]"
            finally:
                event.set()

        thread = threading.Thread(target=_run, daemon=True, name=f"gen-{request_id[:8]}")
        thread.start()

        logger.debug(
            f"generate_async -> request_id={request_id} "
            f"(background thread started, prompt length {len(prompt)})"
        )
        return request_id

    def record_metric(self, name: str, value: float = 1.0) -> None:
        """Record a metric (used by RequestPipeline)."""
        if hasattr(self, '_metrics_collector') and self._metrics_collector is not None:
            try:
                self._metrics_collector.record(name, value)
            except Exception:
                pass

    def wait_for_result(self, request_id: str, timeout: float | None = None) -> str:
        """Wait for an async generation result.

        Checks both the request tracker (batch scheduler path) and
        the legacy event-based path (background thread fallback).
        """
        # Try request tracker first (batch scheduler path)
        if self._request_tracker is not None:
            try:
                return self._request_tracker.wait_for_result(request_id, timeout or 120.0)
            except (ValueError, TimeoutError):
                pass

        # Fallback: legacy event-based path
        event = self._request_events.get(request_id)
        if event is None:
            raise ValueError(f"Unknown request_id: {request_id}")
        event.wait(timeout=timeout)
        result = self._request_results.pop(request_id, None)
        self._request_events.pop(request_id, None)
        if result is None:
            raise TimeoutError(f"Request {request_id} timed out")
        return result

    # ── Start / Stop ──

    def start(self, blocking: bool = True, on_stop: Callable | None = None,
              health_check_interval_s: float = 10.0) -> None:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                revision=self.model_revision,
            )
        self._health_check_interval_s = health_check_interval_s
        self._running.set()
        self._health_event.clear()
        self._health_mgr.start()

        if hasattr(self, '_adaptive_compression_mgr') and self._adaptive_compression_mgr:
            self._adaptive_compression_mgr.start()

        if self._defragmenter is not None:
            self._defrag_task = asyncio.ensure_future(self._defrag_loop())
            logger.info("Defrag background loop started")

        try:
            from distllm.dist.discovery import DiscoveryService
            announce_model = os.environ.get("DISTLLM_DISCOVERY_ANNOUNCE_MODEL") == "1"
            self._discovery = DiscoveryService(
                port=self.port,
                service_id="distllm-coordinator",
                properties={"model": self.model_name} if announce_model else {},
            )
            self._discovery.start()
        except Exception:
            pass

        if (hasattr(self.config, 'federation_config')
                and self.config.federation_config
                and self.config.federation_config.enabled):
            try:
                from distllm.dist.federation import FederationCoordinator
                self._federation = FederationCoordinator(
                    config=self.config.federation_config,
                    local_cluster_id=self.config.federation_config.cluster_id,
                    local_host='localhost',
                    local_port=self.port,
                    coordinator_ref=self,
                )
                self._federation.start()
                logger.info(f"Federation started: cluster={self.config.federation_config.cluster_id}")
            except Exception as e:
                logger.warning(f"Federation failed to start: {e}")

        logger.info(f"Coordinator started on port {self.port} "
                     f"(health check every {health_check_interval_s}s)")
        if blocking:
            try:
                self._running.wait()
            except KeyboardInterrupt:
                logger.info("Coordinator shutting down...")
                self.stop()
        else:
            async def _wait_and_callback_async():
                try:
                    await self._async_shutdown.wait()
                except Exception:
                    pass
                finally:
                    if on_stop:
                        on_stop()
            asyncio.ensure_future(_wait_and_callback_async())

    def stop(self, timeout: float = 30.0) -> None:
        """Graceful shutdown with in-flight request draining.

        Args:
            timeout: Maximum seconds to wait for in-flight requests to complete.

        Shutdown sequence:
        1. Set _shutting_down flag (rejects new requests)
        2. Stop accepting new requests (clear running flag)
        3. Wait for in-flight requests to complete (with timeout)
        4. Checkpoint all active sequences
        5. Release GPU memory
        6. Close gRPC connections
        7. Save state to disk
        8. Stop background threads
        """
        logger.info("Initiating graceful shutdown...")
        self._shutting_down = True

        # 1. Stop accepting new requests
        self._running.clear()
        self._health_event.set()
        self._async_shutdown.set()

        # 2. Wait for in-flight requests to complete
        if self._batch_scheduler is not None:
            active_count = len(self._batch_scheduler.active)
            if active_count > 0:
                logger.info(f"Waiting for {active_count} in-flight requests to complete (timeout={timeout}s)...")
                deadline = time.time() + timeout
                while self._batch_scheduler.active and time.time() < deadline:
                    time.sleep(0.1)
                remaining = len(self._batch_scheduler.active)
                if remaining > 0:
                    logger.warning(f"{remaining} requests still in-flight after timeout, forcing shutdown")

        # 3. Checkpoint active sequences
        if self._batch_scheduler is not None and self._batch_scheduler.active:
            for req_id, seq in list(self._batch_scheduler.active.items()):
                try:
                    if self._recovery_manager is not None:
                        self._recovery_manager.save_checkpoint(
                            request_id=req_id,
                            kv_cache=None,
                            prompt_tokens=getattr(seq, 'prompt_tokens', []),
                            generated_tokens=getattr(seq, 'generated_tokens', []),
                            node_id="coordinator",
                        )
                except Exception as e:
                    logger.warning(f"Failed to checkpoint {req_id}: {e}")

        # 4. Release GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory released")
        except ImportError:
            pass

        # 5. Close gRPC connections and nodes
        for nid, node in list(self.nodes.items()):
            try:
                node.close()
            except Exception as e:
                logger.warning(f"Error closing node {nid}: {e}")

        # 6. Close connection pools
        if hasattr(self, '_resource_mgr') and self._resource_mgr is not None:
            try:
                self._resource_mgr._conn_pool.close_all()
            except Exception:
                pass
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._resource_mgr._async_conn_pool.close_all())
                else:
                    loop.run_until_complete(self._resource_mgr._async_conn_pool.close_all())
            except Exception:
                pass

        # 7. Stop background services
        self._health_mgr.stop()
        if hasattr(self, '_adaptive_compression_mgr') and self._adaptive_compression_mgr:
            self._adaptive_compression_mgr.stop()
        if self._defrag_task is not None:
            self._defrag_task.cancel()
            self._defrag_task = None
            logger.debug("Defrag background loop stopped")
        if hasattr(self, '_federation') and self._federation:
            self._federation.stop()
        if hasattr(self, '_discovery') and self._discovery:
            self._discovery.stop()

        # 8. Shutdown pipeline
        self._pipeline.shutdown()

        # 9. Save state to disk
        try:
            self._save_shutdown_state()
        except Exception as e:
            logger.warning(f"Failed to save shutdown state: {e}")

        logger.info("Graceful shutdown complete")

    def _save_shutdown_state(self) -> None:
        """Save coordinator state to disk for recovery after restart."""
        import json
        state = {
            "model_name": self.model_name,
            "shutdown_time": time.time(),
            "nodes": {
                nid: {"host": n.host, "port": n.port, "healthy": n.healthy}
                for nid, n in self.nodes.items()
            },
            "metrics": self.get_metrics() if hasattr(self, 'get_metrics') else {},
        }
        state_path = os.path.join(os.getcwd(), ".distllm_shutdown_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug(f"Shutdown state saved to {state_path}")


def _resolve_cluster_key() -> str | None:
    """Resolve cluster key from environment variable or file.

    Resolution order:
    1. DISTLLM_CLUSTER_KEY environment variable
    2. ~/.distllm/cluster_key file
    3. None (no key)
    """
    key = os.environ.get("DISTLLM_CLUSTER_KEY", "")
    if key:
        return key
    key_path = os.path.expanduser("~/.distllm/cluster_key")
    if os.path.isfile(key_path):
        try:
            with open(key_path) as f:
                return f.read().strip()
        except OSError:
            pass
    return None


def main():
    from distllm.config.resolver import ConfigResolver

    parser = argparse.ArgumentParser(description="DistLLM Coordinator")
    ConfigResolver._register_args(parser, ConfigResolver.COMMON_ARGS + ConfigResolver.COORDINATOR_ARGS)
    args = parser.parse_args()

    if args.validate_config:
        DistLLMSettings.validate_startup()
        print("Config validation passed")
        return

    if args.debug:
        set_debug_mode(True)

    # Discover config path and load settings if available
    config_path = ConfigResolver._resolve_config_path("coordinator", args)
    settings = DistLLMSettings.from_yaml(config_path=config_path) if config_path else None

    federation_cfg = None
    if args.federate:
        federation_cfg = FederationConfig(
            enabled=True,
            cluster_id=args.federation_cluster_id,
            listen_port=args.federation_port,
            seed_nodes=args.federation_seed or [],
        )

    # Build CoordinatorConfig: YAML/env as base, CLI args override
    if settings is not None:
        config = CoordinatorConfig.from_settings(settings)
        config.model_name = args.model or config.model_name
        config.dtype = args.dtype or config.dtype
        config.port = args.port
        config.trust_remote_code = args.trust_remote_code or None
        config.cluster_key = args.cluster_key or config.cluster_key or _resolve_cluster_key()
        config.model_cache_dir = args.model_cache_dir or config.model_cache_dir
        config.redundancy = args.redundancy
        config.federation_config = federation_cfg or config.federation_config
    else:
        config = CoordinatorConfig(
            model_name=args.model,
            port=args.port,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code or None,
            cluster_key=args.cluster_key or _resolve_cluster_key(),
            model_cache_dir=args.model_cache_dir,
            redundancy=args.redundancy,
            federation_config=federation_cfg,
        )
    coordinator = Coordinator(config=config)
    coordinator._distribute_weights = args.distribute_weights

    # Initialize model router if chat_router config is available
    if settings is not None and getattr(settings, 'chat_router', None):
        cr = settings.chat_router
        if cr.enabled:
            coordinator.init_model_router(cr)

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
                coordinator.manual_register(
                    node_id=f"node_{i}",
                    host=parts[0],
                    port=int(parts[1]),
                    start_layer=int(parts[2]),
                    end_layer=int(parts[3]),
                    total_layers=args.total_layers,
                )
        coordinator.start()


if __name__ == "__main__":
    main()
