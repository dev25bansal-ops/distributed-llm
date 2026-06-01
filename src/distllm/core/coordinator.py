"""Coordinator for distributed LLM inference across multiple devices.

Splits responsibilities across four specialized components:
  - ``InferenceEngine`` — local/distributed text generation
  - ``ClusterManager`` — node registration, topology, weight distribution
  - ``HealthManager`` — periodic health probes, recovery, straggler detection
  - ``MetricsCollector`` — aggregated metrics from all subsystems
"""

import asyncio
import os
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger
from transformers import AutoTokenizer

from distllm.config.settings import DistLLMSettings, NodeRole
from distllm.core.batch_scheduler import BatchScheduler
from distllm.core.cache_manager import CacheManager
from distllm.core.cluster_manager import ClusterManager
from distllm.core.coordinator_config import CoordinatorConfig
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
from distllm.core.param_update_channel import ParamUpdateChannel
from distllm.core.request_tracker import RequestTracker
from distllm.core.resource_manager import ResourceManager
from distllm.dist.federation import FederationConfig
from distllm.dist.latency import LatencyTracker
from distllm.dist.pipeline import PipelineOrchestrator
from distllm.dist.recovery import NodeRecoveryManager
from distllm.dist.reputation import ReputationSystem
from distllm.dist.straggler import DetectionMethod, StragglerDetector
from distllm.models.partitioner import ModelPartitioner
from distllm.security import hf_revision

__all__ = [
    "Coordinator",
]


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

        # High-availability election (optional)
        self._ha_election: Any = None
        self._is_standby = False

        self._running = threading.Event()
        self._async_shutdown = asyncio.Event()
        self._request_results: dict[str, str] = {}
        self._request_events: dict[str, threading.Event] = {}
        self._request_lock = threading.Lock()  # Protects _request_results and _request_events
        self._health_check_interval_s: float = 10.0
        self._straggler_check_counter: int = 0
        self._health_thread: threading.Thread | None = None
        self._health_event = threading.Event()
        self._distribute_weights: bool = True
        self._hot_swap_mgr = None

        # Async batch scheduler support (used by RequestPipeline)
        self._batch_event = threading.Event()
        self._param_update_channel = ParamUpdateChannel()
        self._request_tracker = RequestTracker()
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

    # ── High Availability ──

    def enable_ha(
        self,
        coordinator_id: str | None = None,
        peer_coordinators: list[tuple[str, str, int]] | None = None,
        heartbeat_interval_s: float = 2.0,
        election_timeout_s: float = 10.0,
    ) -> None:
        """Enable high-availability mode with leader election.

        When enabled, this coordinator participates in leader election
        with peer coordinators. Only the leader accepts requests; standbys
        replicate state and wait for failover.

        Args:
            coordinator_id: Unique ID for this coordinator. Defaults to
                hostname:port.
            peer_coordinators: List of (id, host, port) tuples for peers.
            heartbeat_interval_s: Seconds between heartbeats.
            election_timeout_s: Seconds without heartbeat before election.
        """
        from distllm.core.ha_coordinator import RayFaultTolerance

        cid = coordinator_id or f"{self.model_name}:{self.port}"
        self._ha_election = RayFaultTolerance(
            coordinator_id=cid,
            heartbeat_interval_s=heartbeat_interval_s,
            election_timeout_s=election_timeout_s,
        )

        if peer_coordinators:
            for peer_id, peer_host, peer_port in peer_coordinators:
                self._ha_election.add_peer(peer_id, peer_host, peer_port)

        self._ha_election.start()
        logger.info(f"HA enabled for coordinator {cid}")

    @property
    def is_leader(self) -> bool:
        """Return True if this coordinator is the elected leader."""
        if self._ha_election is None:
            return True  # No HA = always leader
        return self._ha_election.is_leader()

    @property
    def ha_status(self) -> dict:
        """Return HA election status."""
        if self._ha_election is None:
            return {"enabled": False}
        return self._ha_election.stats()

    def state_snapshot(self) -> dict[str, Any]:
        """Create a snapshot of coordinator state for replication.

        Standby coordinators can use this to maintain a warm copy
        of the leader's state for fast failover.

        Returns:
            Dict with node registrations, model info, and config.
        """
        return {
            "model_name": self.model_name,
            "total_layers": self.total_layers,
            "nodes": {
                nid: {
                    "host": getattr(n, "host", ""),
                    "port": getattr(n, "port", 0),
                    "start_layer": getattr(n, "start_layer", 0),
                    "end_layer": getattr(n, "end_layer", 0),
                    "healthy": getattr(n, "healthy", False),
                }
                for nid, n in self.nodes.items()
            },
            "node_order": list(self.node_order),
            "timestamp": time.time(),
        }

    def apply_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Apply a state snapshot from the leader (for standby coordinators).

        Re-registers nodes and updates internal state to match the leader.
        """
        if not self._is_standby:
            logger.warning("apply_state_snapshot called on non-standby coordinator")
            return

        nodes = snapshot.get("nodes", {})
        for nid, info in nodes.items():
            if nid not in self.nodes:
                try:
                    self.manual_register(
                        node_id=nid,
                        host=info["host"],
                        port=info["port"],
                        start_layer=info["start_layer"],
                        end_layer=info["end_layer"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply snapshot node {nid}: {e}")

        logger.info(f"Applied state snapshot: {len(nodes)} nodes")

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
            trust_remote_code=getattr(self, 'trust_remote_code', False),
        )

        if utilization_fn is None:
            utilization_fn = self._default_utilization_fn

        compressor = SimpleCompressor(
            output_base=settings.output_dir,
            method=settings.compression_method,
            calibration_samples=settings.calibration_samples,
            trust_remote_code=getattr(self, 'trust_remote_code', False),
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

    def init_graceful_degradation(
        self,
        enabled: bool = True,
        light_threshold: float = 0.3,
        moderate_threshold: float = 0.5,
        severe_threshold: float = 0.7,
        critical_threshold: float = 0.85,
        fallback_model: str | None = None,
    ) -> None:
        """Initialize graceful degradation for overload protection.

        When system load exceeds thresholds, automatically reduces
        response quality instead of returning 503 errors.

        Args:
            enabled: Whether degradation is active.
            light_threshold: Load score for LIGHT degradation (reduce max_tokens).
            moderate_threshold: Load score for MODERATE (smaller model).
            severe_threshold: Load score for SEVERE (cached responses only).
            critical_threshold: Load score for CRITICAL (partial responses).
            fallback_model: Model name for moderate degradation fallback.
        """
        from distllm.core.graceful_degradation import GracefulDegradation
        self._graceful_degradation = GracefulDegradation(
            enabled=enabled,
            light_threshold=light_threshold,
            moderate_threshold=moderate_threshold,
            severe_threshold=severe_threshold,
            critical_threshold=critical_threshold,
            fallback_model=fallback_model,
        )
        logger.info(
            f"Graceful degradation initialized: "
            f"thresholds=[{light_threshold}, {moderate_threshold}, {severe_threshold}, {critical_threshold}]"
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
        except Exception as e:
            logger.warning(f"Failed to compute cluster utilization: {e}")
        return 0.0

    def _load_model_callback(self, name: str, path: str):
        """Load a model for hot-swap with a configurable timeout (default: 5 min)."""
        import concurrent.futures


        timeout = int(os.environ.get("DISTLLM_HOT_SWAP_TIMEOUT", "300"))

        def _load():
            partitioner = ModelPartitioner(model_name=path, dtype=self.dtype)
            partitioner.load_full_model()
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
        with self._request_lock:
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
                with self._request_lock:
                    self._request_results[request_id] = result
            except Exception as e:
                with self._request_lock:
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
            except Exception as e:
                logger.warning(f"Failed to record metric '{name}': {e}")

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
        with self._request_lock:
            event = self._request_events.get(request_id)
        if event is None:
            raise ValueError(f"Unknown request_id: {request_id}")
        event.wait(timeout=timeout)
        with self._request_lock:
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
        except Exception as e:
            logger.warning(
                f"Discovery service failed to start "
                f"(cluster auto-discovery disabled): {e}"
            )

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
        2. Cordon all nodes (mark as draining — stop sending new work)
        3. Stop accepting new requests (clear running flag)
        4. Wait for in-flight requests to complete (with timeout)
        5. Checkpoint all active sequences
        6. Release GPU memory
        7. Close gRPC connections
        8. Save state to disk
        9. Stop background threads
        """
        logger.info("Initiating graceful shutdown...")
        self._shutting_down = True

        # 1. Cordon all nodes — mark as draining so scheduler stops sending work
        for nid in list(self.nodes.keys()):
            try:
                if hasattr(self._resource_mgr, 'mark_node_draining'):
                    self._resource_mgr.mark_node_draining(nid)
                logger.debug(f"Cordoned node {nid}")
            except Exception as e:
                logger.warning(f"Failed to cordon node {nid}: {e}")

        # 2. Stop accepting new requests
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
            except Exception as e:
                logger.warning(f"Error closing sync connection pool: {e}")
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.ensure_future(self._resource_mgr._async_conn_pool.close_all())
                except RuntimeError:
                    asyncio.run(self._resource_mgr._async_conn_pool.close_all())
            except Exception as e:
                logger.warning(f"Error closing async connection pool: {e}")

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


# ── CLI entry point (moved to coordinator_cli.py) ────────────────────────
# Backward-compatible re-exports so existing imports keep working.
from distllm.core.coordinator_cli import _resolve_cluster_key, main  # noqa: E402

if __name__ == "__main__":
    main()
