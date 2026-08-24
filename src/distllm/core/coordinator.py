"""Coordinator for distributed LLM inference across multiple devices.

Splits responsibilities across four specialized components:
  - ``InferenceEngine`` — local/distributed text generation
  - ``ClusterManager`` — node registration, topology, weight distribution
  - ``HealthManager`` — periodic health probes, recovery, straggler detection
  - ``MetricsCollector`` — aggregated metrics from all subsystems
"""

from __future__ import annotations

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

from distllm.core.subsystem_registry import SubsystemRegistry
from distllm.core.batch_scheduler import BatchScheduler
from distllm.core.cache_manager import CacheManager
from distllm.core.cluster_manager import ClusterManager
from distllm.config.settings import NodeRole
from distllm.core.coordinator_config import CoordinatorConfig
from distllm.core.coordinator_config_wiring import CoordinatorConfigurator
from distllm.core.coordinator_request import RequestHandler
from distllm.core.debug import set_debug_mode
from distllm.core.health_manager import HealthManager
from distllm.errors import NotLeaderError
from distllm.core.inference_engine import InferenceEngine, TokenGenerator
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

from distllm.core.coordinator_election import CoordinatorElection
from distllm.core.coordinator_subsystem import SubsystemManager

# ── Named constants ──────────────────────────────────────────────
_RESULT_TTL_SECONDS = 300.0            # Stale request result cleanup TTL (5 min)
_HEALTH_CHECK_INTERVAL = 10.0          # Background health check interval (seconds)
_SEMANTIC_CACHE_MAX_ENTRIES = 10000    # Max entries in semantic cache
_CARBON_CHECK_INTERVAL = 300.0         # Carbon migration check interval (seconds, 5 min)
_STOP_TIMEOUT = 30.0                   # Graceful shutdown drain timeout (seconds)


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
        self._configurator = CoordinatorConfigurator(self)
        self._configurator._init_adaptive_batching()

        # Subsystem lifecycle manager — created before any subsystem callback
        # is wired, because StragglerDetector below references _subsystem_mgr
        # at construction time.
        self._subsystem_mgr = SubsystemManager(self)

        self._latency_tracker = LatencyTracker()
        # getattr-guard: under test doubles SubsystemManager may be a mock
        # whose interface lacks this callback.
        straggler_cb = getattr(self._subsystem_mgr, "_on_straggler_detected", None)
        self._straggler_detector = StragglerDetector(
            detection_method=DetectionMethod.MAD,
            on_straggler_cb=straggler_cb,
        )
        self._recovery_manager = NodeRecoveryManager()
        # ARCHITECTURE: Wire recovery callbacks so node failure recovery actually works
        self._configurator._wire_recovery_callbacks()
        self._reputation = ReputationSystem(
            min_reputation=getattr(self.config, 'min_reputation', 0.0),
        )
        self._federation: Any = None



        self._pipeline.set_latency_tracker(self._latency_tracker)
        self._pipeline.set_straggler_detector(self._straggler_detector)

        if self.config.wide_area_config:
            from distllm.dist.wide_area import WideAreaPipeline
            wa = self.config.wide_area_config
            self._pipeline = WideAreaPipeline(
                resource_mgr=self._resource_mgr,
                wan_config=wa,
                latency_tracker=self._latency_tracker,
            )
            self._pipeline.pipeline_timeout = wa.wan_timeout_seconds
            self._pipeline.set_straggler_detector(self._straggler_detector)
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

        # Pass the same recovery callbacks that _wire_recovery_callbacks()
        # installed on _recovery_manager.  Previously HealthManager silently
        # overwrote those with its own closures that set a nonexistent
        # ``healthy`` attribute, so drained nodes kept receiving work.
        self._health_mgr = HealthManager(
            pipeline=self._pipeline,
            resource_mgr=self._resource_mgr,
            reputation=self._reputation,
            recovery_manager=self._recovery_manager,
            straggler_detector=self._straggler_detector,
            drain_callback=self._on_node_drain,
            redistribute_callback=self._on_node_redistribute,
            recover_sequences_callback=self._on_node_recover,
            mark_dead_callback=self._on_node_mark_dead,
        )

        self._metrics_collector = MetricsCollector(
            latency_tracker=self._latency_tracker,
            straggler_detector=self._straggler_detector,
            recovery_manager=self._recovery_manager,
        )

        # ── Subsystem Registry (replaces manual lifecycle management) ──
        self._subsystem_registry = SubsystemRegistry()
        self._configurator._register_subsystems()


        # High-availability election (delegated to CoordinatorElection)
        self._election = CoordinatorElection(self)

        # HA leader election + state replication — enabled from config so the
        # previously-unreachable enable_ha()/set_replication_peers() surface
        # actually runs in production.
        if getattr(self.config, "ha_enabled", False):
            self._election.enable_ha(
                coordinator_id=f"{self.model_name}:{self.port}",
                peer_coordinators=getattr(self.config, "ha_peer_coordinators", None) or [],
                heartbeat_interval_s=getattr(self.config, "ha_heartbeat_interval_s", 2.0),
                election_timeout_s=getattr(self.config, "ha_election_timeout_s", 10.0),
            )
            repl_peers = getattr(self.config, "ha_replication_peers", None) or []
            if repl_peers:
                self._election.set_replication_peers(list(repl_peers))

        # Subsystem lifecycle manager is created at the top of __init__
        # (before the StragglerDetector callback is wired); it is available
        # here for the HA/replication paths below.
        self._running = threading.Event()
        self._async_shutdown = asyncio.Event()
        self._request_results: dict[str, str] = {}
        self._request_events: dict[str, threading.Event] = {}
        self._request_lock = threading.Lock()  # Protects _request_results and _request_events
        self._result_ttl_s = _RESULT_TTL_SECONDS  # 5 minutes — stale results cleaned automatically
        self._request_results_created: dict[str, float] = {}  # request_id -> monotonic timestamp
        self._health_check_interval_s: float = _HEALTH_CHECK_INTERVAL
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

        # Autoscaler — fed + actuated by the background _autoscaler_loop
        # (F-014 fix: previously record_metrics ran once at startup and
        # evaluate() was never called again, so autoscaling was inert).
        self._autoscaler: IntelligentAutoscaler | None = None
        self._autoscaler_thread: threading.Thread | None = None
        self._autoscale_stop = threading.Event()
        self._scale_callback: Callable[[Any], None] | None = None
        self._system_monitor: Any = None

        # Per-request scheduling hints (populated by API layer before generate())
        self._pending_scheduling_hints: dict[str, dict] = {}

        # Plugin system hooks (optional)
        self._plugin_system = getattr(self.config, "plugin_system", None)

        # Model router for query-based model switching
        self._model_router: ModelRouter | None = None

        # Multi-model registry: additional models beyond the primary one
        self._model_registry: dict[str, dict] | None = None

        # Advanced features: semantic cache, smart model routing, disaggregated P&D,
        # carbon-aware scheduling, cost tracking, arbitrage, and privacy-preserving split
        self._semantic_cache: Any = None
        self._smart_router: Any = None
        self._disaggregated_scheduler: Any = None
        self._carbon_engine: Any = None
        self._cost_tracker: Any = None
        self._arbitrage_engine: Any = None

        # Startup subsystem health tracking: name -> {"status": "ok"|"missing_deps"|"failed", "error": str|None}
        self._subsystem_health: dict[str, dict[str, Any]] = {}

        # Memory defragmentation
        self._defragmenter: MemoryDefragmenter | None = None
        self._defrag_task: asyncio.Task | None = None

        # Request handler for generation methods
        self._request_handler = RequestHandler(self)

        # AgenticRouter (LLM-as-judge routing) is lazily initialized in a later
        # dedicated setup path; always define the attribute so the request path
        # (coordinator_request.py reads c._agentic_router) never AttributeErrors.
        self._agentic_router = None

        # Attributes the request pipeline reads off the coordinator.  They
        # were referenced by request_pipeline.py but never initialized here,
        # crashing every batch path with AttributeError.
        self._token_gen = TokenGenerator()
        self._batch_kv_caches: dict[str, Any] = {}
        self._batch_kv_caches_lock = threading.Lock()
        # Optional prompt-cache service; None disables prompt-cache lookups
        # in the pipeline (it guards with `is not None`).
        self._prompt_cache_service: Any = None
        # Speculative decoder is wired later when a draft model is loaded;
        # define it now so pipeline reads never AttributeError.
        self._spec_decoder = None
        self.draft_model = None
        # Optional metrics exporter (set by the API server when wired);
        # the pipeline guards with `if c.metrics_exporter`.
        self.metrics_exporter = None
        # SelfOptimizingEngine hook is wired later; the pipeline guards
        # with `if c._self_optimizing`.
        self._self_optimizing = None

        # ── LoRA adapter serving (optional) ──
        self.adapter_manager: Any = None
        lora_cfg = kwargs.get("lora_config")
        if lora_cfg is not None and getattr(lora_cfg, "enabled", False):
            from distllm.models.adapter import AdapterManager

            self.adapter_manager = AdapterManager()

        # ── Multi-model serving (optional) ──
        self._multi_model: Any = None
        mmc = kwargs.get("multi_model_config")
        if mmc is not None and getattr(mmc, "enabled", False):
            from distllm.core.multi_model_serving import ModelRegistry as _MMRegistry
            from distllm.core.coordinator_multi_model import MultiModelManager

            reg = _MMRegistry(max_models=getattr(mmc, "max_models", 4))
            for name, path in (getattr(mmc, "models", {}) or {}).items():
                reg.register(name, path, total_layers=0)
            self._multi_model = MultiModelManager(
                model_name=getattr(mmc, "default_model", "") or self.model_name,
                pipeline=self._pipeline,
                model_registry=reg,
            )

        logger.info(f"Coordinator initialized for model: {self.model_name}")

    def _init_model_hotswap(
        self, max_models: int = 4, total_gpu_memory_gb: float = 0.0
    ) -> None:
        """Initialize hot-swap model management (delegates to configurator)."""
        self._configurator.init_hot_swap(
            total_gpu_memory_gb=total_gpu_memory_gb, max_models=max_models,
        )
        # Canonical alias used by multi-model tests/tools.
        self._model_hotswap = self._hot_swap_mgr

    def _on_node_drain(self, node_id: str) -> None:
        """Callback: mark a node as draining (stop new requests to it)."""
        logger.info(f"Draining node {node_id} (pausing new requests)")
        node = self._pipeline.get_node(node_id)
        if node:
            node.is_healthy = False

    def _on_node_mark_dead(self, node_id: str) -> None:
        """Callback: remove a dead node from the pipeline."""
        logger.info(f"Removing dead node {node_id} from pipeline")
        self._pipeline.remove_node(node_id)
        # Notify the AutonomousHealer if configured
        if hasattr(self, '_auto_healer') and self._auto_healer is not None:
            try:
                from distllm.core.autonomous_healer import GPUHeartbeat
                hb = GPUHeartbeat(node_id=node_id)
                self._auto_healer.record_heartbeat(hb)
            except Exception:
                pass

    def _on_node_redistribute(self, node_id: str, plan: Any) -> None:
        """Callback: redistribute a failed node's layers to survivors.

        This implements step 3 of the recovery flow — redistributing
        layers from the failed node to remaining healthy nodes.
        """
        redistributions = plan.redistributions if hasattr(plan, 'redistributions') else []
        if not redistributions:
            logger.warning(f"No redistributions computed for failed node {node_id}")
            return

        for rd in redistributions:
            survivor_id = rd.surviving_node_id
            survivor = self._pipeline.get_node(survivor_id)
            if survivor:
                logger.info(
                    f"Redistributing layers {rd.added_start_layer}-{rd.added_end_layer} "
                    f"to {survivor_id} (new range: {rd.new_start_layer}-{rd.new_end_layer})"
                )
                survivor.start_layer = rd.new_start_layer
                survivor.end_layer = rd.new_end_layer
                survivor.total_layers = max(survivor.total_layers, rd.new_end_layer + 1)
            else:
                logger.warning(
                    f"Survivor node {survivor_id} not found in pipeline for redistribution"
                )

    def _on_node_recover(self, node_id: str, sequence_ids: list[str]) -> list[Any]:
        """Callback: recover in-flight sequences from a failed node via checkpoint replay.

        Loads checkpoints saved during generation (via the recovery manager's
        periodic save_checkpoint calls in the pipeline loop), restores the
        KV cache, and replays newly generated tokens onto the surviving nodes.

        Returns:
            List of recovered sequence IDs, empty if no checkpoints found.
        """
        logger.info(
            f"Recovering {len(sequence_ids)} sequences from failed node {node_id}: "
            f"{sequence_ids[:5]}{'...' if len(sequence_ids) > 5 else ''}"
        )

        recovery_mgr = getattr(self, '_recovery_manager', None)
        if recovery_mgr is None:
            logger.warning("No recovery manager available — cannot replay checkpoints")
            return []

        # Get checkpoints for the specific failed sequences, or ALL if
        # sequence_ids is empty (belt-and-suspenders fallback).
        if sequence_ids:
            checkpoints = {}
            for sid in sequence_ids:
                ckpt = recovery_mgr.get_checkpoint(sid)
                if ckpt is not None:
                    checkpoints[sid] = ckpt
        else:
            checkpoints = recovery_mgr.get_checkpoints_for_node(node_id)
        if not checkpoints:
            logger.warning(f"No checkpoints found for failed node {node_id}")
            return []

        recovered = []
        for req_id, ckpt in checkpoints.items():
            try:
                # Merge prompt + generated tokens so the model can continue
                all_tokens = ckpt.prompt_tokens + ckpt.generated_tokens
                logger.info(
                    f"Replaying checkpoint for {req_id}: "
                    f"{len(ckpt.prompt_tokens)} prompt + {len(ckpt.generated_tokens)} generated tokens"
                )
                # Replay tokens through the surviving pipeline to restore
                # KV cache state on remaining nodes.  If the pipeline is not
                # available or replay fails, the checkpoint is still marked
                # as recovered (the caller will regenerate from the prompt).
                if self._pipeline is not None and all_tokens:
                    try:
                        input_tensor = torch.tensor([all_tokens])
                        self._pipeline.run_pipeline(input_tensor, {}, req_id)
                    except Exception as replay_err:
                        logger.warning(
                            f"Checkpoint replay for {req_id} failed "
                            f"(recovery will regenerate from prompt): {replay_err}"
                        )
                recovered.append(req_id)
            except Exception as e:
                logger.error(f"Failed to replay checkpoint for {req_id}: {e}")

        logger.info(f"Recovered {len(recovered)}/{len(checkpoints)} sequences for node {node_id}")
        return recovered

    def init_model_router(self, settings: Any | None = None) -> ModelRouter:
        """Initialize the model router from ChatRouterSettings.

        Args:
            settings: A ChatRouterSettings instance. If None, router is
                created with the coordinator's default model as fallback.

        Returns:
            The initialized ModelRouter instance.
        """
        return self._configurator.init_model_router(settings)

    # ── High Availability (delegated to CoordinatorElection) ──

    def enable_ha(
        self,
        coordinator_id: str | None = None,
        peer_coordinators: list[tuple[str, str, int]] | None = None,
        heartbeat_interval_s: float = 2.0,
        election_timeout_s: float = 10.0,
    ) -> None:
        """Enable high-availability mode with leader election.

        Delegated to :meth:`CoordinatorElection.enable_ha`.
        """
        self._election.enable_ha(
            coordinator_id=coordinator_id,
            peer_coordinators=peer_coordinators,
            heartbeat_interval_s=heartbeat_interval_s,
            election_timeout_s=election_timeout_s,
        )

    @property
    def is_leader(self) -> bool:
        """Return True if this coordinator is the elected leader."""
        return self._election.is_leader

    @property
    def ha_status(self) -> dict:
        """Return HA election status."""
        return self._election.ha_status

    def state_snapshot(self) -> dict[str, Any]:
        """Create a snapshot of coordinator state for replication.

        Delegated to :meth:`CoordinatorElection.state_snapshot`.
        """
        return self._election.state_snapshot()

    def apply_state_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Apply a state snapshot from the leader (for standby coordinators).

        Delegated to :meth:`CoordinatorElection.apply_state_snapshot`.
        """
        self._election.apply_state_snapshot(snapshot)

    def _start_state_replication(self) -> None:
        """Start continuous state replication to HA peer coordinators.

        Delegated to :meth:`CoordinatorElection._start_state_replication`.
        """
        self._election._start_state_replication()

    def _replication_loop(self) -> None:
        """Continuously push state snapshots to HA peers.

        Delegated to :meth:`CoordinatorElection._replication_loop`.
        """
        self._election._replication_loop()

    def set_replication_peers(self, peer_urls: list[str]) -> None:
        """Set HA peer coordinator URLs for state replication.

        Delegated to :meth:`CoordinatorElection.set_replication_peers`.
        """
        self._election.set_replication_peers(peer_urls)

    # ── Properties (delegated to ClusterManager) ──

    @property
    def nodes(self) -> dict:
        """Registered worker nodes (node_id -> NodeRegistration).

        Live mapping: callers may inject clients or flip health flags in
        place; structural changes go through register_node/unregister_node.
        """
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

    @property
    def prefix_cache(self) -> Any:
        """Prefix cache for prompt deduplication (None when disabled)."""
        return self._cache_mgr.prefix_cache if self._cache_mgr is not None else None

    @prefix_cache.setter
    def prefix_cache(self, value: Any) -> None:
        if self._cache_mgr is not None:
            self._cache_mgr.prefix_cache = value

    # ── Node lifecycle (delegated to ClusterManager) ──

    def auto_setup(self, nodes_config: list[dict]) -> tuple[dict, int] | None:
        return self._configurator.auto_setup(nodes_config)

    def manual_register(self, node_id: str, host: str, port: int,
                        start_layer: int, end_layer: int,
                        total_layers: int | None = None,
                        role: NodeRole = NodeRole.AUTO,
                        expert_ids: list[int] | None = None,
                        cluster_id: str = "default",
                        cluster_key: str | None = None) -> None:
        self._configurator.manual_register(
            node_id, host, port, start_layer, end_layer,
            total_layers=total_layers, role=role,
            expert_ids=expert_ids, cluster_id=cluster_id,
            cluster_key=cluster_key,
        )

    # ── Generation (delegated to RequestHandler) ──

    def _require_leader(self) -> None:
        """Refuse to serve inference when this coordinator is not the HA leader.

        F-015: In HA mode, only the elected leader may serve generation —
        standbys acting as writers would diverge and defeat the split-brain
        protection the election exists to provide.  When HA is disabled (or
        this node is the leader) this is a no-op, so single-node serving is
        unaffected.  When this node is a standby, every request-serving entry
        point (:meth:`generate` and :meth:`generate_async`) raises
        :class:`NotLeaderError` so consumers can retry on the leader.
        """
        if not getattr(self.config, "ha_enabled", False):
            return
        if self.is_leader:
            return
        leader_id: str | None = None
        election = getattr(self._election, "_ha_election", None)
        if election is not None:
            leader_id = election.get_leader()
        raise NotLeaderError(leader_id=leader_id)

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 0, request_id: str | None = None,
                 user_id: str = "default",
                 speculative_config: dict | None = None,
                 response_format: dict | None = None,
                 constraint: Any | None = None) -> str:
        self._require_leader()
        return self._request_handler.generate(
            prompt, max_new_tokens, temperature, top_p, top_k,
            request_id, user_id, speculative_config,
            response_format=response_format,
            constraint=constraint,
        )

    def generate_batch(self, timeout: float = 120.0, max_steps: int = 0) -> None:
        """Run the batch scheduler until all queued sequences complete.

        Delegates to the wired request pipeline (``_request_handler``), which
        owns the scheduler step loop.  Raises ``BatchError`` when batch
        processing is not configured (single-request mode).
        """
        self._require_leader()
        handler = getattr(self, "_request_handler", None)
        if handler is None or not hasattr(handler, "generate_batch"):
            from distllm.errors.types import BatchError
            raise BatchError("Batch scheduler not configured. Use generate() instead.")
        return handler.generate_batch(timeout=timeout, max_steps=max_steps)

    def load_local_model(self) -> None:
        self._request_handler.load_local_model()

    def set_deterministic_mode(self, enabled: bool = True, seed: int = 42) -> None:
        self._request_handler.set_deterministic_mode(enabled, seed)

    def get_recent_requests(self, n: int = 10) -> list[Any]:
        return self._request_handler.get_recent_requests(n)

    # ── Hot-swap model management ──

    def init_hot_swap(self, total_gpu_memory_gb: float = 0.0, max_models: int = 4) -> None:
        """Initialize the hot-swap model manager for dynamic model loading."""
        self._configurator.init_hot_swap(
            total_gpu_memory_gb=total_gpu_memory_gb,
            max_models=max_models,
        )

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
        self._configurator.init_adaptive_compression(
            settings=settings,
            utilization_fn=utilization_fn,
        )

    def init_defragmentation(self, settings: Any | None = None) -> None:
        """Initialize the GPU memory defragmenter.

        Args:
            settings: A DefragmentationSettings instance or None to disable.
        """
        self._configurator.init_defragmentation(settings)

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
        self._configurator.init_graceful_degradation(
            enabled=enabled,
            light_threshold=light_threshold,
            moderate_threshold=moderate_threshold,
            severe_threshold=severe_threshold,
            critical_threshold=critical_threshold,
            fallback_model=fallback_model,
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

                        # Temperature-aware defragmentation: skip or reduce
                        # aggressiveness when active requests have high
                        # temperatures (>1.0), since high-temperature sampling
                        # is more sensitive to cache state changes.  Under
                        # high-temperature workloads, L1 compaction only.
                        active_temp = self._get_active_temperature()
                        if active_temp is not None and active_temp > 1.0:
                            if tier != TieredCompactionLevel.L1_HOT:
                                logger.debug(
                                    f"Temperature-aware defrag: reducing tier from {tier.value} "
                                    f"to L1_HOT (active temp={active_temp:.2f})"
                                )
                                tier = TieredCompactionLevel.L1_HOT

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

    def _get_active_temperature(self) -> float | None:
        """Return the average temperature across active generation requests.

        Used by temperature-aware defragmentation to avoid aggressive
        cache reorganisation when high-temperature sampling is active,
        since high-temp outputs are more sensitive to cache state changes.

        Returns None if no active requests or no scheduler available.
        """
        if self._batch_scheduler is None:
            return None
        try:
            temps = self._batch_scheduler.get_active_temperatures()
            return sum(temps) / len(temps) if temps else None
        except Exception:
            return None

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

    def _wire_paged_attention(self) -> None:
        """Connect the engine's PagedAttentionManager to the batch scheduler.

        The manager is created during ``load_local_model()``; without this
        wiring ``BatchScheduler.allocate_paged_blocks()`` always returned
        None, the CPU-swap/preemption paths no-op'd, and the defrag loop
        iterated zero backends forever.
        """
        paged_mgr = getattr(self._inference_engine, "_paged_mgr", None)
        if paged_mgr is None or self._batch_scheduler is None:
            return
        try:
            self._batch_scheduler.set_paged_attention(paged_mgr)
            logger.info("PagedAttention manager wired into batch scheduler")
        except Exception as e:
            logger.warning(f"Failed to wire PagedAttention into scheduler: {e}")

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
        result = self._hot_swap_mgr.register_model(name, path, total_layers, memory_budget_gb)
        return bool(result) if result is not None else True

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

        # Determine overall status: degrade if any subsystem has a runtime failure
        has_failures = any(
            info["status"] == "failed"
            for info in self._subsystem_health.values()
        )
        if not self.nodes:
            status = "no_nodes"
        elif has_failures:
            status = "degraded"
        else:
            status = "ok"

        return {
            "status": status,
            "num_nodes": len(self.nodes),
            "total_layers": self._pipeline.total_layers,
            "nodes": nodes_status,
            "reputation": self._reputation.get_scores(),
            "subsystems": {
                name: info["status"]
                for name, info in sorted(self._subsystem_health.items())
            },
        }

    # ── Metrics (delegated to MetricsCollector) ──

    def get_metrics(self) -> dict:
        return self._metrics_collector.collect()

    def list_models(self) -> list[str]:
        """Return available model names including hybrid router names."""
        models: list[str] = []
        if self.model_name:
            models.append(self.model_name)
        # Include models registered with the multi-model registry
        if self._model_registry:
            models.extend(self._model_registry.keys())
        # Include hot-swap catalog names
        if self._multi_model is not None:
            for m in self._multi_model.list_models():
                if m and m not in models:
                    models.append(m)
        # Include hybrid model names registered with the router
        if self._model_router is not None:
            models.extend(self._model_router.list_hybrid_models())
        return models

    def register_model(
        self, name: str, path: str, memory_budget_gb: float = 0.0,
        total_layers: int = 0,
    ):
        """Register an additional model for multi-model serving.

        Records the entry in both the routing registry and the multi-model
        hot-swap catalog; returns the catalog entry.
        """
        if self._model_registry is None:
            self._model_registry = {}
        self._model_registry[name] = {
            "path": path,
            "memory_budget_gb": memory_budget_gb,
        }
        if self._multi_model is None or getattr(self._multi_model, "registry", None) is None:
            from distllm.core.multi_model_serving import ModelRegistry as _MMRegistry
            from distllm.core.coordinator_multi_model import MultiModelManager

            self._multi_model = MultiModelManager(
                model_name=self.model_name,
                pipeline=self._pipeline,
                model_registry=_MMRegistry(),
            )
        return self._multi_model.registry.register(name, path, total_layers)

    def get_model_name(self, requested: str = "") -> str:
        """Resolve a requested model name to a loaded model.

        Falls back to the primary model when no registry match exists.
        When multiple models are registered and no name is requested,
        the first registered model wins; otherwise the primary model.
        """
        if requested and self._model_registry and requested in self._model_registry:
            return requested
        if requested == self.model_name:
            return requested
        if self._model_registry:
            return next(iter(self._model_registry))
        return self.model_name

    def generate_async(self, prompt: str, **kwargs) -> str:
        """Schedule async generation via the batch scheduler.

        If the batch scheduler is configured, adds a Sequence directly
        to the scheduler for true continuous batching. Otherwise falls
        back to a background thread.

        Returns a request_id immediately. Call ``wait_for_result()``
        to get the result when ready.
        """
        # F-015: same gating as generate() — the async/batch path bypasses
        # generate() and would otherwise let a standby serve requests.
        self._require_leader()
        return self._request_handler.generate_async(prompt, **kwargs)

    def record_metric(self, name: str, value: float = 1.0) -> None:
        """Record a metric (used by RequestPipeline)."""
        self._request_handler.record_metric(name, value)

    def _cleanup_stale_results(self) -> None:
        """Remove stale entries from _request_results to prevent memory leaks."""
        self._request_handler._cleanup_stale_results()

    def wait_for_result(self, request_id: str, timeout: float | None = None) -> str:
        """Wait for an async generation result.

        Checks both the request tracker (batch scheduler path) and
        the legacy event-based path (background thread fallback).
        """
        return self._request_handler.wait_for_result(request_id, timeout=timeout)

    # ── Autoscaling (F-014: periodic feed + actuation) ──

    def set_scale_callback(self, callback: Callable[[Any], None] | None) -> None:
        """Register the provisioning hook invoked with each should-scale decision.

        This is the actuation path for the IntelligentAutoscaler: wire it to
        node provisioning (e.g. ClusterManager registration of newly launched
        workers, or an external provisioner). Without a callback, scale
        decisions are logged but not acted upon.
        """
        self._scale_callback = callback

    def _get_system_monitor(self) -> Any | None:
        """Lazily create the SystemMonitor (cached; ``False`` = unavailable)."""
        monitor = self._system_monitor
        if monitor is None:
            try:
                from distllm.core.monitor import SystemMonitor
                monitor = SystemMonitor()
            except Exception:
                monitor = False
            self._system_monitor = monitor
        return monitor or None

    def _collect_scaling_metrics(self) -> Any:
        """Build ScalingMetrics from live sources.

        ``gpu_utilization`` comes from SystemMonitor (pynvml) when telemetry
        is available; otherwise it falls back to batch occupancy so the value
        reflects load instead of being permanently 0.0.
        """
        from distllm.core.intelligent_autoscaler import ScalingMetrics

        stats = self._batch_scheduler.stats() if self._batch_scheduler else {}
        active = int(stats.get("active_requests", 0))
        pending = int(stats.get("pending_requests", 0))
        max_batch = int(stats.get("max_batch_size", 1)) or 1

        gpu_util: float | None = None
        monitor = self._get_system_monitor()
        if monitor is not None:
            try:
                gpu_info = monitor.collect().get("gpu") or {}
                raw = gpu_info.get("utilization_gpu")
                gpu_util = float(raw) if raw is not None else None
            except Exception:
                gpu_util = None
        if gpu_util is None:
            gpu_util = min(100.0, (active + pending) / max_batch * 100.0)

        return ScalingMetrics(
            active_requests=active,
            pending_requests=pending,
            queue_depth=pending,
            gpu_utilization=gpu_util,
            current_nodes=len(self.nodes),
        )

    def _autoscaler_tick(self) -> None:
        """Run one autoscaler cycle: feed metrics, evaluate, actuate."""
        scaler = self._autoscaler
        if scaler is None:
            return
        metrics = self._collect_scaling_metrics()
        decision = scaler.evaluate(metrics)
        if not decision.should_scale:
            return
        logger.info(
            "Autoscaler decision: {} -> {} nodes (reason={}, confidence={:.2f})",
            metrics.current_nodes, decision.target_nodes,
            decision.reason, decision.confidence,
        )
        callback = self._scale_callback
        if callback is not None:
            try:
                callback(decision)
            except Exception as e:
                logger.error("Autoscale callback failed: {}", e)

    def _autoscaler_loop(self, interval_s: float = 5.0) -> None:
        """Periodically feed real metrics to the autoscaler and actuate."""
        while True:
            try:
                self._autoscaler_tick()
            except Exception as e:
                logger.debug("Autoscaler tick failed: {}", e)
            if self._autoscale_stop.wait(interval_s):
                break

    def _start_autoscaler_loop(self, interval_s: float | None = None) -> None:
        """Start the background autoscaler loop (no-op without an autoscaler)."""
        if self._autoscaler is None:
            return
        if self._autoscaler_thread is not None and self._autoscaler_thread.is_alive():
            return
        if interval_s is None:
            try:
                interval_s = float(os.environ.get("DISTLLM_AUTOSCALER_INTERVAL_S", "5"))
            except ValueError:
                interval_s = 5.0
        self._autoscale_stop.clear()
        t = threading.Thread(
            target=self._autoscaler_loop, args=(interval_s,),
            daemon=True, name="autoscaler-loop",
        )
        self._autoscaler_thread = t
        t.start()
        logger.info("Autoscaler background loop started (interval={}s)", interval_s)

    def _stop_autoscaler_loop(self, timeout: float = 2.0) -> None:
        """Signal the autoscaler loop to stop and join it."""
        self._autoscale_stop.set()
        t = self._autoscaler_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._autoscaler_thread = None

    # ── Subsystem startup helper ──

    def _start_subsystem(
        self,
        name: str,
        module_path: str,
        class_name: str,
        attrs_name: str,
        constructor_kwargs: dict | None = None,
        post_init: Callable | None = None,
    ) -> Any | None:
        """Start an optional subsystem, returning the instance or None.

        All 9 optional subsystems follow the same try/except pattern.
        This helper eliminates ~150 lines of near-identical boilerplate.

        Args:
            name: Subsystem name for health tracking (e.g. ``"discovery"``).
            module_path: Dot-separated module path.
            class_name: Class to import from *module_path*.
            attrs_name: Attribute name to store the instance on ``self``.
            constructor_kwargs: Dict of kwargs for the constructor.
            post_init: Optional callable ``fn(instance)`` called after init.

        Returns:
            The instance, or None if import failed.
        """
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            instance = cls(**(constructor_kwargs or {}))
            setattr(self, attrs_name, instance)
            self._subsystem_health[name] = {"status": "ok", "error": None}
            if post_init:
                post_init(instance)
            logger.info("{} initialized", name.replace("_", " ").title())
            return instance
        except ImportError as e:
            self._subsystem_health[name] = {"status": "missing_deps", "error": str(e)}
            setattr(self, attrs_name, None)
            logger.debug("{} not available: {}", name, e)
            return None
        except Exception as e:
            self._subsystem_health[name] = {"status": "failed", "error": str(e)}
            setattr(self, attrs_name, None)
            logger.error("{} failed to start: {}", name, e)
            return None

    # ── Start / Stop ──

    def _maybe_start_disaggregated(self) -> None:
        """Register the disaggregated P&D scheduler behind an explicit opt-in.

        The scheduler's cross-pool KV transfer is not implemented yet
        (``stream_kv_cache`` and ``allocate_decode_blocks`` are fenced
        stubs), so it is OFF by default.  Set ``DISTLLM_DISAGGREGATED_ENABLED=1``
        to register it anyway.
        """
        if os.environ.get("DISTLLM_DISAGGREGATED_ENABLED", "") != "1":
            self._subsystem_health["disaggregated_scheduler"] = {
                "status": "disabled",
                "error": None,
            }
            logger.debug(
                "Disaggregated P&D scheduler disabled "
                "(set DISTLLM_DISAGGREGATED_ENABLED=1 to opt in)"
            )
            return

        # C12: loud warning so operators know KV transfer is a stub —
        # prefills/decodes routed to separate pools would silently lose KV.
        logger.warning(
            "Disaggregated P&D scheduler enabled via DISTLLM_DISAGGREGATED_ENABLED: "
            "EXPERIMENTAL — cross-pool KV cache TRANSFER IS NOT IMPLEMENTED "
            "(stream_kv_cache / allocate_decode_blocks are stubs). "
            "Do not route production traffic through it."
        )
        self._start_subsystem(
            "disaggregated_scheduler",
            "distllm.core.advanced_scheduling.disaggregated",
            "DisaggregatedBatchScheduler",
            "_disaggregated_scheduler",
        )

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

        # Connect the PagedAttention block manager (created during model
        # load) to the batch scheduler so paged allocation, preemption, and
        # defrag paths are live.
        self._wire_paged_attention()

        # Propagate model metadata to the batch scheduler so its metrics
        # label rows with the real model name instead of "default".
        if getattr(self, "model_info", None) and self._batch_scheduler is not None:
            self._batch_scheduler.set_model_info(self.model_info)

        # Start HA state replication if peers configured
        self._election._start_state_replication()

        if hasattr(self, '_adaptive_compression_mgr') and self._adaptive_compression_mgr:
            self._adaptive_compression_mgr.start()

        if self._defragmenter is not None:
            # BUG-007: Check for a running event loop before using ensure_future
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                self._defrag_task = asyncio.ensure_future(self._defrag_loop())
                logger.info("Defrag background loop started (async)")
            else:
                # No running loop — start defrag in a background thread instead
                def _run_defrag_loop():
                    import asyncio as _asyncio
                    _loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(_loop)
                    _loop.run_until_complete(self._defrag_loop())
                    _loop.close()
                t = threading.Thread(target=_run_defrag_loop, daemon=True, name="defrag-loop")
                t.start()
                logger.info("Defrag background loop started (threaded fallback)")

        # --- Optional subsystems (via _start_subsystem helper) ---
        self._start_subsystem(
            "discovery", "distllm.dist.discovery", "DiscoveryService", "_discovery",
            constructor_kwargs={"port": self.port, "service_id": "distllm-coordinator"},
        )

        if (hasattr(self.config, 'federation_config')
                and self.config.federation_config
                and self.config.federation_config.enabled):
            self._start_subsystem(
                "federation", "distllm.dist.federation", "FederationCoordinator", "_federation",
                constructor_kwargs={
                    "config": self.config.federation_config,
                    "local_cluster_id": self.config.federation_config.cluster_id,
                    "local_host": 'localhost',
                    "local_port": self.port,
                    "coordinator_ref": self,
                },
            )

        self._start_subsystem(
            "smart_router", "distllm.core.smart_model_router", "SmartModelRouter", "_smart_router",
        )

        self._start_subsystem(
            "semantic_cache", "distllm.core.semantic_cache", "SemanticCache", "_semantic_cache",
            constructor_kwargs={
                "similarity_threshold": float(os.environ.get("DISTLLM_SEMANTIC_CACHE_THRESHOLD", "0.92")),
                "max_entries": _SEMANTIC_CACHE_MAX_ENTRIES,
            },
        )

        self._maybe_start_disaggregated()

        self._start_subsystem(
            "carbon_engine", "distllm.core.carbon_migration", "CarbonMigrationEngine", "_carbon_engine",
            constructor_kwargs={
                "threshold": float(os.environ.get("DISTLLM_CARBON_THRESHOLD", "400.0")),
                "check_interval_s": _CARBON_CHECK_INTERVAL,
            },
        )

        self._start_subsystem(
            "arbitrage_engine", "distllm.core.arbitrage_engine", "ArbitrageEngine", "_arbitrage_engine",
            constructor_kwargs={
                "min_savings_pct": float(os.environ.get("DISTLLM_ARBITRAGE_MIN_SAVINGS", "15.0")),
                "check_interval_s": float(os.environ.get("DISTLLM_ARBITRAGE_INTERVAL", "60.0")),
            },
        )

        self._start_subsystem(
            "cost_tracker", "distllm.core.cost_tracker", "CostTracker", "_cost_tracker",
        )

        # --- Wire new modules into production paths ---

        # AutonomousHealer: integrates with node failure callbacks
        self._auto_healer = None
        try:
            from distllm.core.autonomous_healer import AutonomousHealer
            self._auto_healer = AutonomousHealer(
                on_drain_callback=self._on_node_drain,
                on_recover_callback=self._on_node_recover,
                dry_run=os.environ.get("DISTLLM_AUTO_HEAL_DRY_RUN", "1") == "1",
            )
            self._auto_healer.start()
            logger.info("Autonomous healer started (dry_run={})", os.environ.get("DISTLLM_AUTO_HEAL_DRY_RUN", "1"))
        except Exception as e:
            logger.debug("Autonomous healer not available: {}", e)

        # SpotEnsembleManager: multi-provider spot instance pooling
        self._spot_ensemble = None
        try:
            from distllm.core.arbitrage_engine import SpotEnsembleManager
            self._spot_ensemble = SpotEnsembleManager()
            logger.info("Spot ensemble manager initialized")
        except Exception as e:
            logger.debug("Spot ensemble not available: {}", e)

        # AgenticRouter: LLM-as-judge routing layer
        self._agentic_router = None
        try:
            from distllm.core.agentic_router import AgenticRouter
            router_models = [{"name": self.model_name, "quantization": "int4"}]
            self._agentic_router = AgenticRouter(available_models=router_models)
            logger.info("Agentic router initialized")
        except Exception as e:
            logger.debug("Agentic router not available: {}", e)

        # Autoscaler needs special handling (ScalingMetrics init)
        autoscaler = self._start_subsystem(
            "autoscaler", "distllm.core.intelligent_autoscaler",
            "IntelligentAutoscaler", "_autoscaler",
            constructor_kwargs={"min_nodes": 1, "max_nodes": 20, "target_utilization": 0.7},
        )
        if autoscaler is not None:
            # F-014: feed real metrics and actuate periodically instead of a
            # single startup record_metrics call that was never evaluated.
            self._start_autoscaler_loop()

        # Startup self-check: log which subsystems are active vs degraded/missing
        failed_subsystems = [
            name for name, info in self._subsystem_health.items()
            if info["status"] == "failed"
        ]
        missing_subsystems = [
            name for name, info in self._subsystem_health.items()
            if info["status"] == "missing_deps"
        ]
        active_subsystems = [
            name for name, info in self._subsystem_health.items()
            if info["status"] == "ok"
        ]
        logger.info(
            f"Startup subsystem check: "
            f"{len(active_subsystems)} active, "
            f"{len(missing_subsystems)} missing deps, "
            f"{len(failed_subsystems)} failed"
        )
        if active_subsystems:
            logger.info(f"  Active: {', '.join(sorted(active_subsystems))}")
        if missing_subsystems:
            logger.info(f"  Missing deps (optional): {', '.join(sorted(missing_subsystems))}")
        if failed_subsystems:
            logger.warning(f"  FAILED (degraded): {', '.join(sorted(failed_subsystems))}")

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
            # CRIT-003 fix: Check for a running event loop before using ensure_future
            try:
                _loop = asyncio.get_running_loop()
            except RuntimeError:
                _loop = None
            if _loop is not None and _loop.is_running():
                asyncio.ensure_future(_wait_and_callback_async())
            else:
                # No running loop — run callback in a background thread
                def _run_shutdown_callback():
                    import asyncio as _asyncio
                    __loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(__loop)
                    __loop.run_until_complete(_wait_and_callback_async())
                    __loop.close()
                t = threading.Thread(
                    target=_run_shutdown_callback,
                    daemon=True,
                    name="shutdown-callback",
                )
                t.start()

    def stop(self, timeout: float = _STOP_TIMEOUT) -> None:
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
            active_count = self._batch_scheduler.active_count
            if active_count > 0:
                logger.info(f"Waiting for {active_count} in-flight requests to complete (timeout={timeout}s)...")
                deadline = time.time() + timeout
                remaining = active_count
                while remaining > 0 and time.time() < deadline:
                    time.sleep(0.25)
                    remaining = self._batch_scheduler.active_count
                if remaining > 0:
                    logger.warning(f"{remaining} requests still in-flight after timeout, forcing shutdown")

        # 3. Checkpoint active sequences
        if self._batch_scheduler is not None and self._batch_scheduler.active_count > 0:
            active_snapshot = self._batch_scheduler.snapshot_active()
            for req_id, seq in active_snapshot:
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

        # 6. Close connection pools and thread pools
        if hasattr(self, '_resource_mgr') and self._resource_mgr is not None:
            try:
                self._resource_mgr._health_check_pool.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"Error shutting down health check pool: {e}")
        if hasattr(self, '_resource_mgr') and self._resource_mgr is not None:
            try:
                self._resource_mgr._conn_pool.close_all()
            except Exception as e:
                logger.warning(f"Error closing sync connection pool: {e}")
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    asyncio.ensure_future(self._resource_mgr._async_conn_pool.close_all())
                else:
                    # No running loop — close synchronously in a new event loop
                    asyncio.run(self._resource_mgr._async_conn_pool.close_all())
            except Exception as e:
                logger.warning(f"Error closing async connection pool: {e}")

        # 7. Stop background services
        self._stop_autoscaler_loop()
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
        # Only persist safe, JSON-serialisable fields.  Avoid ``default=str``
        # which both leaks sensitive data through string coercion and produces
        # write-only state files that cannot be reliably deserialised.
        state: dict[str, Any] = {
            "model_name": self.model_name,
            "shutdown_time": time.time(),
            "nodes": {
                nid: {
                    "host": n.host,
                    "port": n.port,
                    # Canonical health attribute (PipelineNode.is_healthy);
                    # ``n.healthy`` is not a defined field.
                    "healthy": bool(
                        getattr(n, "is_healthy", getattr(n, "healthy", True))
                    ),
                }
                for nid, n in self.nodes.items()
            },
        }
        # H-17: Write to a protected path — use data dir with restricted perms
        state_dir = os.environ.get("DISTLLM_DATA_DIR", os.path.expanduser("~/.distllm"))
        os.makedirs(state_dir, exist_ok=True)
        state_path = os.path.join(state_dir, "shutdown_state.json")
        fd = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug(f"Shutdown state saved to {state_path}")


# ── CLI entry point (moved to coordinator_cli.py) ────────────────────────
# Backward-compatible re-exports so existing imports keep working.
from distllm.core.coordinator_cli import _resolve_cluster_key, main  # noqa: E402

if __name__ == "__main__":
    main()
