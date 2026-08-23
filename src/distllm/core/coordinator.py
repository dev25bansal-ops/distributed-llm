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
        # ARCHITECTURE: Wire recovery callbacks so node failure recovery actually works
        self._wire_recovery_callbacks()
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

        # ── Subsystem Registry (replaces manual lifecycle management) ──
        self._subsystem_registry = SubsystemRegistry()
        self._register_subsystems()


        # High-availability election (optional)
        self._ha_election: Any = None
        self._is_standby = False

        self._running = threading.Event()
        self._async_shutdown = asyncio.Event()
        self._replication_thread: threading.Thread | None = None
        self._replication_peers: list[str] = []
        self._request_results: dict[str, str] = {}
        self._request_events: dict[str, threading.Event] = {}
        self._request_lock = threading.Lock()  # Protects _request_results and _request_events
        self._result_ttl_s = 300.0  # 5 minutes — stale results cleaned automatically
        self._request_results_created: dict[str, float] = {}  # request_id -> monotonic timestamp
        self._last_result_cleanup: float = time.monotonic()
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

        # Autoscaler (wire with real metrics from batch scheduler)
        self._autoscaler: IntelligentAutoscaler | None = None

        # Per-request scheduling hints (populated by API layer before generate())
        self._pending_scheduling_hints: dict[str, dict] = {}

        # Plugin system hooks (optional)
        self._plugin_system = getattr(self.config, "plugin_system", None)

        # Model router for query-based model switching
        self._model_router: ModelRouter | None = None

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

        logger.info(f"Coordinator initialized for model: {self.model_name}")

    def _wire_recovery_callbacks(self) -> None:
        """Wire NodeRecoveryManager callbacks so node failure recovery works.

        Previously these callbacks were never set, making the recovery
        manager's redistribution and sequence recovery steps no-ops.
        """
        self._recovery_manager.set_drain_callback(self._on_node_drain)
        self._recovery_manager.set_mark_dead_callback(self._on_node_mark_dead)
        self._recovery_manager.set_redistribute_layers_callback(self._on_node_redistribute)
        self._recovery_manager.set_recover_sequences_callback(self._on_node_recover)

    def _register_subsystems(self) -> None:
        """Register all subsystems with the SubsystemRegistry for lifecycle management."""
        reg = self._subsystem_registry
        reg.register("pipeline", self._pipeline, start_fn=self._pipeline.start if hasattr(self._pipeline, 'start') else None)
        reg.register("batch_scheduler", self._batch_scheduler)
        reg.register("health_manager", self._health_mgr)
        reg.register("metrics_collector", self._metrics_collector)
        reg.register("straggler_detector", self._straggler_detector)
        reg.register("latency_tracker", self._latency_tracker)
        reg.register("reputation", self._reputation)

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

    # ── HA State Replication ──

    def _start_state_replication(self) -> None:
        """Start continuous state replication to HA peer coordinators.

        Runs a background thread that pushes state snapshots to
        all configured peers every 1 second for sub-second failover.
        """
        if not self._replication_peers:
            return
        self._replication_thread = threading.Thread(
            target=self._replication_loop,
            daemon=True,
            name="state-replication",
        )
        self._replication_thread.start()
        logger.info(f"State replication started for {len(self._replication_peers)} peers")

    def _replication_loop(self) -> None:
        """Continuously push state snapshots to HA peers.

        Uses a 1-second interval, but only sends a full snapshot every
        10th call (10s) when the cluster is stable.  Intermediate ticks
        send a lightweight heartbeat to detect peer liveness.  This
        reduces the O(n * peers) serialization + network overhead for
        large clusters by ~90% during steady state.
        """
        import httpx
        tick = 0
        with httpx.Client(timeout=2.0) as client:
            while self._running.is_set():
                tick += 1
                try:
                    # Full snapshot every ~10s, lightweight ping otherwise
                    if tick % 10 == 0:
                        snapshot = self.state_snapshot()
                    else:
                        snapshot = {
                            "heartbeat": True,
                            "node_count": len(self.nodes),
                            "healthy": self._health_mgr.is_healthy(),
                        }
                    for peer_url in self._replication_peers:
                        try:
                            resp = client.post(
                                f"{peer_url.rstrip('/')}/api/v1/ha/snapshot",
                                json=snapshot,
                            )
                            if resp.status_code != 200:
                                logger.debug(f"Replication to {peer_url} returned {resp.status_code}")
                        except Exception as e:
                            logger.debug(f"Replication to {peer_url} failed: {e}")
                except Exception as e:
                    logger.warning(f"State replication error: {e}")
                time.sleep(1.0)

    def set_replication_peers(self, peer_urls: list[str]) -> None:
        """Set HA peer coordinator URLs for state replication.

        Args:
            peer_urls: List of peer API base URLs (e.g. ["http://10.0.0.2:8000"]).
        """
        self._replication_peers = peer_urls
        if self._running.is_set():
            self._start_state_replication()

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
                 response_format: dict | None = None,
                 constraint: Any | None = None) -> str:
        self._inference_engine.tokenizer = self.tokenizer

        # Optional AgenticRouter pre-routing: if configured, use the LLM
        # judge to select the optimal model before delegating to the engine.
        if hasattr(self, '_agentic_router') and self._agentic_router is not None:
            decision = self._agentic_router.route(prompt)
            if decision.model and decision.model != self.model_name:
                logger.info(
                    f"AgenticRouter selected model={decision.model} "
                    f"(confidence={decision.confidence:.2f}) "
                    f"instead of default {self.model_name}"
                )

        # Use caller-provided constraint if given, otherwise build from response_format
        if constraint is None and response_format:
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
                    self._request_results_created[request_id] = time.monotonic()
            except Exception as e:
                with self._request_lock:
                    self._request_results[request_id] = f"[Error: {e}]"
                    self._request_results_created[request_id] = time.monotonic()
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

    def _cleanup_stale_results(self) -> None:
        """Remove stale entries from _request_results to prevent memory leaks."""
        now = time.monotonic()
        if now - self._last_result_cleanup < 60:  # Only run cleanup once per minute
            return
        self._last_result_cleanup = now
        with self._request_lock:
            stale = [
                rid for rid, created in self._request_results_created.items()
                if now - created > self._result_ttl_s
            ]
            for rid in stale:
                self._request_results.pop(rid, None)
                self._request_events.pop(rid, None)
                self._request_results_created.pop(rid, None)
            if stale:
                logger.debug(f"Cleaned {len(stale)} stale request results (TTL={self._result_ttl_s}s)")

    def wait_for_result(self, request_id: str, timeout: float | None = None) -> str:
        """Wait for an async generation result.

        Checks both the request tracker (batch scheduler path) and
        the legacy event-based path (background thread fallback).
        """
        # Periodic cleanup — called opportunistically from wait_for_result
        self._cleanup_stale_results()
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

        # Start HA state replication if peers configured
        self._start_state_replication()

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
                "max_entries": 10000,
            },
        )

        self._start_subsystem(
            "disaggregated_scheduler", "distllm.core.advanced_scheduling.disaggregated",
            "DisaggregatedBatchScheduler", "_disaggregated_scheduler",
        )

        self._start_subsystem(
            "carbon_engine", "distllm.core.carbon_migration", "CarbonMigrationEngine", "_carbon_engine",
            constructor_kwargs={
                "threshold": float(os.environ.get("DISTLLM_CARBON_THRESHOLD", "400.0")),
                "check_interval_s": 300.0,
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
        if autoscaler is not None and self._batch_scheduler:
            try:
                from distllm.core.intelligent_autoscaler import ScalingMetrics
                s = self._batch_scheduler.stats()
                autoscaler.record_metrics(ScalingMetrics(
                    active_requests=s.get("active_requests", 0),
                    pending_requests=s.get("pending_requests", 0),
                    current_nodes=len(getattr(self, 'nodes', {})),
                ))
            except Exception:
                pass

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
                nid: {"host": n.host, "port": n.port, "healthy": n.healthy}
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
