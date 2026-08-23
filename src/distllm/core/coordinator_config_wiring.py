"""CoordinatorConfigurator — wires optional subsystems into the Coordinator.

Extracted from ``coordinator.py`` to reduce class size.  Pure code move with
no logic changes.  Each method operates on the coordinator instance stored
as ``self.coordinator``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from distllm.core.coordinator import Coordinator

from distllm.config.settings import NodeRole
from distllm.core.model_router import ModelRouter


class CoordinatorConfigurator:
    """Wires optional subsystems into a Coordinator instance.

    The configurator stores a reference to the coordinator and configures
    its subsystems by setting attributes on the coordinator directly.
    """

    def __init__(self, coordinator: Coordinator) -> None:
        self.coordinator = coordinator

    # ── Recovery callbacks wiring ──

    def _wire_recovery_callbacks(self) -> None:
        """Wire NodeRecoveryManager callbacks so node failure recovery works.

        Previously these callbacks were never set, making the recovery
        manager's redistribution and sequence recovery steps no-ops.
        """
        coord = self.coordinator
        coord._recovery_manager.set_drain_callback(coord._on_node_drain)
        coord._recovery_manager.set_mark_dead_callback(coord._on_node_mark_dead)
        coord._recovery_manager.set_redistribute_layers_callback(coord._on_node_redistribute)
        coord._recovery_manager.set_recover_sequences_callback(coord._on_node_recover)

    # ── Subsystem registry ──

    def _register_subsystems(self) -> None:
        """Register all subsystems with the SubsystemRegistry for lifecycle management."""
        coord = self.coordinator
        reg = coord._subsystem_registry
        reg.register("pipeline", coord._pipeline, start_fn=coord._pipeline.start if hasattr(coord._pipeline, 'start') else None)  # noqa: E501
        reg.register("batch_scheduler", coord._batch_scheduler)
        reg.register("health_manager", coord._health_mgr)
        reg.register("metrics_collector", coord._metrics_collector)
        reg.register("straggler_detector", coord._straggler_detector)
        reg.register("latency_tracker", coord._latency_tracker)
        reg.register("reputation", coord._reputation)

    # ── Adaptive batching ──

    def _init_adaptive_batching(self) -> None:
        """Connect the adaptive batching engine if the module is available."""
        try:
            from distllm.core.adaptive_batching import AdaptiveBatchingEngine

            engine = AdaptiveBatchingEngine()
            self.coordinator._batch_scheduler.set_adaptive_engine(engine)
            logger.debug("Adaptive batching engine initialized")
        except ImportError:
            logger.debug("Adaptive batching engine not available")

    # ── Model router ──

    def init_model_router(self, settings: Any | None = None) -> ModelRouter:
        """Initialize the model router from ChatRouterSettings.

        Args:
            settings: A ChatRouterSettings instance. If None, router is
                created with the coordinator's default model as fallback.

        Returns:
            The initialized ModelRouter instance.
        """
        coord = self.coordinator
        from distllm.config.settings import ChatRouterSettings

        if settings is None:
            settings = ChatRouterSettings(
                enabled=True,
                default_model=coord.model_name,
            )
        coord._model_router = ModelRouter(settings)
        # Register the router name from settings as a hybrid name
        router_name = getattr(settings, "name", "")
        if router_name:
            coord._model_router.register_hybrid_name(router_name)
        logger.info(
            f"Model router initialized: default={coord._model_router._default_model}, "  # noqa: E501
            f"rules={len(coord._model_router._rules)}, "
            f"hybrid_names={coord._model_router.list_hybrid_models()}"
        )
        return coord._model_router

    # ── Hot-swap model management ──

    def init_hot_swap(self, total_gpu_memory_gb: float = 0.0, max_models: int = 4) -> None:
        """Initialize the hot-swap model manager for dynamic model loading."""
        coord = self.coordinator
        from distllm.core.multi_model_serving import ModelHotSwapManager

        coord._hot_swap_mgr = ModelHotSwapManager(
            total_gpu_memory_gb=total_gpu_memory_gb,
            max_models=max_models,
            on_load_model=coord._load_model_callback,
            on_unload_model=coord._unload_model_callback,
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
        coord = self.coordinator
        from distllm.core.adaptive_compression import (
            AdaptiveCompressionConfig,
            AdaptiveCompressionManager,
            SimpleCompressor,
        )

        if settings is None:
            coord._adaptive_compression_mgr = None
            return

        config = AdaptiveCompressionConfig(
            enabled=settings.enabled,
            idle_threshold_pct=settings.idle_threshold_pct,
            idle_duration_s=settings.idle_duration_s,
            check_interval_s=settings.check_interval_s,
            compression_method=settings.compression_method,
            calibration_samples=settings.calibration_samples,
            output_dir=settings.output_dir,
            trust_remote_code=getattr(coord, 'trust_remote_code', False),
        )

        if utilization_fn is None:
            utilization_fn = coord._default_utilization_fn

        compressor = SimpleCompressor(
            output_base=settings.output_dir,
            method=settings.compression_method,
            calibration_samples=settings.calibration_samples,
            trust_remote_code=getattr(coord, 'trust_remote_code', False),
        )

        coord._adaptive_compression_mgr = AdaptiveCompressionManager(
            config=config,
            utilization_fn=utilization_fn,
            hot_swap_mgr=getattr(coord, "_hot_swap_mgr", None),
            compressor=compressor,
        )

    # ── Memory defragmentation ──

    def init_defragmentation(self, settings: Any | None = None) -> None:
        """Initialize the GPU memory defragmenter.

        Args:
            settings: A DefragmentationSettings instance or None to disable.
        """
        coord = self.coordinator
        from distllm.core.memory_defragmenter import (
            DefragConfig,
            DefragPolicy,
            MemoryDefragmenter,
        )

        if settings is None or not settings.enabled:
            coord._defragmenter = None
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
        coord._defragmenter = MemoryDefragmenter(
            config=config,
            metrics_collector=coord._metrics_collector,
        )
        logger.info(
            f"Defragmenter initialized: policy={settings.policy}, "
            f"threshold={threshold:.0%}, interval={settings.interval_seconds}s"
        )

    # ── Graceful degradation ──

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

        self.coordinator._graceful_degradation = GracefulDegradation(
            enabled=enabled,
            light_threshold=light_threshold,
            moderate_threshold=moderate_threshold,
            severe_threshold=severe_threshold,
            critical_threshold=critical_threshold,
            fallback_model=fallback_model,
        )
        logger.info(
            f"Graceful degradation initialized: "
            f"thresholds=[{light_threshold}, {moderate_threshold}, "
            f"{severe_threshold}, {critical_threshold}]"
        )

    # ── Node lifecycle ──

    def auto_setup(self, nodes_config: list[dict]) -> tuple[dict, int] | None:
        """Auto-discover and register nodes from a config list.

        Args:
            nodes_config: List of node configuration dicts.

        Returns:
            Tuple of (model_info, total_layers) or None.
        """
        coord = self.coordinator
        coord._cluster_mgr.model_revision = coord.model_revision
        result = coord._cluster_mgr.auto_setup(nodes_config)
        coord.tokenizer = coord._cluster_mgr.tokenizer
        coord._inference_engine.tokenizer = coord.tokenizer
        return result

    def manual_register(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        total_layers: int | None = None,
        role: NodeRole = NodeRole.AUTO,
        expert_ids: list[int] | None = None,
        cluster_id: str = "default",
        cluster_key: str | None = None,
    ) -> None:
        """Manually register a node with explicit layer range.

        Args:
            node_id: Unique node identifier.
            host: Node hostname or IP.
            port: Node gRPC port.
            start_layer: First layer index assigned to this node.
            end_layer: Last layer index assigned to this node.
            total_layers: Total layers in the model (optional).
            role: Node role for MoE configurations.
            expert_ids: Expert IDs (only relevant for MoE).
            cluster_id: Cluster this node belongs to.
            cluster_key: Authentication key for cluster.
        """
        coord = self.coordinator
        coord._cluster_mgr.manual_register(
            node_id, host, port, start_layer, end_layer,
            total_layers=total_layers, role=role,
            expert_ids=expert_ids, cluster_id=cluster_id,
            cluster_key=cluster_key,
        )
        coord.tokenizer = coord._cluster_mgr.tokenizer
