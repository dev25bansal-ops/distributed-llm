"""Coordinator creation and gRPC node wrapper for the distributed LLM API server."""
from __future__ import annotations

from typing import Any

from distllm.core.coordinator import Coordinator
from distllm.core.monitor import SystemMonitor
from distllm.config.settings import DistLLMSettings
from distllm.api.server_state import state
from loguru import logger

__all__ = [
    "create_coordinator",
    "_CoordinatorNode",
]


def create_coordinator(
    model_name: str,
    dtype: str = "float16",
    local: bool = False,
    max_batch_size: int = 1,
    max_tokens_per_batch: int = 4096,
    settings: DistLLMSettings | None = None,
) -> Coordinator:
    """Create and configure the coordinator."""
    if settings:
        max_batch_size = settings.batching.max_batch_size
        max_tokens_per_batch = settings.batching.max_tokens_per_batch

    try:
        from distllm.dist.config import WideAreaConfig
    except ImportError:
        WideAreaConfig = None

    wide_area_config = None
    if settings and settings.wide_area.enabled:
        wa = settings.wide_area
        wide_area_config = WideAreaConfig(
            enabled=wa.enabled,
            p2p_forwarding=wa.p2p_forwarding,
            tokens_before_forward=wa.tokens_before_forward,
            wan_timeout_seconds=wa.wan_timeout_seconds,
            max_retries=wa.max_retries,
            backoff_base_seconds=wa.backoff_base_seconds,
        )

    coord = Coordinator(
        model_name=model_name,
        dtype=dtype,
        max_batch_size=max_batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        metrics_exporter=state.metrics_exporter,
        wide_area_config=wide_area_config,
        plugin_system=getattr(state, "plugin_system", None),
    )

    if local:
        coord.load_local_model()
        logger.info(f"Coordinator loaded model locally: {model_name}")
    else:
        logger.info(f"Coordinator ready for distributed mode: {model_name}")

    # Start gRPC server for worker connections
    # Create a minimal node-like object for the gRPC server
    coord_port = 50050  # Default coordinator gRPC port
    try:
        from distllm.dist.node_service import NodeServer

        # Determine TLS config from settings
        coord_tls = False
        coord_cert_file: str | None = None
        coord_key_file: str | None = None
        coord_ca_cert: str | None = None
        if settings and settings.tls.enabled:
            coord_tls = True
            coord_cert_file = settings.tls.cert_file
            coord_key_file = settings.tls.key_file
            coord_ca_cert = settings.tls.ca_cert_file

        # Create a wrapper that provides the interface NodeServer expects
        class _CoordinatorNode:
            def __init__(self, coordinator: Coordinator) -> None:
                self._coord = coordinator
                self.node_id = "coordinator"
                self.host = "0.0.0.0"
                self.port = coord_port
                self.start_layer = 0
                self.end_layer = 0
                self.total_layers = 0
                self.healthy = True
                self.partitioner = None

            def forward_fn(self, **kwargs: Any) -> Any:
                return self._coord.generate(**kwargs)

            def health_check(self) -> bool:
                return True

        coord._node_wrapper = _CoordinatorNode(coord)
        coord._node_server = NodeServer(
            coord._node_wrapper,
            port=coord_port,
            max_workers=4,
            cluster_key=getattr(coord.config, "cluster_key", None),
        )
        coord._node_server.start(
            use_tls=coord_tls,
            cert_file=coord_cert_file,
            key_file=coord_key_file,
            ca_cert=coord_ca_cert,
        )
        if coord_tls:
            logger.info(f"Coordinator gRPC server started on port {coord_port} with TLS for worker connections")
        else:
            logger.info(f"Coordinator gRPC server started on port {coord_port} for worker connections (no TLS)")
    except Exception as e:
        logger.warning(f"Could not start gRPC server on port {coord_port}: {e}")
        logger.warning("Workers will not be able to connect. Run 'system coordinator' separately.")

    monitor_inst = SystemMonitor()
    state.coordinator = coord
    state.monitor = monitor_inst

    return coord
