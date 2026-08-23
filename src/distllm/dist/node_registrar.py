"""Node registrar for distributed inference.

Handles node registration, auto-setup, and expert registration.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from loguru import logger

from distllm.models.partitioner import get_model_info, partition_model_gpu_aware
from distllm.models.partitioner import partition_model_across_nodes
from distllm.config.settings import NodeRole


class NodeRegistrar:
    """Manages node registration and layer assignment.

    Attributes:
        pipeline: PipelineOrchestrator for node registration.
        model_name: Name of the model to partition.
        trust_remote_code: Whether to trust remote code.
        expert_registry: Optional expert registry for MoE.
        federation_manager: Optional federation manager for cross-cluster.
    """

    def __init__(
        self,
        pipeline,
        model_name: str,
        trust_remote_code: bool | None = None,
        expert_registry=None,
        federation_manager=None,
    ):
        self.pipeline = pipeline
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.expert_registry = expert_registry
        self.federation_manager = federation_manager

    def auto_setup(self, nodes_config: list[dict]) -> None:
        """Automatically partition model and assign layers to nodes.

        Uses GPU-aware partitioning when nodes provide GPU info (``gpus`` key),
        otherwise falls back to equal split.

        Args:
            nodes_config: List of node configuration dicts with host, port, node_id
                and optional ``gpus`` key (list of GPUProfile-like objects).
        """
        logger.info(f"Auto-setup: partitioning {self.model_name} across {len(nodes_config)} nodes")

        model_info = get_model_info(self.model_name, self.trust_remote_code)
        total_layers = model_info["num_layers"]
        self.pipeline.total_layers = total_layers
        logger.info(f"Model has {total_layers} layers")

        # Build GPU info dict for GPU-aware partitioning
        node_gpus: dict[str, list] = {}
        has_gpu_info = False
        for config in nodes_config:
            gpus = config.get("gpus")
            if gpus:
                node_id = config.get("node_id", f"node_{len(node_gpus)}")
                node_gpus[node_id] = gpus if isinstance(gpus, list) else [gpus]
                has_gpu_info = True

        if has_gpu_info:
            assignments = partition_model_gpu_aware(
                node_gpus, self.model_name, total_layers, self.trust_remote_code,
            )
        else:
            assignments_raw = partition_model_across_nodes(
                self.model_name, len(nodes_config), self.trust_remote_code,
            )
            assignments = {
                nodes_config[i].get("node_id", f"node_{i}"): assignments_raw[i]
                for i in range(len(nodes_config))
            }

        for i, config in enumerate(nodes_config):
            node_id = config.get("node_id", f"node_{i}")
            start, end = assignments[node_id]

            self.pipeline.register_node(
                node_id=node_id,
                host=config.get("host", "localhost"),
                port=config.get("port", 50051 + i),
                start_layer=start,
                end_layer=end,
            )

            logger.info(f"Assigned {node_id}: layers {start}-{end}")

        return model_info, total_layers

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
        weight_source: str | None = None,
    ) -> None:
        """Manually register a node.

        Args:
            node_id: Unique node identifier.
            host: Node hostname.
            port: Node port.
            start_layer: First layer index.
            end_layer: Last layer index.
            total_layers: Optional total layers in model.
            role: Node role (AUTO, PREFILL, DECODE).
            expert_ids: Expert IDs for MoE.
            cluster_id: Cluster ID for federation.
            cluster_key: Optional shared secret for node authentication.
            weight_source: Optional ``host:port`` of a peer to pull weights from.
        """
        if total_layers:
            self.pipeline.total_layers = total_layers

        self.pipeline.register_node(
            node_id=node_id,
            host=host,
            port=port,
            start_layer=start_layer,
            end_layer=end_layer,
            role=role,
            expert_ids=expert_ids,
            cluster_id=cluster_id,
            cluster_key=cluster_key,
            weight_source=weight_source,
        )

        if self.federation_manager is not None:
            self.federation_manager.register_node(node_id, cluster_id)

        if expert_ids and self.expert_registry is not None:
            for eid in expert_ids:
                self.expert_registry.register_expert(eid, node_id)

        logger.info(f"Registered {node_id}: layers {start_layer}-{end_layer}")

    def register_expert_on_node(
        self, node_id: str, expert_ids: list[int], layer_idx: int = 0
    ) -> None:
        """Register experts on a node in the expert registry.

        Args:
            node_id: Node identifier.
            expert_ids: List of expert IDs hosted by this node.
            layer_idx: Layer index the experts belong to.
        """
        if self.expert_registry is None:
            return
        for eid in expert_ids:
            self.expert_registry.register_expert(eid, node_id, layer_idx)
        logger.info(f"Registered experts {expert_ids} on {node_id}")

    def register_nodes_batch(
        self,
        nodes_config: list[dict[str, Any]],
        cluster_key: str | None = None,
        timeout_s: float = 10.0,
        max_workers: int = 8,
    ) -> dict[str, dict]:
        """Register multiple nodes concurrently, batching Profile RPCs.

        Each node is registered (added to the pipeline topology) and then
        its gRPC client is initialized in parallel.  Profile RPCs run
        concurrently across nodes, reducing total registration time from
        O(N × RTT) to O(max(RTT)).

        Args:
            nodes_config: List of node config dicts with keys:
                node_id, host, port, start_layer, end_layer, and optional
                role, expert_ids, cluster_id, weight_source.
            cluster_key: Optional shared secret for node authentication.
            timeout_s: Per-node connection timeout.
            max_workers: Max concurrent registration threads.

        Returns:
            Dict of node_id -> registration result dict with keys:
                success (bool), gpu_name (str), gpu_memory_gb (float),
                error (str or None).
        """
        results: dict[str, dict] = {}

        def _register_one(config: dict) -> tuple[str, dict]:
            node_id = config.get("node_id", f"node_{config.get('port', 0)}")
            try:
                self.manual_register(
                    node_id=node_id,
                    host=config.get("host", "localhost"),
                    port=config.get("port", 50051),
                    start_layer=config.get("start_layer", 0),
                    end_layer=config.get("end_layer", 0),
                    total_layers=config.get("total_layers"),
                    role=config.get("role", None) or __import__(
                        "distllm.config.settings", fromlist=["NodeRole"]
                    ).NodeRole.AUTO,
                    expert_ids=config.get("expert_ids"),
                    cluster_id=config.get("cluster_id", "default"),
                    cluster_key=cluster_key or config.get("cluster_key"),
                    weight_source=config.get("weight_source"),
                )
                node = self.pipeline.nodes.get(node_id)
                return node_id, {
                    "success": True,
                    "gpu_name": getattr(node, "gpu_name", ""),
                    "gpu_memory_gb": getattr(node, "gpu_memory_total", 0) / (1024 ** 3)
                    if getattr(node, "gpu_memory_total", 0) else 0.0,
                    "error": None,
                }
            except Exception as e:
                logger.error(f"Batch registration failed for {node_id}: {e}")
                return node_id, {
                    "success": False,
                    "gpu_name": "",
                    "gpu_memory_gb": 0.0,
                    "error": str(e),
                }

        workers = min(max_workers, len(nodes_config)) if nodes_config else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_register_one, cfg): cfg
                for cfg in nodes_config
            }
            for future in concurrent.futures.as_completed(futures):
                node_id, result = future.result()
                results[node_id] = result

        succeeded = sum(1 for r in results.values() if r["success"])
        logger.info(
            f"Batch registration complete: {succeeded}/{len(nodes_config)} nodes registered"
        )
        return results
