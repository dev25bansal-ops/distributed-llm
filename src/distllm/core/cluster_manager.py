"""Node lifecycle and cluster management."""

from __future__ import annotations

from typing import Any

from loguru import logger
from transformers import AutoTokenizer

from distllm.config.settings import NodeRole
from distllm.dist.node_registrar import NodeRegistrar
from distllm.dist.pipeline import PipelineOrchestrator
from distllm.models.partitioner import get_model_info


class ClusterManager:
    """Manages worker node lifecycle: registration, topology, weight distribution.

    Delegates to ``NodeRegistrar`` and ``PipelineOrchestrator`` for the
    low-level node management.
    """

    def __init__(
        self,
        pipeline: PipelineOrchestrator,
        model_name: str,
        trust_remote_code: bool | None = None,
        cluster_key: str | None = None,
    ):
        self._pipeline = pipeline
        self._node_registrar = NodeRegistrar(
            pipeline=pipeline,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._cluster_key = cluster_key
        self._model_registry: dict[str, dict] = {}
        self._distribute_weights: bool = True

        self.tokenizer: AutoTokenizer | None = None
        self.model_info: dict | None = None
        self.total_layers = 0
        self.model_revision = "main"

    @property
    def nodes(self) -> dict:
        return self._pipeline.nodes

    @nodes.setter
    def nodes(self, value: dict):
        self._pipeline.nodes = value

    @property
    def node_order(self) -> list[str]:
        return self._pipeline.node_order

    @node_order.setter
    def node_order(self, value: list[str]):
        self._pipeline.node_order = value

    def auto_setup(self, nodes_config: list[dict]) -> None:
        model_info, total_layers = self._node_registrar.auto_setup(nodes_config)
        self.model_info = model_info
        self.total_layers = total_layers
        self._pipeline.total_layers = total_layers
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=self._trust_remote_code,
            revision=self.model_revision,
        )
        logger.info(f"Auto-setup complete: {len(nodes_config)} nodes, {total_layers} layers")

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
        weight_source = None
        if self._distribute_weights:
            source = self._get_weight_source(self._model_name, start_layer, end_layer)
            if source is not None:
                weight_source = f"{source[0]}:{source[1]}"
                logger.info(f"Auto-assigned weight source for {node_id}: {weight_source}")

        self._node_registrar.manual_register(
            node_id, host, port, start_layer, end_layer,
            total_layers=total_layers, role=role,
            expert_ids=expert_ids, cluster_id=cluster_id,
            cluster_key=cluster_key or self._cluster_key,
            weight_source=weight_source,
        )
        if total_layers:
            self._pipeline.total_layers = total_layers
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=self._trust_remote_code,
                revision=self.model_revision,
            )
        if self.model_info is None:
            self.model_info = get_model_info(self._model_name, self._trust_remote_code)
            if total_layers is None:
                self.total_layers = self.model_info["num_layers"]
                self._pipeline.total_layers = self.total_layers

        if self._distribute_weights:
            self._register_weight_source(node_id, self._model_name, start_layer, end_layer)

    def _register_weight_source(self, node_id, model_name, start_layer, end_layer):
        key = f"{model_name}:{start_layer}:{end_layer}"
        node = self._pipeline.nodes.get(node_id)
        if isinstance(node, dict):
            host = node.get("host", "unknown")
            port_b = node.get("port", 0)
        else:
            host = getattr(node, "host", "unknown") if node else "unknown"
            port_b = getattr(node, "port", 0) if node else 0
        self._model_registry[key] = {
            "node_id": node_id, "host": host, "port": port_b,
            "start_layer": start_layer, "end_layer": end_layer,
        }
        logger.info(f"Registered weight source {node_id} for {key}")

    def _get_weight_source(self, model_name, start_layer, end_layer):
        key = f"{model_name}:{start_layer}:{end_layer}"
        entry = self._model_registry.get(key)
        if entry is None:
            return None
        return (entry["host"], entry["port"])

    def register_nodes_batch(
        self,
        nodes_config: list[dict[str, Any]],
        cluster_key: str | None = None,
        max_workers: int = 8,
    ) -> dict[str, dict]:
        """Register multiple nodes concurrently with batched Profile RPCs.

        Args:
            nodes_config: List of node config dicts.
            cluster_key: Optional shared secret.
            max_workers: Max concurrent registration threads.

        Returns:
            Dict of node_id -> registration result.
        """
        return self._node_registrar.register_nodes_batch(
            nodes_config,
            cluster_key=cluster_key or self._cluster_key,
            max_workers=max_workers,
        )

    def scale_pipeline_capacity(self, per_node_limit: int = 16) -> int:
        """Scale the pipeline's concurrent request limit based on node count.

        Args:
            per_node_limit: Max concurrent requests per node.

        Returns:
            The new max concurrent request limit.
        """
        return self._pipeline.scale_concurrent_requests(per_node_limit)

    @property
    def node_count(self) -> int:
        """Return the number of registered nodes."""
        return len(self._pipeline.nodes)

    def get_node_gpu_summary(self) -> dict[str, dict]:
        """Return GPU summary for all registered nodes.

        Returns:
            Dict of node_id -> {gpu_name, memory_total_gb, memory_free_gb}.
        """
        summary = {}
        for nid, node in self._pipeline.nodes.items():
            summary[nid] = {
                "gpu_name": getattr(node, "gpu_name", ""),
                "memory_total_gb": getattr(node, "gpu_memory_total", 0) / (1024 ** 3)
                if getattr(node, "gpu_memory_total", 0) else 0.0,
                "memory_free_gb": getattr(node, "gpu_memory_free", 0) / (1024 ** 3)
                if getattr(node, "gpu_memory_free", 0) else 0.0,
            }
        return summary
