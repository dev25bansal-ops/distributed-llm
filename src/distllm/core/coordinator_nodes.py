"""Node registrar for the Coordinator facade.

Handles node registration, auto-setup, and expert registration.
Extracted from the Coordinator class.
"""


from loguru import logger
from transformers import AutoTokenizer

from distllm.models.partitioner import get_model_info
from distllm.models.partitioner import partition_model_across_nodes
from distllm.config.loader import NodeRole


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

        Args:
            nodes_config: List of node configuration dicts with host, port, node_id.
        """
        logger.info(f"Auto-setup: partitioning {self.model_name} across {len(nodes_config)} nodes")

        model_info = get_model_info(self.model_name, self.trust_remote_code)
        total_layers = model_info["num_layers"]
        self.pipeline.total_layers = total_layers
        logger.info(f"Model has {total_layers} layers")

        assignments = partition_model_across_nodes(
            self.model_name, len(nodes_config), self.trust_remote_code
        )

        for i, config in enumerate(nodes_config):
            start, end = assignments[i]
            node_id = config.get("node_id", f"node_{i}")

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
    ) -> None:
        """Manually register a node.

        Args:
            node_id: Unique node identifier.
            host: Node hostname.
            port: Node gRPC port.
            start_layer: First layer index.
            end_layer: Last layer index.
            total_layers: Optional total layers in model.
            role: Node role (AUTO, PREFILL, DECODE).
            expert_ids: Expert IDs for MoE.
            cluster_id: Cluster ID for federation.
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
        )

        # Register with federation manager
        if self.federation_manager is not None:
            self.federation_manager.register_node(node_id, cluster_id)

        # Register experts on this node if MoE is enabled
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
