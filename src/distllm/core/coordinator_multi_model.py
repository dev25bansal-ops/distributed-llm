"""Multi-model manager for the Coordinator facade.

Handles multi-model serving, model registry, and MoE forward pass.
Extracted from the Coordinator class.
"""


import torch

from distllm.core.model_registry import ModelEntry, ModelRegistry


class MultiModelManager:
    """Manages multi-model serving and MoE orchestration.

    Attributes:
        model_name: Default model name.
        model_registry: Optional model registry for multi-model.
        moe_orchestrator: Optional MoE orchestrator.
        pipeline: PipelineOrchestrator for node access.
    """

    def __init__(
        self,
        model_name: str,
        model_registry: ModelRegistry | None = None,
        moe_orchestrator=None,
        pipeline=None,
    ):
        self.model_name = model_name
        self.model_registry = model_registry
        self.moe_orchestrator = moe_orchestrator
        self.pipeline = pipeline

    def register_model(self, name: str, path: str, total_layers: int) -> ModelEntry:
        """Register an additional model.

        Args:
            name: Model name/alias.
            path: Model path.
            total_layers: Number of layers in the model.

        Returns:
            ModelEntry for the registered model.
        """
        if self.model_registry is None:
            self.model_registry = ModelRegistry()
        return self.model_registry.register(name, path, total_layers)

    def list_models(self) -> list[str]:
        """List all registered model names.

        Returns:
            List of model name strings.
        """
        if self.model_registry is None:
            return [self.model_name]
        return [m.name for m in self.model_registry.list_models()]

    def get_model_name(self, requested: str | None = None) -> str:
        """Resolve model name: requested > registry default > self.model_name.

        Args:
            requested: Optional requested model name.

        Returns:
            Resolved model name.
        """
        if requested and self.model_registry and self.model_registry.is_registered(requested):
            return requested
        if self.model_registry and self.model_registry.default_model:
            return self.model_registry.default_model
        return self.model_name

    def moe_forward(
        self, hidden_states: torch.Tensor, moe_router
    ) -> torch.Tensor:
        """Execute MoE forward pass via distributed expert orchestration.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].
            moe_router: MoERouter instance.

        Returns:
            Aggregated expert output tensor.

        Raises:
            RuntimeError: If MoE orchestrator not initialized.
        """
        if self.moe_orchestrator is None:
            raise RuntimeError("MoE orchestrator not initialized")
        # Build node_clients from registered nodes
        node_clients = {}
        for node_id in self.pipeline.nodes:
            node_clients[node_id] = self.pipeline.nodes[node_id]
        return self.moe_orchestrator.forward(
            hidden_states, moe_router, node_clients
        )
