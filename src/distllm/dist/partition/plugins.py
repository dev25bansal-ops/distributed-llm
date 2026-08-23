"""Plugin API for custom cost models.

Allows users to register custom cost models (for NPUs, TPUs, custom
accelerators, or any hardware) without modifying the core optimizer.

Typical usage::

    class MyNPUCostModel(CostModelPlugin):
        def estimate_node_cost(self, node_id, layers, batch_size, seq_len) -> NodeCost:
            # Custom cost estimation for NPU
            ...
        def supported_devices(self) -> list[str]:
            return ["npu", "my_accelerator"]

    # Register the plugin
    registry = CostModelRegistry()
    registry.register(MyNPUCostModel())

    # Use in optimizer
    cost_model = registry.get_cost_model("npu")
    if cost_model:
        cost = cost_model.estimate_node_cost("npu-0", layers, 1, 4096)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from distllm.dist.partition.cost_model import NodeCost
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import TopologyGraph


class CostModelPlugin(ABC):
    """Abstract base class for custom cost model plugins.

    Subclass this and implement `estimate_node_cost()` and
    `supported_devices()` to add support for custom hardware.
    """

    @abstractmethod
    def estimate_node_cost(
        self,
        node_id: str,
        layers: list[LayerWeights],
        batch_size: int,
        seq_len: int,
        gpu_profile: GPUProfile | None = None,
        topology: TopologyGraph | None = None,
        prev_node_id: str | None = None,
    ) -> NodeCost | None:
        """Estimate the cost of running layers on a node.

        Args:
            node_id: Target node identifier.
            layers: Layer weights to process.
            batch_size: Batch size.
            seq_len: Sequence length.
            gpu_profile: GPU profile for the node (if available).
            topology: Cluster topology (for communication cost).
            prev_node_id: Previous node in pipeline (for comm cost).

        Returns:
            NodeCost estimate, or None if this plugin can't handle it.
        """
        ...

    @abstractmethod
    def supported_devices(self) -> list[str]:
        """Return list of device types this plugin supports.

        Device types are matched against GPUProfile.name or
        custom device identifiers.
        """
        ...

    def name(self) -> str:
        """Return the plugin name (defaults to class name)."""
        return self.__class__.__name__

    def priority(self) -> int:
        """Return plugin priority (higher = checked first).

        Default is 0. Built-in models use -10 so custom plugins
        can override them.
        """
        return 0

    def can_handle(self, node_id: str, gpu_profile: GPUProfile | None) -> bool:
        """Check if this plugin can handle the given node.

        Override for custom matching logic.
        """
        if gpu_profile is None:
            return False
        device_name = gpu_profile.name.lower()
        return any(d in device_name for d in self.supported_devices())


class CostModelRegistry:
    """Registry for cost model plugins.

    Plugins are checked in priority order. The first plugin that
    returns a non-None result for a given node is used.
    """

    def __init__(self) -> None:
        self._plugins: list[CostModelPlugin] = []
        self._by_name: dict[str, CostModelPlugin] = {}

    def register(self, plugin: CostModelPlugin) -> None:
        """Register a cost model plugin.

        Args:
            plugin: The plugin to register.
        """
        self._plugins.append(plugin)
        self._by_name[plugin.name()] = plugin
        self._plugins.sort(key=lambda p: p.priority(), reverse=True)
        logger.info(f"Registered cost model plugin: {plugin.name()} (priority={plugin.priority()})")

    def unregister(self, name: str) -> bool:
        """Unregister a plugin by name.

        Returns True if found and removed.
        """
        plugin = self._by_name.pop(name, None)
        if plugin:
            self._plugins.remove(plugin)
            return True
        return False

    def get(self, name: str) -> CostModelPlugin | None:
        """Get a plugin by name."""
        return self._by_name.get(name)

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all registered plugins."""
        return [
            {
                "name": p.name(),
                "priority": p.priority(),
                "devices": p.supported_devices(),
            }
            for p in self._plugins
        ]

    def estimate_cost(
        self,
        node_id: str,
        layers: list[LayerWeights],
        batch_size: int,
        seq_len: int,
        gpu_profile: GPUProfile | None = None,
        topology: TopologyGraph | None = None,
        prev_node_id: str | None = None,
    ) -> NodeCost | None:
        """Try all registered plugins to estimate cost.

        Returns the first non-None result, or None if no plugin
        can handle the request.
        """
        for plugin in self._plugins:
            if plugin.can_handle(node_id, gpu_profile):
                try:
                    result = plugin.estimate_node_cost(
                        node_id, layers, batch_size, seq_len,
                        gpu_profile, topology, prev_node_id,
                    )
                    if result is not None:
                        return result
                except Exception as e:
                    logger.warning(f"Plugin {plugin.name()} failed for {node_id}: {e}")
        return None


class PluginAwareCostModel:
    """Cost model that delegates to plugins when available.

    Wraps the standard PartitionCostModel and falls back to it
    when no plugin handles a given node.

    Args:
        base_cost_model: The standard analytical cost model.
        registry: Plugin registry with custom cost models.
    """

    def __init__(
        self,
        base_cost_model: Any,
        registry: CostModelRegistry,
    ):
        self._base = base_cost_model
        self._registry = registry

    def evaluate(
        self,
        node_id: str,
        start_layer_id: int,
        end_layer_id: int,
        batch_size: int = 1,
        seq_len: int = 4096,
        prev_node_id: str | None = None,
    ) -> NodeCost:
        """Evaluate cost using plugins first, then base model."""
        layers = self._base._layer_weights[start_layer_id:end_layer_id]
        gpu = self._base._gpu_profiles.get(node_id)

        # Try plugins first
        plugin_cost = self._registry.estimate_cost(
            node_id, layers, batch_size, seq_len,
            gpu_profile=gpu, topology=self._base._topology,
            prev_node_id=prev_node_id,
        )
        if plugin_cost is not None:
            return plugin_cost

        # Fall back to base model
        return self._base.evaluate(
            node_id, start_layer_id, end_layer_id,
            batch_size, seq_len, prev_node_id,
        )

    def evaluate_partition(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> list[NodeCost]:
        costs: list[NodeCost] = []
        for idx, (node_id, start, end) in enumerate(partition):
            prev = partition[idx - 1][0] if idx > 0 else None
            cost = self.evaluate(node_id, start, end, batch_size, seq_len, prev_node_id=prev)
            costs.append(cost)
        return costs

    def combined_throughput(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> float:
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        if not costs:
            return 0.0
        bottleneck = max(c.total_time_ms for c in costs)
        if bottleneck <= 0:
            return 0.0
        return (batch_size * seq_len) / (bottleneck / 1000.0)

    def max_latency(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> float:
        costs = self.evaluate_partition(partition, batch_size, seq_len)
        return max(c.total_time_ms for c in costs) if costs else 0.0

    def pipeline_latency(
        self,
        partition: list[tuple[str, int, int]],
        batch_size: int = 1,
        seq_len: int = 4096,
        num_pipeline_stages: int | None = None,
    ) -> float:
        return self._base.pipeline_latency(partition, batch_size, seq_len, num_pipeline_stages)
