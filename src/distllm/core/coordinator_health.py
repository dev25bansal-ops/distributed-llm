"""Health checker for the Coordinator facade.

Handles health check orchestration and metrics export.
Extracted from the Coordinator class.
"""

from typing import Any, Dict


class HealthChecker:
    """Orchestrates health checks across nodes and exports metrics.

    Attributes:
        resource_mgr: ResourceManager for node health checks.
        metrics_exporter: Optional Prometheus metrics exporter.
    """

    def __init__(self, resource_mgr, metrics_exporter=None):
        self.resource_mgr = resource_mgr
        self.metrics_exporter = metrics_exporter

    def check_all(
        self,
        nodes: Dict[str, Any],
        node_order: list,
        check_circuit_breaker,
    ) -> dict:
        """Check health of all nodes, update metrics.

        Args:
            nodes: Dict of node_id -> NodeRegistration.
            node_order: Ordered list of node IDs.
            check_circuit_breaker: Callable to check circuit breaker state.

        Returns:
            Dict of node_id -> health status.
        """
        results = self.resource_mgr.health_check_all(nodes)

        for node_id, result in results.items():
            if self.metrics_exporter is not None:
                node = nodes[node_id]
                layer_range = f"{node.start_layer}-{node.end_layer}"
                self.metrics_exporter.node_health.labels(node_id, layer_range).set(
                    1 if result.get("healthy") else 0
                )
                if "memory_used" in result:
                    self.metrics_exporter.node_gpu_memory_bytes.labels(node_id).set(
                        result["memory_used"]
                    )

        if self.metrics_exporter is not None:
            for node_id in node_order:
                is_open = check_circuit_breaker(node_id)
                self.metrics_exporter.circuit_breaker_state.labels(target_node=node_id).set(
                    1 if is_open else 0
                )

        return results

    async def check_all_async(
        self,
        nodes: Dict[str, Any],
        node_order: list,
        check_circuit_breaker,
    ) -> dict:
        """Check health of all nodes (async), update metrics.

        Args:
            nodes: Dict of node_id -> NodeRegistration.
            node_order: Ordered list of node IDs.
            check_circuit_breaker: Callable to check circuit breaker state.

        Returns:
            Dict of node_id -> health status.
        """
        results = await self.resource_mgr.health_check_all_async(nodes)

        for node_id, result in results.items():
            if self.metrics_exporter is not None:
                node = nodes[node_id]
                layer_range = f"{node.start_layer}-{node.end_layer}"
                self.metrics_exporter.node_health.labels(node_id, layer_range).set(
                    1 if result.get("healthy") else 0
                )
                if "memory_used" in result:
                    self.metrics_exporter.node_gpu_memory_bytes.labels(node_id).set(
                        result["memory_used"]
                    )

        if self.metrics_exporter is not None:
            for node_id in node_order:
                is_open = check_circuit_breaker(node_id)
                self.metrics_exporter.circuit_breaker_state.labels(target_node=node_id).set(
                    1 if is_open else 0
                )

        return results
