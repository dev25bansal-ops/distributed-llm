"""NodeSystem: node lifecycle, health, registration.

Groups: ResourceManager, HealthChecker, NodeRegistrar
"""

from typing import Any



class NodeSystem:
    """Manages node lifecycle: registration, health, topology.

    Composes ResourceManager, HealthChecker, and NodeRegistrar
    into a single interface for node management.
    """

    def __init__(
        self,
        resource_mgr: Any = None,
        total_layers: int = 0,
        model_name: str = "",
    ):
        from distllm.core.resource_manager import ResourceManager
        from distllm.core.coordinator_health import HealthChecker
        from distllm.core.coordinator_nodes import NodeRegistrar

        self.resource_mgr = resource_mgr or ResourceManager()
        self.total_layers = total_layers
        self._model_name = model_name

        self.health_checker = HealthChecker(
            resource_mgr=self.resource_mgr,
            metrics_exporter=None,
        )

        self.node_registrar = NodeRegistrar(
            pipeline=None,  # Set later via set_pipeline
            model_name=model_name,
            resource_mgr=self.resource_mgr,
        )

    def set_pipeline(self, pipeline: Any) -> None:
        """Wire the pipeline orchestrator for node registration."""
        self.node_registrar._pipeline = pipeline

    def set_metrics_exporter(self, exporter: Any) -> None:
        """Set the metrics exporter for health checks."""
        self.health_checker._metrics_exporter = exporter

    @property
    def nodes(self) -> dict:
        return self.node_registrar.nodes

    @property
    def node_order(self) -> list[str]:
        return self.node_registrar.node_order

    def register_node(
        self,
        node_id: str,
        host: str,
        port: int,
        start_layer: int,
        end_layer: int,
        role: str = "compute",
        gpu_memory_mb: int = 0,
    ) -> None:
        """Register a node in the topology."""
        self.node_registrar.register_node(
            node_id, host, port, start_layer, end_layer, role, gpu_memory_mb,
        )

    def unregister_node(self, node_id: str) -> None:
        self.node_registrar.unregister_node(node_id)

    def health_check(self) -> dict:
        return self.health_checker.run_health_checks()

    def check_circuit_breaker(self, node_id: str) -> bool:
        return self.resource_mgr.check_circuit_breaker(node_id)

    def record_failure(self, node_id: str) -> None:
        self.resource_mgr.record_failure(node_id)

    def record_success(self, node_id: str) -> None:
        self.resource_mgr.record_success(node_id)

    def get_node_count(self) -> int:
        return len(self.nodes)

    def stats(self) -> dict:
        return {
            "node_count": self.get_node_count(),
            "node_order": self.node_order,
            "circuit_breakers": self.resource_mgr.get_circuit_breaker_stats(),
        }
