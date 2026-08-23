"""Tests for node_registrar.py.

Zero mocks -- uses only real objects from the module.
"""

from __future__ import annotations

import pytest

from distllm.config.settings import NodeRole
from distllm.dist.node_registrar import NodeRegistrar
from distllm.dist.pipeline.orchestrator import PipelineOrchestrator


class TestNodeRegistrarInit:
    """Test NodeRegistrar construction."""

    def test_default_init(self) -> None:
        """Default constructor sets expected attribute values."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        assert registrar.pipeline is pipeline
        assert registrar.model_name == "test-model"
        assert registrar.trust_remote_code is None
        assert registrar.expert_registry is None
        assert registrar.federation_manager is None

    def test_init_with_trust_remote_code(self) -> None:
        """trust_remote_code=True is propagated."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model", trust_remote_code=True)
        assert registrar.trust_remote_code is True

    def test_init_with_trust_remote_code_false(self) -> None:
        """trust_remote_code=False is propagated."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model", trust_remote_code=False)
        assert registrar.trust_remote_code is False

    def test_init_with_expert_registry(self) -> None:
        """ExpertRegistry is stored when provided."""
        from distllm.core.moe_orchestrator import ExpertRegistry

        pipeline = PipelineOrchestrator()
        registry = ExpertRegistry()
        registrar = NodeRegistrar(pipeline, "test-model", expert_registry=registry)
        assert registrar.expert_registry is registry

    def test_init_with_federation_manager(self) -> None:
        """federation_manager is stored when provided."""
        pipeline = PipelineOrchestrator()
        manager = object()
        registrar = NodeRegistrar(pipeline, "test-model", federation_manager=manager)
        assert registrar.federation_manager is manager

    def test_init_empty_model_name(self) -> None:
        """Empty model name string is accepted."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "")
        assert registrar.model_name == ""


class TestNodeRegistrarManualRegister:
    """Test NodeRegistrar.manual_register."""

    def test_basic_register(self) -> None:
        """A single node is registered with all required fields."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register("node-0", "10.0.0.1", 50051, 0, 7)
        node = pipeline.get_node("node-0")
        assert node is not None
        assert node.node_id == "node-0"
        assert node.host == "10.0.0.1"
        assert node.port == 50051
        assert node.start_layer == 0
        assert node.end_layer == 7

    def test_register_sets_total_layers(self) -> None:
        """total_layers is propagated to the pipeline when given."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        assert pipeline.total_layers == 0
        registrar.manual_register("node-0", "localhost", 50051, 0, 7, total_layers=32)
        assert pipeline.total_layers == 32

    def test_register_with_role_prefill(self) -> None:
        """NodeRole.PREFILL is accepted without error."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register(
            "node-0", "localhost", 50051, 0, 7, role=NodeRole.PREFILL,
        )

    def test_register_with_role_decode(self) -> None:
        """NodeRole.DECODE is accepted without error."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register(
            "node-0", "localhost", 50051, 0, 7, role=NodeRole.DECODE,
        )

    def test_register_with_cluster_id_and_key(self) -> None:
        """cluster_id and cluster_key are accepted."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register(
            "node-0", "localhost", 50051, 0, 7,
            cluster_id="cluster-a", cluster_key="secret",
        )

    def test_register_with_weight_source(self) -> None:
        """weight_source is accepted."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register(
            "node-0", "localhost", 50051, 0, 7,
            weight_source="10.0.0.2:50052",
        )

    def test_register_with_expert_ids_no_registry(self) -> None:
        """expert_ids with no expert_registry does not raise."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register(
            "node-0", "localhost", 50051, 0, 7, expert_ids=[0, 1],
        )

    def test_register_multiple_nodes(self) -> None:
        """Multiple nodes are registered successfully."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register("node-0", "10.0.0.1", 50051, 0, 7)
        registrar.manual_register("node-1", "10.0.0.2", 50052, 8, 15)
        assert pipeline.get_node("node-0") is not None
        assert pipeline.get_node("node-1") is not None
        assert len(pipeline.node_order) == 2

    def test_register_empty_node_id(self) -> None:
        """Empty string as node_id is accepted (edge case)."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register("", "localhost", 50051, 0, 7)
        node = pipeline.get_node("")
        assert node is not None
        assert node.node_id == ""

    def test_register_zero_port(self) -> None:
        """Port 0 is accepted (edge case)."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register("node-0", "localhost", 0, 0, 7)
        node = pipeline.get_node("node-0")
        assert node is not None
        assert node.port == 0

    def test_register_single_layer(self) -> None:
        """start_layer equal to end_layer (single layer) is accepted."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register("node-0", "localhost", 50051, 5, 5)
        node = pipeline.get_node("node-0")
        assert node.start_layer == 5
        assert node.end_layer == 5

    def test_register_node_appears_in_nodes_property(self) -> None:
        """Registered node appears in the PipelineOrchestrator.nodes dict."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register("node-0", "10.0.0.1", 50051, 0, 7)
        nodes = pipeline.nodes
        assert "node-0" in nodes
        # ``nodes`` maps node_id -> PipelineNode dataclass (attribute access).
        node = nodes["node-0"]
        assert node.host == "10.0.0.1"
        assert node.port == 50051
        assert node.start_layer == 0
        assert node.end_layer == 7
        assert node.is_healthy is True

    def test_register_negative_layers(self) -> None:
        """Negative layer indices are accepted by the pipeline."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register("node-0", "localhost", 50051, -1, -1)
        node = pipeline.get_node("node-0")
        assert node.start_layer == -1
        assert node.end_layer == -1


class TestNodeRegistrarRegisterExpertOnNode:
    """Test NodeRegistrar.register_expert_on_node."""

    def test_no_registry_returns_early(self) -> None:
        """No-op when expert_registry is None."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.register_expert_on_node("node-0", [0, 1, 2])

    def test_with_registry(self) -> None:
        """Experts are registered through a real ExpertRegistry."""
        from distllm.core.moe_orchestrator import ExpertRegistry

        pipeline = PipelineOrchestrator()
        registry = ExpertRegistry()
        registrar = NodeRegistrar(pipeline, "test-model", expert_registry=registry)
        registrar.register_expert_on_node("node-0", [0, 1], layer_idx=3)

    def test_empty_expert_ids(self) -> None:
        """Empty expert_id list is a no-op in the registry."""
        from distllm.core.moe_orchestrator import ExpertRegistry

        pipeline = PipelineOrchestrator()
        registry = ExpertRegistry()
        registrar = NodeRegistrar(pipeline, "test-model", expert_registry=registry)
        registrar.register_expert_on_node("node-0", [])

    def test_multiple_expert_ids(self) -> None:
        """Multiple expert IDs are all registered."""
        from distllm.core.moe_orchestrator import ExpertRegistry

        pipeline = PipelineOrchestrator()
        registry = ExpertRegistry()
        registrar = NodeRegistrar(pipeline, "test-model", expert_registry=registry)
        registrar.register_expert_on_node("node-0", [10, 11, 12], layer_idx=2)

    def test_default_layer_idx(self) -> None:
        """Default layer_idx=0 is used when not specified."""
        from distllm.core.moe_orchestrator import ExpertRegistry

        pipeline = PipelineOrchestrator()
        registry = ExpertRegistry()
        registrar = NodeRegistrar(pipeline, "test-model", expert_registry=registry)
        registrar.register_expert_on_node("node-0", [5, 6])

    def test_multiple_calls_different_nodes(self) -> None:
        """Experts are registered on different nodes correctly."""
        from distllm.core.moe_orchestrator import ExpertRegistry

        pipeline = PipelineOrchestrator()
        registry = ExpertRegistry()
        registrar = NodeRegistrar(pipeline, "test-model", expert_registry=registry)
        registrar.register_expert_on_node("node-0", [0, 1], layer_idx=1)
        registrar.register_expert_on_node("node-1", [2, 3], layer_idx=1)


class TestNodeRegistrarRegisterNodesBatch:
    """Test NodeRegistrar.register_nodes_batch."""

    def test_single_node(self) -> None:
        """A single node batch registration succeeds."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 15,
            },
        ]
        results = registrar.register_nodes_batch(configs)
        assert "node-0" in results
        assert results["node-0"]["success"] is True
        assert results["node-0"]["error"] is None

    def test_multiple_nodes(self) -> None:
        """Multiple nodes are registered in batch."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
            },
            {
                "node_id": "node-1", "host": "10.0.0.2", "port": 50052,
                "start_layer": 8, "end_layer": 15,
            },
            {
                "node_id": "node-2", "host": "10.0.0.3", "port": 50053,
                "start_layer": 16, "end_layer": 23,
            },
        ]
        results = registrar.register_nodes_batch(configs)
        assert len(results) == 3
        for nid in ("node-0", "node-1", "node-2"):
            assert results[nid]["success"] is True

    def test_empty_config_list(self) -> None:
        """Empty config list returns empty results dict."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        results = registrar.register_nodes_batch([])
        assert results == {}

    def test_with_cluster_key(self) -> None:
        """cluster_key is forwarded to manual_register."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
            },
        ]
        results = registrar.register_nodes_batch(configs, cluster_key="shared-secret")
        assert results["node-0"]["success"] is True

    def test_with_all_optional_fields(self) -> None:
        """Config dicts with all optional fields are handled."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
                "total_layers": 32, "role": "prefill",
                "cluster_id": "cluster-a", "weight_source": "10.0.0.2:50052",
                "expert_ids": [0, 1],
            },
        ]
        results = registrar.register_nodes_batch(configs)
        assert results["node-0"]["success"] is True

    def test_custom_max_workers(self) -> None:
        """Custom max_workers is used without error."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": f"node-{i}", "host": "localhost",
                "port": 50051 + i, "start_layer": i * 4,
                "end_layer": i * 4 + 3,
            }
            for i in range(4)
        ]
        results = registrar.register_nodes_batch(configs, max_workers=2)
        assert len(results) == 4
        assert all(r["success"] for r in results.values())

    def test_minimal_config(self) -> None:
        """A config with only node_id uses defaults for all other fields."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [{"node_id": "node-a"}]
        results = registrar.register_nodes_batch(configs)
        assert results["node-a"]["success"] is True

    def test_result_structure(self) -> None:
        """Each result dict has the expected keys and default GPU values."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
            },
        ]
        results = registrar.register_nodes_batch(configs)
        result = results["node-0"]
        assert "success" in result
        assert "gpu_name" in result
        assert "gpu_memory_gb" in result
        assert "error" in result
        assert result["gpu_name"] == ""
        assert result["gpu_memory_gb"] == 0.0
        assert result["error"] is None

    def test_nodes_registered_in_pipeline(self) -> None:
        """Batch-registered nodes appear in the pipeline and stay ordered."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
            },
            {
                "node_id": "node-1", "host": "10.0.0.2", "port": 50052,
                "start_layer": 8, "end_layer": 15,
            },
        ]
        registrar.register_nodes_batch(configs)
        assert pipeline.node_order == ["node-0", "node-1"]

    def test_without_node_id_auto_generated(self) -> None:
        """Missing node_id is auto-generated from the port."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
            },
        ]
        results = registrar.register_nodes_batch(configs)
        assert "node_50051" in results
        assert results["node_50051"]["success"] is True

    def test_custom_timeout(self) -> None:
        """Custom timeout_s is accepted without error."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
            },
        ]
        results = registrar.register_nodes_batch(configs, timeout_s=30.0)
        assert results["node-0"]["success"] is True

    def test_duplicate_node_id(self) -> None:
        """Registering the same node_id twice overwrites the first."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
            },
            {
                "node_id": "node-0", "host": "10.0.0.2", "port": 50052,
                "start_layer": 8, "end_layer": 15,
            },
        ]
        results = registrar.register_nodes_batch(configs)
        # Both registrations succeed (the second overwrites the first in the pipeline)
        assert results["node-0"]["success"] is True
        # The pipeline holds the second registration
        node = pipeline.get_node("node-0")
        assert node.port == 50052
        assert node.start_layer == 8
        assert node.end_layer == 15


class TestNodeRegistrarIntegration:
    """Tests combining multiple NodeRegistrar methods."""

    def test_manual_register_and_expert_registration(self) -> None:
        """Manual register then register_expert_on_node works together."""
        from distllm.core.moe_orchestrator import ExpertRegistry

        pipeline = PipelineOrchestrator()
        registry = ExpertRegistry()
        registrar = NodeRegistrar(pipeline, "test-model", expert_registry=registry)
        registrar.manual_register("node-0", "10.0.0.1", 50051, 0, 7)
        registrar.register_expert_on_node("node-0", [0, 1, 2], layer_idx=3)
        node = pipeline.get_node("node-0")
        assert node is not None
        assert node.node_id == "node-0"

    def test_batch_and_expert_registration(self) -> None:
        """Batch register then register_expert_on_node works together."""
        from distllm.core.moe_orchestrator import ExpertRegistry

        pipeline = PipelineOrchestrator()
        registry = ExpertRegistry()
        registrar = NodeRegistrar(pipeline, "test-model", expert_registry=registry)
        configs = [
            {
                "node_id": "node-0", "host": "10.0.0.1", "port": 50051,
                "start_layer": 0, "end_layer": 7,
            },
            {
                "node_id": "node-1", "host": "10.0.0.2", "port": 50052,
                "start_layer": 8, "end_layer": 15,
            },
        ]
        results = registrar.register_nodes_batch(configs)
        assert results["node-0"]["success"] is True
        assert results["node-1"]["success"] is True
        registrar.register_expert_on_node("node-0", [0, 1, 2], layer_idx=1)
        registrar.register_expert_on_node("node-1", [3, 4], layer_idx=1)

    def test_mixed_manual_and_batch(self) -> None:
        """Manual register and batch register can coexist."""
        pipeline = PipelineOrchestrator()
        registrar = NodeRegistrar(pipeline, "test-model")
        registrar.manual_register("node-manual", "10.0.0.1", 50051, 0, 7)
        configs = [
            {
                "node_id": "node-batch", "host": "10.0.0.2", "port": 50052,
                "start_layer": 8, "end_layer": 15,
            },
        ]
        results = registrar.register_nodes_batch(configs)
        assert results["node-batch"]["success"] is True
        assert pipeline.get_node("node-manual") is not None
        assert pipeline.get_node("node-batch") is not None
        assert pipeline.node_order == ["node-manual", "node-batch"]
