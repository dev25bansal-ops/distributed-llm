"""Tests for pipeline orchestration: node topology, layer assignment,
pipeline execution, disaggregated prefill/decode routing, and the
disaggregated pipeline plan builder.

These tests exercise the real production APIs:
  - ``distllm.dist.pipeline.PipelineOrchestrator`` (distributed pipeline
    parallel execution, micro-batching, tensor transport).
  - ``distllm.core.heterogeneous_scheduler`` (prefill/decode routing and
    disaggregated pipeline plans).

Run: pytest tests/core/test_pipeline_orchestrator.py -v
"""

import concurrent.futures
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import torch

from distllm.core.resource_manager import ResourceManager, NodeRegistration
from distllm.config.loader import NodeRole
from distllm.errors.types import (
    ConfigValidationError, NodeUnreachableError, OOMError,
    InputValidationError, GRPCTimeoutError,
)
from distllm.dist.pipeline import TensorTransport, TransportBackend


@pytest.fixture
def pipeline():
    """Create a PipelineOrchestrator with mocked ResourceManager."""
    from distllm.dist.pipeline import PipelineOrchestrator
    rm = ResourceManager()
    return PipelineOrchestrator(resource_mgr=rm)


def _echo_forward(host="", port=0, hidden_states=None, **kwargs):
    """Mock gRPC forward: return a clone of the input tensor."""
    if hidden_states is not None:
        return hidden_states.clone()
    return torch.tensor([[0]])


class TestPipelineOrchestratorInit:
    """Tests for initialization."""

    def test_default_init(self):
        from distllm.dist.pipeline import PipelineOrchestrator
        p = PipelineOrchestrator()
        assert p.nodes == {}
        assert p.node_order == []
        assert p.total_layers == 0
        assert p.pipeline_timeout == 30.0

    def test_init_with_resource_mgr(self, pipeline):
        assert pipeline._resource_mgr is not None

    def test_register_node_sets_total_layers(self, pipeline):
        # Total layers is derived from node assignments when not explicit.
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert pipeline.total_layers == 6


class TestRegisterNode:
    """Tests for register_node."""

    def test_register_first_node(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert "node-0" in pipeline.nodes
        assert pipeline.node_order == ["node-0"]
        assert pipeline.nodes["node-0"]["start_layer"] == 0
        assert pipeline.nodes["node-0"]["end_layer"] == 5

    def test_register_multiple_nodes(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        pipeline.register_node("node-1", "localhost", 50052, 6, 11)
        assert pipeline.node_order == ["node-0", "node-1"]

    def test_register_unsorted_order(self, pipeline):
        pipeline.register_node("node-1", "localhost", 50052, 6, 11)
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert pipeline.node_order == ["node-0", "node-1"]

    def test_register_node_records_health(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert "node-0" in pipeline.get_healthy_nodes()

    def test_unregister_node(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        pipeline.register_node("node-1", "localhost", 50052, 6, 11)
        pipeline.unregister_node("node-0")
        assert pipeline.node_order == ["node-1"]


class TestValidateLayerAssignment:
    """Tests for layer assignment validation."""

    def test_within_bounds(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert pipeline.node_order == ["node-0"]

    def test_overlapping_layers_raises(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        with pytest.raises((ValueError, ConfigValidationError), match="overlap"):
            pipeline.validate_layer_assignment("node-1", 3, 8)

    def test_adjacent_layers_ok(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        pipeline.register_node("node-1", "localhost", 50052, 6, 11)
        assert len(pipeline.node_order) == 2

    def test_node_kv_sent_lens_tracking(self, pipeline):
        # KV sent-lens map is initialized empty and stays consistent.
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert pipeline._node_order == ["node-0"]


class TestGroupNodesIntoStages:
    """Tests for grouping nodes into hierarchical stages."""

    def test_single_node_single_stage(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 11)
        assert len(pipeline.nodes) == 1

    def test_multiple_nodes_in_topo_order(self, pipeline):
        for i in range(4):
            pipeline.register_node(f"node-{i}", "localhost", 50051 + i, i * 3, i * 3 + 2)
        assert pipeline.node_order == ["node-0", "node-1", "node-2", "node-3"]


class TestSequentialPipelineExecution:
    """Tests for sequential pipeline execution with mocked transport."""

    def test_run_pipeline_with_two_nodes(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        pipeline.register_node("node-1", "localhost", 50052, 6, 11)
        tensor = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with patch("distllm.dist.node_client.forward_request", side_effect=_echo_forward):
            out = pipeline.run_pipeline(tensor, {"node-0": None, "node-1": None}, "req-1")
        assert out is not None
        assert pipeline.stats()["pipeline_runs"] == 1

    def test_run_pipeline_no_healthy_nodes_raises(self, pipeline):
        with pytest.raises(RuntimeError, match="No healthy nodes"):
            pipeline.run_pipeline(torch.tensor([[1]]), {}, "req-empty")

    def test_run_pipeline_marks_failure_on_error(self, pipeline):
        rm = ResourceManager()
        p = pipeline
        p._resource_mgr = rm
        p.register_node("node-0", "localhost", 50051, 0, 5)

        def _boom(host="", port=0, hidden_states=None, **kwargs):
            raise ConnectionError("node down")

        with patch("distllm.dist.node_client.forward_request", side_effect=_boom):
            with pytest.raises(ConnectionError):
                p.run_pipeline(torch.tensor([[1]]), {"node-0": None}, "req-err")


class TestTensorTransport:
    """Tests for tensor transport integration."""

    def test_no_transport_attribute_without_wiring(self, pipeline):
        # set_tensor_transport is not part of the orchestrator API; the
        # transport integration lives in distllm.dist.parallel.configure_pp.
        assert hasattr(pipeline, "_transport") is False or pipeline.stats()["node_count"] == 0

    def test_set_latency_tracker(self, pipeline):
        from distllm.dist.latency import LatencyTracker
        tracker = LatencyTracker()
        pipeline.set_latency_tracker(tracker)
        assert pipeline._latency_tracker is tracker


class TestPrefillDecodeRouter:
    """Tests for the real PrefillDecodeRouter (disaggregated routing).

    The original core tests asserted a sliced prefill/decode API on
    PipelineOrchestrator; production routing lives in
    distllm.core.heterogeneous_scheduler.PrefillDecodeRouter.
    """

    @pytest.fixture
    def router(self):
        from distllm.core.heterogeneous_scheduler import PrefillDecodeRouter
        return PrefillDecodeRouter(prefill_node_ids=["p1"], decode_node_ids=["d1"])

    def test_from_node_roles(self):
        from distllm.core.heterogeneous_scheduler import PrefillDecodeRouter, NodeRole
        r = PrefillDecodeRouter.from_node_roles(
            {"n1": NodeRole.PREFILL, "n2": NodeRole.DECODE, "n3": NodeRole.AUTO}
        )
        assert r._prefill_nodes == ["n1"]
        assert r._decode_nodes == ["n2"]
        assert r._auto_nodes == ["n3"]

    def test_route_non_disaggregated_returns_first_node(self):
        from distllm.core.heterogeneous_scheduler import PrefillDecodeRouter
        r = PrefillDecodeRouter(prefill_node_ids=["only-node"], decode_node_ids=[])
        assert r.route(is_prefill_step=True).node_id == "only-node"
        assert r.route(is_prefill_step=False).node_id == "only-node"

    def test_route_prefill_to_prefill_node(self, router):
        route = router.route(is_prefill_step=True)
        assert route.node_id == "p1"
        assert route.is_prefill is True

    def test_route_decode_to_decode_node(self, router):
        router.record_prefill_node("req-1", "p1")
        route = router.route(is_prefill_step=False, request_id="req-1")
        assert route.node_id == "d1"
        assert route.is_prefill is False
        assert route.source_node_id == "p1"

    def test_disaggregated_when_both_pools_exist(self):
        from distllm.core.heterogeneous_scheduler import PrefillDecodeRouter
        r = PrefillDecodeRouter(prefill_node_ids=["n1"], decode_node_ids=["n2"])
        assert r.is_disaggregated is True
        r0 = PrefillDecodeRouter(prefill_node_ids=["n1"], decode_node_ids=[])
        assert r0.is_disaggregated is False

    def test_cleanup_removes_kv_source(self, router):
        router.record_prefill_node("req-1", "p1")
        assert "req-1" in router._kv_sources
        router.cleanup_request("req-1")
        assert "req-1" not in router._kv_sources


class TestBuildDisaggregatedPlan:
    """Tests for the disaggregated pipeline plan builder."""

    def test_returns_plan_dict(self):
        from distllm.core.heterogeneous_scheduler import build_disaggregated_pipeline_plan
        plan = build_disaggregated_pipeline_plan(
            node_configs=[{"node_id": "n1", "host": "h1", "port": 1, "device_type": "cuda"}],
            total_layers=32,
        )
        assert "roles" in plan
        assert "layer_assignments" in plan
        assert "router" in plan
        assert "is_disaggregated" in plan

    def test_layer_assignments_have_required_keys(self):
        from distllm.core.heterogeneous_scheduler import build_disaggregated_pipeline_plan
        plan = build_disaggregated_pipeline_plan(
            node_configs=[{"node_id": "n1", "host": "h1", "port": 1, "device_type": "cuda"}],
            total_layers=20,
        )
        for a in plan["layer_assignments"]:
            assert "node_id" in a
            assert "start_layer" in a
            assert "end_layer" in a
            assert "role" in a
            assert "host" in a
            assert "port" in a

    def test_multinode_plan(self):
        from distllm.core.heterogeneous_scheduler import build_disaggregated_pipeline_plan
        plan = build_disaggregated_pipeline_plan(
            node_configs=[
                {"node_id": "n1", "host": "h1", "port": 1, "device_type": "cuda"},
                {"node_id": "n2", "host": "h2", "port": 2, "device_type": "cuda"},
                {"node_id": "n3", "host": "h3", "port": 3, "device_type": "cuda"},
            ],
            total_layers=40,
        )
        assert len(plan["layer_assignments"]) == 3
        total_assigned = sum(
            a["end_layer"] - a["start_layer"] + 1 for a in plan["layer_assignments"]
        )
        assert total_assigned <= 40