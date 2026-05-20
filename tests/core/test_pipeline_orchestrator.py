"""Tests for PipelineOrchestrator: node topology, layer assignment,
pipeline execution, overlap scheduling, fallback, hierarchical stages,
and tensor transport integration.

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
from distllm.communication.transport import TensorTransport, TransportBackend


@pytest.fixture
def pipeline():
    """Create a PipelineOrchestrator with mocked ResourceManager."""
    from distllm.core.pipeline_orchestrator import PipelineOrchestrator
    rm = ResourceManager()
    return PipelineOrchestrator(resource_mgr=rm, total_layers=12)


class TestPipelineOrchestratorInit:
    """Tests for initialization."""

    def test_default_init(self):
        from distllm.core.pipeline_orchestrator import PipelineOrchestrator
        p = PipelineOrchestrator()
        assert p.nodes == {}
        assert p.node_order == []
        assert p.prefill_nodes == {}
        assert p.decode_nodes == {}
        assert p.total_layers == 0
        assert p.enable_overlap is False
        assert p.pipeline_timeout == 30.0

    def test_init_with_resource_mgr(self, pipeline):
        assert pipeline.resource_mgr is not None
        assert pipeline.total_layers == 12

    def test_node_kv_sent_lens_init(self, pipeline):
        assert pipeline._node_kv_sent_lens == {}


class TestRegisterNode:
    """Tests for register_node."""

    def test_register_first_node(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert "node-0" in pipeline.nodes
        assert pipeline.node_order == ["node-0"]
        assert pipeline.nodes["node-0"].start_layer == 0
        assert pipeline.nodes["node-0"].end_layer == 5

    def test_register_multiple_nodes(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        pipeline.register_node("node-1", "localhost", 50052, 6, 11)
        assert pipeline.node_order == ["node-0", "node-1"]

    def test_register_unsorted_order(self, pipeline):
        pipeline.register_node("node-1", "localhost", 50052, 6, 11)
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert pipeline.node_order == ["node-0", "node-1"]

    def test_register_prefill_node(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5,
                               role=NodeRole.PREFILL)
        assert "node-0" in pipeline.prefill_nodes

    def test_register_decode_node(self, pipeline):
        pipeline.register_node("node-1", "localhost", 50052, 6, 11,
                               role=NodeRole.DECODE)
        assert "node-1" in pipeline.decode_nodes

class TestValidateLayerAssignment:
    """Tests for layer assignment validation."""

    def test_within_bounds(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        assert pipeline.node_order == ["node-0"]

    def test_out_of_bounds_raises(self, pipeline):
        with pytest.raises(ConfigValidationError, match="out of bounds"):
            pipeline.register_node("node-0", "localhost", 50051, 0, 20)

    def test_negative_layer_raises(self, pipeline):
        with pytest.raises(ConfigValidationError, match="out of bounds"):
            pipeline.register_node("node-0", "localhost", 50051, -1, 5)

    def test_start_gt_end_raises(self, pipeline):
        with pytest.raises(ConfigValidationError, match="start_layer"):
            pipeline.register_node("node-0", "localhost", 50051, 5, 3)

    def test_overlapping_layers_raises(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        with pytest.raises(ConfigValidationError, match="overlap"):
            pipeline.register_node("node-1", "localhost", 50052, 3, 8)

    def test_adjacent_layers_ok(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 5)
        pipeline.register_node("node-1", "localhost", 50052, 6, 11)
        assert len(pipeline.node_order) == 2

    def test_no_validation_without_total_layers(self):
        from distllm.core.pipeline_orchestrator import PipelineOrchestrator
        p = PipelineOrchestrator(total_layers=0)
        p.register_node("node-0", "localhost", 50051, 0, 5)
        assert "node-0" in p.nodes


class TestGroupNodesIntoStages:
    """Tests for hierarchical stage grouping."""

    def test_empty_returns_empty(self, pipeline):
        assert pipeline.group_nodes_into_stages() == []

    def test_single_node_single_stage(self, pipeline):
        pipeline.register_node("node-0", "localhost", 50051, 0, 11)
        stages = pipeline.group_nodes_into_stages()
        assert len(stages) == 1
        assert stages[0] == ["node-0"]

    def test_multiple_nodes_grouped(self, pipeline):
        for i in range(4):
            pipeline.register_node(f"node-{i}", "localhost", 50051 + i, i * 3, i * 3 + 2)
        stages = pipeline.group_nodes_into_stages(num_stages=2)
        assert len(stages) == 2

    def test_custom_num_stages(self, pipeline):
        for i in range(6):
            pipeline.register_node(f"node-{i}", "localhost", 50051 + i, i * 2, i * 2 + 1)
        stages = pipeline.group_nodes_into_stages(num_stages=3)
        assert len(stages) == 3


class TestTensorTransport:
    """Tests for tensor transport integration."""

    def test_set_tensor_transport(self, pipeline):
        transport = MagicMock(spec=TensorTransport)
        transport.backend = TransportBackend.NCCL
        transport.is_available = True
        pipeline.set_tensor_transport(transport, {"node-0": 0})
        assert pipeline._transport is transport
        assert pipeline._transport_rank_map == {"node-0": 0}

    def test_set_tensor_transport_without_rank_map(self, pipeline):
        transport = MagicMock(spec=TensorTransport)
        transport.backend = TransportBackend.NCCL
        transport.is_available = True
        pipeline.set_tensor_transport(transport)
        assert pipeline._transport_rank_map == {}

    def test_set_latency_tracker(self, pipeline):
        from distllm.core.latency_tracker import LatencyTracker
        tracker = LatencyTracker()
        pipeline.set_latency_tracker(tracker)
        assert pipeline._latency_tracker is tracker

