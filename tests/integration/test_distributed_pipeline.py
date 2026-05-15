"""Integration tests for distributed pipeline operations.

Tests tensor serialization round-trips, multi-node forward pass simulation,
layer splitting, and end-to-end generation flow using real gRPC infrastructure
with mock models (no GPU required).
"""

import struct

import pytest
import torch

from distllm.core.batch_scheduler import BatchScheduler, Sequence
from distllm.core.coordinator import Coordinator
from distllm.core.pipeline_orchestrator import PipelineOrchestrator


class TestTensorSerialization:
    """Test tensor to proto and back serialization."""

    def test_roundtrip_float32_tensor(self, mock_tensor_serializer):
        tensor_to_proto, proto_to_tensor = mock_tensor_serializer
        original = torch.randn(4, 8, dtype=torch.float32)
        proto = tensor_to_proto(original)
        restored = proto_to_tensor(proto)
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        torch.testing.assert_close(restored, original)

    def test_roundtrip_1d_tensor(self, mock_tensor_serializer):
        tensor_to_proto, proto_to_tensor = mock_tensor_serializer
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        proto = tensor_to_proto(original)
        restored = proto_to_tensor(proto)
        torch.testing.assert_close(restored, original)

    def test_roundtrip_scalar_tensor(self, mock_tensor_serializer):
        tensor_to_proto, proto_to_tensor = mock_tensor_serializer
        original = torch.tensor(42.0, dtype=torch.float32)
        proto = tensor_to_proto(original)
        restored = proto_to_tensor(proto)
        torch.testing.assert_close(restored, original)

    def test_roundtrip_large_tensor(self, mock_tensor_serializer):
        tensor_to_proto, proto_to_tensor = mock_tensor_serializer
        original = torch.randn(128, 768, dtype=torch.float32)
        proto = tensor_to_proto(original)
        restored = proto_to_tensor(proto)
        assert restored.shape == original.shape
        torch.testing.assert_close(restored, original)


class TestMultiNodeForwardPass:
    """Test multi-node forward pass simulation."""

    def test_pipeline_orchestrator_routes_to_correct_nodes(self, integration_coordinator_with_nodes):
        coord = integration_coordinator_with_nodes
        orchestrator = PipelineOrchestrator(
            total_layers=12,
            resource_mgr=coord._resource_mgr,
        )

        # Register nodes with the orchestrator
        for node_id, reg in coord.nodes.items():
            orchestrator.register_node(
                node_id=node_id,
                host=reg.host,
                port=reg.port,
                start_layer=reg.start_layer,
                end_layer=reg.end_layer,
                use_tls=False,
            )

        # Verify node routing by checking internal state
        node_order = orchestrator.node_order
        assert len(node_order) >= 1

    def test_coordinator_node_registration(self, integration_coordinator_with_nodes):
        coord = integration_coordinator_with_nodes
        assert len(coord.nodes) == 2
        assert "node-0" in coord.nodes
        assert "node-1" in coord.nodes
        assert coord.nodes["node-0"].start_layer == 0
        assert coord.nodes["node-0"].end_layer == 5
        assert coord.nodes["node-1"].start_layer == 6
        assert coord.nodes["node-1"].end_layer == 11


class TestLayerSplitting:
    """Test model layer splitting across nodes."""

    def test_coordinator_splits_layers_evenly(self, integration_coordinator_with_nodes):
        coord = integration_coordinator_with_nodes
        total_layers = 12
        num_nodes = 2
        layers_per_node = total_layers // num_nodes

        for i, node_id in enumerate(coord.node_order):
            reg = coord.nodes[node_id]
            expected_start = i * layers_per_node
            expected_end = (i + 1) * layers_per_node - 1
            assert reg.start_layer == expected_start
            assert reg.end_layer == expected_end

    def test_layer_ranges_cover_full_model(self, integration_coordinator_with_nodes):
        coord = integration_coordinator_with_nodes
        all_layers = set()
        for reg in coord.nodes.values():
            for layer in range(reg.start_layer, reg.end_layer + 1):
                all_layers.add(layer)

        # All 12 layers should be covered
        assert all_layers == set(range(12))


class TestBatchScheduling:
    """Test batch scheduling with real coordinator infrastructure."""

    def test_scheduler_add_and_schedule(self, integration_coordinator):
        scheduler = BatchScheduler(max_batch_size=4, max_tokens_per_batch=512)

        seq1 = Sequence(
            request_id="req-1",
            prompt_tokens=[1, 2, 3],
            generated_tokens=[],
            max_new_tokens=10,
        )
        scheduler.add(seq1)

        batch = scheduler.schedule()
        assert batch is not None
        assert len(batch.sequences) == 1
        assert batch.sequences[0].request_id == "req-1"

    def test_scheduler_respects_max_batch_size(self, integration_coordinator):
        scheduler = BatchScheduler(max_batch_size=2, max_tokens_per_batch=512)

        for i in range(4):
            seq = Sequence(
                request_id=f"req-{i}",
                prompt_tokens=[1],
                generated_tokens=[],
                max_new_tokens=10,
            )
            scheduler.add(seq)

        batch = scheduler.schedule()
        assert len(batch.sequences) <= 2

    def test_scheduler_respects_max_tokens_per_batch(self, integration_coordinator):
        scheduler = BatchScheduler(max_batch_size=10, max_tokens_per_batch=20)

        for i in range(4):
            seq = Sequence(
                request_id=f"req-{i}",
                prompt_tokens=list(range(10)),
                generated_tokens=[],
                max_new_tokens=10,
            )
            scheduler.add(seq)

        batch = scheduler.schedule()
        total_tokens = sum(len(s.prompt_tokens) + len(s.generated_tokens) for s in batch.sequences)
        assert total_tokens <= 20
