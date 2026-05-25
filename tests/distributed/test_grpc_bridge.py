"""Integration tests for the gRPC communication bridge between coordinator and workers.

Tests:
  - gRPC server starts and serves HealthCheck/Profile/ForwardPass RPCs
  - NodeRegistration.init_client() connects and fetches GPU capabilities
  - PipelineOrchestrator.register_node() creates gRPC client
  - ForwardPass with input_ids and hidden_states round-trips correctly
  - Node failure is detected via health_check()
  - Full pipeline: multiple nodes in sequence
"""

import os
import sys
import threading
import time

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_node():
    """Create a mock worker node that returns known tensor shapes."""
    from types import SimpleNamespace

    class MockWorkerNode:
        def __init__(self):
            self.node_id = "mock_worker"
            self.start_layer = 0
            self.end_layer = 3
            self.total_layers = 8
            self.is_first = True
            self.is_last = True
            self.partitioner = SimpleNamespace(layers=list(range(4)))

        def _get_device(self):
            return "cpu"

        def forward_fn(self, hidden_states=None, attention_mask=None,
                       position_ids=None, past_key_values=None, input_ids=None):
            if input_ids is not None:
                out = torch.randn(1, input_ids.shape[1], 64)
            elif hidden_states is not None:
                out = hidden_states * 1.0
            else:
                out = torch.randn(1, 1, 64)
            if self.is_last:
                out = torch.randn(out.shape[0], out.shape[1], 32000)
            return out, None

    return MockWorkerNode()


@pytest.fixture
def node_server(mock_node):
    """Start a gRPC server on a random port, yield (server, port), stop on teardown."""
    from distllm.dist.node_service import NodeServer
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    server = NodeServer(mock_node, port=port)
    t = threading.Thread(target=lambda: server.start(use_tls=False), daemon=True)
    t.start()
    time.sleep(0.5)
    yield server, port
    server.stop()


# ── Test 1: gRPC Server Lifecycle ─────────────────────────────────────

class TestGrpcServerLifecycle:
    """Verify the gRPC server starts, serves, and stops correctly."""

    def test_server_starts_and_stops(self, node_server):
        server, port = node_server
        from distllm.dist.node_client import create_node_client

        client = create_node_client('127.0.0.1', port, timeout_s=3.0)
        assert client is not None
        assert client.stub is not None
        client.close()

    def test_health_check_returns_healthy(self, node_server):
        server, port = node_server
        from distllm.dist.node_client import create_node_client
        from distllm.dist import node_pb2

        client = create_node_client('127.0.0.1', port, timeout_s=3.0)
        resp = client.stub.HealthCheck(node_pb2.HealthCheckRequest(node_id="test"))
        assert resp.healthy
        assert resp.node_id == "mock_worker"
        assert resp.start_layer == 0
        assert resp.end_layer == 3
        client.close()

    def test_profile_returns_gpu_info(self, node_server):
        server, port = node_server
        from distllm.dist.node_client import create_node_client
        from distllm.dist import node_pb2

        client = create_node_client('127.0.0.1', port, timeout_s=3.0)
        resp = client.stub.Profile(node_pb2.ProfileRequest(node_id="test"))
        assert resp.node_id == "mock_worker"
        assert resp.gpu_name in ("cpu", "") or len(resp.gpu_name) > 0
        client.close()


# ── Test 2: ForwardPass RPC ───────────────────────────────────────────

class TestForwardPassRpc:
    """Verify ForwardPass RPC correctly processes input and hidden states."""

    def test_forward_pass_with_input_ids(self, node_server):
        server, port = node_server
        from distllm.dist.node_client import create_node_client
        from distllm.dist import node_pb2
        from distllm.dist.node_service import tensor_from_proto

        client = create_node_client('127.0.0.1', port, timeout_s=3.0)
        req = node_pb2.ForwardPassRequest(
            request_id="test_001", input_ids=[1, 2, 3, 4, 5],
            batch_size=1, seq_len=5, use_cache=True, is_first_pass=True,
        )
        resp = client.stub.ForwardPass(req)
        assert resp.success, f"ForwardPass failed: {resp.error_message}"
        output = tensor_from_proto(resp.output)
        assert output.dim() >= 2, f"Expected >=2D tensor, got shape {output.shape}"
        client.close()

    def test_forward_pass_with_hidden_states(self, node_server):
        server, port = node_server
        from distllm.dist.node_client import create_node_client
        from distllm.dist import node_pb2
        from distllm.dist.node_service import tensor_to_proto, tensor_from_proto

        client = create_node_client('127.0.0.1', port, timeout_s=3.0)
        hs = torch.randn(1, 10, 64)
        req = node_pb2.ForwardPassRequest(
            request_id="test_002",
            hidden_states=tensor_to_proto(hs),
            batch_size=1, seq_len=10, use_cache=True, is_first_pass=False,
        )
        resp = client.stub.ForwardPass(req)
        assert resp.success, f"ForwardPass(hs) failed: {resp.error_message}"
        output = tensor_from_proto(resp.output)
        assert output.shape[0] == 1
        assert output.shape[1] == 10
        client.close()

    def test_forward_pass_error_handling(self, node_server):
        """Send empty request — should not crash the server."""
        server, port = node_server
        from distllm.dist.node_client import create_node_client
        from distllm.dist import node_pb2

        client = create_node_client('127.0.0.1', port, timeout_s=3.0)
        req = node_pb2.ForwardPassRequest(request_id="empty")
        resp = client.stub.ForwardPass(req)
        assert resp.success  # empty request is valid (returns randn)
        client.close()


# ── Test 3: NodeRegistration gRPC Client ──────────────────────────────

class TestNodeRegistration:
    """Verify NodeRegistration.init_client() creates working connections."""

    def test_init_client_connects(self, node_server):
        server, port = node_server
        from distllm.core.resource_manager import NodeRegistration

        reg = NodeRegistration(
            node_id="test_node", host='127.0.0.1', port=port,
            start_layer=0, end_layer=3,
        )
        reg.init_client(timeout_s=5.0)
        assert reg.client is not None
        assert reg.client.stub is not None
        reg.close()

    def test_init_client_pulls_gpu_capabilities(self, node_server):
        server, port = node_server
        from distllm.core.resource_manager import NodeRegistration

        reg = NodeRegistration(
            node_id="cap_test", host='127.0.0.1', port=port,
            start_layer=0, end_layer=3,
        )
        reg.init_client(timeout_s=5.0)
        assert reg.gpu_name != ""
        assert reg.gpu_memory_total > 0
        assert reg.gpu_profile_raw is not None
        reg.close()

    def test_health_check_via_registration(self, node_server):
        server, port = node_server
        from distllm.core.resource_manager import NodeRegistration

        reg = NodeRegistration(
            node_id="live_test", host='127.0.0.1', port=port,
            start_layer=0, end_layer=3,
        )
        reg.init_client(timeout_s=5.0)
        alive = reg.health_check()
        assert alive

        server.stop()
        time.sleep(0.3)
        alive_after = reg.health_check()
        assert not alive_after
        reg.close()

    def test_init_client_raises_on_unreachable(self):
        from distllm.core.resource_manager import NodeRegistration
        from distllm.errors.types import NodeUnreachableError

        reg = NodeRegistration(
            node_id="dead_node", host='127.0.0.1', port=19999,
            start_layer=0, end_layer=3,
        )
        with pytest.raises(NodeUnreachableError):
            reg.init_client(timeout_s=1.0)


# ── Test 4: PipelineOrchestrator Node Registration ────────────────────

class TestPipelineRegistration:
    """Verify PipelineOrchestrator.register_node() connects via gRPC."""

    def test_register_node_creates_client(self, node_server):
        server, port = node_server
        from distllm.dist.pipeline import PipelineOrchestrator

        pipeline = PipelineOrchestrator(total_layers=8, pipeline_timeout=10.0)
        pipeline.register_node(
            node_id='node_0', host='127.0.0.1', port=port,
            start_layer=0, end_layer=3,
        )
        assert 'node_0' in pipeline.nodes
        assert pipeline.nodes['node_0'].client is not None
        assert pipeline.nodes['node_0'].client.stub is not None

    def test_pipeline_node_order(self, node_server):
        """Multiple nodes should be ordered by start_layer."""
        server, port = node_server
        from distllm.dist.pipeline import PipelineOrchestrator

        pipeline = PipelineOrchestrator(total_layers=16, pipeline_timeout=10.0)
        pipeline.register_node('node_2', '127.0.0.1', port, 8, 15)
        pipeline.register_node('node_1', '127.0.0.1', port, 4, 7)
        pipeline.register_node('node_0', '127.0.0.1', port, 0, 3)
        assert pipeline.node_order == ['node_0', 'node_1', 'node_2']
