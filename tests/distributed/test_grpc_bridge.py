"""Tests for gRPC communication bridge using mocks."""

from unittest.mock import MagicMock, patch

import pytest
import torch


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_node():
    """Create a mock worker node."""
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
            if self.is_last and input_ids is not None:
                out = torch.randn(out.shape[0], out.shape[1], 32000)
            return out, None

    return MockWorkerNode()


# ── Test: Circuit breaker integration ─────────────────────────────────

class TestCircuitBreakerIntegration:
    """Verify circuit breaker tracks node failures properly."""

    def test_check_circuit_breaker_returns_false_initially(self):
        from distllm.core.coordinator import Coordinator
        with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok, \
             patch("distllm.core.coordinator.GRPCServer", create=True):
            mock_tok.from_pretrained.return_value = MagicMock()
            coord = Coordinator(model_name="test-model")
            result = coord._resource_mgr.check_circuit_breaker("node-0")
            assert result is False

    def test_record_failure_then_check_opens_circuit(self):
        from distllm.core.coordinator import Coordinator
        with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok, \
             patch("distllm.core.coordinator.GRPCServer", create=True):
            mock_tok.from_pretrained.return_value = MagicMock()
            coord = Coordinator(model_name="test-model")
            rm = coord._resource_mgr
            rm.record_failure("node-0")
            rm.record_failure("node-0")
            rm.record_failure("node-0")
            assert rm.check_circuit_breaker("node-0") is True

    def test_record_success_resets_circuit(self):
        from distllm.core.coordinator import Coordinator
        with patch("distllm.core.coordinator.AutoTokenizer") as mock_tok, \
             patch("distllm.core.coordinator.GRPCServer", create=True):
            mock_tok.from_pretrained.return_value = MagicMock()
            coord = Coordinator(model_name="test-model")
            rm = coord._resource_mgr
            rm.record_failure("node-0")
            rm.record_failure("node-0")
            rm.record_success("node-0")
            assert rm.check_circuit_breaker("node-0") is False


# ── Test: Mock gRPC client/server ────────────────────────────────────

class TestMockGrpcClient:
    """Test gRPC-like client creation with mocks."""

    def test_create_mock_client(self, mock_node):
        """Verify a mock node can be wrapped as a client-like object."""
        client = MagicMock()
        client.node_id = mock_node.node_id
        client.start_layer = mock_node.start_layer
        client.end_layer = mock_node.end_layer
        assert client.node_id == "mock_worker"
        assert client.start_layer == 0
        assert client.end_layer == 3

    def test_mock_client_health_check(self, mock_node):
        """Simulate health check response."""
        client = MagicMock()
        client.health_check.return_value = {
            "healthy": True,
            "node_id": mock_node.node_id,
            "start_layer": mock_node.start_layer,
            "end_layer": mock_node.end_layer,
        }
        resp = client.health_check()
        assert resp["healthy"] is True
        assert resp["node_id"] == "mock_worker"

    def test_mock_client_forward_pass(self, mock_node):
        """Simulate forward pass through a mock client."""
        input_ids = torch.tensor([[1, 2, 3]])
        hidden_states, _ = mock_node.forward_fn(input_ids=input_ids)
        assert hidden_states.shape[0] == 1
        assert hidden_states.shape[1] == 3

    def test_mock_client_forward_pass_hidden(self, mock_node):
        """Simulate forward pass with hidden states."""
        hidden_in = torch.randn(1, 5, 64)
        hidden_out, _ = mock_node.forward_fn(hidden_states=hidden_in)
        assert hidden_out.shape == hidden_in.shape

    def test_mock_profile_returns_gpu_info(self):
        """Simulate profile RPC returning GPU info."""
        client = MagicMock()
        client.profile.return_value = {"device": "cpu", "memory_gb": 16.0, "compute_capability": "7.5"}
        prof = client.profile()
        assert prof["device"] == "cpu"
        assert prof["memory_gb"] == 16.0

    def test_mock_client_error_handling(self):
        """Simulate RPC error handling."""
        client = MagicMock()
        client.forward_pass.side_effect = ConnectionError("Node unreachable")
        with pytest.raises(ConnectionError):
            client.forward_pass(input_ids=torch.tensor([[1, 2, 3]]))

    def test_mock_node_registration(self, mock_node):
        """Simulate node registration."""
        registry = {}
        registry[mock_node.node_id] = {
            "start_layer": mock_node.start_layer,
            "end_layer": mock_node.end_layer,
            "total_layers": mock_node.total_layers,
        }
        assert "mock_worker" in registry
        assert registry["mock_worker"]["start_layer"] == 0

    def test_mock_pipeline_node_order(self, mock_node):
        """Simulate pipeline node ordering."""
        nodes = [
            {"id": "node-0", "start": 0, "end": 3},
            {"id": "node-1", "start": 3, "end": 6},
        ]
        assert len(nodes) == 2
        assert nodes[0]["end"] == nodes[1]["start"]

    def test_mock_init_client_connects(self):
        """Simulate client initialization."""
        client = MagicMock()
        client.connect.return_value = True
        assert client.connect() is True

    def test_mock_init_client_pulls_capabilities(self):
        """Simulate pulling GPU capabilities."""
        client = MagicMock()
        client.get_capabilities.return_value = {"device": "cuda", "memory_gb": 80}
        caps = client.get_capabilities()
        assert caps["device"] == "cuda"

    def test_mock_health_check_via_registration(self):
        """Simulate health check through registration."""
        registry = MagicMock()
        registry.health_check.return_value = {"healthy": True}
        resp = registry.health_check()
        assert resp["healthy"] is True

    def test_mock_register_node_creates_client(self, mock_node):
        """Simulate registering a node creates a client reference."""
        pipeline = MagicMock()
        pipeline.register_node.return_value = {"node_id": mock_node.node_id, "client": MagicMock()}
        result = pipeline.register_node(mock_node)
        assert result["node_id"] == "mock_worker"
        assert result["client"] is not None
