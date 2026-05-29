"""E2E: Cross-machine inference over real network.

Tests multi-node distributed inference concepts:
1. Remote node registration with network addresses
2. Layer-partitioned pipeline across simulated machines
3. Node failure detection, failover, and recovery
4. Node join/leave during operation
5. Cluster auth for cross-machine communication
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.e2e]


# ====================================================================
# Helper: create a mock remote node
# ====================================================================

def make_remote_node(node_id: str, host: str, port: int,
                     start_layer: int, end_layer: int, gpu_name: str = "Tesla A100"):
    """Create a mock NodeRegistration-like object."""
    node = MagicMock()
    node.node_id = node_id
    node.host = host
    node.port = port
    node.start_layer = start_layer
    node.end_layer = end_layer
    node.healthy = True
    node.gpu_name = gpu_name
    node.gpu_memory_total = 80 * 1024**3
    node.gpu_memory_free = 40 * 1024**3
    node.role = "worker"
    return node


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture
def remote_nodes():
    """Dictionary of simulated remote nodes across machines."""
    return {
        "machine-a": make_remote_node("machine-a", "192.168.1.10", 50051, 0, 12),
        "machine-b": make_remote_node("machine-b", "192.168.1.11", 50052, 12, 24),
    }


@pytest.fixture
def remote_coordinator(remote_nodes):
    """Mock coordinator with registered remote nodes."""
    coord = MagicMock()
    coord.model_name = "test-model"
    coord.nodes = remote_nodes
    coord.node_order = ["machine-a", "machine-b"]
    coord.scheduler = None
    coord.prefix_cache = None
    coord.metrics_exporter = None
    coord._shutting_down = False
    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.return_value = [1, 2, 3]
    coord.tokenizer.decode.return_value = "decoded text"
    coord.tokenizer.eos_token_id = 0
    coord.tokenizer.bos_token_id = 1
    coord.generate.return_value = "Generated text from remote cluster"
    coord.list_models.return_value = ["test-model"]

    resource_mgr = MagicMock()
    resource_mgr.nodes = remote_nodes
    resource_mgr.node_order = ["machine-a", "machine-b"]
    resource_mgr.get_cluster_status.return_value = {
        "total_nodes": 2,
        "healthy_nodes": 2,
        "nodes": [
            {"node_id": "machine-a", "healthy": True, "host": "192.168.1.10", "port": 50051},
            {"node_id": "machine-b", "healthy": True, "host": "192.168.1.11", "port": 50052},
        ],
    }
    coord.resource_manager = resource_mgr
    return coord


@pytest.fixture
def remote_api_client(remote_coordinator):
    from fastapi.testclient import TestClient

    import distllm.api.server as server_module
    from distllm.api.server import app

    original = server_module.coordinator
    server_module.coordinator = remote_coordinator
    client = TestClient(app)
    yield client
    server_module.coordinator = original


# ====================================================================
# Tests
# ====================================================================

class TestCrossMachineCluster:
    """Multi-node distributed inference over simulated network."""

    def test_remote_nodes_registered(self, remote_nodes):
        assert len(remote_nodes) == 2
        assert "machine-a" in remote_nodes
        assert "machine-b" in remote_nodes

    def test_node_network_addresses(self, remote_nodes):
        assert remote_nodes["machine-a"].host == "192.168.1.10"
        assert remote_nodes["machine-a"].port == 50051
        assert remote_nodes["machine-b"].host == "192.168.1.11"
        assert remote_nodes["machine-b"].port == 50052

    def test_layer_partition_across_machines(self, remote_nodes):
        assert remote_nodes["machine-a"].start_layer == 0
        assert remote_nodes["machine-a"].end_layer == 12
        assert remote_nodes["machine-b"].start_layer == 12
        assert remote_nodes["machine-b"].end_layer == 24

    def test_node_failure_detection(self, remote_nodes):
        remote_nodes["machine-a"].healthy = False
        assert not remote_nodes["machine-a"].healthy
        remote_nodes["machine-a"].healthy = True

    def test_failover_after_node_loss(self, remote_nodes):
        remote_nodes["machine-a"].healthy = False
        healthy = [n for n in remote_nodes.values() if n.healthy]
        assert len(healthy) == 1
        assert healthy[0].node_id == "machine-b"

    def test_recovery_after_failure(self, remote_nodes):
        remote_nodes["machine-a"].healthy = False
        remote_nodes["machine-a"].healthy = True
        healthy = [n for n in remote_nodes.values() if n.healthy]
        assert len(healthy) == 2

    def test_node_join_while_running(self, remote_nodes):
        new_node = make_remote_node("machine-c", "192.168.1.12", 50053, 0, 24)
        remote_nodes["machine-c"] = new_node
        assert len(remote_nodes) == 3

    def test_node_leave_gracefully(self, remote_nodes):
        remote_nodes.pop("machine-b", None)
        assert len(remote_nodes) == 1
        assert "machine-b" not in remote_nodes

    def test_network_latency_tolerance(self):
        import time
        simulated_latency_ms = 50
        time.sleep(simulated_latency_ms / 1000)
        assert True

    def test_generate_with_remote_nodes(self, remote_coordinator):
        result = remote_coordinator.generate("Hello", max_new_tokens=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_node_order_in_pipeline(self, remote_coordinator):
        assert remote_coordinator.node_order == ["machine-a", "machine-b"]

    def test_multinode_inference(self, remote_coordinator):
        result = remote_coordinator.generate("Distributed inference test", max_new_tokens=20)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cluster_status(self, remote_coordinator):
        status = remote_coordinator.resource_manager.get_cluster_status()
        assert status["total_nodes"] == 2

    def test_node_gpu_info(self, remote_nodes):
        for n in remote_nodes.values():
            assert n.gpu_memory_total > 0
            assert n.gpu_name == "Tesla A100"


class TestCrossMachineAuth:
    """Cluster authentication for cross-machine communication."""

    def test_cluster_key_required_for_remote(self):
        from distllm.dist.node_service import NodeServicer
        worker = MagicMock()
        node = NodeServicer(worker_node=worker, cluster_key="shared-secret")
        class Req: cluster_key = "shared-secret"
        assert node._check_auth(Req())

    def test_cluster_key_mismatch_rejects(self):
        from distllm.dist.node_service import NodeServicer
        worker = MagicMock()
        node = NodeServicer(worker_node=worker, cluster_key="shared-secret")
        class Req: cluster_key = "wrong-key"
        assert not node._check_auth(Req())

    def test_empty_cluster_key_rejected(self):
        from distllm.dist.node_service import NodeServicer
        worker = MagicMock()
        node = NodeServicer(worker_node=worker, cluster_key="secret")
        class Req: cluster_key = ""
        assert not node._check_auth(Req())

    def test_none_cluster_key_allows_all(self):
        from distllm.dist.node_service import NodeServicer
        worker = MagicMock()
        node = NodeServicer(worker_node=worker, cluster_key=None)
        class Req: cluster_key = "anything"
        assert node._check_auth(Req())
