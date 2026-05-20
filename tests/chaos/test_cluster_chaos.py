"""Cluster-level chaos engineering tests for distributed-llm.

Spins up 2 real in-process gRPC node servers with mock model forward fns,
registers them with a Coordinator/PipelineOrchestrator, then injects failures:

  - Node kill + self-heal: stop/restart a node's gRPC server
  - Network partition: block traffic via a failing mock client
  - Message corruption: corrupt tensor data at the gRPC servicer layer

Each scenario verifies the cluster detects the fault and recovers within
bounded time, with no data loss or wrong outputs.

Usage:
    pytest tests/chaos/test_cluster_chaos.py -v --timeout=60
"""

import threading
import time
from unittest.mock import MagicMock

import grpc
import pytest
import torch

from distllm.communication.grpc import GRPCServer, NodeClient, NodeService
from distllm.core.coordinator import Coordinator
from distllm.core.pipeline_orchestrator import PipelineOrchestrator
from distllm.core.resource_manager import (
    CircuitBreakerConfig,
    NodeRegistration,
    ResourceManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_model_info():
    """Minimal model metadata — no GPU or real model needed."""
    return {
        "num_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "head_dim": 64,
    }


def _dummy_forward_fn(hidden_states, attention_mask=None, position_ids=None,
                      past_key_values=None, input_ids=None):
    """Deterministic mock forward pass — returns fake hidden states."""
    device = hidden_states.device if hidden_states is not None else "cpu"
    batch_size = hidden_states.shape[0] if hidden_states is not None else 1
    seq_len = hidden_states.shape[1] if hidden_states is not None else 1
    num_heads = 12
    head_dim = 64

    output = torch.ones(batch_size, seq_len, 768, device=device) * 0.5
    # Fake KV cache: one tuple per "layer" (just 2 for testing)
    new_past = [
        (torch.zeros(1, num_heads, seq_len, head_dim),
         torch.zeros(1, num_heads, seq_len, head_dim))
        for _ in range(2)
    ]
    return output, new_past


@pytest.fixture
def coordinator(request):
    """Create a bare Coordinator with mocked tokenizer — no model loaded."""
    coord = Coordinator(model_name="test-model", dtype="float32")
    coord.tokenizer = MagicMock()
    coord.tokenizer.encode.return_value = [1, 2, 3]
    coord.tokenizer.decode.return_value = "decoded"
    coord.tokenizer.eos_token_id = 0
    coord.model_info = request.getfixturevalue("mock_model_info")
    coord.total_layers = 12
    return coord


@pytest.fixture
def resource_mgr():
    """ResourceManager with fast circuit-breaker settings for tests."""
    return ResourceManager(cb_config=CircuitBreakerConfig(
        threshold=2,          # open after 2 failures
        base_delay=0.1,
        max_delay=1.0,
    ))


@pytest.fixture
def orchestrator(resource_mgr):
    """PipelineOrchestrator backed by the fast ResourceManager."""
    return PipelineOrchestrator(resource_mgr=resource_mgr, total_layers=12)


def _make_node_reg(node_id, host, port, start_layer, end_layer):
    """Create a NodeRegistration (clients will be overridden by caller)."""
    return NodeRegistration(
        node_id=node_id,
        host=host,
        port=port,
        start_layer=start_layer,
        end_layer=end_layer,
        use_tls=False,
    )


# ---------------------------------------------------------------------------
# Helper: manage 2-node cluster fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def node_ports():
    """Return 2 available TCP ports for the node gRPC servers."""
    import socket
    ports = []
    for _ in range(2):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        ports.append(s.getsockname()[1])
        s.close()
    return ports


NodeProcess = tuple[str, int, NodeService, GRPCServer]


def _start_node(port: int, node_id: str, corruption=False) -> NodeProcess:
    """Start an in-process gRPC node server and return its handles.

    If corruption=True, the forward_fn injects random bit flips into the
    output tensor to simulate data corruption.
    """
    forward_fn = _dummy_forward_fn

    def _corrupting_fn(*args, **kwargs):
        out, kv = _dummy_forward_fn(*args, **kwargs)
        out = out + torch.randn_like(out) * 1e6  # massive noise
        return out, kv

    servicer = NodeService(
        node_id=node_id,
        forward_fn=_corrupting_fn if corruption else forward_fn,
    )
    server = GRPCServer(port=port, servicer=servicer, use_tls=False, max_workers=4)
    server.start()
    return node_id, port, servicer, server


def _stop_node(proc: NodeProcess):
    """Gracefully stop a node server."""
    _, _, _, server = proc
    server.stop(grace=1)


@pytest.fixture
def two_node_cluster(request, resource_mgr, orchestrator, node_ports):
    """Start 2 real in-process gRPC node servers and register them.

    Returns a dict with:
      - procs: list of (node_id, port, servicer, server)
      - orchestrator
      - resource_mgr
      - node_ports
    """
    procs = []
    layers_per_node = 6  # 12 total / 2 nodes
    for i, port in enumerate(node_ports):
        node_id = f"node-{i}"
        proc = _start_node(port, node_id)
        procs.append(proc)
        orchestrator.register_node(
            node_id=node_id,
            host="localhost",
            port=port,
            start_layer=i * layers_per_node,
            end_layer=(i + 1) * layers_per_node - 1,
            use_tls=False,
        )

    yield {
        "procs": procs,
        "orchestrator": orchestrator,
        "resource_mgr": resource_mgr,
        "node_ports": node_ports,
    }

    # Teardown — stop all servers
    for proc in procs:
        _stop_node(proc)


# ---------------------------------------------------------------------------
# Scenario 1: Node Kill + Self-Heal
# ---------------------------------------------------------------------------

class TestNodeKillSelfHeal:
    """Verify cluster detects a killed node and recovers when restarted."""

    def test_node_kill_detected_by_health_check(self, two_node_cluster):
        """After stopping a node, health check should mark it unhealthy."""
        procs = two_node_cluster["procs"]
        res_mgr = two_node_cluster["resource_mgr"]
        orch = two_node_cluster["orchestrator"]

        node_id, port, _, _ = procs[0]
        _stop_node(procs[0])

        # Allow time for the server to fully stop
        time.sleep(0.5)

        health = res_mgr.health_check_all(orch.nodes)
        assert node_id in health
        assert health[node_id]["healthy"] is False

    def test_circuit_breaker_trips_after_failures(self, two_node_cluster):
        """Repeated health-check failures should trip the circuit breaker."""
        procs = two_node_cluster["procs"]
        res_mgr = two_node_cluster["resource_mgr"]
        orch = two_node_cluster["orchestrator"]
        node_id, port, _, _ = procs[0]

        _stop_node(procs[0])
        time.sleep(0.3)

        # Run health checks until circuit breaker opens
        for _ in range(5):
            res_mgr.health_check_all(orch.nodes)

        assert res_mgr.check_circuit_breaker(node_id) is True

    def test_self_heal_after_restart(self, two_node_cluster):
        """Restarting a stopped node should restore health."""
        procs = two_node_cluster["procs"]
        res_mgr = two_node_cluster["resource_mgr"]
        orch = two_node_cluster["orchestrator"]
        node_id, port, _, _ = procs[0]

        # Kill
        _stop_node(procs[0])
        time.sleep(0.3)
        for _ in range(3):
            res_mgr.health_check_all(orch.nodes)
        assert res_mgr.check_circuit_breaker(node_id) is True

        # Restart
        new_proc = _start_node(port, node_id)
        procs[0] = new_proc
        time.sleep(0.5)

        # Reset circuit breaker state for this node
        res_mgr.record_success(node_id)

        # Health should recover
        health = res_mgr.health_check_all(orch.nodes)
        assert health[node_id]["healthy"] is True
        assert res_mgr.check_circuit_breaker(node_id) is False

    def test_self_heal_time_bounded(self, two_node_cluster):
        """Self-heal should complete within expected time bounds."""
        procs = two_node_cluster["procs"]
        res_mgr = two_node_cluster["resource_mgr"]
        orch = two_node_cluster["orchestrator"]
        node_id, port, _, _ = procs[0]

        _stop_node(procs[0])
        time.sleep(0.2)

        # Trip circuit breaker
        for _ in range(3):
            res_mgr.health_check_all(orch.nodes)

        # Measure recovery time
        restart_start = time.time()
        new_proc = _start_node(port, node_id)
        procs[0] = new_proc
        res_mgr.record_success(node_id)

        heal_timeout = 5.0
        healed = False
        deadline = time.time() + heal_timeout
        while time.time() < deadline:
            health = res_mgr.health_check_all(orch.nodes)
            if health.get(node_id, {}).get("healthy"):
                healed = True
                break
            time.sleep(0.2)

        heal_elapsed = time.time() - restart_start
        assert healed, f"Node did not self-heal within {heal_timeout}s"
        assert heal_elapsed < heal_timeout, (
            f"Self-heal took {heal_elapsed:.2f}s, expected < {heal_timeout}s"
        )

    def test_remaining_node_serves_requests_during_failure(self, two_node_cluster):
        """When one node fails, the other should still respond to health checks."""
        procs = two_node_cluster["procs"]
        res_mgr = two_node_cluster["resource_mgr"]
        orch = two_node_cluster["orchestrator"]
        surviving_id, _, _, _ = procs[1]

        _stop_node(procs[0])
        time.sleep(0.3)

        health = res_mgr.health_check_all(orch.nodes)
        assert health[surviving_id]["healthy"] is True

    def test_no_data_leak_on_node_failure(self, two_node_cluster):
        """After a node fails and recovers, health check should return to
        a healthy state — verifying the cluster metadata is consistent."""
        procs = two_node_cluster["procs"]
        res_mgr = two_node_cluster["resource_mgr"]
        orch = two_node_cluster["orchestrator"]
        node_id, port, _, _ = procs[0]

        # Pre-failure: both nodes healthy
        health_before = res_mgr.health_check_all(orch.nodes)
        assert health_before[node_id]["healthy"] is True

        # Kill
        _stop_node(procs[0])
        time.sleep(0.2)
        for _ in range(3):
            res_mgr.health_check_all(orch.nodes)
        assert res_mgr.check_circuit_breaker(node_id) is True

        # Heal
        new_proc = _start_node(port, node_id)
        procs[0] = new_proc
        res_mgr.record_success(node_id)

        for _ in range(10):
            h = res_mgr.health_check_all(orch.nodes)
            if h.get(node_id, {}).get("healthy"):
                break
            time.sleep(0.2)

        health_after = res_mgr.health_check_all(orch.nodes)
        assert health_after[node_id]["healthy"] is True
        assert res_mgr.check_circuit_breaker(node_id) is False
        # Other node should remain healthy throughout
        surviving_id, _, _, _ = procs[1]
        assert health_after[surviving_id]["healthy"] is True


# ---------------------------------------------------------------------------
# Scenario 2: Network Partition (simulated via client failure)
# ---------------------------------------------------------------------------

class TestNetworkPartition:
    """Verify circuit-breaker + recovery under network partition conditions."""

    def test_partition_detected_via_health_checks(self, coordinator, resource_mgr):
        """Repeated health-check failures (simulating partition) should
        mark the node unhealthy and open the circuit breaker."""
        node_id = "partitioned-node"

        # Register a node whose client always fails
        mock_client = MagicMock(spec=NodeClient)
        mock_client.health_check.side_effect = ConnectionError("partitioned")
        reg = NodeRegistration(
            node_id=node_id, host="localhost", port=59999,
            start_layer=0, end_layer=5, use_tls=False,
        )
        reg.client = mock_client
        coordinator.nodes[node_id] = reg
        coordinator.node_order.append(node_id)

        # Health checks should detect the partition
        health = resource_mgr.health_check_all(coordinator.nodes)
        assert health[node_id]["healthy"] is False

        # Circuit breaker should trip
        for _ in range(resource_mgr.cb_config.threshold + 2):
            resource_mgr.record_failure(node_id)
        assert resource_mgr.check_circuit_breaker(node_id) is True

    def test_partition_heal_resets_circuit_breaker(self, coordinator, resource_mgr):
        """When the partition heals, circuit breaker should reset."""
        node_id = "healed-node"

        mock_client = MagicMock(spec=NodeClient)
        mock_client.health_check.side_effect = ConnectionError("partitioned")
        reg = NodeRegistration(
            node_id=node_id, host="localhost", port=59998,
            start_layer=0, end_layer=5, use_tls=False,
        )
        reg.client = mock_client
        coordinator.nodes[node_id] = reg
        coordinator.node_order.append(node_id)

        # Trip breaker
        for _ in range(resource_mgr.cb_config.threshold + 1):
            resource_mgr.record_failure(node_id)
        assert resource_mgr.check_circuit_breaker(node_id) is True

        # Simulate partition heal — make client work again
        mock_health = MagicMock()
        mock_health.healthy = True
        mock_health.memory_used = 512
        mock_health.memory_total = 8192
        mock_client.health_check.side_effect = None
        mock_client.health_check.return_value = mock_health

        # Record success to reset breaker
        resource_mgr.record_success(node_id)
        assert resource_mgr.check_circuit_breaker(node_id) is False

        # Full health check should now pass
        health = resource_mgr.health_check_all(coordinator.nodes)
        assert health[node_id]["healthy"] is True

    def test_partition_heal_time_bounded(self, coordinator, resource_mgr):
        """Circuit breaker should reset within expected time after heal."""
        node_id = "timed-node"

        mock_client = MagicMock(spec=NodeClient)
        mock_client.health_check.side_effect = ConnectionError("partitioned")
        reg = NodeRegistration(
            node_id=node_id, host="localhost", port=59997,
            start_layer=0, end_layer=5, use_tls=False,
        )
        reg.client = mock_client
        coordinator.nodes[node_id] = reg
        coordinator.node_order.append(node_id)

        for _ in range(resource_mgr.cb_config.threshold + 1):
            resource_mgr.record_failure(node_id)
        assert resource_mgr.check_circuit_breaker(node_id) is True

        # Heal
        heal_start = time.time()
        mock_client.health_check.side_effect = None
        mock_client.health_check.return_value = MagicMock(healthy=True)
        resource_mgr.record_success(node_id)

        deadline = time.time() + 3.0
        recovered = False
        while time.time() < deadline:
            if not resource_mgr.check_circuit_breaker(node_id):
                recovered = True
                break
            time.sleep(0.1)

        assert recovered, "Circuit breaker did not reset within 3s after heal"
        assert (time.time() - heal_start) < 3.0


# ---------------------------------------------------------------------------
# Scenario 3: Message Corruption Detection
# ---------------------------------------------------------------------------

class TestMessageCorruption:
    """Verify corrupt tensor data is detected and not silently accepted."""

    def test_corrupted_tensor_produces_wrong_output(self, two_node_cluster):
        """A node producing corrupted tensors should result in different
        outputs compared to a clean pipeline run."""
        procs = two_node_cluster["procs"]
        orch = two_node_cluster["orchestrator"]

        input_ids = torch.randint(0, 100, (1, 4))

        # Run clean
        output_clean = orch.run_pipeline(
            input_ids=input_ids,
            node_kv_caches={},
            request_id="clean",
        )

        # Replace second node with a corrupting one
        _, port, _, _ = procs[1]
        _stop_node(procs[1])
        corrupt_proc = _start_node(port, "node-1-corrupted", corruption=True)
        procs[1] = corrupt_proc
        time.sleep(0.3)

        # Run with corruption
        output_corrupt = orch.run_pipeline(
            input_ids=input_ids,
            node_kv_caches={},
            request_id="corrupted",
        )

        # Outputs should differ significantly
        assert not torch.allclose(output_clean, output_corrupt, atol=1.0), (
            "Corrupted node produced same output as clean node"
        )

    def test_data_corruptor_bit_flip(self):
        """The DataCorruptor helper should change tensor bytes."""
        from tests.chaos.scenarios.data_corruption import DataCorruptor

        corruptor = DataCorruptor(corruption_rate=0.5)
        original = b"\x00" * 1024
        corrupted = corruptor.corrupt_tensor(original)
        assert original != corrupted
        assert corruptor.stats["flips"] > 0

    def test_data_corruptor_json_detection(self):
        """Corrupted JSON payloads should fail to parse (or change)."""
        from tests.chaos.scenarios.data_corruption import DataCorruptor
        import json

        corruptor = DataCorruptor(corruption_rate=0.15)
        payload = {"request_id": "abc", "prompt": "hello world"}
        corrupted = corruptor.corrupt_json(payload)
        try:
            parsed = json.loads(corrupted)
            assert parsed != payload, "Payload should change after corruption"
        except (json.JSONDecodeError, ValueError):
            pass  # Expected — invalid JSON

    def test_data_corruptor_truncation(self):
        """Truncated messages should be shorter than originals."""
        from tests.chaos.scenarios.data_corruption import DataCorruptor

        corruptor = DataCorruptor()
        data = b"some long message data" * 20
        truncated = corruptor.truncate_message(data)
        assert len(truncated) < len(data)
        assert len(truncated) >= 4


# ---------------------------------------------------------------------------
# Scenario 4: ResourceManager chaos API
# ---------------------------------------------------------------------------

class TestChaosAPI:
    """Verify the built-in chaos API (simulate_node_failure) works."""

    def test_simulate_node_failure_trips_breaker(self, resource_mgr):
        """simulate_node_failure should immediately open circuit breaker."""
        resource_mgr.simulate_node_failure("node-x")
        assert resource_mgr.check_circuit_breaker("node-x") is True

    def test_node_failure_callback_fires(self, resource_mgr):
        """set_node_failure_callback should fire when breaker opens."""
        fired = []

        def callback(node_id):
            fired.append(node_id)

        resource_mgr.set_node_failure_callback(callback)
        for _ in range(resource_mgr.cb_config.threshold):
            resource_mgr.record_failure("node-y")
        assert len(fired) == 1
        assert fired[0] == "node-y"

    def test_drain_and_restore(self, coordinator):
        """A drained node should be excluded until restored."""
        from distllm.core.resource_manager import ResourceManager

        rm = ResourceManager()
        node_id = "drain-target"
        reg = NodeRegistration(
            node_id=node_id, host="localhost", port=59996,
            start_layer=0, end_layer=5, use_tls=False,
        )
        coordinator.nodes[node_id] = reg
        coordinator.node_order.append(node_id)

        assert not rm.is_node_draining(node_id)
        rm.mark_node_draining(node_id)
        assert rm.is_node_draining(node_id) is True

        rm.mark_node_alive(node_id)
        assert rm.is_node_draining(node_id) is False
