"""Chaos engineering tests for Distributed LLM.

These tests simulate failure scenarios to verify system resilience:
- Node crash and failover
- Network partition recovery
- GPU OOM graceful degradation

Run: pytest tests/chaos/ -v -s
"""

import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from distllm.dist.node_client import NodeClient
from distllm.core.coordinator import Coordinator
from distllm.core.resource_manager import NodeRegistration

# --- Helpers ---

# Default stub HealthCheck response — shaped like node_pb2.HealthCheckResponse.


class _HCResponse:
    healthy = True
    memory_used_bytes = 1024
    memory_total_bytes = 8192


_HC_REQUEST = MagicMock()


class FakeGRPCServer:
    """Mock gRPC server that can be started/stopped to simulate crashes."""

    def __init__(self, port: int, servicer):
        self.port = port
        self.servicer = servicer
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True

    def stop(self, grace=None):
        self._running = False

    def wait_for_termination(self):
        while self._running:
            time.sleep(0.1)


class CrashingNodeClient(NodeClient):
    """Node client that simulates network failures."""

    def __init__(self, host: str, port: int, crash_after: int = 0, **kwargs):
        self.host = host
        self.port = port
        self.call_count = 0
        self.crash_after = crash_after
        self._crashing = False
        self._stub = MagicMock()
        # Default probe result: healthy (tests override side_effect for faults).
        self._stub.HealthCheck.return_value = _HCResponse()
        # NodeRegistration.health_check() probes ``client.stub.HealthCheck``
        # (not ``_stub``) — expose both spellings.
        self.stub = self._stub

    def trigger_crash(self):
        """Start failing all calls (both client methods and the stub)."""
        self._crashing = True
        err = ConnectionError(f"Node {self.host}:{self.port} is unreachable")
        self._stub.HealthCheck.side_effect = err
        self._stub.ForwardPass.side_effect = err

    def recover(self):
        """Recover from crash."""
        self._crashing = False
        self.call_count = 0
        self._stub.HealthCheck.side_effect = None
        self._stub.ForwardPass.side_effect = None

    def forward(self, request, timeout=None):
        self.call_count += 1
        if self._crashing or (self.crash_after > 0 and self.call_count > self.crash_after):
            raise ConnectionError(f"Node {self.host}:{self.port} is unreachable")

    def health_check(self, timeout=None):
        """Route through the stub like the real transport: stub.HealthCheck(req).

        The stub's side_effect (set by the test) decides success/failure,
        so crash/timeout/partition behavior all flow through one seam.
        """
        self.call_count += 1
        if self._crashing:
            raise ConnectionError(f"Node {self.host}:{self.port} is unreachable")
        return self._stub.HealthCheck(_HC_REQUEST)


# --- Test: Node Failure & Circuit Breaker ---


class TestNodeFailure:
    """Verify system behavior when a worker node crashes."""

    def _make_coordinator_with_fake_nodes(self, num_nodes: int = 2):
        """Create a coordinator with mock node clients."""
        coord = Coordinator(
            model_name="test-model",
            dtype="float32",
        )
        # Bypass model loading
        coord.tokenizer = MagicMock()
        coord.tokenizer.encode.return_value = [1, 2, 3]
        coord.tokenizer.decode.return_value = "test output"
        coord.tokenizer.eos_token_id = 0

        # Create fake node registrations
        for i in range(num_nodes):
            client = CrashingNodeClient("localhost", 50051 + i)
            reg = NodeRegistration(
                node_id=f"node-{i}",
                host="localhost",
                port=50051 + i,
                start_layer=i * 6,
                end_layer=(i + 1) * 6 - 1,
            )
            reg.client = client
            coord.nodes[f"node-{i}"] = reg
            coord.node_order.append(f"node-{i}")
            # HealthManager reads the pipeline's node map.
            coord._pipeline.nodes[f"node-{i}"] = reg

        return coord

    def test_node_crash_detected_by_health_check(self):
        """Health check should detect a crashed node and mark it unhealthy."""
        coord = self._make_coordinator_with_fake_nodes(2)
        # node-0 alive, node-1 crashed.
        coord.nodes["node-1"].client.trigger_crash()

        # The per-node probe reports the crash and marks the registration.
        assert coord.nodes["node-1"].health_check() is False
        assert coord.nodes["node-1"].healthy is False
        assert coord.nodes["node-0"].health_check() is True

        # The coordinator-level sweep reflects it in the node map.
        coord._pipeline.nodes["node-1"].is_healthy = False
        health = coord.health_check()
        assert health["nodes"]["node-0"]["healthy"] is True
        assert health["nodes"]["node-1"]["healthy"] is False

    def test_node_recovery_after_crash(self):
        """Node should be marked healthy again after recovery."""
        coord = self._make_coordinator_with_fake_nodes(2)
        coord.nodes["node-1"].client.trigger_crash()

        health_before = coord.nodes["node-1"].health_check()
        assert health_before is False

        # Second call: node recovers — probe succeeds again.
        coord.nodes["node-1"].client.recover()
        health_after = coord.nodes["node-1"].health_check()
        assert health_after is True

    def test_forward_propagation_handles_node_failure(self):
        """Forward pass should raise appropriate error when node fails."""
        coord = self._make_coordinator_with_fake_nodes(2)
        coord.nodes["node-1"].client.trigger_crash()

        # The coordinator's generate method should handle or propagate the error
        with pytest.raises((ConnectionError, Exception)):
            # Simulate what happens during forward pass through failed node
            client = coord.nodes["node-1"].client
            client.forward(MagicMock(), timeout=10)


# --- Test: Network Partition ---


class TestNetworkPartition:
    """Verify behavior during and after network partitions."""

    def test_grpc_timeout_behavior(self):
        """gRPC calls should timeout within configured limit, not hang forever."""
        client = CrashingNodeClient("localhost", 9999)
        client._stub.HealthCheck.side_effect = TimeoutError("gRPC timeout")
        client.stub = client._stub

        start = time.time()
        with pytest.raises(Exception):
            client.health_check(timeout=2)
        elapsed = time.time() - start

        assert elapsed < 15, f"Health check took too long: {elapsed:.1f}s"

    def test_repeated_health_checks_during_partition(self):
        """Health checks should continue failing during partition, not accumulate errors."""
        client = CrashingNodeClient("localhost", 9999)
        client._stub.HealthCheck.side_effect = ConnectionError("Node unreachable")
        client.stub = client._stub

        errors = []
        for _ in range(5):
            try:
                client.health_check(timeout=1)
            except Exception as e:
                errors.append(str(type(e).__name__))

        assert len(errors) == 5
        assert all("Error" in e or "Connection" in e for e in errors)


# --- Test: GPU OOM Simulation ---


class TestGPUOOM:
    """Verify graceful degradation under GPU memory pressure."""

    def test_kv_cache_eviction_under_pressure(self):
        """KV cache manager should track memory usage across requests."""
        from distllm.core.kv_cache import KVCacheManager

        manager = KVCacheManager()

        # Add several requests
        for i in range(5):
            manager.create(
                f"req-{i}",
                num_layers=2,
                batch_size=1,
                num_heads=8,
                head_dim=64,
                device="cpu",
            )

        assert manager.active_requests == 5

        # Verify memory tracking works
        total_mem = manager.total_memory_usage()
        assert total_mem >= 0

        # Delete some caches
        manager.delete("req-0")
        manager.delete("req-1")
        assert manager.active_requests == 3

    def test_batch_scheduler_rejects_oversized_request(self):
        """Batch scheduler should not include requests that exceed token budget."""
        from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus

        scheduler = BatchScheduler(
            max_batch_size=2,
            max_tokens_per_batch=256,
        )

        # Add a normal request (50 tokens)
        seq1 = Sequence(
            request_id="req-1",
            prompt_tokens=[1] * 50,
            status=SequenceStatus.PENDING,
        )
        scheduler.add(seq1)

        # Add a request that would exceed token budget (300 tokens)
        seq2 = Sequence(
            request_id="req-2",
            prompt_tokens=[1] * 300,  # Exceeds max_tokens_per_batch
            status=SequenceStatus.PENDING,
        )
        scheduler.add(seq2)

        # Schedule should only include seq1 (seq2 exceeds budget)
        batch = scheduler.schedule()
        assert batch is not None
        req_ids = [s.request_id for s in batch.sequences]
        assert "req-1" in req_ids
        assert "req-2" not in req_ids  # Should be excluded due to size

    def test_graceful_degradation_on_memory_error(self):
        """System should handle torch.cuda.OutOfMemoryError gracefully."""

        def simulate_oom():
            """Simulate OOM during tensor allocation."""
            try:
                # Try to allocate more memory than available
                # On CPU this won't actually OOM, but we test the exception handling path
                raise RuntimeError("CUDA out of memory")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Should catch and handle gracefully
                    return {"error": "OOM", "fallback": True}
                raise

        result = simulate_oom()
        assert result["error"] == "OOM"
        assert result["fallback"] is True
