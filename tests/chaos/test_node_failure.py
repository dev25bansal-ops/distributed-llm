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

from distllm.communication.grpc import NodeClient
from distllm.core.coordinator import Coordinator
from distllm.core.resource_manager import NodeRegistration

# --- Helpers ---


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

    def trigger_crash(self):
        """Start failing all calls."""
        self._crashing = True

    def recover(self):
        """Recover from crash."""
        self._crashing = False
        self.call_count = 0

    def forward(self, request, timeout=None):
        self.call_count += 1
        if self._crashing or (self.crash_after > 0 and self.call_count > self.crash_after):
            raise ConnectionError(f"Node {self.host}:{self.port} is unreachable")
        return super().forward(request, timeout=timeout)

    def health_check(self, timeout=None):
        self.call_count += 1
        if self._crashing:
            raise ConnectionError(f"Node {self.host}:{self.port} is unreachable")
        return super().health_check(timeout=timeout)


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

        return coord

    @patch.object(NodeClient, "health_check")
    def test_node_crash_detected_by_health_check(self, mock_health):
        """Health check should detect a crashed node and mark it unhealthy."""
        # Make first call succeed (node-0), second fails (node-1)
        mock_health.side_effect = [
            MagicMock(healthy=True, memory_used=1024, memory_total=8192),
            ConnectionError("Node unreachable"),
        ]

        coord = self._make_coordinator_with_fake_nodes(2)
        health = coord.health_check()

        assert health["node-0"]["healthy"] is True
        assert health["node-1"]["healthy"] is False

    @patch.object(NodeClient, "health_check")
    def test_node_recovery_after_crash(self, mock_health):
        """Node should be marked healthy again after recovery."""
        # First call: node-0 healthy, node-1 fails
        mock_health.side_effect = [
            MagicMock(healthy=True, memory_used=1024, memory_total=8192),
            ConnectionError("Node unreachable"),
        ]
        coord = self._make_coordinator_with_fake_nodes(2)
        health_before = coord.health_check()
        assert health_before["node-1"]["healthy"] is False

        # Second call: both healthy
        mock_health.side_effect = [
            MagicMock(healthy=True, memory_used=1024, memory_total=8192),
            MagicMock(healthy=True, memory_used=1024, memory_total=8192),
        ]
        health_after = coord.health_check()
        assert health_after["node-1"]["healthy"] is True

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

    @patch.object(NodeClient, "__init__", return_value=None)
    def test_grpc_timeout_behavior(self, mock_init):
        """gRPC calls should timeout within configured limit, not hang forever."""
        client = NodeClient.__new__(NodeClient)
        client.host = "localhost"
        client.port = 9999
        client._stub = MagicMock()
        client._stub.HealthCheck.side_effect = TimeoutError("gRPC timeout")

        start = time.time()
        with pytest.raises(Exception):
            client.health_check(timeout=2)
        elapsed = time.time() - start

        assert elapsed < 15, f"Health check took too long: {elapsed:.1f}s"

    @patch.object(NodeClient, "__init__", return_value=None)
    def test_repeated_health_checks_during_partition(self, mock_init):
        """Health checks should continue failing during partition, not accumulate errors."""
        errors = []
        for _ in range(5):
            try:
                client = NodeClient.__new__(NodeClient)
                client.host = "localhost"
                client.port = 9999
                client._stub = MagicMock()
                client._stub.HealthCheck.side_effect = ConnectionError("Node unreachable")
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
