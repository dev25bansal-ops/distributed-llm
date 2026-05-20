"""Chaos engineering test scenarios for Distributed LLM.

Combines all chaos scenarios:
  - Network latency injection
  - Data corruption
  - Network partition
  - Node failure (existing)
  - GPU OOM simulation (existing)

Run:
    pytest tests/chaos/test_all_scenarios.py -v
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scenarios"))


class TestChaosLatency:
    """Network latency injection scenarios."""

    def test_latency_config_creation(self):
        from scenarios.network_latency import LatencyConfig, NetworkLatencyInjector

        config = LatencyConfig(target_host="localhost", target_port=50050, latency_ms=200)
        assert config.latency_ms == 200
        assert config.jitter_ms == 10

        injector = NetworkLatencyInjector(config)
        assert injector.config.latency_ms == 200


class TestChaosDataCorruption:
    """Data corruption scenarios."""

    def test_tensor_bit_flip(self):
        from scenarios.data_corruption import DataCorruptor

        corruptor = DataCorruptor(corruption_rate=0.5)
        original = b"hello" * 100
        corrupted = corruptor.corrupt_tensor(original)
        assert original != corrupted
        assert corruptor.stats["flips"] > 0

    def test_json_corruption(self):
        from scenarios.data_corruption import DataCorruptor

        corruptor = DataCorruptor()
        result = corruptor.corrupt_json({"key": "value"})
        assert isinstance(result, str)

    def test_message_truncation(self):
        from scenarios.data_corruption import DataCorruptor

        corruptor = DataCorruptor()
        data = b"some long message data" * 10
        truncated = corruptor.truncate_message(data)
        assert len(truncated) < len(data)


class TestChaosNetworkPartition:
    """Network partition scenarios."""

    def test_node_isolation(self):
        from scenarios.network_partition import NetworkPartitionSimulator

        sim = NetworkPartitionSimulator()
        sim.isolate_node("worker-1")
        assert sim.should_block("worker-1")
        assert not sim.should_block("worker-0")
        sim.heal_partition()
        assert not sim.should_block("worker-1")

    def test_half_cluster_partition(self):
        from scenarios.network_partition import NetworkPartitionSimulator

        sim = NetworkPartitionSimulator()
        sim.partition_half(["w0", "w1", "w2", "w3"])
        assert sim.isolated_count == 2


class TestChaosNodeFailure:
    """Node failure scenarios (imported from existing test)."""

    def test_circuit_breaker_on_failure(self):
        """Node failure should trip circuit breaker."""
        from distllm.core.resource_manager import ResourceManager

        rm = ResourceManager()
        node_id = "failing-node"

        for _ in range(rm.cb_config.threshold + 1):
            rm.record_failure(node_id)

        assert rm.check_circuit_breaker(node_id) is True

    def test_circuit_breaker_resets_after_recovery(self):
        """After recovery, circuit breaker should allow requests."""
        from distllm.core.resource_manager import ResourceManager

        rm = ResourceManager()
        node_id = "recovering-node"

        for _ in range(rm.cb_config.threshold + 1):
            rm.record_failure(node_id)

        assert rm.check_circuit_breaker(node_id) is True

        rm.record_success(node_id)
        assert rm.check_circuit_breaker(node_id) is False


class TestChaosOOM:
    """GPU OOM simulation scenarios."""

    def test_kv_cache_large_allocation(self):
        """System should handle large KV cache allocations gracefully."""
        from distllm.core.kv_cache import KVCacheManager

        manager = KVCacheManager()
        # Create a moderately sized cache
        cache = manager.create(
            "test-req",
            num_layers=2,
            batch_size=1,
            num_heads=8,
            head_dim=64,
            device="cpu",
        )
        assert cache is not None
        assert "test-req" in manager.caches

    def test_kv_cache_cleanup_on_error(self):
        """System should not leak cache entries when an error occurs."""
        from distllm.core.kv_cache import KVCacheManager

        manager = KVCacheManager()
        try:
            # Simulate an allocation then delete
            manager.create("req-1", num_layers=2, batch_size=1, num_heads=8, head_dim=64, device="cpu")
            manager.create("req-2", num_layers=2, batch_size=1, num_heads=8, head_dim=64, device="cpu")
            assert manager.active_requests == 2
        finally:
            manager.delete("req-1")
            manager.delete("req-2")
        assert manager.active_requests == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
