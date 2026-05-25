"""Tests for expert capacity management."""

import pytest
import torch

from distllm.core.moe_capacity import (
    CapacityConfig,
    ExpertCapacityManager,
)


class TestCapacityConfig:
    def test_defaults(self):
        config = CapacityConfig()
        assert config.capacity_factor == 1.25
        assert config.min_capacity == 4
        assert config.max_overflow_ratio == 0.1
        assert config.drop_on_overflow is True
        assert config.use_overflow_routing is True

    def test_custom(self):
        config = CapacityConfig(
            capacity_factor=2.0,
            min_capacity=8,
            drop_on_overflow=False,
        )
        assert config.capacity_factor == 2.0
        assert config.min_capacity == 8
        assert config.drop_on_overflow is False


class TestExpertCapacityManager:
    @pytest.fixture
    def manager(self):
        return ExpertCapacityManager(num_experts=4)

    def test_init(self, manager):
        stats = manager.stats()
        assert stats["total_tokens"] == 0
        assert stats["overflow_tokens"] == 0
        assert stats["dropped_tokens"] == 0
        assert stats["buffer_size"] == 0

    def test_compute_capacity_default(self, manager):
        capacities = manager.compute_capacity(num_tokens=32)
        assert len(capacities) == 4
        # avg = 32/4 = 8, capacity = max(int(8*1.25), 4) = 10
        for eid in range(4):
            assert capacities[eid] == 10

    def test_compute_capacity_min(self, manager):
        capacities = manager.compute_capacity(num_tokens=4)
        for eid in range(4):
            assert capacities[eid] >= manager._config.min_capacity

    def test_check_overflow_no_overflow(self, manager):
        manager.compute_capacity(num_tokens=16)
        overflow = manager.check_overflow({0: 4, 1: 4, 2: 4, 3: 4})
        assert all(v == 0 for v in overflow.values())

    def test_check_overflow_detected(self, manager):
        manager.compute_capacity(num_tokens=16)
        # avg = 4, capacity = max(int(4*1.25),4) = 5
        overflow = manager.check_overflow({0: 10, 1: 4, 2: 4, 3: 4})
        assert overflow[0] == 5  # 10 - 5
        assert overflow[1] == 0

    def test_handle_overflow_with_routing(self, manager):
        manager._config.use_overflow_routing = True
        manager._config.overflow_fallback_ratio = 1.0
        manager.compute_capacity(num_tokens=16)
        # capacity = 5 per expert
        token_map = {
            0: [(i, 1.0) for i in range(10)],
            1: [(i, 1.0) for i in range(3)],
            2: [(i, 1.0) for i in range(3)],
            3: [(i, 1.0) for i in range(3)],
        }
        overflow = manager.check_overflow({0: 10, 1: 3, 2: 3, 3: 3})
        assert 0 in overflow

        accepted, overflow_map = manager.handle_overflow(overflow, token_map)
        assert 0 in accepted
        assert len(accepted[0]) <= 5
        assert manager.stats()["rerouted_tokens"] > 0

    def test_handle_overflow_with_drop(self, manager):
        manager._config.use_overflow_routing = False
        manager._config.drop_on_overflow = True
        manager.compute_capacity(num_tokens=16)

        token_map = {0: [(i, 1.0) for i in range(10)]}
        overflow = manager.check_overflow({0: 10})
        accepted, overflow_map = manager.handle_overflow(overflow, token_map)
        assert manager.stats()["dropped_tokens"] > 0

    def test_handle_overflow_with_buffering(self, manager):
        manager._config.use_overflow_routing = False
        manager._config.drop_on_overflow = False
        manager.compute_capacity(num_tokens=16)

        token_map = {0: [(i, 1.0) for i in range(10)]}
        overflow = manager.check_overflow({0: 10})
        accepted, overflow_map = manager.handle_overflow(overflow, token_map)
        assert manager.stats()["buffered_tokens"] > 0

    def test_drain_buffer(self, manager):
        manager._config.drop_on_overflow = False
        manager.compute_capacity(num_tokens=16)
        token_map = {0: [(i, 1.0) for i in range(10)]}
        overflow = manager.check_overflow({0: 10})
        manager.handle_overflow(overflow, token_map)
        buffer = manager.drain_buffer()
        assert len(buffer) > 0
        assert manager.stats()["buffer_size"] == 0

    def test_record_usage(self, manager):
        manager.record_usage(0, 5)
        manager.record_usage(1, 3)
        assert manager._expert_usage[0] == 5
        assert manager._expert_usage[1] == 3

    def test_get_available_capacity(self, manager):
        manager.compute_capacity(num_tokens=16)
        manager.record_usage(0, 3)
        avail = manager.get_available_capacity(0)
        assert avail > 0

    def test_reset_usage(self, manager):
        manager.record_usage(0, 5)
        manager.reset_usage()
        assert len(manager._expert_usage) == 0

    def test_compute_overflow_loss_no_overflow(self, manager):
        manager.compute_capacity(num_tokens=16)
        manager.record_usage(0, 4)
        routing_probs = torch.randn(8, 4).softmax(dim=-1)
        expert_indices = torch.randint(0, 4, (8, 2))
        loss = manager.compute_overflow_loss(routing_probs, expert_indices)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_stats(self, manager):
        stats = manager.stats()
        assert "total_tokens" in stats
        assert "overflow_tokens" in stats
        assert "capacity_factor" in stats
        assert "drop_on_overflow" in stats
        assert "overflow_routing" in stats

    def test_large_capacity(self):
        manager = ExpertCapacityManager(num_experts=64)
        capacities = manager.compute_capacity(num_tokens=1024)
        assert len(capacities) == 64
        for eid in range(64):
            assert capacities[eid] > 0

    def test_overflow_multiple_experts(self, manager):
        manager.compute_capacity(num_tokens=4)
        # capacity = max(int(1*1.25), 4) = 4
        overflow = manager.check_overflow({0: 10, 1: 8, 2: 4, 3: 4})
        assert overflow[0] > 0  # 10 - 4 = 6
        assert overflow[1] > 0  # 8 - 4 = 4
        assert overflow[2] == 0
        assert overflow[3] == 0

    def test_overflow_routing_to_multiple_targets(self, manager):
        manager._config.use_overflow_routing = True
        manager._config.overflow_fallback_ratio = 1.0
        manager.compute_capacity(num_tokens=32)
        # Only expert 0 is overloaded, rest have spare capacity
        token_map = {0: [(i, 1.0) for i in range(20)]}
        overflow = manager.check_overflow({0: 20})
        accepted, overflow_map = manager.handle_overflow(overflow, token_map)
        # Overflow should be distributed across experts 1,2,3
        total_rerouted = manager.stats()["rerouted_tokens"]
        assert total_rerouted > 0

    def test_drop_when_all_experts_full(self, manager):
        manager._config.use_overflow_routing = True
        manager.compute_capacity(num_tokens=4)
        # All experts at capacity, overflow has nowhere to go
        for eid in range(4):
            manager.record_usage(eid, 5)
        token_map = {0: [(i, 1.0) for i in range(10)]}
        overflow = manager.check_overflow({0: 10})
        accepted, overflow_map = manager.handle_overflow(overflow, token_map)
        dropped = manager.stats()["dropped_tokens"]
        assert dropped > 0
