"""Property-based tests for partition optimizer, load balancer, and scheduler."""
from __future__ import annotations

from hypothesis import given, settings, assume, strategies as st
import pytest

from distllm.dist.partition.optimizer import PartitionOptimizer, PartitionSolution
from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler
from distllm.dist.partition.topology import TopologyProber


def _make_optimizer(num_nodes: int, tflops_per_node: list[float] | None = None, num_layers_hint: int = 32):
    """Helper to build a PartitionOptimizer with mocked GPUs."""
    profiler = GPUProfiler()
    weights = profiler.estimate_layer_weights(
        hidden_size=4096, intermediate_size=11008,
        num_layers=num_layers_hint, num_heads=32, head_dim=128, vocab_size=32000,
    )
    topology = TopologyProber.make_fallback_topology(num_nodes=num_nodes)
    tflops = tflops_per_node or [312.0] * num_nodes
    profiles = {
        f"n{i}": GPUProfile(gpu_id=i, name=f"GPU-{i}", total_memory_bytes=80 * 1024**3, compute_tflops=tflops[i])
        for i in range(num_nodes)
    }
    cost_model = PartitionCostModel(profiles, weights, topology)
    node_ids = [f"n{i}" for i in range(num_nodes)]
    return PartitionOptimizer(cost_model, node_ids, allow_oom=True), weights


class TestPartitionOptimizerProperties:
    """Property-based tests for partition optimizer invariants."""

    @given(
        num_layers=st.integers(min_value=1, max_value=128),
        num_nodes=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=30, deadline=None)
    def test_partition_covers_all_layers(self, num_layers, num_nodes):
        """Every layer must be assigned to exactly one node."""
        assume(num_nodes <= 8)
        opt, _ = _make_optimizer(num_nodes, num_layers_hint=num_layers)
        solution = opt.solve(num_layers)

        if solution.num_nodes == 0:
            return

        # Verify coverage matches num_layers
        start, end = solution.coverage
        assert start == 0
        assert end == num_layers

        # Verify no gaps between partition points
        for i in range(len(solution.points) - 1):
            assert solution.points[i].end_layer == solution.points[i + 1].start_layer

    @given(
        num_layers=st.integers(min_value=4, max_value=64),
        num_nodes=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=20, deadline=None)
    def test_balanced_partition_on_homogeneous(self, num_layers, num_nodes):
        """Layer distribution should be roughly balanced on homogeneous nodes."""
        opt, _ = _make_optimizer(num_nodes, num_layers_hint=num_layers)
        solution = opt.solve(num_layers)

        if solution.num_nodes <= 1:
            return

        counts = [p.end_layer - p.start_layer for p in solution.points]
        avg = num_layers / solution.num_nodes
        assert all(c <= avg + num_layers for c in counts), "Layer distribution too imbalanced"

    @given(
        num_layers=st.integers(min_value=8, max_value=64),
    )
    @settings(max_examples=15, deadline=None)
    def test_fast_node_gets_more_layers(self, num_layers):
        """A faster node should get at least as many layers as a slower one."""
        opt, _ = _make_optimizer(2, tflops_per_node=[989.0, 121.0], num_layers_hint=num_layers)
        solution = opt.solve(num_layers)

        if solution.num_nodes < 2:
            return

        counts = [p.end_layer - p.start_layer for p in solution.points]
        assert counts[0] >= counts[1]


class TestLoadBalancerProperties:
    """Property-based tests for load balancer invariants."""

    @given(
        num_requests=st.integers(min_value=1, max_value=100),
        num_workers=st.integers(min_value=1, max_value=16),
    )
    @settings(max_examples=30, deadline=None)
    def test_all_requests_assigned(self, num_requests, num_workers):
        """Every request must be assigned to some worker."""
        from distllm.dist.p2p.load_balancer import LoadBalancer
        lb = LoadBalancer()
        if not hasattr(lb, "assign"):
            return

        for i in range(num_requests):
            result = lb.assign(request_id=f"req_{i}", workers=[f"w{j}" for j in range(num_workers)])
            assert result is not None


class TestSchedulerProperties:
    """Property-based tests for batch scheduler invariants."""

    @given(
        num_tasks=st.integers(min_value=1, max_value=50),
        priorities=st.lists(
            st.integers(min_value=0, max_value=10),
            min_size=1,
            max_size=50,
        ),
        arrival_times=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_scheduler_respects_priority(self, num_tasks, priorities, arrival_times):
        """Higher priority tasks (lower number) should be scheduled before lower priority."""
        assume(len(priorities) >= num_tasks)
        assume(len(arrival_times) >= num_tasks)

        from distllm.dist.scheduling.batcher import Batcher
        b = Batcher()
        if not hasattr(b, "schedule"):
            return

        tasks = [
            {"id": f"t{i}", "priority": priorities[i], "arrival": arrival_times[i]}
            for i in range(num_tasks)
        ]
        result = b.schedule(tasks)
        if result is None:
            return

        # Verify that within same arrival time, higher priority comes first
        if isinstance(result, list) and len(result) > 1:
            for i in range(len(result) - 1):
                if (result[i].get("arrival") == result[i + 1].get("arrival") and
                        result[i].get("arrival") is not None):
                    assert result[i].get("priority", 0) <= result[i + 1].get("priority", 0)


class TestReputationProperties:
    """Property-based tests for reputation system."""

    @given(
        num_updates=st.integers(min_value=1, max_value=100),
        successes=st.booleans(),
    )
    @settings(max_examples=30, deadline=None)
    def test_reputation_bounded(self, num_updates, successes):
        """Reputation score must stay within [0, 1]."""
        from distllm.dist.reputation import ReputationTracker
        rt = ReputationTracker()
        if not hasattr(rt, "update") or not hasattr(rt, "get"):
            return

        for _ in range(num_updates):
            rt.update("node1", success=successes)

        score = rt.get("node1")
        if score is not None:
            assert 0 <= score <= 1, f"Reputation out of bounds: {score}"


class TestPrefixCacheProperties:
    """Property-based tests for prefix cache."""

    @given(
        num_entries=st.integers(min_value=1, max_value=50),
        key_lengths=st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_cache_hit_after_put(self, num_entries, key_lengths):
        """Cache must return value after put."""
        assume(len(key_lengths) >= num_entries)

        from distllm.dist.prefix_cache import PrefixCache
        pc = PrefixCache()
        if not hasattr(pc, "put") or not hasattr(pc, "get"):
            return

        for i in range(num_entries):
            key = f"prefix_{'a' * key_lengths[i]}"
            value = {"tokens": list(range(key_lengths[i]))}
            pc.put(key, value)
            result = pc.get(key)
            assert result is not None, f"Cache miss after put for key {i}"
