"""Regression tests: known partition quality benchmarks."""

from __future__ import annotations

import pytest

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import PartitionOptimizer
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


class TestDPQualityRegression:
    """DP solution quality must meet known baselines."""

    def test_7b_two_a100_improvement_over_equal(self):
        """7B model on 2x A100: DP must beat equal split."""
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(
            hidden_size=4096, intermediate_size=11008,
            num_layers=32, num_heads=32, head_dim=128, vocab_size=32000,
        )
        profiles = {
            "gpu-0": GPUProfile(gpu_id=0, name="A100", total_memory_bytes=80 * 1024**3, compute_tflops=312.0),
            "gpu-1": GPUProfile(gpu_id=1, name="A100", total_memory_bytes=80 * 1024**3, compute_tflops=312.0),
        }
        topo = TopologyGraph(
            node_ids=["gpu-0", "gpu-1"],
            gpu_counts={"gpu-0": 1, "gpu-1": 1},
            links=[LinkProfile(source="gpu-0", target="gpu-1", bandwidth_gbps=25.0)],
        )
        cm = PartitionCostModel(profiles, weights, topo)
        opt = PartitionOptimizer(cm, ["gpu-0", "gpu-1"])
        comparison = opt.compare_strategies(len(weights))

        dp_lat = comparison["dp_minimax"]["max_latency_ms"]
        eq_lat = comparison["equal_split"]["max_latency_ms"]
        assert dp_lat <= eq_lat + 0.01

    def test_13b_heterogeneous_fast_gets_more(self):
        """13B on H100+A100: fast node should get more layers."""
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(
            hidden_size=5120, intermediate_size=13824,
            num_layers=40, num_heads=40, head_dim=128, vocab_size=32000,
        )
        profiles = {
            "fast": GPUProfile(gpu_id=0, name="H100", total_memory_bytes=80 * 1024**3, compute_tflops=989.0),
            "slow": GPUProfile(gpu_id=1, name="A100", total_memory_bytes=80 * 1024**3, compute_tflops=312.0),
        }
        topo = TopologyGraph(
            node_ids=["fast", "slow"],
            gpu_counts={"fast": 1, "slow": 1},
            links=[LinkProfile(source="fast", target="slow", bandwidth_gbps=25.0)],
        )
        cm = PartitionCostModel(profiles, weights, topo)
        opt = PartitionOptimizer(cm, ["fast", "slow"], allow_oom=True)
        sol = opt.solve(len(weights))

        if sol.num_nodes >= 2:
            fast_layers = next(p.end_layer - p.start_layer for p in sol.points if p.node_id == "fast")
            slow_layers = next(p.end_layer - p.start_layer for p in sol.points if p.node_id == "slow")
            assert fast_layers >= slow_layers

    def test_single_node_latency_reasonable(self):
        """Single node latency should be within expected bounds."""
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(
            hidden_size=4096, intermediate_size=11008,
            num_layers=32, num_heads=32, head_dim=128, vocab_size=32000,
        )
        profiles = {"gpu-0": GPUProfile(gpu_id=0, name="A100", total_memory_bytes=80 * 1024**3, compute_tflops=312.0)}
        topo = TopologyGraph(node_ids=["gpu-0"], gpu_counts={"gpu-0": 1})
        cm = PartitionCostModel(profiles, weights, topo)
        opt = PartitionOptimizer(cm, ["gpu-0"])
        sol = opt.solve(len(weights))

        assert sol.max_node_time_ms > 0
        assert sol.max_node_time_ms < 10000  # Should be well under 10s for 7B

    def test_throughput_positive(self):
        """Throughput must always be positive for valid partitions."""
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(num_layers=8, hidden_size=2048)
        profiles = {"gpu-0": GPUProfile(gpu_id=0, name="A100", total_memory_bytes=80 * 1024**3, compute_tflops=312.0)}
        topo = TopologyGraph(node_ids=["gpu-0"], gpu_counts={"gpu-0": 1})
        cm = PartitionCostModel(profiles, weights, topo)
        opt = PartitionOptimizer(cm, ["gpu-0"])
        sol = opt.solve(len(weights))
        assert sol.estimated_throughput_tok_s > 0
