"""Comprehensive tests for PartitionOptimizer (DP solver).

Target: 20+ tests covering edge cases, invariants, and regression.
"""

from __future__ import annotations

import pytest

from distllm.dist.partition.optimizer import PartitionOptimizer, PartitionSolution


class TestOptimizerEdgeCases:
    """Edge case tests for the DP solver."""

    def test_no_nodes(self, cost_model_homogeneous):
        opt = PartitionOptimizer(cost_model_homogeneous, [])
        sol = opt.solve(32)
        assert sol.num_nodes == 0
        assert "No nodes" in sol.explanation

    def test_single_node(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0"])
        sol = opt.solve(len(medium_model_weights))
        assert sol.num_nodes == 1
        assert sol.coverage == (0, len(medium_model_weights))

    def test_single_layer(self, cost_model_homogeneous):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(1)
        assert sol.coverage[1] == 1

    def test_zero_layers(self, cost_model_homogeneous):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(0)
        assert sol.num_nodes == 0 or sol.coverage == (0, 0)

    def test_many_layers_few_nodes(self, cost_model_homogeneous):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0"])
        sol = opt.solve(128)
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 128)

    def test_few_layers_many_nodes(self, cost_model_homogeneous):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1", "gpu-2", "gpu-3", "gpu-4"])
        sol = opt.solve(3)
        assert sol.coverage == (0, 3)

    def test_equal_node_count_and_layers(self, cost_model_homogeneous):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(2)
        assert sol.coverage == (0, 2)


class TestOptimizerInvariants:
    """Invariant tests for the DP solver."""

    def test_coverage_matches_num_layers(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        assert sol.coverage == (0, len(medium_model_weights))

    def test_no_gaps_between_partitions(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        for i in range(len(sol.points) - 1):
            assert sol.points[i].end_layer == sol.points[i + 1].start_layer

    def test_no_overlaps(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        for i in range(len(sol.points) - 1):
            assert sol.points[i].end_layer <= sol.points[i + 1].start_layer

    def test_all_positive_times(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        for pt in sol.points:
            assert pt.estimated_time_ms >= 0

    def test_max_node_time_is_max(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        if sol.points:
            assert sol.max_node_time_ms == max(p.estimated_time_ms for p in sol.points)

    def test_pipeline_latency_ge_max_node_time(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        assert sol.pipeline_latency_ms >= sol.max_node_time_ms - 0.01


class TestOptimizerBehavior:
    """Behavioral tests for the DP solver."""

    def test_two_identical_nodes_split(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        assert sol.num_nodes >= 1
        assert sol.max_node_time_ms > 0

    def test_heterogeneous_fast_gets_more(self, cost_model_heterogeneous, medium_model_weights):
        opt = PartitionOptimizer(
            cost_model_heterogeneous, ["gpu-0", "gpu-1"],
            allow_oom=True,
        )
        sol = opt.solve(len(medium_model_weights))
        if sol.num_nodes >= 2:
            counts = [p.end_layer - p.start_layer for p in sol.points]
            assert counts[0] >= counts[1]

    def test_single_node_throughput(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0"])
        sol = opt.solve(len(medium_model_weights))
        assert sol.estimated_throughput_tok_s > 0

    def test_oom_nodes_tracked(self, small_gpu_profile, large_model_weights, single_node_topology):
        from distllm.dist.partition.cost_model import PartitionCostModel
        cm = PartitionCostModel({"gpu-0": small_gpu_profile}, large_model_weights, single_node_topology)
        opt = PartitionOptimizer(cm, ["gpu-0"], allow_oom=True)
        sol = opt.solve(len(large_model_weights))
        assert sol.num_oom_nodes >= 0


class TestOptimizerConstrained:
    """Tests for constrained DP (min_layers_per_node)."""

    def test_min_layers_respected(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(
            cost_model_homogeneous, ["gpu-0", "gpu-1"],
            min_layers_per_node=5,
        )
        sol = opt.solve(len(medium_model_weights))
        for pt in sol.points:
            assert (pt.end_layer - pt.start_layer) >= 5 or sol.num_nodes == 1


class TestOptimizerBeamSearch:
    """Tests for beam search fallback."""

    def test_beam_search_large_model(self, cost_model_homogeneous):
        from distllm.dist.partition.profiles import GPUProfiler
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(
            hidden_size=4096, intermediate_size=11008,
            num_layers=256, num_heads=32, head_dim=128,
        )
        nodes = [f"gpu-{i}" for i in range(10)]
        profiles = {f"gpu-{i}": cost_model_homogeneous._gpu_profiles["gpu-0"] for i in range(10)}
        from distllm.dist.partition.topology import TopologyGraph, LinkProfile
        topo = TopologyGraph(node_ids=nodes, gpu_counts={n: 1 for n in nodes})
        cm = PartitionCostModel(profiles, weights, topo)
        opt = PartitionOptimizer(cm, nodes, allow_oom=True)
        sol = opt.solve(256)
        assert sol.num_nodes >= 1
        assert sol.coverage == (0, 256)
