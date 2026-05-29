"""Tests for the hardware-aware auto-partition package.

Tests: GPUProfiler, TopologyProber, PartitionCostModel,
PartitionOptimizer, HardwareAwarePartitioner.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.optimizer import (
    PartitionOptimizer,
    PartitionPoint,
    PartitionSolution,
)
from distllm.dist.partition.partitioner import HardwareAwarePartitioner
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph, TopologyProber


# ─── GPUProfile / LayerWeights ────────────────────────────────────────────────


class TestGPUProfile:
    def test_defaults(self):
        p = GPUProfile(gpu_id=0, name="TestGPU")
        assert p.total_memory_bytes == 0
        assert p.compute_tflops == 0.0
        assert p.memory_bandwidth_gbps == 0.0
        assert p.free_memory_bytes == 0

    def test_custom(self):
        p = GPUProfile(
            gpu_id=1, name="H100", total_memory_bytes=80 * 1024**3,
            compute_tflops=989.0, memory_bandwidth_gbps=3350.0,
        )
        assert p.compute_tflops == 989.0
        assert p.total_memory_bytes == 80 * 1024**3


class TestLayerWeights:
    def test_total_memory(self):
        lw = LayerWeights(layer_id=0, layer_type="embed", weight_memory_bytes=100, activation_memory_bytes=50)
        assert lw.total_memory_bytes == 150

    def test_defaults(self):
        lw = LayerWeights(layer_id=5)
        assert lw.layer_type == "transformer"


# ─── GPUProfiler ──────────────────────────────────────────────────────────────


class TestGPUProfiler:
    def test_estimate_layer_weights(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(
            hidden_size=1024, intermediate_size=4096,
            num_layers=4, num_heads=8, head_dim=64,
            vocab_size=10000,
        )
        # layers: embed(0) + 4 transformer + lm_head(5) = 6
        assert len(weights) == 6
        assert weights[0].layer_type == "embed"
        assert weights[1].layer_type == "transformer"
        assert weights[-1].layer_type == "lm_head"

    def test_weight_memory_scales_with_hidden_size(self):
        profiler = GPUProfiler()
        small = profiler.estimate_layer_weights(hidden_size=1024, num_layers=2)
        large = profiler.estimate_layer_weights(hidden_size=4096, num_layers=2)
        # Larger hidden -> more weight memory (scales roughly 4-16x)
        assert large[1].weight_memory_bytes > small[1].weight_memory_bytes * 4

    def test_layer_ids_sequential(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(hidden_size=512, num_layers=3)
        for i, w in enumerate(weights):
            assert w.layer_id == i

    def test_flops_per_token_positive(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights()
        for w in weights:
            if w.layer_type == "transformer":
                assert w.flops_per_token > 0

    def test_kv_cache_nonzero_for_transformer(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(hidden_size=1024, num_heads=8, head_dim=64)
        for w in weights:
            if w.layer_type == "transformer":
                assert w.kv_cache_bytes_per_token > 0
            else:
                assert w.kv_cache_bytes_per_token == 0

    def test_first_last_layer_extra_norm_memory(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(hidden_size=4096, num_layers=4)
        first = weights[1]
        middle = weights[2]
        last = weights[4]
        # first/last have extra norm weights
        assert first.weight_memory_bytes > middle.weight_memory_bytes
        assert last.weight_memory_bytes > middle.weight_memory_bytes

    def test_embed_memory_scales_with_vocab(self):
        profiler = GPUProfiler()
        small = profiler.estimate_layer_weights(vocab_size=1000)
        large = profiler.estimate_layer_weights(vocab_size=32000)
        assert large[0].weight_memory_bytes > small[0].weight_memory_bytes

    @patch.object(GPUProfiler, "_device_count", return_value=2)
    @patch.object(GPUProfiler, "_get_device_name", side_effect=["MockGPU", "MockGPU2"])
    @patch.object(GPUProfiler, "_get_memory_info", return_value=(16 * 1024**3, 8 * 1024**3))
    def test_profile_all_gpus(self, mock_mem, mock_name, mock_count):
        profiler = GPUProfiler()
        profiles = profiler.profile_all_gpus()
        assert len(profiles) == 2
        assert profiles[0].name == "MockGPU"

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    def test_profile_all_gpus_empty_fallback(self, mock_count):
        profiler = GPUProfiler()
        profiles = profiler.profile_all_gpus()
        assert len(profiles) == 1
        assert profiles[0].name == "cpu_fallback"


# ─── LinkProfile / TopologyGraph / TopologyProber ─────────────────────────────


class TestLinkProfile:
    def test_defaults(self):
        link = LinkProfile(source="a", target="b")
        assert link.bandwidth_gbps == 12.5
        assert link.latency_us == 100.0
        assert not link.is_nvlink

    def test_custom(self):
        link = LinkProfile(source="a", target="b", bandwidth_gbps=600.0, latency_us=5.0, is_nvlink=True)
        assert link.bandwidth_gbps == 600.0
        assert link.is_nvlink


class TestTopologyGraph:
    def test_get_bandwidth(self):
        graph = TopologyGraph(
            node_ids=["a", "b"],
            links=[LinkProfile(source="a", target="b", bandwidth_gbps=50.0)],
        )
        assert graph.get_bandwidth("a", "b") == 50.0
        assert graph.get_bandwidth("a", "c") == 1.0  # fallback

    def test_get_latency(self):
        graph = TopologyGraph(
            node_ids=["a", "b"],
            links=[LinkProfile(source="a", target="b", latency_us=200.0)],
        )
        assert graph.get_latency("a", "b") == 200.0
        assert graph.get_latency("unknown", "other") == 1000.0  # fallback

    def test_total_gpus(self):
        graph = TopologyGraph(gpu_counts={"a": 4, "b": 8})
        assert graph.total_gpus() == 12

    def test_to_dict(self):
        graph = TopologyGraph(node_ids=["x"], gpu_counts={"x": 2}, node_hostnames={"x": "10.0.0.1"})
        d = graph.to_dict()
        assert d["total_gpus"] == 2
        assert len(d["nodes"]) == 1


class TestTopologyProber:
    @patch.object(TopologyProber, "_detect_inter_node_bandwidth", return_value=25.0)
    @patch.object(TopologyProber, "_measure_latency_async", return_value=300.0)
    async def test_probe_two_nodes(self, mock_latency, mock_bw):
        prober = TopologyProber()
        graph = await prober.probe(
            node_ids=["node-0", "node-1"],
            hostnames={"node-0": "10.0.0.1", "node-1": "10.0.0.2"},
            gpu_counts={"node-0": 1, "node-1": 2},
        )
        assert len(graph.node_ids) == 2
        assert len(graph.links) == 1
        assert graph.links[0].bandwidth_gbps == 25.0
        assert graph.gpu_counts["node-0"] == 1

    @patch.object(TopologyProber, "_detect_nvlink", return_value=True)
    def test_probe_local_topology(self, mock_nvlink):
        prober = TopologyProber()
        links = prober.probe_local_topology(num_gpus=4)
        # 4 GPUs -> C(4,2) = 6 links
        assert len(links) == 6
        assert all(l.is_nvlink for l in links)

    def test_fallback_topology(self):
        graph = TopologyProber.make_fallback_topology(num_nodes=3, gpus_per_node=2)
        assert len(graph.node_ids) == 3
        assert graph.total_gpus() == 6
        assert len(graph.links) == 3  # C(3,2)

    def test_fallback_topology_links(self):
        graph = TopologyProber.make_fallback_topology(num_nodes=2)
        assert len(graph.links) == 1
        assert graph.links[0].bandwidth_gbps == 12.5


# ─── NodeCost / PartitionCostModel ────────────────────────────────────────────


class TestNodeCost:
    def test_memory_utilization(self):
        cost = NodeCost(node_id="n0", start_layer=0, end_layer=10, memory_bytes=500, memory_available_bytes=1000)
        assert cost.memory_utilization == 0.5

    def test_memory_utilization_zero_available(self):
        cost = NodeCost(node_id="n0", start_layer=0, end_layer=0)
        assert cost.memory_utilization == 0.0


class TestPartitionCostModel:
    @pytest.fixture
    def setup(self):
        gpu_profiles = {
            "node-0": GPUProfile(gpu_id=0, name="H100", total_memory_bytes=80 * 1024**3, compute_tflops=989.0),
        }
        profiler = GPUProfiler()
        layer_weights = profiler.estimate_layer_weights(
            hidden_size=4096, intermediate_size=11008,
            num_layers=4, num_heads=32, head_dim=128,
            vocab_size=32000,
        )
        topology = TopologyProber.make_fallback_topology(num_nodes=1)
        model = PartitionCostModel(gpu_profiles, layer_weights, topology)
        return model, layer_weights

    def test_evaluate_all_layers(self, setup):
        model, weights = setup
        cost = model.evaluate("node-0", 0, len(weights))
        assert cost.compute_time_ms > 0
        assert cost.total_time_ms > 0
        assert cost.memory_bytes > 0

    def test_fits_in_memory(self, setup):
        model, _ = setup
        cost = model.evaluate("node-0", 0, 2)
        assert cost.fits_in_memory

    def test_empty_layers_fits(self, setup):
        model, _ = setup
        cost = model.evaluate("node-0", 0, 0)
        assert cost.fits_in_memory
        assert cost.total_time_ms == 0.0

    def test_evaluate_partition_list(self, setup):
        model, weights = setup
        costs = model.evaluate_partition([("node-0", 0, len(weights))])
        assert len(costs) == 1
        assert costs[0].total_time_ms > 0

    def test_max_latency(self, setup):
        model, weights = setup
        lat = model.max_latency([("node-0", 0, len(weights))])
        assert lat > 0

    def test_combined_throughput(self, setup):
        model, weights = setup
        tp = model.combined_throughput([("node-0", 0, len(weights))])
        assert tp > 0

    def test_throughput_zero_on_empty(self, setup):
        model, _ = setup
        tp = model.combined_throughput([])
        assert tp == 0.0

    def test_more_layers_more_time(self, setup):
        model, weights = setup
        small = model.evaluate("node-0", 0, 2)
        large = model.evaluate("node-0", 0, len(weights))
        assert large.compute_time_ms > small.compute_time_ms

    def test_unknown_node_id(self, setup):
        model, weights = setup
        cost = model.evaluate("unknown", 0, len(weights))
        assert cost.compute_time_ms > 0  # falls back to CPU estimate
        assert cost.fits_in_memory

    def test_cost_summary(self, setup):
        model, _ = setup
        summary = model.cost_summary("node-0", 0, 6)
        assert "node-0" in summary
        assert "compute" in summary
        assert "comm" in summary

    def test_more_layers_increase_memory(self, setup):
        model, weights = setup
        small = model.evaluate("node-0", 0, 2)
        large = model.evaluate("node-0", 0, len(weights))
        assert large.memory_bytes >= small.memory_bytes


# ─── PartitionPoint / PartitionSolution ───────────────────────────────────────


class TestPartitionPoint:
    def test_defaults(self):
        p = PartitionPoint(node_id="n0", start_layer=0, end_layer=10)
        assert p.estimated_time_ms == 0.0

    def test_custom(self):
        p = PartitionPoint(node_id="n0", start_layer=0, end_layer=10, estimated_time_ms=50.0)
        assert p.estimated_time_ms == 50.0


class TestPartitionSolution:
    def test_empty(self):
        s = PartitionSolution()
        assert s.num_nodes == 0
        assert s.coverage == (0, 0)  # type: ignore

    def test_with_points(self):
        s = PartitionSolution(
            points=[
                PartitionPoint(node_id="a", start_layer=0, end_layer=5),
                PartitionPoint(node_id="b", start_layer=5, end_layer=10),
            ],
            max_node_time_ms=100.0,
            estimated_throughput_tok_s=50.0,
        )
        assert s.num_nodes == 2
        assert s.coverage == (0, 10)

    def test_summary(self):
        s = PartitionSolution(
            points=[PartitionPoint(node_id="n0", start_layer=0, end_layer=10, estimated_time_ms=50.0)],
            max_node_time_ms=50.0,
            estimated_throughput_tok_s=100.0,
        )
        summary = s.summary()
        assert "1 nodes" in summary
        assert "10 layers" in summary
        assert "50.0ms" in summary
        assert "100 tok/s" in summary


# ─── PartitionOptimizer ───────────────────────────────────────────────────────


class TestPartitionOptimizer:
    @pytest.fixture
    def setup(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(
            hidden_size=4096, intermediate_size=11008,
            num_layers=4, num_heads=32, head_dim=128,
            vocab_size=32000,
        )
        topology = TopologyProber.make_fallback_topology(num_nodes=2)
        # Use identical GPUs so splitting is beneficial
        profiles = {
            "node-0": GPUProfile(gpu_id=0, name="H100", total_memory_bytes=80 * 1024**3, compute_tflops=989.0),
            "node-1": GPUProfile(gpu_id=1, name="H100", total_memory_bytes=80 * 1024**3, compute_tflops=989.0),
        }
        cost_model = PartitionCostModel(profiles, weights, topology)
        return cost_model, weights

    def test_solve_two_nodes(self, setup):
        """With identical GPUs and enough layers, splitting can beat single-node."""
        cost_model, _ = setup
        # Use many layers so compute dominates communication overhead
        optimizer = PartitionOptimizer(cost_model, ["node-0", "node-1"])
        solution = optimizer.solve(32)
        # DP should find a valid solution
        assert solution.num_nodes >= 1
        assert solution.max_node_time_ms > 0
        assert solution.coverage == (0, 32)

    def test_solve_fast_node_gets_more(self, setup):
        cost_model, weights = setup
        # Create a second fixture with different-speed GPUs
        fast_profile = GPUProfile(gpu_id=0, name="H100", total_memory_bytes=80 * 1024**3, compute_tflops=989.0)
        slow_profile = GPUProfile(gpu_id=1, name="L4", total_memory_bytes=24 * 1024**3, compute_tflops=121.0)
        mixed_profiles = {"node-fast": fast_profile, "node-slow": slow_profile}
        topo = TopologyProber.make_fallback_topology(2)
        topo.node_ids = ["node-fast", "node-slow"]
        mixed_cost = PartitionCostModel(mixed_profiles, weights, topo)
        optimizer = PartitionOptimizer(mixed_cost, ["node-fast", "node-slow"], allow_oom=True)
        solution = optimizer.solve(len(weights))
        # Fast node should get more layers (or the same if all fit on one)
        try:
            n_fast = next(p for p in solution.points if p.node_id == "node-fast")
            n_slow = next(p for p in solution.points if p.node_id == "node-slow")
            assert (n_fast.end_layer - n_fast.start_layer) >= (
                n_slow.end_layer - n_slow.start_layer
            )
        except StopIteration:
            # May collapse to single node if fast node fits all layers
            assert solution.num_nodes == 1
            assert solution.points[0].node_id == "node-fast"

    def test_solve_single_node(self, setup):
        cost_model, weights = setup
        optimizer = PartitionOptimizer(cost_model, ["node-0"])
        solution = optimizer.solve(len(weights))
        assert solution.num_nodes == 1
        assert solution.points[0].start_layer == 0
        assert solution.points[0].end_layer == len(weights)
        assert solution.estimated_throughput_tok_s > 0

    def test_solve_no_nodes(self, setup):
        cost_model, _ = setup
        optimizer = PartitionOptimizer(cost_model, [])
        solution = optimizer.solve(32)
        assert solution.num_nodes == 0
        assert "No nodes available" in solution.explanation

    def test_compare_strategies(self, setup):
        cost_model, weights = setup
        optimizer = PartitionOptimizer(cost_model, ["node-0", "node-1"])
        comparison = optimizer.compare_strategies(len(weights))
        assert "dp_minimax" in comparison
        assert "equal_split" in comparison
        assert "proportional_split" in comparison
        assert "improvement_over_equal" in comparison
        assert comparison["dp_minimax"]["max_latency_ms"] > 0
        assert comparison["equal_split"]["max_latency_ms"] > 0


# ─── HardwareAwarePartitioner ──────────────────────────────────────────────────


class TestHardwareAwarePartitioner:
    def test_init_defaults(self):
        p = HardwareAwarePartitioner()
        assert p._batch_size == 1
        assert p._seq_len == 4096
        assert not p._allow_oom

    def test_init_custom(self):
        p = HardwareAwarePartitioner(batch_size=4, seq_len=8192, allow_oom=True)
        assert p._batch_size == 4
        assert p._seq_len == 8192
        assert p._allow_oom

    def test_solution_returns_none_before_partition(self):
        p = HardwareAwarePartitioner()
        assert p.solution() is None

    def test_get_layer_assignments_before_partition(self):
        p = HardwareAwarePartitioner()
        assert p.get_layer_assignments() is None

    def test_get_node_summaries_before_partition(self):
        p = HardwareAwarePartitioner()
        assert p.get_node_summaries() is None

    def test_compare_to_baselines_before_partition(self):
        p = HardwareAwarePartitioner()
        assert p.compare_to_baselines() is None

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(1))
    async def test_partition_single_node(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        solution = await p.partition(
            model_name="test", num_layers=4,
            hidden_size=1024, intermediate_size=4096,
            num_heads=8, head_dim=64, vocab_size=10000,
        )
        assert solution.num_nodes == 1
        assert solution.max_node_time_ms > 0
        assert solution.estimated_throughput_tok_s > 0

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(2))
    async def test_partition_two_nodes(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        solution = await p.partition(
            model_name="test", node_ids=["node-0", "node-1"],
            num_layers=32, hidden_size=1024, intermediate_size=4096,
            num_heads=8, head_dim=64, vocab_size=10000,
        )
        # DP may find that 1 node is still faster due to CPU fallback
        assert solution.num_nodes >= 1
        assert solution.max_node_time_ms > 0
        assert solution.coverage[0] == 0
        assert solution.coverage[1] == 34  # 32 layers + embed + lm_head

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(2))
    async def test_get_node_summaries_after_partition(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        await p.partition(
            model_name="test", node_ids=["node-0", "node-1"],
            num_layers=32, hidden_size=1024, intermediate_size=4096,
            num_heads=8, head_dim=64, vocab_size=10000,
        )
        summaries = p.get_node_summaries()
        assert summaries is not None
        assert len(summaries) >= 1
        for s in summaries:
            assert "node_id" in s
            assert "num_layers" in s
            assert "total_time_ms" in s
            assert "memory_gb" in s

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(2))
    async def test_compare_to_baselines_after_partition(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        await p.partition(
            model_name="test", node_ids=["node-0", "node-1"],
            num_layers=4, hidden_size=1024, intermediate_size=4096,
            num_heads=8, head_dim=64, vocab_size=10000,
        )
        comparison = p.compare_to_baselines()
        assert comparison is not None
        assert "dp_minimax" in comparison
        assert "equal_split" in comparison

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(2))
    async def test_partition_summary(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        await p.partition(
            model_name="test", node_ids=["node-0"],
            num_layers=4, hidden_size=1024,
        )
        s = p.summary()
        assert "HardwareAwarePartitioner" in s
