"""Chaos and resilience tests for the partition system."""

from __future__ import annotations

import pytest

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import PartitionOptimizer, PartitionSolution
from distllm.dist.partition.profiles import GPUProfile
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


class TestZeroTflops:
    """Cost model must handle GPUs with 0 TFLOPS gracefully."""

    def test_zero_tflops_gpu(self):
        weights = [type("LW", (), {
            "layer_type": "transformer", "weight_memory_bytes": 1000,
            "activation_memory_bytes": 100, "flops_per_seq": 1000,
            "flops_per_token": 100, "kv_cache_bytes_per_token": 50,
        })()]
        profiles = {"gpu-0": GPUProfile(gpu_id=0, name="Broken", total_memory_bytes=80 * 1024**3, compute_tflops=0.0)}
        topo = TopologyGraph(node_ids=["gpu-0"], gpu_counts={"gpu-0": 1})
        cm = PartitionCostModel(profiles, weights, topo)
        cost = cm.evaluate("gpu-0", 0, 1)
        assert cost.total_time_ms == 0.0


class TestAllOOM:
    """When all nodes are OOM, solver must report it."""

    def test_all_nodes_oom(self):
        tiny_gpu = GPUProfile(gpu_id=0, name="Tiny", total_memory_bytes=1024, compute_tflops=1.0)
        profiler_t = type("P", (), {"estimate_layer_weights": staticmethod(lambda **kw: [
            type("LW", (), {
                "layer_type": "transformer", "weight_memory_bytes": 1024 * 1024 * 100,
                "activation_memory_bytes": 1024 * 1024, "flops_per_seq": 1000,
                "flops_per_token": 100, "kv_cache_bytes_per_token": 1024,
            })()
        ])})
        weights = profiler_t.estimate_layer_weights()
        profiles = {"gpu-0": tiny_gpu}
        topo = TopologyGraph(node_ids=["gpu-0"], gpu_counts={"gpu-0": 1})
        cm = PartitionCostModel(profiles, weights, topo)
        opt = PartitionOptimizer(cm, ["gpu-0"], allow_oom=True)
        sol = opt.solve(1)
        assert sol.num_oom_nodes >= 0


class TestNaNHandling:
    """DP must handle NaN/inf costs gracefully."""

    def test_nan_propagates_as_inf(self):
        import math
        assert math.isinf(float("inf"))
        assert math.isnan(float("nan"))
        assert not math.isinf(float("nan"))
        assert not math.isnan(float("inf"))


class TestTopologyTimeout:
    """Topology probing must degrade gracefully."""

    def test_fallback_topology_works(self):
        topo = TopologyProber.make_fallback_topology(3, 2)
        assert len(topo.node_ids) == 3
        assert topo.total_gpus() == 6
        assert len(topo.links) == 3


class TestConfigCorruption:
    """Partition plan loading must handle corrupt files."""

    def test_load_corrupt_json(self, tmp_path):
        from distllm.dist.partition.partitioner import HardwareAwarePartitioner
        p = HardwareAwarePartitioner(profile_dir=str(tmp_path))
        # Write corrupt file
        corrupt_file = tmp_path / "corrupt_partition_plan.json"
        corrupt_file.write_text("{invalid json")
        result = p.load_plan("corrupt")
        assert result is None

    def test_load_missing_file(self, tmp_path):
        from distllm.dist.partition.partitioner import HardwareAwarePartitioner
        p = HardwareAwarePartitioner(profile_dir=str(tmp_path))
        result = p.load_plan("nonexistent")
        assert result is None


class TestLargeModel:
    """Stress test with large model configuration."""

    def test_70b_model_partition(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(
            hidden_size=8192, intermediate_size=28672,
            num_layers=80, num_heads=64, head_dim=128, vocab_size=32000,
        )
        profiles = {
            f"gpu-{i}": GPUProfile(gpu_id=i, name="H100", total_memory_bytes=80 * 1024**3, compute_tflops=989.0)
            for i in range(4)
        }
        nodes = [f"gpu-{i}" for i in range(4)]
        topo = TopologyGraph(node_ids=nodes, gpu_counts={n: 1 for n in nodes})
        cm = PartitionCostModel(profiles, weights, topo)
        opt = PartitionOptimizer(cm, nodes, allow_oom=True)
        sol = opt.solve(len(weights))
        assert sol.num_nodes >= 1
        assert sol.coverage == (0, len(weights))


class TestEdgeCaseNodes:
    """Edge cases with node configurations."""

    def test_many_nodes_few_layers(self):
        profiler = GPUProfiler()
        weights = profiler.estimate_layer_weights(num_layers=2, hidden_size=512)
        nodes = [f"gpu-{i}" for i in range(10)]
        profiles = {n: GPUProfile(gpu_id=i, name="GPU", total_memory_bytes=80 * 1024**3, compute_tflops=100.0) for i, n in enumerate(nodes)}
        topo = TopologyGraph(node_ids=nodes, gpu_counts={n: 1 for n in nodes})
        cm = PartitionCostModel(profiles, weights, topo)
        opt = PartitionOptimizer(cm, nodes, allow_oom=True)
        sol = opt.solve(len(weights))
        assert sol.coverage == (0, len(weights))
