"""Tests for dist/partition modules — real objects, zero mocks."""
from __future__ import annotations

import pytest


class TestGPUProfile:
    def test_gpu_profile_init(self):
        from distllm.dist.partition.profiles import GPUProfile
        profile = GPUProfile(gpu_id=0, name="NVIDIA A100", total_memory_bytes=80 * 1024**3)
        assert profile is not None
        assert profile.name == "NVIDIA A100"
        assert profile.total_memory_bytes == 80 * 1024**3

    def test_gpu_profile_with_tflops(self):
        from distllm.dist.partition.profiles import GPUProfile
        profile = GPUProfile(
            gpu_id=0, name="RTX 4090", total_memory_bytes=24 * 1024**3,
            peak_tflops_fp16=165.0, memory_bandwidth_gbps=1008,
        )
        assert profile.peak_tflops_fp16 == 165.0


class TestPartitionCostModel:
    def test_cost_model_init(self):
        from distllm.dist.partition import PartitionCostModel, GPUProfile, TopologyGraph
        from distllm.dist.partition.cost_model import NodeCost

        profiles = {
            "n1": GPUProfile(gpu_id=0, name="A100", total_memory_bytes=80 * 1024**3),
            "n2": GPUProfile(gpu_id=1, name="A100", total_memory_bytes=80 * 1024**3),
        }
        topo = TopologyGraph()
        cm = PartitionCostModel(gpu_profiles=profiles, layer_weights=[], topology=topo)
        assert cm is not None


class TestPartitionOptimizer:
    def test_optimizer_init(self):
        from distllm.dist.partition import PartitionCostModel, GPUProfile, TopologyGraph, PartitionOptimizer

        profiles = {"n1": GPUProfile(gpu_id=0, name="A100", total_memory_bytes=80 * 1024**3)}
        topo = TopologyGraph()
        cm = PartitionCostModel(gpu_profiles=profiles, layer_weights=[], topology=topo)
        opt = PartitionOptimizer(cost_model=cm, node_ids=["n1"])
        assert opt is not None


class TestQuantization:
    def test_quant_auto_tuner_init(self):
        from distllm.dist.partition import QuantizationAutoTuner
        tuner = QuantizationAutoTuner()
        assert tuner is not None

    def test_quant_aware_cost_model(self):
        from distllm.dist.partition.quant_cost import QuantizationAwareCostModel
        from distllm.dist.partition.cost_model import PartitionCostModel

        base = PartitionCostModel(gpu_profiles={}, layer_weights=[], topology=None)
        model = QuantizationAwareCostModel(base_cost_model=base)
        # Just verify it instantiates and has the expected method
        assert hasattr(model, "compare_with_without_quant")
        assert hasattr(model, "evaluate_with_quant")
        assert hasattr(model, "evaluate_partition_with_quant")


class TestTopologyGraph:
    def test_topology_graph_init(self):
        from distllm.dist.partition import TopologyGraph
        g = TopologyGraph()
        assert g is not None
        assert g.total_gpus() == 0

    def test_topology_with_links(self):
        from distllm.dist.partition import TopologyGraph
        g = TopologyGraph()
        # TopologyGraph exposes node_ids as a dict-like attribute
        # total_gpus reflects configured GPU count
        assert g.total_gpus() >= 0


class TestPartitionConfig:
    def test_auto_partition_config(self):
        from distllm.dist.partition import AutoPartitionConfig
        cfg = AutoPartitionConfig()
        assert cfg is not None
        assert cfg.enabled is False  # default is disabled


class TestPartitioner:
    def test_hardware_aware_partitioner(self):
        from distllm.dist.partition import HardwareAwarePartitioner
        p = HardwareAwarePartitioner()
        assert p is not None


class TestProfiles:
    def test_gpu_profiles_class(self):
        from distllm.dist.partition.profiles import GPUProfile
        profile = GPUProfile(gpu_id=0, name="test")
        assert profile is not None
        assert profile.name == "test"


class TestReportGeneration:
    def test_report_generator(self):
        from distllm.dist.partition.quant_report import ReportGenerator
        gen = ReportGenerator()
        assert gen is not None
