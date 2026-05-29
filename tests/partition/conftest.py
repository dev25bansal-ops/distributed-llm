"""Shared fixtures for partition tests."""

from __future__ import annotations

import pytest

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


# ── GPU Profiles ─────────────────────────────────────────────────────────────

@pytest.fixture
def h100_profile():
    return GPUProfile(
        gpu_id=0, name="H100", total_memory_bytes=80 * 1024**3,
        compute_tflops=989.0, memory_bandwidth_gbps=3350.0,
        sm_count=132, peak_tflops_fp16=989.0,
    )


@pytest.fixture
def a100_profile():
    return GPUProfile(
        gpu_id=1, name="A100", total_memory_bytes=80 * 1024**3,
        compute_tflops=312.0, memory_bandwidth_gbps=2039.0,
        sm_count=108, peak_tflops_fp16=312.0,
    )


@pytest.fixture
def l4_profile():
    return GPUProfile(
        gpu_id=2, name="L4", total_memory_bytes=24 * 1024**3,
        compute_tflops=121.0, memory_bandwidth_gbps=300.0,
        sm_count=24, peak_tflops_fp16=121.0,
    )


@pytest.fixture
def t4_profile():
    return GPUProfile(
        gpu_id=3, name="T4", total_memory_bytes=16 * 1024**3,
        compute_tflops=65.0, memory_bandwidth_gbps=320.0,
        sm_count=40, peak_tflops_fp16=65.0,
    )


@pytest.fixture
def small_gpu_profile():
    """GPU with very limited memory for OOM testing."""
    return GPUProfile(
        gpu_id=99, name="SmallGPU", total_memory_bytes=2 * 1024**3,
        compute_tflops=50.0, memory_bandwidth_gbps=200.0,
    )


# ── GPU Profile Dicts ────────────────────────────────────────────────────────

@pytest.fixture
def homogeneous_profiles(h100_profile):
    return {"gpu-0": h100_profile, "gpu-1": h100_profile}


@pytest.fixture
def heterogeneous_profiles(h100_profile, a100_profile, l4_profile):
    return {"gpu-0": h100_profile, "gpu-1": a100_profile, "gpu-2": l4_profile}


# ── Layer Weights ────────────────────────────────────────────────────────────

@pytest.fixture
def small_model_weights():
    profiler = GPUProfiler()
    return profiler.estimate_layer_weights(
        hidden_size=1024, intermediate_size=4096,
        num_layers=4, num_heads=8, head_dim=64, vocab_size=10000,
    )


@pytest.fixture
def medium_model_weights():
    profiler = GPUProfiler()
    return profiler.estimate_layer_weights(
        hidden_size=4096, intermediate_size=11008,
        num_layers=32, num_heads=32, head_dim=128, vocab_size=32000,
    )


@pytest.fixture
def large_model_weights():
    profiler = GPUProfiler()
    return profiler.estimate_layer_weights(
        hidden_size=8192, intermediate_size=28672,
        num_layers=80, num_heads=64, head_dim=128, vocab_size=32000,
    )


# ── Topologies ───────────────────────────────────────────────────────────────

@pytest.fixture
def single_node_topology():
    return TopologyGraph(
        node_ids=["gpu-0"],
        gpu_counts={"gpu-0": 1},
        links=[],
    )


@pytest.fixture
def two_node_topology():
    return TopologyGraph(
        node_ids=["gpu-0", "gpu-1"],
        gpu_counts={"gpu-0": 1, "gpu-1": 1},
        links=[
            LinkProfile(source="gpu-0", target="gpu-1", bandwidth_gbps=25.0, latency_us=500.0),
        ],
    )


@pytest.fixture
def three_node_nvlink_topology():
    return TopologyGraph(
        node_ids=["gpu-0", "gpu-1", "gpu-2"],
        gpu_counts={"gpu-0": 1, "gpu-1": 1, "gpu-2": 1},
        links=[
            LinkProfile(source="gpu-0", target="gpu-1", bandwidth_gbps=600.0, latency_us=5.0, is_nvlink=True),
            LinkProfile(source="gpu-1", target="gpu-2", bandwidth_gbps=600.0, latency_us=5.0, is_nvlink=True),
            LinkProfile(source="gpu-0", target="gpu-2", bandwidth_gbps=600.0, latency_us=5.0, is_nvlink=True),
        ],
    )


# ── Cost Models ──────────────────────────────────────────────────────────────

@pytest.fixture
def cost_model_homogeneous(homogeneous_profiles, medium_model_weights, two_node_topology):
    return PartitionCostModel(homogeneous_profiles, medium_model_weights, two_node_topology)


@pytest.fixture
def cost_model_heterogeneous(heterogeneous_profiles, medium_model_weights, three_node_nvlink_topology):
    return PartitionCostModel(heterogeneous_profiles, medium_model_weights, three_node_nvlink_topology)


@pytest.fixture
def cost_model_single(h100_profile, small_model_weights, single_node_topology):
    return PartitionCostModel({"gpu-0": h100_profile}, small_model_weights, single_node_topology)
