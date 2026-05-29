"""Comprehensive tests for PartitionCostModel.

Target: 25+ tests covering all cost model features.
"""

from __future__ import annotations

import pytest

from distllm.dist.partition.cost_model import NodeCost, PartitionCostModel
from distllm.dist.partition.profiles import GPUProfile
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


class TestNodeCost:
    def test_defaults(self):
        nc = NodeCost(node_id="n0", start_layer=0, end_layer=10)
        assert nc.compute_time_ms == 0.0
        assert nc.fits_in_memory is True

    def test_memory_utilization(self):
        nc = NodeCost(node_id="n0", start_layer=0, end_layer=10, memory_bytes=500, memory_available_bytes=1000)
        assert nc.memory_utilization == 0.5

    def test_memory_utilization_zero(self):
        nc = NodeCost(node_id="n0", start_layer=0, end_layer=0)
        assert nc.memory_utilization == 0.0


class TestCostModelEvaluate:
    def test_evaluate_basic(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        cost = cm.evaluate("gpu-0", 0, len(medium_model_weights))
        assert cost.compute_time_ms > 0
        assert cost.total_time_ms > 0
        assert cost.memory_bytes > 0

    def test_evaluate_empty_layers(self, cost_model_homogeneous):
        cm = cost_model_homogeneous
        cost = cm.evaluate("gpu-0", 0, 0)
        assert cost.fits_in_memory is True
        assert cost.total_time_ms == 0.0

    def test_evaluate_unknown_node(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        cost = cm.evaluate("unknown", 0, len(medium_model_weights))
        assert cost.compute_time_ms > 0  # CPU fallback

    def test_more_layers_more_time(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        small = cm.evaluate("gpu-0", 0, 2)
        large = cm.evaluate("gpu-0", 0, len(medium_model_weights))
        assert large.compute_time_ms > small.compute_time_ms

    def test_more_layers_more_memory(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        small = cm.evaluate("gpu-0", 0, 2)
        large = cm.evaluate("gpu-0", 0, len(medium_model_weights))
        assert large.memory_bytes >= small.memory_bytes

    def test_faster_gpu_less_time(self, heterogeneous_profiles, medium_model_weights, two_node_topology):
        cm = PartitionCostModel(heterogeneous_profiles, medium_model_weights, two_node_topology)
        h100_cost = cm.evaluate("gpu-0", 0, len(medium_model_weights))
        l4_cost = cm.evaluate("gpu-2", 0, len(medium_model_weights))
        assert h100_cost.compute_time_ms < l4_cost.compute_time_ms

    def test_communication_cost(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        cost = cm.evaluate("gpu-1", 0, len(medium_model_weights))
        assert cost.communication_time_ms >= 0

    def test_first_node_no_comm(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        cost = cm.evaluate("gpu-0", 0, len(medium_model_weights))
        assert cost.communication_time_ms == 0.0

    def test_batch_size_scales_memory(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        cost_b1 = cm.evaluate("gpu-0", 0, len(medium_model_weights), batch_size=1)
        cost_b4 = cm.evaluate("gpu-0", 0, len(medium_model_weights), batch_size=4)
        assert cost_b4.memory_bytes > cost_b1.memory_bytes

    def test_seq_len_scales_time(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        cost_s1 = cm.evaluate("gpu-0", 0, len(medium_model_weights), seq_len=1024)
        cost_s4 = cm.evaluate("gpu-0", 0, len(medium_model_weights), seq_len=4096)
        assert cost_s4.compute_time_ms > cost_s1.compute_time_ms


class TestCostModelPartition:
    def test_evaluate_partition(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        partition = [("gpu-0", 0, 16), ("gpu-1", 16, len(medium_model_weights))]
        costs = cm.evaluate_partition(partition)
        assert len(costs) == 2
        assert all(c.total_time_ms > 0 for c in costs)

    def test_max_latency(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        partition = [("gpu-0", 0, 16), ("gpu-1", 16, len(medium_model_weights))]
        lat = cm.max_latency(partition)
        assert lat > 0

    def test_combined_throughput(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        partition = [("gpu-0", 0, 16), ("gpu-1", 16, len(medium_model_weights))]
        tp = cm.combined_throughput(partition)
        assert tp > 0

    def test_throughput_zero_empty(self, cost_model_homogeneous):
        assert cost_model_homogeneous.combined_throughput([]) == 0.0


class TestCostModelTokenAware:
    def test_token_aware_basic(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        cost = cm.evaluate_token_aware("gpu-0", 0, len(medium_model_weights))
        assert cost.compute_time_ms > 0
        assert cost.total_time_ms > 0

    def test_token_aware_decode_scales(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        cost_1 = cm.evaluate_token_aware("gpu-0", 0, len(medium_model_weights), num_decode_tokens=1)
        cost_10 = cm.evaluate_token_aware("gpu-0", 0, len(medium_model_weights), num_decode_tokens=10)
        assert cost_10.total_time_ms > cost_1.total_time_ms

    def test_token_aware_throughput(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        partition = [("gpu-0", 0, 16), ("gpu-1", 16, len(medium_model_weights))]
        tp = cm.combined_throughput_token_aware(partition, num_decode_tokens=10)
        assert tp > 0


class TestCostModelPipeline:
    def test_pipeline_latency(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        partition = [("gpu-0", 0, 16), ("gpu-1", 16, len(medium_model_weights))]
        lat = cm.pipeline_latency(partition)
        assert lat > 0

    def test_pipeline_latency_single_node(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        partition = [("gpu-0", 0, len(medium_model_weights))]
        lat = cm.pipeline_latency(partition)
        assert lat > 0


class TestCostModelFragmentation:
    def test_high_utilization_reduces_fit(self, small_gpu_profile, medium_model_weights, single_node_topology):
        cm = PartitionCostModel({"gpu-0": small_gpu_profile}, medium_model_weights, single_node_topology)
        cost = cm.evaluate("gpu-0", 0, len(medium_model_weights))
        assert not cost.fits_in_memory


class TestCostModelContention:
    def test_nvlink_low_contention(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        factor = cm._estimate_contention_factor("gpu-0", "gpu-1")
        assert factor >= 0.8

    def test_ethernet_high_contention(self, cost_model_homogeneous, medium_model_weights):
        cm = cost_model_homogeneous
        factor = cm._estimate_contention_factor("gpu-0", "gpu-1", num_active_links=4)
        assert 0.0 < factor <= 1.0


class TestCostModelUtilization:
    def test_utilization_factor_range(self, cost_model_homogeneous):
        cm = cost_model_homogeneous
        for h in [256, 1024, 4096, 8192]:
            for bs in [1, 4, 8, 16]:
                for sl in [512, 2048, 4096, 8192]:
                    f = cm._compute_utilization_factor(h, bs, sl, 312.0)
                    assert 0.1 < f <= 1.0


class TestCostModelSummary:
    def test_cost_summary(self, cost_model_homogeneous):
        cm = cost_model_homogeneous
        s = cm.cost_summary("gpu-0", 0, 10)
        assert "gpu-0" in s
        assert "compute" in s
