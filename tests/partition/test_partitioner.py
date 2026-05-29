"""Tests for HardwareAwarePartitioner (orchestrator integration)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from distllm.dist.partition.partitioner import HardwareAwarePartitioner
from distllm.dist.partition.profiles import GPUProfiler
from distllm.dist.partition.topology import TopologyProber


class TestPartitionerInit:
    def test_defaults(self):
        p = HardwareAwarePartitioner()
        assert p._batch_size == 1
        assert p._seq_len == 4096
        assert not p._allow_oom

    def test_custom(self):
        p = HardwareAwarePartitioner(batch_size=4, seq_len=8192, allow_oom=True)
        assert p._batch_size == 4
        assert p._seq_len == 8192
        assert p._allow_oom


class TestPartitionerPrePartition:
    def test_solution_none_before(self):
        p = HardwareAwarePartitioner()
        assert p.solution() is None

    def test_assignments_none_before(self):
        p = HardwareAwarePartitioner()
        assert p.get_layer_assignments() is None

    def test_summaries_none_before(self):
        p = HardwareAwarePartitioner()
        assert p.get_node_summaries() is None

    def test_baselines_none_before(self):
        p = HardwareAwarePartitioner()
        assert p.compare_to_baselines() is None


class TestPartitionerPartition:
    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(1))
    async def test_single_node(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            model_name="test", num_layers=4,
            hidden_size=1024, intermediate_size=4096,
            num_heads=8, head_dim=64, vocab_size=10000,
        )
        assert sol.num_nodes == 1
        assert sol.max_node_time_ms > 0

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(2))
    async def test_two_nodes(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        sol = await p.partition(
            model_name="test", node_ids=["node-0", "node-1"],
            num_layers=32, hidden_size=1024, intermediate_size=4096,
            num_heads=8, head_dim=64, vocab_size=10000,
        )
        assert sol.num_nodes >= 1
        assert sol.coverage[0] == 0
        assert sol.coverage[1] == 34  # 32 + embed + lm_head

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(1))
    async def test_summaries_after(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        await p.partition(
            model_name="test", num_layers=4,
            hidden_size=1024, intermediate_size=4096,
        )
        summaries = p.get_node_summaries()
        assert summaries is not None
        assert len(summaries) >= 1

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(1))
    async def test_baselines_after(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        await p.partition(
            model_name="test", num_layers=4,
            hidden_size=1024, intermediate_size=4096,
        )
        comparison = p.compare_to_baselines()
        assert comparison is not None
        assert "dp_minimax" in comparison

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(1))
    async def test_summary_after(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        await p.partition(model_name="test", num_layers=4, hidden_size=1024)
        s = p.summary()
        assert "HardwareAwarePartitioner" in s


class TestPartitionerCache:
    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(1))
    async def test_cache_hit(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        sol1 = await p.partition(model_name="test", num_layers=4, hidden_size=1024)
        sol2 = await p.partition(model_name="test", num_layers=4, hidden_size=1024)
        assert sol1 is sol2
        assert mock_probe.call_count == 1

    @patch.object(GPUProfiler, "_device_count", return_value=0)
    @patch.object(TopologyProber, "probe", return_value=TopologyProber.make_fallback_topology(1))
    async def test_cache_miss_on_config_change(self, mock_probe, mock_count):
        p = HardwareAwarePartitioner()
        await p.partition(model_name="test", num_layers=4, hidden_size=1024)
        await p.partition(model_name="test", num_layers=8, hidden_size=1024)
        assert mock_probe.call_count == 2


class TestPartitionerPersistence:
    def test_load_plan_missing(self):
        p = HardwareAwarePartitioner()
        assert p.load_plan("nonexistent_model") is None
