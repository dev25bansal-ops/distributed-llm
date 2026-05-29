"""End-to-end integration tests for the partition system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import PartitionOptimizer, PartitionSolution
from distllm.dist.partition.partitioner import HardwareAwarePartitioner
from distllm.dist.partition.profiles import GPUProfile, GPUProfiler
from distllm.dist.partition.topology import LinkProfile, TopologyGraph, TopologyProber
from distllm.dist.partition.persistence import PartitionStore
from distllm.dist.partition.validator import PartitionValidator
from distllm.dist.partition.visualizer import ClusterVisualizer


class TestEndToEndPartitionFlow:
    """Full partition pipeline: profile → optimize → validate → persist."""

    def test_full_flow(self, cost_model_homogeneous, medium_model_weights):
        # Optimize
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        assert sol.num_nodes >= 1
        assert sol.coverage == (0, len(medium_model_weights))

        # Validate
        validator = PartitionValidator(cost_model_homogeneous)
        report = validator.validate(sol, num_layers=len(medium_model_weights))
        assert report.simulation is not None
        assert report.simulation.throughput_tok_s > 0

        # Visualize
        viz = ClusterVisualizer()
        output = viz.print_partition(sol)
        assert "Partition" in output

        # Persist
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = PartitionStore(f.name)
            run_id = store.save_run(
                model_name="test-model",
                solution=sol,
                config={"num_layers": len(medium_model_weights)},
            )
            assert run_id > 0
            store.close()


class TestStrategyComparisonConsistency:
    """DP should always beat or tie equal/proportional splits."""

    def test_dp_beats_equal(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        comparison = opt.compare_strategies(len(medium_model_weights))
        dp_lat = comparison["dp_minimax"]["max_latency_ms"]
        eq_lat = comparison["equal_split"]["max_latency_ms"]
        assert dp_lat <= eq_lat + 0.01

    def test_dp_beats_proportional(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        comparison = opt.compare_strategies(len(medium_model_weights))
        dp_lat = comparison["dp_minimax"]["max_latency_ms"]
        pr_lat = comparison["proportional_split"]["max_latency_ms"]
        assert dp_lat <= pr_lat + 0.01


class TestPersistenceRoundTrip:
    """Save and load partition plans."""

    def test_save_and_load_run(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = PartitionStore(f.name)
            sol = PartitionSolution(
                points=[],
                max_node_time_ms=50.0,
                estimated_throughput_tok_s=100.0,
            )
            run_id = store.save_run(
                model_name="test", solution=sol,
                config={"hidden_size": 4096},
            )
            loaded = store.get_run(run_id)
            assert loaded is not None
            assert loaded.model_name == "test"
            assert loaded.solution["max_node_time_ms"] == 50.0
            store.close()

    def test_metric_recording(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = PartitionStore(f.name)
            run_id = store.save_run(model_name="test", solution={}, config={})
            store.record_metric(run_id, "actual_latency_ms", 42.5)
            run = store.get_run(run_id)
            assert run is not None
            assert run.metrics["actual_latency_ms"] == 42.5
            store.close()

    def test_best_run(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = PartitionStore(f.name)
            id1 = store.save_run(model_name="test", solution={}, config={})
            store.record_metric(id1, "actual_latency_ms", 100.0)
            id2 = store.save_run(model_name="test", solution={}, config={})
            store.record_metric(id2, "actual_latency_ms", 50.0)
            best = store.get_best_run("test")
            assert best is not None
            assert best.run_id == id2
            store.close()


class TestValidatorIntegration:
    """Validator produces meaningful reports."""

    def test_valid_solution(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        validator = PartitionValidator(cost_model_homogeneous)
        report = validator.validate(sol, num_layers=len(medium_model_weights))
        assert report.is_valid

    def test_incomplete_coverage_detected(self, cost_model_homogeneous):
        sol = PartitionSolution(
            points=[],
            max_node_time_ms=0.0,
        )
        validator = PartitionValidator(cost_model_homogeneous)
        report = validator.validate(sol, num_layers=32)
        assert not report.is_valid


class TestVisualizerIntegration:
    """Visualizer produces output without crashing."""

    def test_topology_output(self, three_node_nvlink_topology):
        viz = ClusterVisualizer()
        output = viz.print_topology(three_node_nvlink_topology)
        assert "3 nodes" in output
        assert "NVLink" in output

    def test_partition_output(self, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        viz = ClusterVisualizer()
        output = viz.print_partition(sol)
        assert "Partition" in output

    def test_json_export(self, three_node_nvlink_topology, cost_model_homogeneous, medium_model_weights):
        opt = PartitionOptimizer(cost_model_homogeneous, ["gpu-0", "gpu-1"])
        sol = opt.solve(len(medium_model_weights))
        viz = ClusterVisualizer()
        json_str = viz.to_json(topology=three_node_nvlink_topology, solution=sol)
        data = json.loads(json_str)
        assert "topology" in data
        assert "solution" in data
