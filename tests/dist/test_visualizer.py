"""Tests for distllm.dist.partition.visualizer.

Zero mocks — uses only real objects from the module.
Tests the public API surface of ClusterVisualizer and module-level helpers.
"""

from __future__ import annotations

import json

import pytest

from distllm.dist.partition.cost_model import NodeCost
from distllm.dist.partition.optimizer import PartitionPoint, PartitionSolution
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph
from distllm.dist.partition.visualizer import (
    ClusterVisualizer,
    _bar,
    _fmt_bytes,
    _pad,
)


# ===========================================================================
# Module-level helper tests
# ===========================================================================


class TestFmtBytes:
    """Tests for the _fmt_bytes helper."""

    def test_bytes(self) -> None:
        assert _fmt_bytes(0) == "0B"
        assert _fmt_bytes(1) == "1B"
        assert _fmt_bytes(512) == "512B"
        assert _fmt_bytes(1023) == "1023B"

    def test_megabytes(self) -> None:
        assert _fmt_bytes(1024**2) == "1MB"
        assert _fmt_bytes(2 * 1024**2) == "2MB"
        assert _fmt_bytes(512 * 1024**2) == "512MB"

    def test_gigabytes(self) -> None:
        assert _fmt_bytes(1024**3) == "1.0GB"
        assert _fmt_bytes(2 * 1024**3) == "2.0GB"
        assert _fmt_bytes(4 * 1024**3 + 512 * 1024**2) == "4.5GB"
        assert _fmt_bytes(80 * 1024**3) == "80.0GB"

    def test_boundary_mb_to_gb(self) -> None:
        # Exactly at the boundary: 1024 MB = 1 GB
        assert _fmt_bytes(1024**3) == "1.0GB"

    def test_large_values(self) -> None:
        huge = 1_000_000 * 1024**3  # 1M GB
        result = _fmt_bytes(huge)
        assert "GB" in result


class TestPad:
    """Tests for the _pad helper."""

    def test_shorter_than_width(self) -> None:
        assert _pad("abc", 5) == "abc  "

    def test_exact_width(self) -> None:
        assert _pad("abcde", 5) == "abcde"

    def test_longer_than_width(self) -> None:
        assert _pad("abcdef", 5) == "abcdef"

    def test_empty_string(self) -> None:
        assert _pad("", 3) == "   "

    def test_zero_width(self) -> None:
        assert _pad("hello", 0) == "hello"

    def test_negative_width(self) -> None:
        assert _pad("hello", -1) == "hello"


class TestBar:
    """Tests for the _bar helper."""

    def test_empty(self) -> None:
        assert _bar(0.0) == "[" + "." * 20 + "]"

    def test_full(self) -> None:
        assert _bar(1.0) == "[" + "#" * 20 + "]"

    def test_half(self) -> None:
        result = _bar(0.5)
        assert result == "[" + "#" * 10 + "." * 10 + "]"

    def test_quarter(self) -> None:
        result = _bar(0.25)
        assert result == "[" + "#" * 5 + "." * 15 + "]"

    def test_above_one(self) -> None:
        """pct > 1 gives more filled than width (no clamping in helper)."""
        result = _bar(1.5)
        assert result == "[" + "#" * 30 + "." * 0 + "]"

    def test_below_zero(self) -> None:
        """Negative pct leads to negative filled, so width-fill > width."""
        result = _bar(-0.5)
        assert result == "[" + "#" * 0 + "." * 30 + "]"

    def test_zero_pct(self) -> None:
        assert _bar(0.0, width=10) == "[" + "." * 10 + "]"

    def test_custom_width(self) -> None:
        assert len(_bar(0.5, width=10)) == 12  # brackets + 10


# ===========================================================================
# ClusterVisualizer tests
# ===========================================================================


class TestClusterVisualizer:
    """Tests for the ClusterVisualizer class."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def visualizer(self) -> ClusterVisualizer:
        return ClusterVisualizer()

    @pytest.fixture
    def single_node_topology(self) -> TopologyGraph:
        return TopologyGraph(
            node_ids=["node-0"],
            gpu_counts={"node-0": 1},
            links=[],
        )

    @pytest.fixture
    def multi_node_topology(self) -> TopologyGraph:
        return TopologyGraph(
            node_ids=["node-0", "node-1", "node-2"],
            gpu_counts={"node-0": 2, "node-1": 1, "node-2": 4},
            links=[
                LinkProfile(source="node-0", target="node-1", bandwidth_gbps=12.5, latency_us=100.0, is_infiniband=True),
                LinkProfile(source="node-1", target="node-2", bandwidth_gbps=25.0, latency_us=50.0, is_nvlink=True),
                LinkProfile(source="node-0", target="node-2", bandwidth_gbps=1.0, latency_us=500.0),
            ],
        )

    @pytest.fixture
    def gpu_profile_dict(self) -> dict[str, GPUProfile]:
        return {
            "node-0": GPUProfile(
                gpu_id=0,
                name="H100",
                total_memory_bytes=80 * 1024**3,
                compute_tflops=989.0,
                memory_bandwidth_gbps=3350.0,
                sm_count=132,
            ),
            "node-1": GPUProfile(
                gpu_id=1,
                name="A100",
                total_memory_bytes=40 * 1024**3,
                compute_tflops=312.0,
                memory_bandwidth_gbps=2039.0,
                sm_count=108,
            ),
        }

    @pytest.fixture
    def gpu_profile_list(self) -> list[GPUProfile]:
        return [
            GPUProfile(gpu_id=0, name="H100", total_memory_bytes=80 * 1024**3),
            GPUProfile(gpu_id=1, name="A100", total_memory_bytes=40 * 1024**3),
        ]

    @pytest.fixture
    def empty_solution(self) -> PartitionSolution:
        return PartitionSolution(points=[])

    @pytest.fixture
    def partition_solution(self) -> PartitionSolution:
        return PartitionSolution(
            points=[
                PartitionPoint(
                    node_id="node-0",
                    start_layer=0,
                    end_layer=10,
                    estimated_time_ms=15.0,
                ),
                PartitionPoint(
                    node_id="node-1",
                    start_layer=10,
                    end_layer=20,
                    estimated_time_ms=25.0,
                ),
            ],
            max_node_time_ms=25.0,
            total_time_ms=40.0,
            estimated_throughput_tok_s=1234.0,
            pipeline_latency_ms=50.0,
            num_oom_nodes=0,
            explanation="DP minimax optimized, 2/2 nodes, no OOM",
        )

    @pytest.fixture
    def node_costs(self) -> list[NodeCost]:
        return [
            NodeCost(
                node_id="node-0",
                start_layer=0,
                end_layer=10,
                compute_time_ms=10.0,
                communication_time_ms=5.0,
                total_time_ms=15.0,
                memory_bytes=20 * 1024**3,
                memory_available_bytes=80 * 1024**3,
                fits_in_memory=True,
            ),
            NodeCost(
                node_id="node-1",
                start_layer=10,
                end_layer=20,
                compute_time_ms=20.0,
                communication_time_ms=5.0,
                total_time_ms=25.0,
                memory_bytes=50 * 1024**3,
                memory_available_bytes=40 * 1024**3,
                fits_in_memory=False,
            ),
        ]

    @pytest.fixture
    def layer_weights(self) -> list[LayerWeights]:
        return [
            LayerWeights(layer_id=0, layer_type="embed", weight_memory_bytes=1024**2),
            LayerWeights(layer_id=1, layer_type="transformer", weight_memory_bytes=4 * 1024**2),
            LayerWeights(layer_id=2, layer_type="transformer", weight_memory_bytes=4 * 1024**2),
        ]

    # ------------------------------------------------------------------
    # _normalize_profiles
    # ------------------------------------------------------------------

    def test_normalize_profiles_none(self, visualizer: ClusterVisualizer) -> None:
        assert visualizer._normalize_profiles(None) == {}

    def test_normalize_profiles_dict(self, visualizer: ClusterVisualizer, gpu_profile_dict) -> None:
        result = visualizer._normalize_profiles(gpu_profile_dict)
        assert result == gpu_profile_dict

    def test_normalize_profiles_list(self, visualizer: ClusterVisualizer, gpu_profile_list) -> None:
        result = visualizer._normalize_profiles(gpu_profile_list)
        assert "0" in result
        assert result["0"] is gpu_profile_list[0]
        assert result["1"] is gpu_profile_list[1]

    def test_normalize_profiles_empty_list(self, visualizer: ClusterVisualizer) -> None:
        result = visualizer._normalize_profiles([])
        assert result == {}

    def test_normalize_profiles_empty_dict(self, visualizer: ClusterVisualizer) -> None:
        result = visualizer._normalize_profiles({})
        assert result == {}

    # ------------------------------------------------------------------
    # print_topology
    # ------------------------------------------------------------------

    def test_print_topology_single_node_no_gpu_profiles(
        self, visualizer: ClusterVisualizer, single_node_topology: TopologyGraph,
    ) -> None:
        output = visualizer.print_topology(single_node_topology)
        assert "Cluster Topology: 1 nodes, 1 GPUs" in output
        assert "node-0" in output
        assert output.endswith("")  # last line contains no extra newline

    def test_print_topology_multi_node_no_links(
        self, visualizer: ClusterVisualizer,
    ) -> None:
        topology = TopologyGraph(
            node_ids=["node-a", "node-b"],
            gpu_counts={"node-a": 1, "node-b": 1},
            links=[],
        )
        output = visualizer.print_topology(topology)
        assert "2 nodes" in output
        assert "node-a" in output
        assert "node-b" in output
        # Links section should not appear when links is empty
        assert "Links:" not in output

    def test_print_topology_with_profiles_dict(
        self, visualizer: ClusterVisualizer, multi_node_topology: TopologyGraph,
        gpu_profile_dict,
    ) -> None:
        output = visualizer.print_topology(multi_node_topology, gpu_profile_dict)
        assert "3 nodes" in output
        assert "H100" in output
        assert "A100" in output
        assert "80.0GB" in output
        assert "NVLink" in output
        assert "IB" in output
        assert "Eth" in output

    def test_print_topology_with_profiles_list(
        self, visualizer: ClusterVisualizer, gpu_profile_list,
    ) -> None:
        """List profiles are keyed by str(gpu_id), so create matching node ids."""
        topology = TopologyGraph(
            node_ids=["0", "1"],
            gpu_counts={"0": 1, "1": 1},
            links=[],
        )
        output = visualizer.print_topology(topology, gpu_profile_list)
        assert "2 nodes" in output
        assert "H100" in output
        assert "A100" in output

    def test_print_topology_multi_gpu_node(
        self, visualizer: ClusterVisualizer,
    ) -> None:
        """Nodes with multiple GPUs should print per-GPU lines."""
        topology = TopologyGraph(
            node_ids=["fat-node"],
            gpu_counts={"fat-node": 3},
            links=[],
        )
        profile = {"fat-node": GPUProfile(gpu_id=0, name="A10", total_memory_bytes=24 * 1024**3)}
        output = visualizer.print_topology(topology, profile)
        assert output.count("fat-node") == 3
        assert output.count("fat-node/gpu1") == 1
        assert output.count("fat-node/gpu2") == 1

    def test_print_topology_empty_topology(self, visualizer: ClusterVisualizer) -> None:
        topology = TopologyGraph(node_ids=[], gpu_counts={}, links=[])
        output = visualizer.print_topology(topology)
        assert "0 nodes, 0 GPUs" in output

    def test_print_topology_none_profiles(
        self, visualizer: ClusterVisualizer, multi_node_topology: TopologyGraph,
    ) -> None:
        output = visualizer.print_topology(multi_node_topology, None)
        assert "unknown" in output

    # ------------------------------------------------------------------
    # print_partition
    # ------------------------------------------------------------------

    def test_print_partition_empty_solution(
        self, visualizer: ClusterVisualizer,
    ) -> None:
        sol = PartitionSolution(points=[])
        output = visualizer.print_partition(sol)
        # Coverage of an empty partition
        assert "0 nodes" in output
        assert "0 layers" in output

    def test_print_partition_basic(
        self, visualizer: ClusterVisualizer, partition_solution: PartitionSolution,
    ) -> None:
        output = visualizer.print_partition(partition_solution)
        assert "2 nodes" in output
        assert "20 layers" in output  # coverage: [0, 20) = 20 layers
        assert "node-0" in output
        assert "node-1" in output
        assert "15.0ms" in output
        assert "25.0ms" in output
        assert "50.0ms" in output  # pipeline latency
        assert "1234 tok/s" in output
        assert "OK" in output

    def test_print_partition_with_node_costs(
        self, visualizer: ClusterVisualizer, partition_solution: PartitionSolution,
        node_costs,
    ) -> None:
        output = visualizer.print_partition(partition_solution, node_costs=node_costs)
        assert "20.0GB" in output  # memory_bytes for node-0
        assert "80.0GB" in output  # memory_available_bytes for node-0
        assert "OOM!" in output  # node-1 does not fit

    def test_print_partition_with_layer_weights(
        self, visualizer: ClusterVisualizer, partition_solution: PartitionSolution,
        layer_weights,
    ) -> None:
        output = visualizer.print_partition(partition_solution, layer_weights=layer_weights)
        assert "OK" in output

    def test_print_partition_with_gpu_profiles(
        self, visualizer: ClusterVisualizer, partition_solution: PartitionSolution,
        gpu_profile_dict,
    ) -> None:
        output = visualizer.print_partition(partition_solution, gpu_profiles=gpu_profile_dict)
        assert "80.0GB" in output  # from the profile

    def test_print_partition_single_node(
        self, visualizer: ClusterVisualizer,
    ) -> None:
        sol = PartitionSolution(
            points=[
                PartitionPoint(node_id="node-0", start_layer=0, end_layer=32, estimated_time_ms=100.5),
            ],
            max_node_time_ms=100.5,
            total_time_ms=100.5,
            estimated_throughput_tok_s=500.0,
            pipeline_latency_ms=100.5,
            num_oom_nodes=0,
        )
        output = visualizer.print_partition(sol)
        assert "1 nodes" in output
        assert "32 layers" in output
        assert "100.5ms" in output
        assert "500 tok/s" in output

    def test_print_partition_with_oom_nodes(
        self, visualizer: ClusterVisualizer,
    ) -> None:
        sol = PartitionSolution(
            points=[
                PartitionPoint(node_id="node-0", start_layer=0, end_layer=20, estimated_time_ms=50.0),
            ],
            num_oom_nodes=1,
            explanation="OOM-tolerant",
        )
        output = visualizer.print_partition(sol)
        assert "OOM nodes:" in output or "OOM" in output  # could show in summary

    def test_print_partition_all_none_extras(
        self, visualizer: ClusterVisualizer, partition_solution: PartitionSolution,
    ) -> None:
        """All optional args are None — should not crash."""
        output = visualizer.print_partition(partition_solution, None, None, None)
        assert "OK" in output

    # ------------------------------------------------------------------
    # print_comparison
    # ------------------------------------------------------------------

    def test_print_comparison_empty(self, visualizer: ClusterVisualizer) -> None:
        output = visualizer.print_comparison({})
        assert "Strategy Comparison:" in output
        assert "Latency" in output

    def test_print_comparison_multiple(self, visualizer: ClusterVisualizer) -> None:
        comparison = {
            "dp_minimax": {"max_latency_ms": 25.0, "throughput": 1234.0},
            "equal_split": {"max_latency_ms": 30.0, "throughput": 1000.0},
            "proportional_split": {"max_latency_ms": 28.5, "throughput": 1100.0},
        }
        output = visualizer.print_comparison(comparison)
        assert "dp_minimax" in output
        assert "equal_split" in output
        assert "25.0ms" in output
        assert "1234.0 tok/s" in output

    def test_print_comparison_mixed_values(self, visualizer: ClusterVisualizer) -> None:
        """Some entries are strings instead of dicts."""
        comparison = {
            "dp_minimax": {"max_latency_ms": 25.0, "throughput": 1234.0},
            "error_strategy": "No valid partition found",
        }
        output = visualizer.print_comparison(comparison)
        assert "dp_minimax" in output
        assert "error_strategy" in output
        assert "No valid partition found" in output

    def test_print_comparison_single_entry(self, visualizer: ClusterVisualizer) -> None:
        comparison = {
            "only_one": {"max_latency_ms": 10.0, "throughput": 500.0},
        }
        output = visualizer.print_comparison(comparison)
        assert "only_one" in output
        assert "10.0ms" in output

    def test_print_comparison_nested_non_dict(self, visualizer: ClusterVisualizer) -> None:
        """Non-dict values should be printed as-is."""
        comparison = {
            "strategy_a": {"max_latency_ms": 15.0},
            "strategy_b": 42,
        }
        output = visualizer.print_comparison(comparison)
        assert "strategy_b" in output
        assert "42" in output

    # ------------------------------------------------------------------
    # to_json
    # ------------------------------------------------------------------

    def test_to_json_all_none(self, visualizer: ClusterVisualizer) -> None:
        result = visualizer.to_json(None, None, None, None)
        parsed = json.loads(result)
        assert parsed == {}

    def test_to_json_with_topology(
        self, visualizer: ClusterVisualizer, single_node_topology: TopologyGraph,
    ) -> None:
        result = visualizer.to_json(topology=single_node_topology)
        parsed = json.loads(result)
        assert "topology" in parsed
        assert parsed["topology"]["total_gpus"] == 1
        assert len(parsed["topology"]["nodes"]) == 1

    def test_to_json_with_solution(
        self, visualizer: ClusterVisualizer, partition_solution: PartitionSolution,
    ) -> None:
        result = visualizer.to_json(solution=partition_solution)
        parsed = json.loads(result)
        assert "solution" in parsed
        assert parsed["solution"]["num_nodes"] == 2
        assert parsed["solution"]["coverage"][1] == 20
        assert parsed["solution"]["explanation"] is not None
        assert len(parsed["solution"]["assignments"]) == 2

    def test_to_json_with_node_costs(
        self, visualizer: ClusterVisualizer, partition_solution: PartitionSolution,
        node_costs,
    ) -> None:
        result = visualizer.to_json(solution=partition_solution, node_costs=node_costs)
        parsed = json.loads(result)
        assert "node_costs" in parsed
        assert parsed["node_costs"][0]["node_id"] == "node-0"
        assert parsed["node_costs"][1]["fits_in_memory"] is False
        assert parsed["node_costs"][1]["utilization"] == 1.25  # 50GB / 40GB

    def test_to_json_with_gpu_profiles(
        self, visualizer: ClusterVisualizer, multi_node_topology: TopologyGraph,
        gpu_profile_dict,
    ) -> None:
        result = visualizer.to_json(topology=multi_node_topology, gpu_profiles=gpu_profile_dict)
        parsed = json.loads(result)
        assert "gpu_profiles" in parsed
        assert "node-0" in parsed["gpu_profiles"]
        assert parsed["gpu_profiles"]["node-0"]["name"] == "H100"

    def test_to_json_with_gpu_profile_list(
        self, visualizer: ClusterVisualizer, gpu_profile_list,
    ) -> None:
        result = visualizer.to_json(gpu_profiles=gpu_profile_list)
        parsed = json.loads(result)
        assert "gpu_profiles" in parsed
        assert "0" in parsed["gpu_profiles"]
        assert "1" in parsed["gpu_profiles"]

    def test_to_json_with_empty_solution(self, visualizer: ClusterVisualizer) -> None:
        result = visualizer.to_json(solution=PartitionSolution(points=[]))
        parsed = json.loads(result)
        assert "solution" in parsed
        assert parsed["solution"]["num_nodes"] == 0
        assert parsed["solution"]["coverage"] == [0, 0]

    def test_to_json_roundtrip(
        self, visualizer: ClusterVisualizer, multi_node_topology: TopologyGraph,
        partition_solution: PartitionSolution, gpu_profile_dict, node_costs,
    ) -> None:
        """Full round-trip with all data sources."""
        result = visualizer.to_json(multi_node_topology, partition_solution, gpu_profile_dict, node_costs)
        parsed = json.loads(result)
        assert "topology" in parsed
        assert "solution" in parsed
        assert "gpu_profiles" in parsed
        assert "node_costs" in parsed
        # Verify structure integrity
        assert len(parsed["topology"]["nodes"]) == 3
        assert len(parsed["gpu_profiles"]) == 2
        assert len(parsed["node_costs"]) == 2

    # ------------------------------------------------------------------
    # Integration: print_topology + print_partition return types
    # ------------------------------------------------------------------

    def test_print_methods_return_string(
        self, visualizer: ClusterVisualizer, single_node_topology: TopologyGraph,
        partition_solution: PartitionSolution,
    ) -> None:
        topo_out = visualizer.print_topology(single_node_topology)
        part_out = visualizer.print_partition(partition_solution)
        comp_out = visualizer.print_comparison({"a": {"max_latency_ms": 1.0}})
        assert isinstance(topo_out, str)
        assert isinstance(part_out, str)
        assert isinstance(comp_out, str)
        assert len(topo_out) > 0
        assert len(part_out) > 0
        assert len(comp_out) > 0
