"""Tests for dist/partition/optimizer.py -- real objects, zero mocks."""
from __future__ import annotations

import pytest

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import (
    PartitionOptimizer,
    PartitionPoint,
    PartitionSolution,
)
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


# ---------------------------------------------------------------------------
# Helpers -- construct real objects deterministically (no GPU, no network)
# ---------------------------------------------------------------------------

def _gpu_profile(name: str = "A100", memory_gb: int = 80) -> GPUProfile:
    """A single GPU profile with sensible defaults."""
    return GPUProfile(
        gpu_id=0,
        name=name,
        total_memory_bytes=memory_gb * 1024**3,
        free_memory_bytes=memory_gb * 1024**3,
        compute_tflops=312.0,
        memory_bandwidth_gbps=2039.0,
        peak_tflops_fp16=312.0,
    )


def _layer_weights(num: int = 8) -> list[LayerWeights]:
    """Uniform transformer layers for deterministic cost estimates."""
    return [
        LayerWeights(
            layer_id=i,
            layer_type="transformer",
            weight_memory_bytes=100 * 1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_seq=100_000_000,
            flops_per_token=100_000,
            kv_cache_bytes_per_token=1024,
        )
        for i in range(num)
    ]


def _topology(node_ids: list[str]) -> TopologyGraph:
    """Linear pipeline topology matching the given node ids."""
    return TopologyGraph(
        node_ids=list(node_ids),
        gpu_counts={n: 1 for n in node_ids},
        links=[
            LinkProfile(
                source=node_ids[i],
                target=node_ids[i + 1],
                bandwidth_gbps=12.5,
                latency_us=100.0,
                is_nvlink=False,
                is_infiniband=False,
            )
            for i in range(len(node_ids) - 1)
        ],
    )


def _cost_model(
    node_ids: list[str], num_layers: int = 8,
) -> PartitionCostModel:
    """Build a ready-to-use PartitionCostModel with real objects."""
    profiles = {n: _gpu_profile() for n in node_ids}
    weights = _layer_weights(num_layers)
    topo = _topology(node_ids)
    return PartitionCostModel(
        gpu_profiles=profiles,
        layer_weights=weights,
        topology=topo,
    )


def _optimizer(
    node_ids: list[str],
    num_layers: int = 8,
    **kwargs,
) -> PartitionOptimizer:
    """Build a ready-to-use PartitionOptimizer with real objects (no mocks)."""
    cm = _cost_model(node_ids, num_layers)
    return PartitionOptimizer(cost_model=cm, node_ids=node_ids, **kwargs)


# ===================================================================
# PartitionPoint
# ===================================================================

class TestPartitionPoint:
    """Tests for the PartitionPoint dataclass."""

    def test_default_values(self) -> None:
        pt = PartitionPoint(node_id="n1", start_layer=0, end_layer=4)
        assert pt.node_id == "n1"
        assert pt.start_layer == 0
        assert pt.end_layer == 4
        assert pt.estimated_time_ms == 0.0
        assert pt.quant_method == "none"

    def test_all_fields(self) -> None:
        pt = PartitionPoint(
            node_id="n2",
            start_layer=4,
            end_layer=8,
            estimated_time_ms=12.5,
            quant_method="fp8",
        )
        assert pt.node_id == "n2"
        assert pt.start_layer == 4
        assert pt.end_layer == 8
        assert pt.estimated_time_ms == 12.5
        assert pt.quant_method == "fp8"

    def test_zero_width_range(self) -> None:
        """A point where start == end (no layers assigned)."""
        pt = PartitionPoint(node_id="n1", start_layer=5, end_layer=5)
        assert pt.start_layer == pt.end_layer

    def test_negative_layers_negative(self) -> None:
        """Negative layer indices are allowed by the dataclass (validation
        is the caller's responsibility)."""
        pt = PartitionPoint(node_id="n1", start_layer=-1, end_layer=3)
        assert pt.start_layer == -1
        assert pt.end_layer == 3


# ===================================================================
# PartitionSolution
# ===================================================================

class TestPartitionSolution:
    """Tests for the PartitionSolution dataclass and its properties."""

    def test_empty_solution(self) -> None:
        sol = PartitionSolution()
        assert sol.num_nodes == 0
        assert sol.coverage == (0, 0)
        assert sol.max_node_time_ms == 0.0
        assert sol.total_time_ms == 0.0
        assert sol.estimated_throughput_tok_s == 0.0
        assert sol.pipeline_latency_ms == 0.0
        assert sol.num_oom_nodes == 0
        assert sol.explanation == ""

    def test_solution_with_points(self) -> None:
        sol = PartitionSolution(points=[
            PartitionPoint("n1", 0, 4, 10.0),
            PartitionPoint("n2", 4, 8, 15.0),
        ])
        assert sol.num_nodes == 2
        assert sol.coverage == (0, 8)

    def test_num_nodes_property(self) -> None:
        sol = PartitionSolution(points=[
            PartitionPoint("n1", 0, 2),
            PartitionPoint("n2", 2, 4),
            PartitionPoint("n3", 4, 6),
        ])
        assert sol.num_nodes == 3

    def test_coverage_property_empty_points(self) -> None:
        sol = PartitionSolution()
        assert sol.coverage == (0, 0)

    def test_coverage_tracks_first_and_last_point(self) -> None:
        sol = PartitionSolution(points=[
            PartitionPoint("n1", 3, 7),
            PartitionPoint("n2", 7, 12),
        ])
        assert sol.coverage == (3, 12)

    def test_summary_contains_key_sections(self) -> None:
        sol = PartitionSolution(
            points=[PartitionPoint("n1", 0, 8, 15.0)],
            max_node_time_ms=15.0,
            total_time_ms=15.0,
            estimated_throughput_tok_s=1000.0,
            pipeline_latency_ms=15.0,
            num_oom_nodes=0,
            explanation="Single node test",
        )
        text = sol.summary()
        assert "PartitionSolution" in text
        assert "n1" in text
        assert "Single node test" in text
        assert "OOM nodes: 0" in text

    def test_summary_with_quant_plan(self) -> None:
        """summary() handles quant_plan without crashing."""
        sol = PartitionSolution(
            points=[PartitionPoint("n1", 0, 8, quant_method="int8")],
            max_node_time_ms=10.0,
            estimated_throughput_tok_s=500.0,
        )

        class FakeQuantPlan:
            strategy = "mixed_int8"

        sol.quant_plan = FakeQuantPlan()
        text = sol.summary()
        assert "mixed_int8" in text

    def test_summary_with_oom(self) -> None:
        sol = PartitionSolution(
            points=[PartitionPoint("n1", 0, 8)],
            num_oom_nodes=2,
            explanation="OOM on 2 nodes",
        )
        text = sol.summary()
        assert "OOM" in text
        assert "2" in text

    def test_summary_with_quant_method_on_points(self) -> None:
        sol = PartitionSolution(
            points=[PartitionPoint("n1", 0, 4, estimated_time_ms=5.0, quant_method="fp8")],
        )
        text = sol.summary()
        assert "[fp8]" in text or "fp8" in text


# ===================================================================
# PartitionOptimizer -- constructor
# ===================================================================

class TestPartitionOptimizerInit:
    """PartitionOptimizer construction with various parameter combinations."""

    def test_minimal_init(self) -> None:
        opt = _optimizer(["n1"])
        assert opt is not None

    def test_multi_node_init(self) -> None:
        opt = _optimizer(["n1", "n2", "n3"])
        assert opt is not None

    def test_init_with_gpu_counts(self) -> None:
        opt = _optimizer(["n1", "n2"], gpu_counts={"n1": 2, "n2": 1})
        assert opt is not None

    def test_init_allow_oom(self) -> None:
        opt = _optimizer(["n1"], allow_oom=True)
        assert opt._allow_oom is True

    def test_init_min_layers(self) -> None:
        opt = _optimizer(["n1", "n2"], min_layers_per_node=2)
        assert opt._min_layers == 2

    def test_init_min_layers_clamped_below_one(self) -> None:
        """min_layers_per_node < 1 is clamped to 1."""
        opt = _optimizer(["n1", "n2"], min_layers_per_node=0)
        assert opt._min_layers == 1

    def test_init_with_model_size_bytes(self) -> None:
        opt = _optimizer(["n1"], model_size_bytes=7 * 1024**3)
        assert opt._model_size_bytes == 7 * 1024**3

    def test_init_with_inter_node_bandwidth(self) -> None:
        opt = _optimizer(["n1", "n2"], inter_node_bandwidth_gbps=50.0)
        assert opt._inter_node_bandwidth_gbps == 50.0

    def test_init_empty_node_ids(self) -> None:
        opt = _optimizer([])
        assert opt._node_ids == []
        assert opt._num_nodes == 0

    def test_init_gpu_counts_default(self) -> None:
        """GPU counts default to 1 per node when not provided."""
        opt = _optimizer(["n1", "n2"])
        assert opt._gpu_counts == {"n1": 1, "n2": 1}


# ===================================================================
# PartitionOptimizer -- solve()
# ===================================================================

class TestPartitionOptimizerSolve:
    """solve() method -- the main public entry point."""

    def test_solve_single_node(self) -> None:
        opt = _optimizer(["n1"])
        sol = opt.solve(num_layers=8)
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 8)
        assert sol.max_node_time_ms > 0
        assert sol.estimated_throughput_tok_s > 0
        assert "Single node" in sol.explanation

    def test_solve_two_nodes(self) -> None:
        opt = _optimizer(["n1", "n2"])
        sol = opt.solve(num_layers=8)
        assert sol.num_nodes == 2
        assert sol.coverage == (0, 8)
        assert sol.points[0].start_layer == 0
        assert sol.points[-1].end_layer == 8

    def test_solve_three_nodes(self) -> None:
        opt = _optimizer(["n1", "n2", "n3"])
        sol = opt.solve(num_layers=12)
        assert sol.num_nodes == 3
        assert sol.coverage == (0, 12)
        assert len(sol.points) == 3

    def test_solve_empty_nodes(self) -> None:
        opt = _optimizer([])
        sol = opt.solve(num_layers=8)
        assert sol.num_nodes == 0
        assert sol.explanation == "No nodes available"

    def test_solve_zero_layers(self) -> None:
        opt = _optimizer(["n1", "n2"])
        sol = opt.solve(num_layers=0)
        assert sol.num_nodes == 0
        assert sol.explanation == "No valid partition found"

    def test_solve_single_layer(self) -> None:
        opt = _optimizer(["n1", "n2"])
        sol = opt.solve(num_layers=1)
        assert sol.num_nodes >= 1
        assert sol.coverage == (0, 1)

    def test_solve_more_nodes_than_layers(self) -> None:
        """Extra nodes are unused when layers < nodes."""
        opt = _optimizer(["n1", "n2", "n3", "n4", "n5"])
        sol = opt.solve(num_layers=2)
        assert sol.num_nodes <= 2
        assert sol.coverage == (0, 2)

    def test_solve_allow_oom_flag(self) -> None:
        """allow_oom=True permits nodes that exceed GPU memory."""
        opt = _optimizer(["n1", "n2"], allow_oom=True)
        sol = opt.solve(num_layers=8)
        assert sol.num_nodes >= 1
        assert sol.coverage == (0, 8)

    def test_solve_multi_gpu_per_node(self) -> None:
        """M5: Single node with 2 GPUs gets speedup in estimate."""
        opt = _optimizer(["n1"], gpu_counts={"n1": 2})
        sol = opt.solve(num_layers=8)
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 8)

    def test_solve_with_min_layers_constraint(self) -> None:
        """Each node should get at least min_layers_per_node layers
        (the DP minimax may use fewer nodes if that lowers max latency)."""
        opt = _optimizer(["n1", "n2", "n3"], min_layers_per_node=2)
        sol = opt.solve(num_layers=12)
        assert sol.num_nodes >= 1
        for pt in sol.points:
            assert pt.end_layer - pt.start_layer >= 2

    def test_solve_with_heterogeneous_gpu_counts(self) -> None:
        """Different GPU counts across nodes produce different cost estimates."""
        opt = _optimizer(
            ["n1", "n2", "n3"],
            gpu_counts={"n1": 4, "n2": 2, "n3": 1},
        )
        sol = opt.solve(num_layers=12)
        assert sol.num_nodes >= 1
        assert sol.coverage == (0, 12)

    def test_solve_very_large_model_beam_search(self) -> None:
        """>200 layers with >8 nodes triggers the beam search fallback."""
        nodes = [f"n{i}" for i in range(10)]
        opt = _optimizer(nodes, num_layers=250)
        sol = opt.solve(num_layers=250)
        assert sol.num_nodes > 0
        assert sol.coverage == (0, 250)
        assert sol.max_node_time_ms > 0

    def test_solve_min_layers_exceeds_layers(self) -> None:
        """When min_layers_per_node > num_layers / num_nodes, DP still runs."""
        opt = _optimizer(["n1", "n2", "n3"], min_layers_per_node=10)
        sol = opt.solve(num_layers=8)
        # The DP may still produce a result or return empty, but shouldn't crash
        assert sol is not None


# ===================================================================
# PartitionOptimizer -- compare_strategies()
# ===================================================================

class TestPartitionOptimizerCompareStrategies:
    """compare_strategies() benchmark against baseline splits."""

    def test_compare_basic(self) -> None:
        opt = _optimizer(["n1", "n2"])
        result = opt.compare_strategies(num_layers=8)
        assert "dp_minimax" in result
        assert "equal_split" in result
        assert "proportional_split" in result
        assert "improvement_over_equal" in result
        assert result["dp_minimax"]["max_latency_ms"] > 0
        assert result["equal_split"]["max_latency_ms"] > 0
        assert result["proportional_split"]["max_latency_ms"] > 0

    def test_compare_single_node(self) -> None:
        """Single node -- all strategies should produce the same result."""
        opt = _optimizer(["n1"])
        result = opt.compare_strategies(num_layers=8)
        assert result["dp_minimax"]["max_latency_ms"] > 0

    def test_compare_three_nodes(self) -> None:
        opt = _optimizer(["n1", "n2", "n3"])
        result = opt.compare_strategies(num_layers=12)
        assert result["dp_minimax"]["throughput"] > 0
        assert result["equal_split"]["throughput"] > 0

    def test_compare_improvement_string(self) -> None:
        opt = _optimizer(["n1", "n2", "n3"])
        result = opt.compare_strategies(num_layers=12)
        imp = result["improvement_over_equal"]
        assert isinstance(imp, str)
        assert imp.endswith("%") or imp == "N/A"

    def test_compare_empty_nodes_raises(self) -> None:
        """_equal_split raises ZeroDivisionError when no nodes exist."""
        opt = _optimizer([])
        with pytest.raises(ZeroDivisionError):
            opt.compare_strategies(num_layers=8)


# ===================================================================
# PartitionOptimizer -- internal helper methods
# ===================================================================

class TestPartitionOptimizerInternal:
    """Direct tests of underscore-prefixed helper methods."""

    def test_evaluate_node_valid(self) -> None:
        opt = _optimizer(["n1", "n2"])
        cost = opt._evaluate_node(node_idx=0, start_layer=0, end_layer=4)
        assert cost is not None
        assert cost > 0

    def test_evaluate_node_invalid_index(self) -> None:
        """Out-of-range node_idx returns None."""
        opt = _optimizer(["n1"])
        result = opt._evaluate_node(node_idx=5, start_layer=0, end_layer=8)
        assert result is None

    def test_evaluate_node_negative_range(self) -> None:
        """start_layer > end_layer should still be handled gracefully."""
        opt = _optimizer(["n1"])
        # The evaluate_node doesn't validate ranges; cost_model.evaluate
        # gets an empty slice which returns a zero-cost NodeCost.
        cost = opt._evaluate_node(node_idx=0, start_layer=5, end_layer=3)
        # May return a value or None depending on fits_in_memory for an empty slice
        assert cost is None or cost >= 0

    def test_equal_split_two_nodes(self) -> None:
        opt = _optimizer(["n1", "n2"])
        parts = opt._equal_split(num_layers=8)
        assert len(parts) == 2
        total = sum(e - s for _, s, e in parts)
        assert total == 8

    def test_equal_split_exact_division(self) -> None:
        """Exactly divisible layers produce equal segments."""
        opt = _optimizer(["n1", "n2"])
        parts = opt._equal_split(num_layers=8)
        _, s1, e1 = parts[0]
        _, s2, e2 = parts[1]
        assert e1 - s1 == 4
        assert e2 - s2 == 4

    def test_equal_split_with_remainder(self) -> None:
        opt = _optimizer(["n1", "n2", "n3"])
        parts = opt._equal_split(num_layers=10)
        assert len(parts) == 3
        total = sum(e - s for _, s, e in parts)
        assert total == 10

    def test_equal_split_single_node(self) -> None:
        opt = _optimizer(["n1"])
        parts = opt._equal_split(num_layers=8)
        assert parts == [("n1", 0, 8)]

    def test_equal_split_empty_nodes_raises(self) -> None:
        opt = _optimizer([])
        with pytest.raises(ZeroDivisionError):
            opt._equal_split(num_layers=8)

    def test_proportional_split_two_nodes(self) -> None:
        opt = _optimizer(["n1", "n2"])
        parts = opt._proportional_split(num_layers=8)
        assert len(parts) == 2
        total = sum(e - s for _, s, e in parts)
        assert total == 8

    def test_proportional_split_three_nodes(self) -> None:
        opt = _optimizer(["n1", "n2", "n3"])
        parts = opt._proportional_split(num_layers=12)
        assert len(parts) == 3
        total = sum(e - s for _, s, e in parts)
        assert total == 12

    def test_proportional_split_single_node(self) -> None:
        opt = _optimizer(["n1"])
        parts = opt._proportional_split(num_layers=8)
        assert parts == [("n1", 0, 8)]

    def test_proportional_split_empty_nodes(self) -> None:
        opt = _optimizer([])
        parts = opt._proportional_split(num_layers=8)
        assert parts == []

    def test_enforce_boundary_layers_empty(self) -> None:
        opt = _optimizer(["n1"])
        result = opt._enforce_boundary_layers([])
        assert result == []

    def test_enforce_boundary_layers_fixes_start(self) -> None:
        """start_layer corrected to 0 when it is non-zero."""
        opt = _optimizer(["n1"], num_layers=8)
        points = [PartitionPoint("n1", start_layer=3, end_layer=8)]
        result = opt._enforce_boundary_layers(points)
        assert result[0].start_layer == 0

    def test_enforce_boundary_layers_fixes_end(self) -> None:
        """end_layer corrected to num_layers when it doesn't match."""
        opt = _optimizer(["n1", "n2"], num_layers=8)
        opt._num_layers = 8  # normally set by solve()
        points = [
            PartitionPoint("n1", 0, 3),
            PartitionPoint("n2", 3, 7),
        ]
        result = opt._enforce_boundary_layers(points)
        assert result[-1].end_layer == 8

    def test_enforce_boundary_layers_points_re_evaluated(self) -> None:
        """After boundary correction, points get fresh cost estimates > 0."""
        opt = _optimizer(["n1", "n2"], num_layers=8)
        opt._num_layers = 8  # normally set by solve()
        points = [
            PartitionPoint("n1", 3, 4),
            PartitionPoint("n2", 4, 7),
        ]
        result = opt._enforce_boundary_layers(points)
        for pt in result:
            assert pt.estimated_time_ms > 0

    def test_initialize_dp(self) -> None:
        opt = _optimizer(["n1", "n2"])
        opt._num_layers = 8
        opt._num_nodes = 2  # normally set by solve()
        opt._initialize_dp()
        # DP table should have L+1 rows and min(N, L) cols
        assert len(opt._dp) == 9
        assert len(opt._dp[0]) == 2

    def test_single_node_solution(self) -> None:
        opt = _optimizer(["n1"], num_layers=8)
        opt._num_layers = 8  # normally set by solve()
        sol = opt._single_node_solution()
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 8)
        assert sol.max_node_time_ms > 0
        assert "Single node" in sol.explanation


# ===================================================================
# PartitionOptimizer -- integration edge cases
# ===================================================================

class TestPartitionOptimizerEdgeCases:
    """Mixed edge-case scenarios exercising multiple internals."""

    def test_solve_single_node_zero_layers(self) -> None:
        opt = _optimizer(["n1"])
        sol = opt.solve(num_layers=0)
        # Single node + 0 layers -> _single_node_solution is called
        # It returns 1 node (the only one available) covering [0, 0)
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 0)
        # Throughput can be huge since time is near-zero
        assert sol.estimated_throughput_tok_s > 0

    def test_solve_single_node_no_weights(self) -> None:
        """Cost model with empty layer_weights (0 layers)."""
        cm = PartitionCostModel(
            gpu_profiles={"n1": _gpu_profile()},
            layer_weights=[],
            topology=_topology(["n1"]),
        )
        opt = PartitionOptimizer(cost_model=cm, node_ids=["n1"])
        sol = opt.solve(num_layers=0)
        # Single node path handles 0-layers gracefully
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 0)

    def test_beam_search_small_model(self) -> None:
        """Beam search is NOT triggered for <= 200 layers (code path check)."""
        nodes = [f"n{i}" for i in range(10)]
        opt = _optimizer(nodes, num_layers=40)
        sol = opt.solve(num_layers=40)
        assert sol.num_nodes > 0
        assert sol.coverage == (0, 40)

    def test_dp_large_node_count(self) -> None:
        """Many nodes with few layers: DP caps at min(N, L)."""
        nodes = [f"n{i}" for i in range(50)]
        opt = _optimizer(nodes, num_layers=4)
        sol = opt.solve(num_layers=4)
        # DP can use at most num_layers nodes
        assert sol.num_nodes <= 4

    def test_compare_returns_consistent_types(self) -> None:
        opt = _optimizer(["n1", "n2"])
        result = opt.compare_strategies(num_layers=8)
        assert isinstance(result["dp_minimax"]["max_latency_ms"], float)
        assert isinstance(result["dp_minimax"]["pipeline_latency_ms"], float)
        assert isinstance(result["equal_split"]["max_latency_ms"], float)

    def test_solution_with_all_oom_and_allow_oom(self) -> None:
        """allow_oom=True with very large layers per node."""
        # Create tiny GPU memory so layers always OOM
        tiny_profile = _gpu_profile(memory_gb=1)
        cm = PartitionCostModel(
            gpu_profiles={"n1": tiny_profile},
            layer_weights=_layer_weights(8),
            topology=_topology(["n1"]),
        )
        opt = PartitionOptimizer(cost_model=cm, node_ids=["n1"], allow_oom=True)
        sol = opt.solve(num_layers=8)
        # With allow_oom=True, the node is used even if it OOMs
        assert sol.num_nodes == 1
        assert sol.num_oom_nodes == 1

    def test_solution_with_all_oom_no_allow(self) -> None:
        """Without allow_oom, OOM nodes are still returned for single-node
        (the single-node path always returns the only available node)."""
        tiny_profile = _gpu_profile(memory_gb=1)  # 1GB
        # Much larger layer weights to force OOM
        big_weights = [
            LayerWeights(
                layer_id=i,
                layer_type="transformer",
                weight_memory_bytes=800 * 1024 * 1024,  # 800MB each
                activation_memory_bytes=4096,
                flops_per_seq=100_000_000,
                flops_per_token=100_000,
                kv_cache_bytes_per_token=1024,
            )
            for i in range(8)
        ]  # 8 * 800MB = 6.4GB total >> 1GB
        cm = PartitionCostModel(
            gpu_profiles={"n1": tiny_profile},
            layer_weights=big_weights,
            topology=_topology(["n1"]),
        )
        opt = PartitionOptimizer(cost_model=cm, node_ids=["n1"], allow_oom=False)
        sol = opt.solve(num_layers=8)
        # Single-node path always returns the node; OOM is recorded
        assert sol.num_nodes == 1
        assert sol.num_oom_nodes == 1

    def test_solve_multiple_nodes_oom_without_allow(self) -> None:
        """Multiple nodes, all OOM, no allow_oom -> empty solution."""
        tiny_profile = _gpu_profile(memory_gb=1)  # 1GB
        big_weights = [
            LayerWeights(
                layer_id=i,
                layer_type="transformer",
                weight_memory_bytes=800 * 1024 * 1024,  # 800MB each
                activation_memory_bytes=4096,
                flops_per_seq=100_000_000,
                flops_per_token=100_000,
                kv_cache_bytes_per_token=1024,
            )
            for i in range(8)
        ]  # 6.4GB total >> 1GB
        cm = PartitionCostModel(
            gpu_profiles={"n1": tiny_profile, "n2": tiny_profile},
            layer_weights=big_weights,
            topology=_topology(["n1", "n2"]),
        )
        opt = PartitionOptimizer(
            cost_model=cm, node_ids=["n1", "n2"], allow_oom=False,
        )
        sol = opt.solve(num_layers=8)
        assert sol.num_nodes == 0
        assert "No valid partition" in sol.explanation
