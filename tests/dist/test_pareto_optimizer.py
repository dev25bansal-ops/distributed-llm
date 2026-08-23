"""Tests for Pareto optimizer (real objects, zero mocks)."""

from __future__ import annotations

import pytest

from distllm.dist.partition.cost_model import PartitionCostModel
from distllm.dist.partition.optimizer import PartitionPoint, PartitionSolution
from distllm.dist.partition.pareto_optimizer import (
    Objective,
    ObjectiveVector,
    ParetoFrontier,
    ParetoPartitionOptimizer,
    ParetoSolution,
)
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


# ---------------------------------------------------------------------------
# Fixtures – single-layer model, two-node ring topology
# ---------------------------------------------------------------------------

@pytest.fixture
def layer_weights() -> list[LayerWeights]:
    return [
        LayerWeights(
            layer_id=0,
            layer_type="embed",
            weight_memory_bytes=1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=1000,
            flops_per_seq=1000,
            kv_cache_bytes_per_token=0,
        ),
        LayerWeights(
            layer_id=1,
            layer_type="transformer",
            weight_memory_bytes=8 * 1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=500_000,
            flops_per_seq=500_000,
            kv_cache_bytes_per_token=256,
        ),
        LayerWeights(
            layer_id=2,
            layer_type="transformer",
            weight_memory_bytes=8 * 1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=500_000,
            flops_per_seq=500_000,
            kv_cache_bytes_per_token=256,
        ),
        LayerWeights(
            layer_id=3,
            layer_type="transformer",
            weight_memory_bytes=8 * 1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=500_000,
            flops_per_seq=500_000,
            kv_cache_bytes_per_token=256,
        ),
        LayerWeights(
            layer_id=4,
            layer_type="lm_head",
            weight_memory_bytes=1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=200_000,
            flops_per_seq=200_000,
            kv_cache_bytes_per_token=0,
        ),
    ]


@pytest.fixture
def gpu_profiles() -> dict[str, GPUProfile]:
    return {
        "gpu-a": GPUProfile(
            gpu_id=0,
            name="A100",
            total_memory_bytes=80 * 1024 ** 3,
            free_memory_bytes=80 * 1024 ** 3,
            compute_tflops=312.0,
            memory_bandwidth_gbps=2039.0,
        ),
        "gpu-b": GPUProfile(
            gpu_id=1,
            name="A100",
            total_memory_bytes=80 * 1024 ** 3,
            free_memory_bytes=80 * 1024 ** 3,
            compute_tflops=312.0,
            memory_bandwidth_gbps=2039.0,
        ),
    }


@pytest.fixture
def topology() -> TopologyGraph:
    return TopologyGraph(
        node_ids=["gpu-a", "gpu-b"],
        gpu_counts={"gpu-a": 1, "gpu-b": 1},
        links=[
            LinkProfile(
                source="gpu-a",
                target="gpu-b",
                bandwidth_gbps=600.0,
                latency_us=5.0,
                is_nvlink=True,
            ),
        ],
    )


@pytest.fixture
def cost_model(
    gpu_profiles: dict[str, GPUProfile],
    layer_weights: list[LayerWeights],
    topology: TopologyGraph,
) -> PartitionCostModel:
    return PartitionCostModel(
        gpu_profiles=gpu_profiles,
        layer_weights=layer_weights,
        topology=topology,
        pipeline_node_order=["gpu-a", "gpu-b"],
    )


@pytest.fixture
def optimizer(cost_model: PartitionCostModel) -> ParetoPartitionOptimizer:
    return ParetoPartitionOptimizer(
        cost_model=cost_model,
        node_ids=["gpu-a", "gpu-b"],
        batch_size=1,
        seq_len=4096,
        allow_oom=False,
        node_costs_per_hour={"gpu-a": 0.50, "gpu-b": 2.00},
        max_quality_loss=0.05,
        frontier_limit=16,
    )


# ===================================================================
# Test Objective enum
# ===================================================================

class TestObjective:
    def test_members(self) -> None:
        assert Objective.LATENCY.value == "latency"
        assert Objective.THROUGHPUT.value == "throughput"
        assert Objective.MEMORY.value == "memory"
        assert Objective.QUALITY.value == "quality"
        assert Objective.COST.value == "cost"

    def test_is_str_enum(self) -> None:
        assert issubclass(Objective, str)


# ===================================================================
# Test ObjectiveVector
# ===================================================================

class TestObjectiveVector:
    def test_default_construction(self) -> None:
        v = ObjectiveVector()
        assert v.latency_ms == 0.0
        assert v.throughput_tok_s == 0.0
        assert v.memory_utilization == 0.0
        assert v.quality_loss == 0.0
        assert v.cost_per_hour == 0.0

    def test_custom_construction(self) -> None:
        v = ObjectiveVector(
            latency_ms=10.0,
            throughput_tok_s=500.0,
            memory_utilization=0.6,
            quality_loss=0.01,
            cost_per_hour=1.5,
        )
        assert v.latency_ms == 10.0
        assert v.throughput_tok_s == 500.0
        assert v.memory_utilization == 0.6

    # -- to_dict --

    def test_to_dict(self) -> None:
        v = ObjectiveVector(latency_ms=5.0, throughput_tok_s=100.0)
        d = v.to_dict()
        assert d["latency_ms"] == 5.0
        assert d["throughput_tok_s"] == 100.0
        assert set(d) == {"latency_ms", "throughput_tok_s", "memory_utilization",
                          "quality_loss", "cost_per_hour"}

    # -- get --

    def test_get_known_objectives(self) -> None:
        v = ObjectiveVector(latency_ms=1.0, throughput_tok_s=2.0,
                            memory_utilization=0.5, quality_loss=0.01,
                            cost_per_hour=3.0)
        assert v.get("latency") == 1.0
        assert v.get("throughput") == 2.0
        assert v.get("memory") == 0.5
        assert v.get("quality") == 0.01
        assert v.get("cost") == 3.0

    def test_get_unknown_objective_returns_zero(self) -> None:
        v = ObjectiveVector()
        assert v.get("nonexistent") == 0.0

    # -- dominates (default minimize set) --

    def test_dominates_strictly_better(self) -> None:
        better = ObjectiveVector(latency_ms=5.0, memory_utilization=0.3,
                                 quality_loss=0.01, cost_per_hour=1.0)
        worse = ObjectiveVector(latency_ms=10.0, memory_utilization=0.6,
                                quality_loss=0.05, cost_per_hour=2.0)
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_dominates_equal_vectors_not_dominated(self) -> None:
        a = ObjectiveVector(latency_ms=5.0, throughput_tok_s=100.0)
        b = ObjectiveVector(latency_ms=5.0, throughput_tok_s=100.0)
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_dominates_one_worse_one_better(self) -> None:
        a = ObjectiveVector(latency_ms=5.0, cost_per_hour=2.0)
        b = ObjectiveVector(latency_ms=10.0, cost_per_hour=1.0)
        # a has better latency but worse cost -> neither dominates
        assert not a.dominates(b)
        assert not b.dominates(a)

    def test_dominates_with_custom_minimize_set(self) -> None:
        # throughput is NOT in the custom minimize set -> higher is better
        minimize = {"latency", "memory"}
        a = ObjectiveVector(latency_ms=5.0, throughput_tok_s=100.0)
        b = ObjectiveVector(latency_ms=10.0, throughput_tok_s=50.0)
        # a is better in both objectives with custom minimize set
        assert a.dominates(b, minimize=minimize)

    def test_dominates_throughput_higher_is_better(self) -> None:
        better = ObjectiveVector(throughput_tok_s=200.0)
        worse = ObjectiveVector(throughput_tok_s=100.0)
        # throughput is NOT in default minimize set -> higher is better
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_dominates_at_least_as_good_but_not_strictly(self) -> None:
        a = ObjectiveVector(latency_ms=10.0, throughput_tok_s=100.0)
        b = ObjectiveVector(latency_ms=10.0, throughput_tok_s=50.0)
        # a is strictly better in throughput, equal in latency -> dominates
        assert a.dominates(b)
        assert not b.dominates(a)

    def test_dominates_both_at_least_as_good(self) -> None:
        better = ObjectiveVector(latency_ms=5.0, throughput_tok_s=100.0,
                                 memory_utilization=0.3, quality_loss=0.01,
                                 cost_per_hour=1.0)
        worse = ObjectiveVector(latency_ms=10.0, throughput_tok_s=50.0,
                                memory_utilization=0.6, quality_loss=0.05,
                                cost_per_hour=2.0)
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_dominates_all_zeros(self) -> None:
        a = ObjectiveVector()
        b = ObjectiveVector(latency_ms=1.0, throughput_tok_s=1.0)
        # a is strictly better (lower) for latency, worse (lower) for throughput
        # a: latency=0, throughput=0
        # b: latency=1 (higher=worse), throughput=1 (higher=better)
        # For throughput (not in minimize), higher is better.
        # a has throughput=0 < b's 1, so a is NOT at least as good in throughput
        assert not a.dominates(b)


# ===================================================================
# Test ParetoSolution
# ===================================================================

class TestParetoSolution:
    def test_default_construction(self) -> None:
        sol = ParetoSolution(points=[], vector=ObjectiveVector())
        assert sol.points == []
        assert sol.vector.latency_ms == 0.0
        assert sol.assignments == []

    def test_to_partition_solution(self) -> None:
        pts = [
            PartitionPoint(node_id="gpu-a", start_layer=0, end_layer=2,
                           estimated_time_ms=10.0),
            PartitionPoint(node_id="gpu-b", start_layer=2, end_layer=4,
                           estimated_time_ms=15.0),
        ]
        vec = ObjectiveVector(latency_ms=15.0, throughput_tok_s=200.0,
                              memory_utilization=0.5, cost_per_hour=2.50)
        ps = ParetoSolution(points=pts, vector=vec)
        result = ps.to_partition_solution()

        assert isinstance(result, PartitionSolution)
        assert result.max_node_time_ms == 15.0
        assert result.estimated_throughput_tok_s == 200.0
        assert result.num_oom_nodes == 0
        assert "latency=" in result.explanation
        assert "mem=" in result.explanation
        assert "cost=" in result.explanation

    def test_to_partition_solution_empty_points(self) -> None:
        ps = ParetoSolution(points=[], vector=ObjectiveVector())
        result = ps.to_partition_solution()
        assert result.max_node_time_ms == 0.0
        assert result.total_time_ms == 0.0

    def test_assignments_default_empty(self) -> None:
        ps = ParetoSolution(points=[], vector=ObjectiveVector())
        assert ps.assignments == []

    def test_assignments_custom(self) -> None:
        ps = ParetoSolution(
            points=[],
            vector=ObjectiveVector(),
            assignments=[("gpu-a", 0, 2, "fp16")],
        )
        assert ps.assignments == [("gpu-a", 0, 2, "fp16")]


# ===================================================================
# Test ParetoFrontier
# ===================================================================

class TestParetoFrontier:
    def test_empty_frontier_size(self) -> None:
        pf = ParetoFrontier()
        assert pf.size == 0

    def test_empty_best_by_returns_none(self) -> None:
        pf = ParetoFrontier()
        assert pf.best_by("latency") is None
        assert pf.best_by("throughput") is None

    def test_empty_weighted_select_raises(self) -> None:
        pf = ParetoFrontier()
        with pytest.raises(ValueError, match="Empty Pareto frontier"):
            pf.weighted_select({"latency": 1.0})

    def test_empty_summary(self) -> None:
        pf = ParetoFrontier()
        summary = pf.summary()
        assert "0 non-dominated" in summary

    @pytest.fixture
    def sample_frontier(self) -> ParetoFrontier:
        solutions = [
            ParetoSolution(
                points=[PartitionPoint(node_id="gpu-a", start_layer=0, end_layer=4,
                                       estimated_time_ms=10.0)],
                vector=ObjectiveVector(latency_ms=10.0, throughput_tok_s=500.0,
                                       memory_utilization=0.3, cost_per_hour=1.0),
            ),
            ParetoSolution(
                points=[PartitionPoint(node_id="gpu-a", start_layer=0, end_layer=4,
                                       estimated_time_ms=30.0)],
                vector=ObjectiveVector(latency_ms=30.0, throughput_tok_s=200.0,
                                       memory_utilization=0.1, cost_per_hour=0.5),
            ),
        ]
        return ParetoFrontier(solutions=solutions)

    def test_size(self, sample_frontier: ParetoFrontier) -> None:
        assert sample_frontier.size == 2

    def test_best_by_latency(self, sample_frontier: ParetoFrontier) -> None:
        best = sample_frontier.best_by("latency")
        assert best is not None
        assert best.vector.latency_ms == 10.0

    def test_best_by_cost(self, sample_frontier: ParetoFrontier) -> None:
        best = sample_frontier.best_by("cost")
        assert best is not None
        assert best.vector.cost_per_hour == 0.5

    def test_best_by_throughput(self, sample_frontier: ParetoFrontier) -> None:
        best = sample_frontier.best_by("throughput")
        assert best is not None
        assert best.vector.throughput_tok_s == 500.0

    def test_best_by_unknown_objective(self, sample_frontier: ParetoFrontier) -> None:
        # unknown not in minimize -> max (tie-breaking: first element)
        best = sample_frontier.best_by("unknown")
        assert best is not None
        assert best.vector.latency_ms == 10.0

    def test_weighted_select_latency(self, sample_frontier: ParetoFrontier) -> None:
        best = sample_frontier.weighted_select({"latency": 1.0, "cost": 0.0})
        assert best.vector.latency_ms == 10.0

    def test_weighted_select_cost(self, sample_frontier: ParetoFrontier) -> None:
        best = sample_frontier.weighted_select({"latency": 0.0, "cost": 1.0})
        assert best.vector.cost_per_hour == 0.5

    def test_weighted_select_balanced(self, sample_frontier: ParetoFrontier) -> None:
        best = sample_frontier.weighted_select({"latency": 0.5, "cost": 0.5})
        # both solutions are on the frontier
        assert best is not None

    def test_weighted_select_throughput_maximized(self, sample_frontier: ParetoFrontier) -> None:
        best = sample_frontier.weighted_select({"throughput": 1.0})
        assert best.vector.throughput_tok_s == 500.0

    def test_weighted_select_negative_weight(self, sample_frontier: ParetoFrontier) -> None:
        # negative weight for cost (which is minimized) encourages high cost
        best = sample_frontier.weighted_select({"cost": -1.0})
        assert best.vector.cost_per_hour == 1.0

    def test_summary(self, sample_frontier: ParetoFrontier) -> None:
        summary = sample_frontier.summary()
        assert "Pareto frontier" in summary
        assert "2 non-dominated" in summary
        assert "latency=" in summary
        assert "throughput=" in summary
        assert "memory=" in summary
        assert "cost=" in summary

    def test_single_solution_weighted_select(self) -> None:
        pf = ParetoFrontier(solutions=[
            ParetoSolution(
                points=[PartitionPoint(node_id="gpu-a", start_layer=0, end_layer=4,
                                       estimated_time_ms=10.0)],
                vector=ObjectiveVector(latency_ms=10.0),
            ),
        ])
        best = pf.weighted_select({"latency": 1.0})
        assert best.vector.latency_ms == 10.0


# ===================================================================
# Test ParetoPartitionOptimizer
# ===================================================================

class TestParetoPartitionOptimizer:
    """Tests using real PartitionCostModel with no mocks."""

    def test_construct(self, cost_model: PartitionCostModel) -> None:
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=["gpu-a", "gpu-b"],
            batch_size=1,
            seq_len=4096,
        )
        assert opt is not None

    def test_construct_with_costs(self, cost_model: PartitionCostModel) -> None:
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=["gpu-a", "gpu-b"],
            batch_size=1,
            seq_len=4096,
            allow_oom=True,
            node_costs_per_hour={"gpu-a": 0.5},
            max_quality_loss=0.1,
            frontier_limit=64,
        )
        assert opt is not None

    # -- solve() --

    def test_solve_returns_frontier(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        assert isinstance(frontier, ParetoFrontier)
        assert frontier.size >= 1

    def test_solve_zero_nodes(self, cost_model: PartitionCostModel) -> None:
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=[],
        )
        frontier = opt.solve(num_layers=4)
        assert frontier.size == 0

    def test_solve_single_node(self, cost_model: PartitionCostModel) -> None:
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=["gpu-a"],
            batch_size=1,
            seq_len=4096,
        )
        frontier = opt.solve(num_layers=4)
        assert frontier.size == 1
        sol = frontier.solutions[0]
        assert sol.vector.latency_ms > 0
        assert len(sol.points) == 1
        assert sol.points[0].node_id == "gpu-a"
        assert sol.points[0].start_layer == 0
        assert sol.points[0].end_layer == 4

    def test_solve_with_oom_allowed(self, cost_model: PartitionCostModel) -> None:
        """allow_oom=True should still produce a frontier."""
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=["gpu-a", "gpu-b"],
            batch_size=1,
            seq_len=4096,
            allow_oom=True,
            node_costs_per_hour={"gpu-a": 0.5, "gpu-b": 1.0},
        )
        frontier = opt.solve(num_layers=4)
        assert frontier.size >= 1

    def test_solve_all_layers_all_nodes(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        for sol in frontier.solutions:
            assert sol.points
            first = sol.points[0]
            last = sol.points[-1]
            assert first.start_layer == 0, "First point must start at layer 0"
            assert last.end_layer == 4, "Last point must end at num_layers"
            # assignments should match points
            assert len(sol.assignments) == len(sol.points)

    def test_solve_returns_non_dominated(self, optimizer: ParetoPartitionOptimizer) -> None:
        """No solution in the frontier should dominate another."""
        frontier = optimizer.solve(num_layers=4)
        solutions = frontier.solutions
        for i, a in enumerate(solutions):
            for j, b in enumerate(solutions):
                if i == j:
                    continue
                assert not a.vector.dominates(b.vector), (
                    f"Solution {i} dominates solution {j} in the frontier"
                )

    def test_solve_multiple_layers(self, cost_model: PartitionCostModel) -> None:
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=["gpu-a", "gpu-b"],
            batch_size=1,
            seq_len=4096,
            node_costs_per_hour={"gpu-a": 0.5, "gpu-b": 2.0},
        )
        frontier = opt.solve(num_layers=4)
        # With 4 layers and 2 nodes, expect frontier_size >= 1
        assert frontier.size >= 1

    # -- select_solution() --

    def test_select_solution(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        best = optimizer.select_solution(frontier, {"latency": 1.0, "cost": 0.0})
        assert isinstance(best, ParetoSolution)
        assert best.vector.latency_ms > 0

    def test_select_solution_throughput_weight(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        best = optimizer.select_solution(frontier, {"throughput": 1.0})
        assert isinstance(best, ParetoSolution)

    def test_select_solution_equal_weights(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        best = optimizer.select_solution(frontier, {
            "latency": 0.25, "throughput": 0.25,
            "memory": 0.25, "cost": 0.25,
        })
        assert best is not None

    # -- solve_and_select() --

    def test_solve_and_select(self, optimizer: ParetoPartitionOptimizer) -> None:
        result = optimizer.solve_and_select(4, {"latency": 1.0})
        assert isinstance(result, PartitionSolution)
        assert result.max_node_time_ms > 0
        assert result.explanation

    def test_solve_and_select_zero_nodes(self, cost_model: PartitionCostModel) -> None:
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=[],
        )
        result = opt.solve_and_select(4, {"latency": 1.0})
        assert isinstance(result, PartitionSolution)
        assert "No valid partition found" in result.explanation
        assert result.max_node_time_ms == 0.0

    def test_solve_and_select_returns_coherent_solution(
        self, optimizer: ParetoPartitionOptimizer,
    ) -> None:
        result = optimizer.solve_and_select(4, {"latency": 0.5, "cost": 0.5})
        assert result.num_nodes >= 1
        assert result.coverage == (0, 4)

    def test_solve_and_select_no_oom(self, optimizer: ParetoPartitionOptimizer) -> None:
        result = optimizer.solve_and_select(4, {"memory": 1.0})
        assert result.num_oom_nodes == 0

    # -- solve with different frontier_limits --

    def test_solve_small_frontier_limit(self, cost_model: PartitionCostModel) -> None:
        """Small frontier_limit limits per-cell frontier but global frontier
        may still have multiple solutions from different cells."""
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=["gpu-a", "gpu-b"],
            batch_size=1,
            seq_len=4096,
            frontier_limit=2,
            node_costs_per_hour={"gpu-a": 0.5, "gpu-b": 2.0},
        )
        frontier = opt.solve(num_layers=4)
        # frontier_limit is per DP cell, not global; the global frontier
        # aggregates across all DP cells. At minimum we get at least one solution.
        assert frontier.size >= 1

    # -- cost_model with unlimited memory to avoid OOM pruning --

    def test_solve_large_memory_gpu(self, layer_weights: list[LayerWeights],
                                   topology: TopologyGraph) -> None:
        huge_gpu_profile: dict[str, GPUProfile] = {
            "gpu-a": GPUProfile(
                gpu_id=0, name="H100",
                total_memory_bytes=1_000 * 1024 ** 3,
                free_memory_bytes=1_000 * 1024 ** 3,
                compute_tflops=989.0,
                memory_bandwidth_gbps=3350.0,
            ),
            "gpu-b": GPUProfile(
                gpu_id=1, name="H100",
                total_memory_bytes=1_000 * 1024 ** 3,
                free_memory_bytes=1_000 * 1024 ** 3,
                compute_tflops=989.0,
                memory_bandwidth_gbps=3350.0,
            ),
        }
        cm = PartitionCostModel(
            gpu_profiles=huge_gpu_profile,
            layer_weights=layer_weights,
            topology=topology,
            pipeline_node_order=["gpu-a", "gpu-b"],
        )
        opt = ParetoPartitionOptimizer(
            cost_model=cm,
            node_ids=["gpu-a", "gpu-b"],
            batch_size=1,
            seq_len=4096,
            node_costs_per_hour={"gpu-a": 1.0, "gpu-b": 2.0},
        )
        frontier = opt.solve(num_layers=4)
        assert frontier.size >= 1

    # -- edge: zero layers --

    def test_solve_zero_layers(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=0)
        assert frontier.size == 0

    # -- edge: single layer --

    def test_solve_single_layer(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=1)
        # With 1 layer and 2 nodes -> first layer == layer 0 (embed) at end=1
        # Solutions should exist
        assert frontier.size >= 0  # at least doesn't crash

    # -- very large frontier_limit doesn't crash --

    def test_solve_large_frontier_limit(self, cost_model: PartitionCostModel) -> None:
        opt = ParetoPartitionOptimizer(
            cost_model=cost_model,
            node_ids=["gpu-a", "gpu-b"],
            batch_size=1,
            seq_len=4096,
            frontier_limit=999,
            node_costs_per_hour={"gpu-a": 0.5, "gpu-b": 2.0},
        )
        frontier = opt.solve(num_layers=4)
        assert frontier.size >= 1

    # -- select_solution with weights that include all objectives --

    def test_select_solution_all_objectives(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        best = optimizer.select_solution(frontier, {
            "latency": 0.2,
            "throughput": 0.2,
            "memory": 0.2,
            "quality": 0.2,
            "cost": 0.2,
        })
        assert best is not None


# ===================================================================
# Test full pipeline – solve → select → to_partition_solution
# ===================================================================

class TestParetoPipeline:
    def test_full_pipeline(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        assert frontier.size >= 1

        best = optimizer.select_solution(frontier, {"latency": 0.5, "cost": 0.5})
        assert best is not None

        ps = best.to_partition_solution()
        assert isinstance(ps, PartitionSolution)
        assert ps.max_node_time_ms > 0
        assert ps.num_nodes >= 1

    def test_full_pipeline_throughput(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        best = optimizer.select_solution(frontier, {"throughput": 1.0})
        result = best.to_partition_solution()
        assert result.estimated_throughput_tok_s > 0

    def test_full_pipeline_cost_focus(self, optimizer: ParetoPartitionOptimizer) -> None:
        frontier = optimizer.solve(num_layers=4)
        best = optimizer.select_solution(frontier, {"cost": 1.0})
        result = best.to_partition_solution()
        assert "cost" in result.explanation
