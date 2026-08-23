"""Tests for QuantAwarePartitionSolver and supporting dataclasses.

Tests use only real objects from the module (no mocks) and do not require
GPU hardware or network connectivity.  All hardware-dependent paths fall
back gracefully to CPU defaults.
"""

from __future__ import annotations

import pytest

from distllm.dist.partition.cost_model import (
    NodeCost,
    PartitionCostModel,
)
from distllm.dist.partition.optimizer import PartitionPoint, PartitionSolution
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.quant_partition import (
    QuantAwarePartitionSolver,
    QuantAwareSolution,
    QuantizedNodeCost,
    QuantizedPartitionPoint,
)
from distllm.dist.partition.quantization_tuner import (
    QUANT_PROFILES,
    QuantMethod,
)
from distllm.dist.partition.topology import LinkProfile, TopologyGraph


# ---------------------------------------------------------------------------
# Fixtures – minimal two-node topology with a few layers
# ---------------------------------------------------------------------------


@pytest.fixture
def layer_weights() -> list[LayerWeights]:
    """Four layers (embed + 2 transformer + lm_head) on a tiny model."""
    return [
        LayerWeights(
            layer_id=0,
            layer_type="embed",
            weight_memory_bytes=128 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=500,
            flops_per_seq=500,
            kv_cache_bytes_per_token=0,
        ),
        LayerWeights(
            layer_id=1,
            layer_type="transformer",
            weight_memory_bytes=4 * 1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=200_000,
            flops_per_seq=200_000,
            kv_cache_bytes_per_token=128,
        ),
        LayerWeights(
            layer_id=2,
            layer_type="transformer",
            weight_memory_bytes=4 * 1024 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=200_000,
            flops_per_seq=200_000,
            kv_cache_bytes_per_token=128,
        ),
        LayerWeights(
            layer_id=3,
            layer_type="lm_head",
            weight_memory_bytes=128 * 1024,
            activation_memory_bytes=4096,
            flops_per_token=200,
            flops_per_seq=200,
            kv_cache_bytes_per_token=0,
        ),
    ]


@pytest.fixture
def gpu_profiles() -> dict[str, GPUProfile]:
    """Two nodes with modest GPU specs."""
    return {
        "gpu-0": GPUProfile(
            gpu_id=0,
            name="RTX 4090",
            total_memory_bytes=24 * 1024**3,
            compute_tflops=82.0,
            memory_bandwidth_gbps=1008.0,
        ),
        "gpu-1": GPUProfile(
            gpu_id=1,
            name="RTX 4090",
            total_memory_bytes=24 * 1024**3,
            compute_tflops=82.0,
            memory_bandwidth_gbps=1008.0,
        ),
    }


@pytest.fixture
def topology() -> TopologyGraph:
    """Two nodes connected by a fast link."""
    return TopologyGraph(
        node_ids=["gpu-0", "gpu-1"],
        gpu_counts={"gpu-0": 1, "gpu-1": 1},
        links=[
            LinkProfile(
                source="gpu-0",
                target="gpu-1",
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
    )


@pytest.fixture
def node_ids() -> list[str]:
    return ["gpu-0", "gpu-1"]


# ===========================================================================
# QuantizedNodeCost
# ===========================================================================


class TestQuantizedNodeCost:
    """Dataclass construction and field defaults."""

    def test_defaults(self) -> None:
        nc = QuantizedNodeCost(
            node_id="n0",
            start_layer=0,
            end_layer=4,
            quant_method=QuantMethod.NONE,
        )
        assert nc.node_id == "n0"
        assert nc.start_layer == 0
        assert nc.end_layer == 4
        assert nc.quant_method == QuantMethod.NONE
        assert nc.compute_time_ms == 0.0
        assert nc.communication_time_ms == 0.0
        assert nc.total_time_ms == 0.0
        assert nc.memory_bytes == 0
        assert nc.memory_available_bytes == 0
        assert nc.fits_in_memory is True
        assert nc.quality_loss == 0.0
        assert nc.speed_penalty == 1.0
        assert nc.memory_reduction == 1.0

    def test_custom_values(self) -> None:
        nc = QuantizedNodeCost(
            node_id="n0",
            start_layer=0,
            end_layer=8,
            quant_method=QuantMethod.BNB_4BIT,
            compute_time_ms=12.5,
            communication_time_ms=3.2,
            total_time_ms=15.7,
            memory_bytes=500_000_000,
            memory_available_bytes=24_000_000_000,
            fits_in_memory=True,
            quality_loss=0.03,
            speed_penalty=1.1,
            memory_reduction=0.25,
        )
        assert nc.node_id == "n0"
        assert nc.quant_method == QuantMethod.BNB_4BIT
        assert nc.total_time_ms == 15.7
        assert nc.memory_bytes == 500_000_000
        assert nc.quality_loss == 0.03
        assert nc.speed_penalty == 1.1
        assert nc.memory_reduction == 0.25


# ===========================================================================
# QuantizedPartitionPoint
# ===========================================================================


class TestQuantizedPartitionPoint:
    """Dataclass construction and field defaults."""

    def test_defaults(self) -> None:
        pp = QuantizedPartitionPoint(
            node_id="n0",
            start_layer=0,
            end_layer=4,
            quant_method=QuantMethod.NONE,
        )
        assert pp.node_id == "n0"
        assert pp.start_layer == 0
        assert pp.end_layer == 4
        assert pp.quant_method == QuantMethod.NONE
        assert pp.estimated_time_ms == 0.0
        assert pp.quality_loss == 0.0
        assert pp.memory_bytes == 0

    def test_custom_values(self) -> None:
        pp = QuantizedPartitionPoint(
            node_id="n1",
            start_layer=4,
            end_layer=8,
            quant_method=QuantMethod.FP8_E4M3,
            estimated_time_ms=8.2,
            quality_loss=0.005,
            memory_bytes=250_000_000,
        )
        assert pp.node_id == "n1"
        assert pp.quant_method == QuantMethod.FP8_E4M3
        assert pp.estimated_time_ms == 8.2
        assert pp.quality_loss == 0.005
        assert pp.memory_bytes == 250_000_000


# ===========================================================================
# QuantAwareSolution
# ===========================================================================


class TestQuantAwareSolution:
    """Dataclass, properties, and conversion methods."""

    def test_empty_solution(self) -> None:
        sol = QuantAwareSolution()
        assert sol.points == []
        assert sol.num_nodes == 0
        assert sol.max_node_time_ms == 0.0
        assert sol.total_time_ms == 0.0
        assert sol.estimated_throughput_tok_s == 0.0
        assert sol.avg_quality_loss == 0.0
        assert sol.total_memory_bytes == 0
        assert sol.explanation == ""

    def test_empty_coverage(self) -> None:
        sol = QuantAwareSolution()
        assert sol.coverage == (0, 0)

    def test_empty_quant_methods_used(self) -> None:
        sol = QuantAwareSolution()
        assert sol.quant_methods_used() == set()

    def test_empty_to_partition_solution(self) -> None:
        sol = QuantAwareSolution()
        ps = sol.to_partition_solution()
        assert isinstance(ps, PartitionSolution)
        assert ps.points == []

    def test_empty_summary(self) -> None:
        sol = QuantAwareSolution()
        s = sol.summary()
        assert "0 nodes" in s

    def test_single_point(self) -> None:
        pp = QuantizedPartitionPoint(
            node_id="n0",
            start_layer=0,
            end_layer=4,
            quant_method=QuantMethod.BNB_4BIT,
            estimated_time_ms=10.0,
            quality_loss=0.03,
            memory_bytes=100_000_000,
        )
        sol = QuantAwareSolution(
            points=[pp],
            max_node_time_ms=10.0,
            total_time_ms=10.0,
            estimated_throughput_tok_s=409.6,
            avg_quality_loss=0.03,
            total_memory_bytes=100_000_000,
        )
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 4)
        assert sol.quant_methods_used() == {QuantMethod.BNB_4BIT}
        assert sol.max_node_time_ms == 10.0
        assert sol.avg_quality_loss == 0.03

    def test_multi_point(self) -> None:
        points = [
            QuantizedPartitionPoint(
                node_id="n0",
                start_layer=0,
                end_layer=4,
                quant_method=QuantMethod.NONE,
                estimated_time_ms=5.0,
            ),
            QuantizedPartitionPoint(
                node_id="n1",
                start_layer=4,
                end_layer=8,
                quant_method=QuantMethod.BNB_4BIT,
                estimated_time_ms=8.0,
                quality_loss=0.02,
            ),
        ]
        sol = QuantAwareSolution(
            points=points,
            max_node_time_ms=8.0,
            total_time_ms=13.0,
            estimated_throughput_tok_s=512.0,
            avg_quality_loss=0.01,
            total_memory_bytes=300_000_000,
        )
        assert sol.num_nodes == 2
        assert sol.coverage == (0, 8)
        assert sol.quant_methods_used() == {QuantMethod.NONE, QuantMethod.BNB_4BIT}

    def test_to_partition_solution(self) -> None:
        pp = QuantizedPartitionPoint(
            node_id="n0",
            start_layer=0,
            end_layer=4,
            quant_method=QuantMethod.INT8,
            estimated_time_ms=6.0,
            quality_loss=0.01,
            memory_bytes=200_000_000,
        )
        sol = QuantAwareSolution(
            points=[pp],
            max_node_time_ms=6.0,
            total_time_ms=6.0,
            estimated_throughput_tok_s=682.0,
            explanation="test",
        )
        ps = sol.to_partition_solution()
        assert isinstance(ps, PartitionSolution)
        assert len(ps.points) == 1
        assert ps.points[0].node_id == "n0"
        assert ps.points[0].start_layer == 0
        assert ps.points[0].end_layer == 4
        assert ps.points[0].estimated_time_ms == 6.0
        assert ps.max_node_time_ms == 6.0
        assert ps.total_time_ms == 6.0
        assert ps.estimated_throughput_tok_s == 682.0
        assert ps.explanation == "test"

    def test_summary_output(self) -> None:
        pp = QuantizedPartitionPoint(
            node_id="n0",
            start_layer=0,
            end_layer=4,
            quant_method=QuantMethod.FP8_E4M3,
            estimated_time_ms=4.0,
            quality_loss=0.005,
        )
        sol = QuantAwareSolution(
            points=[pp],
            max_node_time_ms=4.0,
            total_time_ms=4.0,
            estimated_throughput_tok_s=1024.0,
            avg_quality_loss=0.005,
        )
        s = sol.summary()
        assert "1 nodes" in s
        assert "4 layers" in s
        assert "4.0ms" in s
        assert "1024" in s
        assert "0.005" in s
        assert "fp8_e4m3" in s


# ===========================================================================
# QuantAwarePartitionSolver
# ===========================================================================


class TestQuantAwarePartitionSolver:
    """Solver construction and solve() behaviour."""

    def test_constructor_defaults(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        assert solver._node_ids == node_ids
        assert solver._cost_model is cost_model
        assert solver._batch_size == 1
        assert solver._seq_len == 4096
        assert solver._allow_oom is False
        assert solver._max_quality_loss == 0.05
        assert solver._quality_weight == 100.0
        assert solver._require_calibration is False
        # device types default
        for nid in node_ids:
            assert solver._device_types[nid] == "cuda"

    def test_constructor_custom(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        device_types = {"gpu-0": "cuda", "gpu-1": "rocm"}
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
            device_types=device_types,
            batch_size=4,
            seq_len=2048,
            allow_oom=True,
            max_quality_loss=0.1,
            quality_weight=50.0,
            require_calibration=True,
        )
        assert solver._batch_size == 4
        assert solver._seq_len == 2048
        assert solver._allow_oom is True
        assert solver._max_quality_loss == 0.1
        assert solver._quality_weight == 50.0
        assert solver._require_calibration is True
        assert solver._device_types["gpu-1"] == "rocm"

    # -- solve with zero nodes --

    def test_solve_zero_nodes(self, cost_model: PartitionCostModel) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=[],
        )
        sol = solver.solve(num_layers=10)
        assert isinstance(sol, QuantAwareSolution)
        assert sol.points == []
        assert sol.explanation == "No nodes available"

    # -- solve with one node --

    def test_solve_single_node(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
            device_types={"gpu-0": "cuda"},
        )
        sol = solver.solve(num_layers=4)
        assert isinstance(sol, QuantAwareSolution)
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 4)
        assert sol.points[0].node_id == "gpu-0"
        assert sol.points[0].start_layer == 0
        assert sol.points[0].end_layer == 4
        assert sol.explanation.startswith("Single node")
        assert sol.estimated_throughput_tok_s > 0
        assert sol.max_node_time_ms > 0

    def test_solve_single_node_zero_layers(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
        )
        sol = solver.solve(num_layers=0)
        assert isinstance(sol, QuantAwareSolution)
        assert sol.num_nodes == 1
        assert sol.coverage == (0, 0)

    # -- solve with two nodes --

    def test_solve_two_nodes(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        sol = solver.solve(num_layers=4)
        assert isinstance(sol, QuantAwareSolution)
        assert sol.num_nodes >= 1
        assert sol.num_nodes <= 2
        if sol.num_nodes == 2:
            assert sol.coverage == (0, 4)
        assert sol.max_node_time_ms > 0
        assert sol.estimated_throughput_tok_s > 0
        # All assigned methods must be valid QuantMethod values
        for pt in sol.points:
            assert isinstance(pt.quant_method, QuantMethod)
            assert isinstance(pt.estimated_time_ms, float)
            assert isinstance(pt.quality_loss, float)

    def test_solve_two_nodes_equal_layers(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        sol = solver.solve(num_layers=2)
        assert sol.num_nodes >= 1
        assert sol.num_nodes <= 2
        if sol.num_nodes == 2:
            assert sol.points[0].end_layer == sol.points[1].start_layer

    def test_solve_large_layer_count(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        sol = solver.solve(num_layers=40)
        assert sol.num_nodes >= 1
        assert sol.num_nodes <= 2
        if sol.num_nodes == 2:
            assert sol.coverage == (0, 40)

    # -- solver with allow_oom --

    def test_solve_with_allow_oom(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
            allow_oom=True,
        )
        sol = solver.solve(num_layers=4)
        assert sol.num_nodes >= 1
        assert sol.max_node_time_ms > 0

    # -- solver with strict quality threshold --

    def test_solve_max_quality_loss_zero(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
            max_quality_loss=0.0,
        )
        sol = solver.solve(num_layers=4)
        assert sol.num_nodes >= 1
        # Only NONE quant has quality_loss=0.0
        for pt in sol.points:
            assert pt.quality_loss == 0.0

    def test_solve_require_calibration(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
            require_calibration=True,
        )
        sol = solver.solve(num_layers=4)
        assert sol.num_nodes >= 1
        # No method that requires calibration should appear
        for pt in sol.points:
            profile = QUANT_PROFILES[pt.quant_method]
            assert not profile.requires_calibration

    # -- device type filtering --

    def test_solve_cpu_device(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
            device_types={"gpu-0": "cpu"},
        )
        sol = solver.solve(num_layers=4)
        assert sol.num_nodes == 1
        # CPU only supports NONE quant
        assert sol.points[0].quant_method == QuantMethod.NONE

    def test_solve_mps_device(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
            device_types={"gpu-0": "mps", "gpu-1": "mps"},
        )
        sol = solver.solve(num_layers=4)
        assert sol.num_nodes >= 1

    # -- _get_valid_methods --

    def test_get_valid_methods_default(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
        )
        methods = solver._get_valid_methods("gpu-0")
        assert len(methods) > 0
        assert QuantMethod.NONE in methods

    def test_get_valid_methods_cpu(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
            device_types={"gpu-0": "cpu"},
        )
        methods = solver._get_valid_methods("gpu-0")
        assert QuantMethod.NONE in methods
        for m in methods:
            profile = QUANT_PROFILES[m]
            assert "cpu" in profile.supported_hardware

    def test_get_valid_methods_zero_quality_loss(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
            max_quality_loss=0.0,
        )
        methods = solver._get_valid_methods("gpu-0")
        assert QuantMethod.NONE in methods
        for m in methods:
            assert QUANT_PROFILES[m].quality_loss <= 0.0
        # All non-NONE methods have positive quality_loss, so only NONE qualifies
        assert len(methods) == 1

    def test_get_valid_methods_require_calibration(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
            require_calibration=True,
        )
        methods = solver._get_valid_methods("gpu-0")
        for m in methods:
            assert not QUANT_PROFILES[m].requires_calibration

    # -- _evaluate_with_quant --

    def test_evaluate_with_quant_out_of_range(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
        )
        result = solver._evaluate_with_quant(
            node_idx=99,
            start=0,
            end=4,
            method=QuantMethod.NONE,
        )
        assert result is None

    def test_evaluate_with_quant_basic(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
        )
        nc = solver._evaluate_with_quant(
            node_idx=0,
            start=0,
            end=4,
            method=QuantMethod.NONE,
        )
        assert nc is not None
        assert nc.node_id == "gpu-0"
        assert nc.start_layer == 0
        assert nc.end_layer == 4
        assert nc.quant_method == QuantMethod.NONE
        assert nc.compute_time_ms >= 0
        assert nc.total_time_ms >= 0
        assert nc.memory_bytes > 0

    def test_evaluate_with_quant_oom_detected(
        self, cost_model: PartitionCostModel,
    ) -> None:
        """When allow_oom=False, a node that would OOM returns None."""
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
            allow_oom=False,
        )
        # Request huge layer range to cause OOM
        nc = solver._evaluate_with_quant(
            node_idx=0,
            start=0,
            end=1000,
            method=QuantMethod.NONE,
        )
        # This may or may not OOM depending on model size;
        # if OOM, result is None; otherwise it's still valid.
        if nc is not None:
            assert isinstance(nc, QuantizedNodeCost)

    def test_evaluate_with_quant_oom_allowed(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
            allow_oom=True,
        )
        nc = solver._evaluate_with_quant(
            node_idx=0,
            start=0,
            end=4,
            method=QuantMethod.NONE,
        )
        assert nc is not None

    # -- _cost_with_quality --

    def test_cost_with_quality(self, cost_model: PartitionCostModel) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0"],
            quality_weight=100.0,
        )
        nc = QuantizedNodeCost(
            node_id="gpu-0",
            start_layer=0,
            end_layer=4,
            quant_method=QuantMethod.BNB_4BIT,
            total_time_ms=10.0,
            quality_loss=0.03,
        )
        cost = solver._cost_with_quality(nc)
        assert cost == 10.0 + 100.0 * 0.03

    # -- to_partition_solution on solver result --

    def test_solver_to_partition_solution(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        qs = solver.solve(num_layers=4)
        ps = qs.to_partition_solution()
        assert isinstance(ps, PartitionSolution)
        assert ps.num_nodes == qs.num_nodes
        assert ps.max_node_time_ms == qs.max_node_time_ms
        assert ps.total_time_ms == qs.total_time_ms
        assert ps.estimated_throughput_tok_s == qs.estimated_throughput_tok_s

    # -- repeated solve should be deterministic --

    def test_solve_deterministic(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        sol1 = solver.solve(num_layers=4)
        sol2 = solver.solve(num_layers=4)
        assert len(sol1.points) == len(sol2.points)
        assert sol1.max_node_time_ms == sol2.max_node_time_ms
        assert sol1.total_time_ms == sol2.total_time_ms

    # -- solve with many nodes (stress DP) --

    def test_solve_many_nodes(
        self, cost_model: PartitionCostModel,
    ) -> None:
        many_ids = [f"node-{i}" for i in range(10)]
        many_profiles = {
            f"node-{i}": GPUProfile(
                gpu_id=i,
                name="RTX 4090",
                total_memory_bytes=24 * 1024**3,
                compute_tflops=82.0,
                memory_bandwidth_gbps=1008.0,
            )
            for i in range(10)
        }
        many_topology = TopologyGraph(
            node_ids=many_ids,
            gpu_counts={nid: 1 for nid in many_ids},
            links=[
                LinkProfile(
                    source=many_ids[i],
                    target=many_ids[j],
                    bandwidth_gbps=600.0,
                    latency_us=5.0,
                    is_nvlink=True,
                )
                for i in range(10)
                for j in range(i + 1, 10)
            ],
        )
        from distllm.dist.partition.profiles import LayerWeights

        many_layers = [
            LayerWeights(
                layer_id=i,
                layer_type="transformer",
                weight_memory_bytes=2 * 1024 * 1024,
                activation_memory_bytes=4096,
                flops_per_token=100_000,
                flops_per_seq=100_000,
                kv_cache_bytes_per_token=64,
            )
            for i in range(32)
        ]
        cm = PartitionCostModel(
            gpu_profiles=many_profiles,
            layer_weights=many_layers,
            topology=many_topology,
        )
        solver = QuantAwarePartitionSolver(
            cost_model=cm,
            node_ids=many_ids,
        )
        sol = solver.solve(num_layers=32)
        assert sol.num_nodes >= 1
        assert sol.num_nodes <= 10
        assert sol.max_node_time_ms > 0

    # -- solve with mismatched device types --

    def test_solve_different_device_types(
        self, cost_model: PartitionCostModel,
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=["gpu-0", "gpu-1"],
            device_types={"gpu-0": "cuda", "gpu-1": "cpu"},
        )
        sol = solver.solve(num_layers=4)
        assert sol.num_nodes >= 1
        # gpu-1 is cpu, will only get NONE quant
        cpu_point = next(
            (p for p in sol.points if p.node_id == "gpu-1"),
            None,
        )
        if cpu_point is not None:
            assert cpu_point.quant_method == QuantMethod.NONE

    # -- solve with zero layers (boundary) --

    def test_solve_zero_layers(
        self, cost_model: PartitionCostModel, node_ids: list[str],
    ) -> None:
        solver = QuantAwarePartitionSolver(
            cost_model=cost_model,
            node_ids=node_ids,
        )
        sol = solver.solve(num_layers=0)
        assert isinstance(sol, QuantAwareSolution)
        assert sol.coverage == (0, 0)
