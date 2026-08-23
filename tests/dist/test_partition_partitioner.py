"""Tests for HardwareAwarePartitioner with mocked hardware backends.

Uses mocks for GPUProfiler, TopologyProber, PartitionCostModel, and
PartitionOptimizer to isolate the partitioner's orchestration logic
from real hardware dependencies.
"""

from __future__ import annotations

from typing import Any

import pytest

from distllm.dist.partition.partitioner import HardwareAwarePartitioner
from distllm.dist.partition.profiles import GPUProfile, LayerWeights
from distllm.dist.partition.topology import LinkProfile, TopologyGraph
from distllm.dist.partition.optimizer import PartitionPoint, PartitionSolution


# ---------------------------------------------------------------------------
# Test helpers — reusable mock data builders
# ---------------------------------------------------------------------------


def _make_topology(node_ids: list[str]) -> TopologyGraph:
    """Create a full-mesh TopologyGraph for the given node IDs."""
    links: list[LinkProfile] = []
    for i, a in enumerate(node_ids):
        for b in node_ids[i + 1 :]:
            links.append(
                LinkProfile(source=a, target=b, bandwidth_gbps=12.5, latency_us=100.0)
            )
    return TopologyGraph(
        node_ids=node_ids,
        gpu_counts={n: 1 for n in node_ids},
        links=links,
    )


def _make_solution(
    node_ids: list[str],
    total_layers: int,
    per_node: int | None = None,
) -> PartitionSolution:
    """Build a PartitionSolution that evenly distributes layers."""
    if per_node is None:
        per_node = total_layers // len(node_ids)
    points: list[PartitionPoint] = []
    start = 0
    for nid in node_ids:
        end = min(start + per_node, total_layers)
        points.append(
            PartitionPoint(
                node_id=nid,
                start_layer=start,
                end_layer=end,
                estimated_time_ms=100.0,
            )
        )
        start = end
        if start >= total_layers:
            break
    return PartitionSolution(
        points=points,
        max_node_time_ms=100.0,
        total_time_ms=100.0 * len(points),
        estimated_throughput_tok_s=8192.0,
        pipeline_latency_ms=200.0,
        num_oom_nodes=0,
        explanation="DP minimax optimized",
    )


# ---------------------------------------------------------------------------
# Fixtures — reusable mock data objects
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gpu_profile() -> GPUProfile:
    """Standard A100-like GPU profile."""
    return GPUProfile(
        gpu_id=0,
        name="A100",
        total_memory_bytes=80 * 1024**3,
        free_memory_bytes=72 * 1024**3,
        compute_tflops=312.0,
        memory_bandwidth_gbps=2039.0,
        sm_count=108,
        peak_tflops_fp16=312.0,
        peak_tflops_fp32=156.0,
    )


@pytest.fixture
def mock_layer_weights_32() -> list[LayerWeights]:
    """32 transformer layer weights."""
    return [
        LayerWeights(
            layer_id=i,
            layer_type="transformer",
            weight_memory_bytes=1024**3,
            activation_memory_bytes=256 * 1024**2,
            flops_per_token=100_000,
            flops_per_seq=100_000 * 4096,
            kv_cache_bytes_per_token=1024,
        )
        for i in range(32)
    ]


@pytest.fixture
def mock_layer_weights_8() -> list[LayerWeights]:
    """8 transformer layer weights (small model)."""
    return [
        LayerWeights(
            layer_id=i,
            layer_type="transformer",
            weight_memory_bytes=512 * 1024**2,
            activation_memory_bytes=128 * 1024**2,
            flops_per_token=50_000,
            flops_per_seq=50_000 * 4096,
            kv_cache_bytes_per_token=512,
        )
        for i in range(8)
    ]


# ---------------------------------------------------------------------------
# 1.  Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """HardwareAwarePartitioner construction with mocked backends."""

    def test_gpu_profiler_and_topology_prober_created(self, mocker) -> None:
        """__init__ instantiates GPUProfiler and TopologyProber."""
        mock_gpu_cls = mocker.patch(
            "distllm.dist.partition.partitioner.GPUProfiler"
        )
        mock_topo_cls = mocker.patch(
            "distllm.dist.partition.partitioner.TopologyProber"
        )
        p = HardwareAwarePartitioner()
        mock_gpu_cls.assert_called_once()
        mock_topo_cls.assert_called_once()
        assert p._gpu_profiler is mock_gpu_cls.return_value
        assert p._topology_prober is mock_topo_cls.return_value

    def test_custom_parameters_propagated(self, mocker) -> None:
        """Constructor params are stored on the instance."""
        mocker.patch("distllm.dist.partition.partitioner.GPUProfiler")
        mocker.patch("distllm.dist.partition.partitioner.TopologyProber")
        p = HardwareAwarePartitioner(
            batch_size=32,
            seq_len=8192,
            allow_oom=True,
            enable_quant_tuning=True,
            max_quality_loss=0.15,
            prefer_speed=True,
        )
        assert p._batch_size == 32
        assert p._seq_len == 8192
        assert p._allow_oom is True
        assert p._enable_quant_tuning is True
        assert p._max_quality_loss == 0.15
        assert p._prefer_speed is True

    def test_empty_initial_state(self, mocker) -> None:
        """Instance starts with empty profiles, no topology, no solution."""
        mocker.patch("distllm.dist.partition.partitioner.GPUProfiler")
        mocker.patch("distllm.dist.partition.partitioner.TopologyProber")
        p = HardwareAwarePartitioner()
        assert p._gpu_profiles == []
        assert p._topology is None
        assert p._layer_weights == []
        assert p._solution is None
        assert p._last_model_name is None
        assert p._last_partition_time == 0.0

    def test_init_with_defaults(self, mocker) -> None:
        """Default parameter values are correct."""
        mocker.patch("distllm.dist.partition.partitioner.GPUProfiler")
        mocker.patch("distllm.dist.partition.partitioner.TopologyProber")
        p = HardwareAwarePartitioner()
        assert p._batch_size == 1
        assert p._seq_len == 4096
        assert p._allow_oom is False
        assert p._enable_quant_tuning is False
        assert p._max_quality_loss == 0.05
        assert p._prefer_speed is False


# ---------------------------------------------------------------------------
# 2.  Partition orchestration with mocked hardware
# ---------------------------------------------------------------------------


class TestPartitionMocked:
    """partition() orchestration verified with fully mocked backends."""

    @pytest.fixture(autouse=True)
    def _mock_all(
        self,
        mocker: Any,
        mock_gpu_profile: GPUProfile,
        mock_layer_weights_32: list[LayerWeights],
    ) -> None:
        """Replace all hardware backends with mocks before each test.

        Sets up mocks for GPUProfiler, TopologyProber, PartitionCostModel,
        and PartitionOptimizer so that partition() exercises its orchestration
        flow without touching real hardware.
        """
        # -- GPUProfiler --
        self._mock_gpu_cls = mocker.patch(
            "distllm.dist.partition.partitioner.GPUProfiler",
        )
        self._mock_gpu = self._mock_gpu_cls.return_value
        self._mock_gpu._device_count.return_value = 2
        self._mock_gpu.profile_all_gpus.return_value = [
            mock_gpu_profile,
            mock_gpu_profile,
        ]
        self._mock_gpu._profile_single_gpu.return_value = mock_gpu_profile
        self._mock_gpu.estimate_layer_weights.return_value = (
            mock_layer_weights_32
        )
        self._mock_gpu.profile_to_dict.return_value = {
            "gpu_id": 0,
            "name": "A100",
            "total_memory_gb": 80.0,
            "free_memory_gb": 72.0,
            "compute_tflops": 312.0,
            "memory_bandwidth_gbps": 2039.0,
            "sm_count": 108,
            "measured_tflops_fp16": 0.0,
        }

        # -- TopologyProber --
        self._mock_topo_cls = mocker.patch(
            "distllm.dist.partition.partitioner.TopologyProber",
        )
        self._mock_topo = self._mock_topo_cls.return_value
        self._mock_topo.probe = mocker.AsyncMock()

        # -- TopologyProber.make_fallback_topology (static method) --
        self._mock_fallback = mocker.patch(
            "distllm.dist.partition.partitioner.TopologyProber"
            ".make_fallback_topology",
            return_value=_make_topology(["fallback-0"]),
        )

        # -- PartitionCostModel (constructed inside partition()) --
        self._mock_cost_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionCostModel",
        )

        # -- PartitionOptimizer (constructed inside partition()) --
        self._mock_opt_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionOptimizer",
        )
        self._mock_opt = self._mock_opt_cls.return_value

    # --- Happy path ---

    @pytest.mark.asyncio
    async def test_returns_solution(self) -> None:
        """partition() returns a valid PartitionSolution."""
        node_ids = ["node-a", "node-b"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 32)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(
            model_name="gpt-test",
            node_ids=node_ids,
            hidden_size=4096,
            num_layers=32,
        )
        assert result is sol
        assert p._last_model_name == "gpt-test"
        assert p._solution is sol

    @pytest.mark.asyncio
    async def test_solution_fields_populated(self) -> None:
        """Returned solution has expected non-negative numeric fields."""
        node_ids = ["n0", "n1"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 32)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=32)

        assert result.max_node_time_ms >= 0
        assert result.estimated_throughput_tok_s >= 0
        assert result.pipeline_latency_ms >= 0
        assert result.num_nodes == 2
        assert result.points[0].node_id == "n0"

    @pytest.mark.asyncio
    async def test_custom_params_forwarded_to_optimizer(self) -> None:
        """batch_size, seq_len, allow_oom reach the optimizer."""
        node_ids = ["n0", "n1"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 16)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner(batch_size=8, seq_len=2048, allow_oom=True)
        await p.partition(node_ids=node_ids, num_layers=16)

        _, call_kwargs = self._mock_opt_cls.call_args
        assert call_kwargs["batch_size"] == 8
        assert call_kwargs["seq_len"] == 2048
        assert call_kwargs["allow_oom"] is True

    @pytest.mark.asyncio
    async def test_single_gpu_path(self) -> None:
        """When only 1 GPU, partition() uses profile_all_gpus shortcut."""
        self._mock_gpu._device_count.return_value = 1

        node_ids = ["solo"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=8)

        assert result is sol
        # _profile_gpus_parallel() takes the num_gpus <= 1 branch
        self._mock_gpu.profile_all_gpus.assert_called_once()

    # --- Cache invalidation ---

    @pytest.mark.asyncio
    async def test_cache_returns_same_object(self) -> None:
        """Repeated partition() with same config returns cached solution."""
        node_ids = ["n0"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        self._mock_opt.solve.return_value = _make_solution(node_ids, 8)

        p = HardwareAwarePartitioner()
        sol1 = await p.partition(
            model_name="m", node_ids=node_ids, num_layers=8
        )
        self._mock_opt.solve.reset_mock()
        sol2 = await p.partition(
            model_name="m", node_ids=node_ids, num_layers=8
        )

        assert sol1 is sol2
        # Cache hit => solve() must NOT be called the second time
        self._mock_opt.solve.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_invalidated_by_model_name(self) -> None:
        """Different model name invalidates cache."""
        node_ids = ["n0"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol1 = _make_solution(node_ids, 8)
        sol2 = _make_solution(node_ids, 8)
        self._mock_opt.solve.return_value = sol1

        p = HardwareAwarePartitioner()
        await p.partition(
            model_name="m1", node_ids=node_ids, num_layers=8
        )
        self._mock_opt.solve.return_value = sol2
        result = await p.partition(
            model_name="m2", node_ids=node_ids, num_layers=8
        )

        assert result is sol2

    @pytest.mark.asyncio
    async def test_cache_invalidated_by_hidden_size(self) -> None:
        """Different model dimensions invalidate cache."""
        node_ids = ["n0"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol1 = _make_solution(node_ids, 8)
        sol2 = _make_solution(node_ids, 8)
        self._mock_opt.solve.return_value = sol1

        p = HardwareAwarePartitioner()
        await p.partition(
            model_name="m", node_ids=node_ids,
            hidden_size=4096, num_layers=8,
        )
        self._mock_opt.solve.return_value = sol2
        result = await p.partition(
            model_name="m", node_ids=node_ids,
            hidden_size=2048, num_layers=8,
        )

        assert result is sol2

    @pytest.mark.asyncio
    async def test_cache_invalidated_by_node_ids(self) -> None:
        """Different node_ids list invalidates cache."""
        self._mock_topo.probe.return_value = _make_topology(["n0"])
        sol1 = _make_solution(["n0"], 8)
        sol2 = _make_solution(["n1"], 8)
        self._mock_opt.solve.return_value = sol1

        p = HardwareAwarePartitioner()
        await p.partition(
            model_name="m", node_ids=["n0"], num_layers=8
        )
        self._mock_topo.probe.return_value = _make_topology(["n1"])
        self._mock_opt.solve.return_value = sol2
        result = await p.partition(
            model_name="m", node_ids=["n1"], num_layers=8
        )
        assert result is sol2

    # --- Topology probing failure fallback ---

    @pytest.mark.asyncio
    async def test_topology_probe_fallback(self) -> None:
        """When probe() raises, make_fallback_topology is used."""
        node_ids = ["n0", "n1"]
        self._mock_topo.probe.side_effect = ConnectionError("timeout")
        sol = _make_solution(node_ids, 16)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=16)

        assert result is sol
        self._mock_fallback.assert_called_once_with(
            num_nodes=2, gpus_per_node=1,
        )

    @pytest.mark.asyncio
    async def test_topology_probe_fallback_different_nodes(self) -> None:
        """Fallback topology handles varying node counts."""
        node_ids = [f"n{i}" for i in range(5)]
        self._mock_topo.probe.side_effect = RuntimeError("probe failed")
        sol = _make_solution(node_ids, 25)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=25)

        assert result is sol
        self._mock_fallback.assert_called_once_with(
            num_nodes=5, gpus_per_node=1,
        )

    # --- Default node naming ---

    @pytest.mark.asyncio
    async def test_default_node_naming(self) -> None:
        """When node_ids is None, uses 'node-N' naming."""
        self._mock_topo.probe.return_value = _make_topology(["node-0"])
        sol = _make_solution(["node-0"], 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(num_layers=8)

        assert isinstance(result, PartitionSolution)

    @pytest.mark.asyncio
    async def test_empty_node_ids_list(self) -> None:
        """Empty node_ids list also falls back to default naming."""
        self._mock_topo.probe.return_value = _make_topology(["node-0"])
        sol = _make_solution(["node-0"], 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=[], num_layers=8)

        assert isinstance(result, PartitionSolution)


# ---------------------------------------------------------------------------
# 3.  Edge cases: single node, max nodes, unbalanced hardware
# ---------------------------------------------------------------------------


class TestPartitionEdgeCases:
    """Single node, more nodes than layers, unbalanced GPU counts."""

    @pytest.fixture(autouse=True)
    def _mock_all(
        self,
        mocker: Any,
        mock_gpu_profile: GPUProfile,
        mock_layer_weights_8: list[LayerWeights],
    ) -> None:
        self._mock_gpu_cls = mocker.patch(
            "distllm.dist.partition.partitioner.GPUProfiler",
        )
        self._mock_gpu = self._mock_gpu_cls.return_value
        self._mock_gpu._device_count.return_value = 2
        self._mock_gpu.profile_all_gpus.return_value = [
            mock_gpu_profile,
            mock_gpu_profile,
        ]
        self._mock_gpu._profile_single_gpu.return_value = mock_gpu_profile
        self._mock_gpu.estimate_layer_weights.return_value = (
            mock_layer_weights_8
        )
        self._mock_gpu.profile_to_dict.return_value = {
            "gpu_id": 0,
            "name": "A100",
            "total_memory_gb": 80.0,
            "free_memory_gb": 72.0,
            "compute_tflops": 312.0,
            "memory_bandwidth_gbps": 2039.0,
            "sm_count": 108,
            "measured_tflops_fp16": 0.0,
        }

        self._mock_topo_cls = mocker.patch(
            "distllm.dist.partition.partitioner.TopologyProber",
        )
        self._mock_topo = self._mock_topo_cls.return_value
        self._mock_topo.probe = mocker.AsyncMock()

        self._mock_cost_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionCostModel",
        )
        self._mock_opt_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionOptimizer",
        )
        self._mock_opt = self._mock_opt_cls.return_value

    # --- Single node ---

    @pytest.mark.asyncio
    async def test_single_node_partition(self) -> None:
        """Single-node partition returns solution with 1 node."""
        node_ids = ["single"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=8)

        assert result.num_nodes == 1
        assert result.points[0].node_id == "single"
        assert result.points[0].start_layer == 0
        assert result.points[0].end_layer == 8

    @pytest.mark.asyncio
    async def test_single_node_with_gpu_counts(self) -> None:
        """Single node with multi-GPU count is forwarded to optimizer."""
        node_ids = ["fat-node"]
        gpu_cnt = {"fat-node": 8}
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 16)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(
            node_ids=node_ids,
            gpu_counts=gpu_cnt,
            num_layers=16,
        )
        assert result is sol
        _, call_kwargs = self._mock_opt_cls.call_args
        assert call_kwargs["gpu_counts"] == gpu_cnt

    # --- More nodes than layers ---

    @pytest.mark.asyncio
    async def test_more_nodes_than_layers(self) -> None:
        """10 nodes for 3 layers — optimizer receives all 10 but uses <= 3."""
        node_ids = [f"n{i}" for i in range(10)]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids[:3], 3, 1)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=3)

        assert result.num_nodes == 3
        assert len(result.points) == 3
        assert result.coverage == (0, 3)

    @pytest.mark.asyncio
    async def test_single_layer_many_nodes(self) -> None:
        """1 layer with many nodes — optimizer handles extreme ratio."""
        node_ids = [f"n{i}" for i in range(20)]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids[:1], 1, 1)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=1)

        assert result.num_nodes == 1
        assert result.coverage == (0, 1)

    # --- Unbalanced hardware ---

    @pytest.mark.asyncio
    async def test_unbalanced_gpu_counts(self) -> None:
        """Different GPU counts per node forwarded to optimizer."""
        node_ids = ["strong", "weak"]
        gpu_cnt = {"strong": 8, "weak": 1}
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(
            node_ids=node_ids,
            gpu_counts=gpu_cnt,
            num_layers=8,
        )
        assert result is sol
        _, call_kwargs = self._mock_opt_cls.call_args
        assert call_kwargs["gpu_counts"] == gpu_cnt

    @pytest.mark.asyncio
    async def test_zero_gpu_count_node(self) -> None:
        """A node with zero GPUs is passed through to optimizer."""
        node_ids = ["n0", "n1"]
        gpu_cnt = {"n0": 0, "n1": 1}
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(
            node_ids=node_ids,
            gpu_counts=gpu_cnt,
            num_layers=8,
        )
        assert result is sol
        _, call_kwargs = self._mock_opt_cls.call_args
        assert call_kwargs["gpu_counts"]["n0"] == 0

    # --- Hostnames forwarded ---

    @pytest.mark.asyncio
    async def test_hostnames_forwarded_to_prober(self) -> None:
        """Hostnames dict is passed through to TopologyProber.probe()."""
        node_ids = ["n0", "n1"]
        hosts = {"n0": "10.0.0.1", "n1": "10.0.0.2"}
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(
            node_ids=node_ids,
            hostnames=hosts,
            num_layers=8,
        )
        assert result is sol
        _, call_kwargs = self._mock_topo.probe.call_args
        assert call_kwargs["hostnames"] == hosts

    # --- Large model ---

    @pytest.mark.asyncio
    async def test_large_num_layers(self) -> None:
        """500 layers flows through correctly."""
        node_ids = ["n0", "n1"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 500, 250)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=500)

        assert result.num_nodes == 2
        assert result.coverage == (0, 500)


# ---------------------------------------------------------------------------
# 4.  Error handling / invalid configurations
# ---------------------------------------------------------------------------


class TestPartitionErrorHandling:
    """Exception propagation and invalid configuration handling."""

    @pytest.fixture(autouse=True)
    def _mock_all(
        self,
        mocker: Any,
        mock_gpu_profile: GPUProfile,
        mock_layer_weights_8: list[LayerWeights],
    ) -> None:
        self._mock_gpu_cls = mocker.patch(
            "distllm.dist.partition.partitioner.GPUProfiler",
        )
        self._mock_gpu = self._mock_gpu_cls.return_value
        self._mock_gpu._device_count.return_value = 2
        self._mock_gpu.profile_all_gpus.return_value = [
            mock_gpu_profile,
            mock_gpu_profile,
        ]
        self._mock_gpu._profile_single_gpu.return_value = mock_gpu_profile
        self._mock_gpu.estimate_layer_weights.return_value = (
            mock_layer_weights_8
        )
        self._mock_gpu.profile_to_dict.return_value = {
            "gpu_id": 0,
            "name": "A100",
            "total_memory_gb": 80.0,
            "free_memory_gb": 72.0,
            "compute_tflops": 312.0,
            "memory_bandwidth_gbps": 2039.0,
            "sm_count": 108,
            "measured_tflops_fp16": 0.0,
        }

        self._mock_topo_cls = mocker.patch(
            "distllm.dist.partition.partitioner.TopologyProber",
        )
        self._mock_topo = self._mock_topo_cls.return_value
        self._mock_topo.probe = mocker.AsyncMock()
        self._mock_topo.probe.return_value = _make_topology(["n0"])

        self._mock_cost_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionCostModel",
        )
        self._mock_opt_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionOptimizer",
        )
        self._mock_opt = self._mock_opt_cls.return_value

    @pytest.mark.asyncio
    async def test_optimizer_solve_exception_propagates(self) -> None:
        """Exception from optimizer.solve() propagates to caller."""
        self._mock_opt.solve.side_effect = ValueError("bad layer count")

        p = HardwareAwarePartitioner()
        with pytest.raises(ValueError, match="bad layer count"):
            await p.partition(num_layers=8)

    @pytest.mark.asyncio
    async def test_profile_all_gpus_exception_propagates(self) -> None:
        """Exception from profile_all_gpus (num_gpus <= 1) propagates."""
        self._mock_gpu._device_count.return_value = 1
        self._mock_gpu.profile_all_gpus.side_effect = RuntimeError("no GPUs")

        p = HardwareAwarePartitioner()
        with pytest.raises(RuntimeError, match="no GPUs"):
            await p.partition(num_layers=4)

    @pytest.mark.asyncio
    async def test_estimate_layer_weights_exception_propagates(self) -> None:
        """Exception from estimate_layer_weights propagates."""
        self._mock_topo.probe.return_value = _make_topology(["n0"])
        self._mock_gpu.estimate_layer_weights.side_effect = ValueError(
            "invalid dims"
        )

        p = HardwareAwarePartitioner()
        with pytest.raises(ValueError, match="invalid dims"):
            await p.partition(num_layers=4)

    @pytest.mark.asyncio
    async def test_negative_layer_count(self) -> None:
        """Negative num_layers produces an empty (or no) solution."""
        node_ids = ["n0"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        self._mock_opt.solve.return_value = PartitionSolution(
            points=[],
            explanation="No valid partition found",
        )

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=-1)
        assert len(result.points) == 0

    @pytest.mark.asyncio
    async def test_cost_model_construction_exception_propagates(self) -> None:
        """Exception from PartitionCostModel constructor propagates."""
        self._mock_cost_cls.side_effect = RuntimeError("cost model failed")

        p = HardwareAwarePartitioner()
        with pytest.raises(RuntimeError, match="cost model failed"):
            await p.partition(num_layers=4)

    @pytest.mark.asyncio
    async def test_optimizer_construction_exception_propagates(self) -> None:
        """Exception from PartitionOptimizer constructor propagates."""
        self._mock_opt_cls.side_effect = TypeError("invalid argument")

        p = HardwareAwarePartitioner()
        with pytest.raises(TypeError, match="invalid argument"):
            await p.partition(num_layers=4)

    @pytest.mark.asyncio
    async def test_empty_solution_from_optimizer(self) -> None:
        """Optimizer returning empty points list is handled."""
        node_ids = ["n0"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        self._mock_opt.solve.return_value = PartitionSolution(
            points=[],
            max_node_time_ms=0.0,
            total_time_ms=0.0,
            estimated_throughput_tok_s=0.0,
            pipeline_latency_ms=0.0,
            explanation="Nothing to partition",
        )

        p = HardwareAwarePartitioner()
        result = await p.partition(node_ids=node_ids, num_layers=0)
        assert len(result.points) == 0
        assert "Nothing" in result.explanation


# ---------------------------------------------------------------------------
# 5.  Accessor methods (solution, layer_assignments, node_summaries)
# ---------------------------------------------------------------------------


class TestAccessors:
    """solution(), get_layer_assignments(), get_node_summaries(), and
    compare_to_baselines() after a partition."""

    @pytest.fixture(autouse=True)
    def _mock_all(
        self,
        mocker: Any,
        mock_gpu_profile: GPUProfile,
        mock_layer_weights_8: list[LayerWeights],
    ) -> None:
        self._mock_gpu_cls = mocker.patch(
            "distllm.dist.partition.partitioner.GPUProfiler",
        )
        self._mock_gpu = self._mock_gpu_cls.return_value
        self._mock_gpu._device_count.return_value = 2
        self._mock_gpu.profile_all_gpus.return_value = [
            mock_gpu_profile,
            mock_gpu_profile,
        ]
        self._mock_gpu._profile_single_gpu.return_value = mock_gpu_profile
        self._mock_gpu.estimate_layer_weights.return_value = (
            mock_layer_weights_8
        )
        self._mock_gpu.profile_to_dict.return_value = {
            "gpu_id": 0,
            "name": "A100",
            "total_memory_gb": 80.0,
            "free_memory_gb": 72.0,
            "compute_tflops": 312.0,
            "memory_bandwidth_gbps": 2039.0,
            "sm_count": 108,
            "measured_tflops_fp16": 0.0,
        }

        self._mock_topo_cls = mocker.patch(
            "distllm.dist.partition.partitioner.TopologyProber",
        )
        self._mock_topo = self._mock_topo_cls.return_value
        self._mock_topo.probe = mocker.AsyncMock()
        self._mock_topo.probe.return_value = _make_topology(["n0", "n1"])

        self._mock_cost_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionCostModel",
        )
        self._mock_opt_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionOptimizer",
        )
        self._mock_opt = self._mock_opt_cls.return_value

    def test_solution_before_partition(self) -> None:
        """solution() returns None before partition()."""
        p = HardwareAwarePartitioner()
        assert p.solution() is None

    @pytest.mark.asyncio
    async def test_solution_after_partition(self) -> None:
        """solution() returns the partition result."""
        node_ids = ["n0", "n1"]
        sol = _make_solution(node_ids, 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        await p.partition(node_ids=node_ids, num_layers=8)
        assert p.solution() is sol

    def test_layer_assignments_none_before_partition(self) -> None:
        """get_layer_assignments() returns None before partition()."""
        p = HardwareAwarePartitioner()
        assert p.get_layer_assignments() is None

    @pytest.mark.asyncio
    async def test_layer_assignments_after_partition(self) -> None:
        """get_layer_assignments() returns one entry per layer."""
        node_ids = ["n0", "n1"]
        sol = _make_solution(node_ids, 8, 4)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        await p.partition(node_ids=node_ids, num_layers=8)

        assigns = p.get_layer_assignments()
        assert assigns is not None
        assert len(assigns) == 8
        assert assigns[0]["layer_id"] == 0
        assert assigns[7]["layer_id"] == 7
        assert assigns[0]["node_id"] == "n0"
        assert assigns[4]["node_id"] == "n1"
        # Every entry has required keys
        for a in assigns:
            assert "layer_id" in a
            assert "node_id" in a
            assert "layer_type" in a

    def test_node_summaries_none_before_partition(self) -> None:
        """get_node_summaries() returns None before partition()."""
        p = HardwareAwarePartitioner()
        assert p.get_node_summaries() is None

    @pytest.mark.asyncio
    async def test_node_summaries_after_partition(self) -> None:
        """get_node_summaries() returns one summary per node."""
        node_ids = ["n0", "n1"]
        sol = _make_solution(node_ids, 8, 4)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        await p.partition(node_ids=node_ids, num_layers=8)

        summaries = p.get_node_summaries()
        assert summaries is not None
        assert len(summaries) == 2
        required = {
            "node_id", "layers", "num_layers",
            "compute_time_ms", "comm_time_ms", "total_time_ms",
            "memory_bytes", "memory_gb", "fits_in_memory",
        }
        for s in summaries:
            assert required.issubset(s.keys())

    def test_compare_to_baselines_none_before_partition(self) -> None:
        """compare_to_baselines() returns None before partition()."""
        p = HardwareAwarePartitioner()
        assert p.compare_to_baselines() is None

    @pytest.mark.asyncio
    async def test_compare_to_baselines_after_partition(self) -> None:
        """compare_to_baselines() returns dict with all strategy keys."""
        node_ids = ["n0", "n1"]
        self._mock_topo.probe.return_value = _make_topology(node_ids)
        sol = _make_solution(node_ids, 8, 4)
        self._mock_opt.solve.return_value = sol
        self._mock_opt.compare_strategies.return_value = {
            "dp_minimax": {
                "max_latency_ms": 50.0,
                "pipeline_latency_ms": 100.0,
                "throughput": 16000.0,
            },
            "equal_split": {
                "max_latency_ms": 80.0,
                "throughput": 10000.0,
            },
            "proportional_split": {
                "max_latency_ms": 60.0,
                "throughput": 13000.0,
            },
            "improvement_over_equal": "37.5%",
        }

        p = HardwareAwarePartitioner()
        await p.partition(node_ids=node_ids, num_layers=8)

        result = p.compare_to_baselines()
        assert result is not None
        assert "dp_minimax" in result
        assert "equal_split" in result
        assert "proportional_split" in result
        assert "improvement_over_equal" in result
        assert result["dp_minimax"]["max_latency_ms"] == 50.0

    @pytest.mark.asyncio
    async def test_summary_output(self) -> None:
        """summary() reflects state before and after partition."""
        node_ids = ["n0", "n1"]
        sol = _make_solution(node_ids, 8, 4)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        before = p.summary()
        assert "No partition computed" in before
        assert "HardwareAwarePartitioner" in before

        await p.partition(node_ids=node_ids, num_layers=8)
        after = p.summary()
        assert "Partition" in after
        assert "No partition computed" not in after


# ---------------------------------------------------------------------------
# 6.  _profile_gpus_parallel edge cases
# ---------------------------------------------------------------------------


class TestGPUProfilingEdgeCases:
    """Internal _profile_gpus_parallel behaviour under unusual conditions."""

    @pytest.fixture(autouse=True)
    def _mock_all(
        self,
        mocker: Any,
        mock_gpu_profile: GPUProfile,
        mock_layer_weights_8: list[LayerWeights],
    ) -> None:
        self._mock_gpu_cls = mocker.patch(
            "distllm.dist.partition.partitioner.GPUProfiler",
        )
        self._mock_gpu = self._mock_gpu_cls.return_value
        self._mock_gpu._device_count.return_value = 2
        self._mock_gpu.profile_all_gpus.return_value = [
            mock_gpu_profile,
            mock_gpu_profile,
        ]
        self._mock_gpu._profile_single_gpu.return_value = mock_gpu_profile
        self._mock_gpu.estimate_layer_weights.return_value = (
            mock_layer_weights_8
        )
        self._mock_gpu.profile_to_dict.return_value = {
            "gpu_id": 0,
            "name": "A100",
            "total_memory_gb": 80.0,
            "free_memory_gb": 72.0,
            "compute_tflops": 312.0,
            "memory_bandwidth_gbps": 2039.0,
            "sm_count": 108,
            "measured_tflops_fp16": 0.0,
        }
        self._mock_topo_cls = mocker.patch(
            "distllm.dist.partition.partitioner.TopologyProber",
        )
        self._mock_topo = self._mock_topo_cls.return_value
        self._mock_topo.probe = mocker.AsyncMock()
        self._mock_topo.probe.return_value = _make_topology(["n0"])
        self._mock_cost_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionCostModel",
        )
        self._mock_opt_cls = mocker.patch(
            "distllm.dist.partition.partitioner.PartitionOptimizer",
        )
        self._mock_opt = self._mock_opt_cls.return_value

    @pytest.mark.asyncio
    async def test_zero_gpus_fallback(self) -> None:
        """_device_count() == 0 triggers profile_all_gpus fallback."""
        self._mock_gpu._device_count.return_value = 0
        self._mock_gpu.profile_all_gpus.return_value = [
            GPUProfile(gpu_id=0, name="cpu_fallback"),
        ]
        sol = _make_solution(["n0"], 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(num_layers=8)

        assert isinstance(result, PartitionSolution)
        self._mock_gpu.profile_all_gpus.assert_called_once()

    @pytest.mark.asyncio
    async def test_many_gpus_threaded(self) -> None:
        """8 GPUs uses thread pool with max_workers=4."""
        self._mock_gpu._device_count.return_value = 8
        self._mock_gpu._profile_single_gpu.return_value = GPUProfile(
            gpu_id=0,
            name="H100",
            total_memory_bytes=80 * 1024**3,
            free_memory_bytes=72 * 1024**3,
            compute_tflops=989.0,
            memory_bandwidth_gbps=3350.0,
            sm_count=132,
            peak_tflops_fp16=989.0,
            peak_tflops_fp32=494.0,
        )
        sol = _make_solution(["n0"], 8)
        self._mock_opt.solve.return_value = sol

        p = HardwareAwarePartitioner()
        result = await p.partition(num_layers=8)

        assert isinstance(result, PartitionSolution)
        # With 8 GPUs, profile_all_gpus should NOT be called
        self._mock_gpu.profile_all_gpus.assert_not_called()
        # _profile_single_gpu should have been called 8 times
        assert self._mock_gpu._profile_single_gpu.call_count == 8
