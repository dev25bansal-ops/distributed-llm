"""Tests for DeviceClass, NodeCapabilityInfo, and HeterogeneousBudgetComputer."""

from __future__ import annotations

from types import SimpleNamespace

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_hetero = load_module("distllm/core/advanced_scheduling/heterogeneous.py")
DeviceClass = _hetero.DeviceClass
NodeCapabilityInfo = _hetero.NodeCapabilityInfo
HeterogeneousBudgetComputer = _hetero.HeterogeneousBudgetComputer


class TestDeviceClass:
    """Test suite for DeviceClass enum."""

    def test_members(self) -> None:
        assert DeviceClass.DATA_CENTER.value == "data_center"
        assert DeviceClass.WORKSTATION.value == "workstation"
        assert DeviceClass.CONSUMER.value == "consumer"
        assert DeviceClass.MOBILE.value == "mobile"
        assert DeviceClass.CPU.value == "cpu"

    def test_all_distinct(self) -> None:
        values = [m.value for m in DeviceClass]
        assert len(values) == len(set(values))


class TestNodeCapabilityInfo:
    """Test suite for NodeCapabilityInfo dataclass."""

    def test_default_construction(self) -> None:
        info = NodeCapabilityInfo(node_id="gpu-0")
        assert info.node_id == "gpu-0"
        assert info.device_class == DeviceClass.CONSUMER
        assert info.gpu_tflops == 0.0
        assert info.gpu_memory_gb == 0.0
        assert info.bandwidth_gbps == 0.0
        assert info.num_layers_assigned == 0
        assert info.current_load == 0.0
        assert info.is_spot is False
        assert info.cost_per_hour == 0.0

    def test_custom_values(self) -> None:
        info = NodeCapabilityInfo(
            node_id="a100-0",
            device_class=DeviceClass.DATA_CENTER,
            gpu_tflops=312.0,
            gpu_memory_gb=80.0,
            bandwidth_gbps=2039.0,
            num_layers_assigned=32,
            current_load=0.5,
            is_spot=True,
            cost_per_hour=3.0,
        )
        assert info.gpu_tflops == 312.0
        assert info.is_spot is True
        assert info.cost_per_hour == 3.0


class TestHeterogeneousBudgetComputer:
    """Test suite for HeterogeneousBudgetComputer."""

    def test_default_construction(self) -> None:
        computer = HeterogeneousBudgetComputer()
        assert computer._nodes == {}

    def test_construction_with_nodes(self) -> None:
        nodes = {
            "a": NodeCapabilityInfo(node_id="a", gpu_tflops=100.0),
            "b": NodeCapabilityInfo(node_id="b", gpu_tflops=200.0),
        }
        computer = HeterogeneousBudgetComputer(nodes=nodes)
        assert len(computer._nodes) == 2

    def test_update_nodes(self) -> None:
        computer = HeterogeneousBudgetComputer()
        nodes = {"x": NodeCapabilityInfo(node_id="x", gpu_tflops=300.0)}
        computer.update_nodes(nodes)
        assert computer._nodes["x"].gpu_tflops == 300.0

    def test_compute_budget_no_nodes_returns_base(self) -> None:
        computer = HeterogeneousBudgetComputer()
        result = computer.compute_budget(4096, 512, 32, 32768)
        assert result.max_prefill_tokens == 4096
        assert result.max_decode_tokens == 512
        assert result.max_batch_size == 32
        assert result.max_total_tokens == 32768

    def test_compute_budget_homogeneous_no_change(self) -> None:
        nodes = {
            "a": NodeCapabilityInfo(node_id="a", gpu_tflops=200.0),
            "b": NodeCapabilityInfo(node_id="b", gpu_tflops=200.0),
        }
        computer = HeterogeneousBudgetComputer(nodes=nodes)
        # ratio = 200/200 = 1.0, not < 0.5, so unchanged
        result = computer.compute_budget(4096, 512, 32, 32768)
        assert result.max_prefill_tokens == 4096

    def test_compute_budget_returns_fresh_iteration_budget(self) -> None:
        """compute_budget must return a new IterationBudget, never mutate inputs."""
        from distllm.core.scheduler.budget import IterationBudget

        nodes = {
            "fast": NodeCapabilityInfo(node_id="fast", gpu_tflops=400.0),
            "slow": NodeCapabilityInfo(node_id="slow", gpu_tflops=100.0),
        }
        computer = HeterogeneousBudgetComputer(nodes=nodes)
        result = computer.compute_budget(1000, 2000, 3000, 4000)
        assert isinstance(result, IterationBudget)
        # Only prefill is scaled; every other field passes through.
        assert result.max_prefill_tokens == int(1000 * 0.25)
        assert result.max_decode_tokens == 2000
        assert result.max_batch_size == 3000
        assert result.max_total_tokens == 4000

    def test_compute_budget_heterogeneous_scales_down(self) -> None:
        nodes = {
            "fast": NodeCapabilityInfo(node_id="fast", gpu_tflops=400.0),
            "slow": NodeCapabilityInfo(node_id="slow", gpu_tflops=100.0),
        }
        computer = HeterogeneousBudgetComputer(nodes=nodes)
        # ratio = 100/400 = 0.25 < 0.5 -> scales
        result = computer.compute_budget(
            base_prefill_tokens=4096,
            base_decode_tokens=512,
            base_batch_size=32,
            base_total_tokens=32768,
        )
        assert result.max_prefill_tokens == int(4096 * 0.25)

    def test_compute_budget_zero_tflops_skipped(self) -> None:
        nodes = {
            "a": NodeCapabilityInfo(node_id="a", gpu_tflops=0.0),
            "b": NodeCapabilityInfo(node_id="b", gpu_tflops=200.0),
        }
        computer = HeterogeneousBudgetComputer(nodes=nodes)
        result = computer.compute_budget(4096, 512, 32, 32768)
        # min/max only look at > 0, so min=max=200, ratio=1.0, no change
        assert result.max_prefill_tokens == 4096

    def test_compute_budget_all_zero_tflops_passthrough(self) -> None:
        """All-zero TFLOPS must pass the budget through, not crash.

        Regression (C4): min()/max() over an empty filtered sequence used
        to raise ValueError inside the serving loop whenever nodes were
        registered without compute info (e.g. bandwidth-only).
        """
        nodes = {
            "a": NodeCapabilityInfo(node_id="a", gpu_tflops=0.0),
        }
        computer = HeterogeneousBudgetComputer(nodes=nodes)
        result = computer.compute_budget(4096, 512, 32, 32768)
        assert result.max_prefill_tokens == 4096
        assert result.max_batch_size == 32

    def test_compute_budget_mixed_zero_and_positive(self) -> None:
        nodes = {
            "a": NodeCapabilityInfo(node_id="a", gpu_tflops=0.0),
            "b": NodeCapabilityInfo(node_id="b", gpu_tflops=100.0),
            "c": NodeCapabilityInfo(node_id="c", gpu_tflops=500.0),
        }
        computer = HeterogeneousBudgetComputer(nodes=nodes)
        # min (positive) = 100, max = 500, ratio = 0.2 < 0.5 -> scales
        result = computer.compute_budget(5000, 512, 32, 32768)
        assert result.max_prefill_tokens == int(5000 * 0.2)
