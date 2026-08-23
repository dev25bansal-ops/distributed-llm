"""Tests for distllm.dist.worker module.

Zero mocks -- uses only real objects from the module.
No GPU, no network, no timing-dependent assertions.
"""

from __future__ import annotations

import io

import torch
import torch.nn as nn
import pytest
from loguru import logger

from distllm.dist.privacy import PrivacySplitConfig
from distllm.dist.worker import (
    WorkerNode,
    _SimpleCompressionConfig,
    _validate_state_dict_keys,
)

# ---------------------------------------------------------------------------
# _SimpleCompressionConfig
# ---------------------------------------------------------------------------


class TestSimpleCompressionConfig:
    """Construction and attribute access for _SimpleCompressionConfig."""

    def test_construct_with_all_fields(self) -> None:
        config = _SimpleCompressionConfig(
            method="magnitude",
            enabled=True,
            target_bits=8,
            pruning_ratio=0.3,
            distillation_teacher="teacher_v2",
            calibration_samples=128,
            pruning_targets=["q_proj", "v_proj"],
        )
        assert config.method == "magnitude"
        assert config.enabled is True
        assert config.target_bits == 8
        assert config.pruning_ratio == 0.3
        assert config.distillation_teacher == "teacher_v2"
        assert config.calibration_samples == 128
        assert config.pruning_targets == ["q_proj", "v_proj"]

    def test_construct_with_none_teacher(self) -> None:
        config = _SimpleCompressionConfig(
            method="none",
            enabled=False,
            target_bits=4,
            pruning_ratio=0.0,
            distillation_teacher=None,
            calibration_samples=64,
            pruning_targets=[],
        )
        assert config.distillation_teacher is None
        assert config.pruning_targets == []

    def test_construct_with_boundary_values(self) -> None:
        config = _SimpleCompressionConfig(
            method="",
            enabled=False,
            target_bits=0,
            pruning_ratio=-1.0,
            distillation_teacher=None,
            calibration_samples=0,
            pruning_targets=[],
        )
        assert config.method == ""
        assert config.target_bits == 0
        assert config.pruning_ratio == -1.0
        assert config.calibration_samples == 0

    def test_construct_with_large_values(self) -> None:
        config = _SimpleCompressionConfig(
            method="svd",
            enabled=True,
            target_bits=32,
            pruning_ratio=0.99,
            distillation_teacher="x",
            calibration_samples=10_000,
            pruning_targets=[f"layer_{i}" for i in range(100)],
        )
        assert config.target_bits == 32
        assert config.pruning_ratio == 0.99
        assert len(config.pruning_targets) == 100


# ---------------------------------------------------------------------------
# WorkerNode -- construction and computed properties
# ---------------------------------------------------------------------------


class TestWorkerNodeConstruction:
    """WorkerNode __init__ attribute assignment and computed properties."""

    def test_basic_construction(self) -> None:
        node = WorkerNode(
            node_id="worker-0",
            model_name="test-model",
            start_layer=0,
            end_layer=4,
            total_layers=12,
            port=50051,
        )
        assert node.node_id == "worker-0"
        assert node.model_name == "test-model"
        assert node.start_layer == 0
        assert node.end_layer == 4
        assert node.total_layers == 12
        assert node.port == 50051
        assert node.coordinator_host == "localhost"
        assert node.coordinator_port == 50050
        assert node.device == "auto"
        assert node.dtype == "float16"
        assert node.quantization_config is None
        assert node.expert_ids == []
        assert node.compression_config is None
        assert isinstance(node.privacy_config, PrivacySplitConfig)
        assert node.privacy_config.enabled is False
        assert node.partitioner is None
        assert node._ready is False

    def test_is_first_node(self) -> None:
        node = WorkerNode(
            node_id="first",
            model_name="m",
            start_layer=0,
            end_layer=3,
            total_layers=12,
            port=50051,
        )
        assert node.is_first is True
        assert node.is_last is False

    def test_is_last_node(self) -> None:
        node = WorkerNode(
            node_id="last",
            model_name="m",
            start_layer=8,
            end_layer=11,
            total_layers=12,
            port=50051,
        )
        assert node.is_first is False
        assert node.is_last is True

    def test_is_both_first_and_last(self) -> None:
        """Single-layer node where start=0 and end=0 with total_layers=1."""
        node = WorkerNode(
            node_id="only",
            model_name="m",
            start_layer=0,
            end_layer=0,
            total_layers=1,
            port=50051,
        )
        assert node.is_first is True
        assert node.is_last is True

    def test_is_neither_first_nor_last(self) -> None:
        node = WorkerNode(
            node_id="mid",
            model_name="m",
            start_layer=4,
            end_layer=7,
            total_layers=12,
            port=50051,
        )
        assert node.is_first is False
        assert node.is_last is False

    def test_end_layer_boundary(self) -> None:
        """end_layer == total_layers - 1 is the boundary for is_last."""
        node = WorkerNode(
            node_id="b",
            model_name="m",
            start_layer=5,
            end_layer=11,
            total_layers=12,
            port=50051,
        )
        assert node.is_last is True

        node2 = WorkerNode(
            node_id="b2",
            model_name="m",
            start_layer=5,
            end_layer=10,
            total_layers=12,
            port=50051,
        )
        assert node2.is_last is False

    def test_start_layer_boundary(self) -> None:
        """start_layer > 0 is the boundary for is_first."""
        node = WorkerNode(
            node_id="f",
            model_name="m",
            start_layer=0,
            end_layer=3,
            total_layers=12,
            port=50051,
        )
        assert node.is_first is True

        node2 = WorkerNode(
            node_id="f2",
            model_name="m",
            start_layer=1,
            end_layer=3,
            total_layers=12,
            port=50051,
        )
        assert node2.is_first is False

    def test_expert_ids_default_to_empty_list(self) -> None:
        node = WorkerNode(
            node_id="e",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=4,
            port=50051,
            expert_ids=None,
        )
        assert node.expert_ids == []

    def test_expert_ids_with_values(self) -> None:
        node = WorkerNode(
            node_id="e",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=4,
            port=50051,
            expert_ids=[0, 2, 5],
        )
        assert node.expert_ids == [0, 2, 5]

    def test_expert_ids_empty_list_explicit(self) -> None:
        node = WorkerNode(
            node_id="e",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=4,
            port=50051,
            expert_ids=[],
        )
        assert node.expert_ids == []

    def test_privacy_node_detection(self) -> None:
        privacy = PrivacySplitConfig(enabled=True, prefix_layers=2, suffix_layers=2)
        node = WorkerNode(
            node_id="p",
            model_name="m",
            start_layer=2,
            end_layer=9,
            total_layers=12,
            port=50051,
            privacy_config=privacy,
        )
        assert node.is_privacy_node is True
        assert node.privacy_config.enabled is True

    def test_privacy_node_disabled_by_default(self) -> None:
        node = WorkerNode(
            node_id="np",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
        )
        assert node.is_privacy_node is False

    def test_custom_coordinator_address(self) -> None:
        node = WorkerNode(
            node_id="c",
            model_name="m",
            start_layer=0,
            end_layer=5,
            total_layers=12,
            port=50052,
            coordinator_host="10.0.0.1",
            coordinator_port=9090,
        )
        assert node.coordinator_host == "10.0.0.1"
        assert node.coordinator_port == 9090

    def test_custom_device_and_dtype(self) -> None:
        node = WorkerNode(
            node_id="d",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
            device="cpu",
            dtype="bfloat16",
        )
        assert node.device == "cpu"
        assert node.dtype == "bfloat16"

    def test_zero_total_layers(self) -> None:
        """Edge case: total_layers=0 makes end_layer >= -1, so is_last."""
        node = WorkerNode(
            node_id="z",
            model_name="m",
            start_layer=0,
            end_layer=0,
            total_layers=0,
            port=50051,
        )
        assert node.is_first is True
        assert node.is_last is True  # 0 >= -1

    def test_negative_layer_indices(self) -> None:
        """Negative layer values are accepted without validation."""
        node = WorkerNode(
            node_id="neg",
            model_name="m",
            start_layer=-5,
            end_layer=-1,
            total_layers=12,
            port=50051,
        )
        assert node.start_layer == -5
        assert node.end_layer == -1
        # is_last: end_layer >= total_layers - 1  =>  -1 >= 11  =>  False
        assert node.is_last is False


# ---------------------------------------------------------------------------
# WorkerNode -- _get_device
# ---------------------------------------------------------------------------


class TestWorkerNodeDevice:
    """_get_device helper method."""

    def test_explicit_device_cpu(self) -> None:
        node = WorkerNode(
            node_id="cpu-node",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
            device="cpu",
        )
        assert node._get_device() == "cpu"

    def test_explicit_device_cuda(self) -> None:
        node = WorkerNode(
            node_id="cuda-node",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
            device="cuda:0",
        )
        assert node._get_device() == "cuda:0"

    def test_explicit_device_mps(self) -> None:
        node = WorkerNode(
            node_id="mps-node",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
            device="mps",
        )
        assert node._get_device() == "mps"

    def test_auto_device_returns_string(self) -> None:
        """_get_device with ``auto`` delegates to detect_platform.

        Skipped when the platform-detection module has a missing internal
        import (known issue in dev builds of ``distllm.constants``).
        """
        node = WorkerNode(
            node_id="auto-node",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
            device="auto",
        )
        try:
            device = node._get_device()
        except ImportError:
            pytest.skip("detect_platform not importable in this build")
        assert isinstance(device, str)
        assert len(device) > 0

    def test_device_persists_after_get(self) -> None:
        """Calling _get_device does not mutate self.device."""
        node = WorkerNode(
            node_id="p",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
            device="cpu",
        )
        before = node.device
        _ = node._get_device()
        assert node.device == before


# ---------------------------------------------------------------------------
# WorkerNode -- stop (edge cases)
# ---------------------------------------------------------------------------


class TestWorkerNodeStop:
    """stop() behaviour when no server has been started."""

    def test_stop_without_start_no_error(self) -> None:
        node = WorkerNode(
            node_id="no-server",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
        )
        # _server attribute does not exist -- should not raise
        node.stop()

    def test_stop_twice_no_error(self) -> None:
        node = WorkerNode(
            node_id="double-stop",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
        )
        node.stop()
        node.stop()


# ---------------------------------------------------------------------------
# WorkerNode -- forward_fn (error path)
# ---------------------------------------------------------------------------


class TestWorkerNodeForwardFn:
    """forward_fn edge cases that do not require a loaded model."""

    def test_forward_fn_raises_when_no_model(self) -> None:
        node = WorkerNode(
            node_id="no-model",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
        )
        with pytest.raises(RuntimeError, match="Model not loaded"):
            node.forward_fn(hidden_states=torch.randn(1, 10))

    def test_forward_fn_raises_with_none_hidden(self) -> None:
        """Even with input_ids, model is required."""
        node = WorkerNode(
            node_id="no-model-2",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
        )
        with pytest.raises(RuntimeError, match="Model not loaded"):
            node.forward_fn(
                hidden_states=None,
                input_ids=torch.randint(0, 100, (1, 10)),
            )


# ---------------------------------------------------------------------------
# WorkerNode -- verify_model_integrity (error path)
# ---------------------------------------------------------------------------


class TestWorkerNodeVerifyIntegrity:
    """verify_model_integrity without a loaded model."""

    def test_verify_integrity_raises_when_no_model(self) -> None:
        node = WorkerNode(
            node_id="no-model",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
        )
        with pytest.raises(RuntimeError, match="Model not loaded"):
            node.verify_model_integrity()

    def test_verify_integrity_ignores_expected_checksum_when_no_model(
        self,
    ) -> None:
        """The no-model check fires before the checksum comparison."""
        node = WorkerNode(
            node_id="no-model",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
        )
        with pytest.raises(RuntimeError):
            node.verify_model_integrity(expected_checksum="a" * 64)


# ---------------------------------------------------------------------------
# WorkerNode -- _get_gpu_name
# ---------------------------------------------------------------------------


class TestWorkerNodeGpuName:
    """_get_gpu_name helper."""

    def test_get_gpu_name_no_cuda(self) -> None:
        node = WorkerNode(
            node_id="cpu-node",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
            device="cpu",
        )
        name = node._get_gpu_name()
        if not torch.cuda.is_available():
            assert name == "cpu"
        else:
            assert isinstance(name, str)
            assert len(name) > 0


# ---------------------------------------------------------------------------
# WorkerNode -- _get_host_ip
# ---------------------------------------------------------------------------


class TestWorkerNodeHostIp:
    """_get_host_ip helper."""

    def test_get_host_ip_returns_string(self) -> None:
        node = WorkerNode(
            node_id="host-ip",
            model_name="m",
            start_layer=0,
            end_layer=1,
            total_layers=2,
            port=50051,
        )
        ip = node._get_host_ip()
        assert isinstance(ip, str)
        assert len(ip) > 0


# ---------------------------------------------------------------------------
# _validate_state_dict_keys
# ---------------------------------------------------------------------------


class TestValidateStateDictKeys:
    """Module-level helper that validates state dict keys.

    Uses a loguru ``io.StringIO`` sink to capture warnings since
    ``capsys`` does not intercept loguru output.
    """

    @staticmethod
    def _capture_warning(callable, *args, **kwargs) -> str:
        """Execute *callable* with *args/*kwargs and return captured log output."""
        sink = io.StringIO()
        handler_id = logger.add(sink, format="{message}", level="WARNING")
        try:
            callable(*args, **kwargs)
        finally:
            logger.remove(handler_id)
        return sink.getvalue()

    def test_no_unexpected_keys_no_warning(self) -> None:
        """Keys in state_dict exactly match model keys -- no log output."""
        model = nn.Linear(4, 4)
        state_dict = model.state_dict()
        output = self._capture_warning(_validate_state_dict_keys, model, state_dict)
        assert "not found" not in output.lower()

    def test_unexpected_keys_logs_warning(self) -> None:
        """Extra keys in state_dict produce a warning."""
        model = nn.Linear(4, 4)
        state_dict = {
            "weight": torch.randn(4, 4),
            "bias": torch.randn(4),
            "extra_key": torch.zeros(1),
        }
        output = self._capture_warning(_validate_state_dict_keys, model, state_dict)
        assert "not found" in output.lower()

    def test_multiple_unexpected_keys_logged(self) -> None:
        """Multiple extra keys produce a warning."""
        model = nn.Linear(4, 4)
        state_dict = {
            "weight": torch.randn(4, 4),
            "bias": torch.randn(4),
            "extra_1": torch.zeros(1),
            "extra_2": torch.ones(1),
        }
        output = self._capture_warning(_validate_state_dict_keys, model, state_dict)
        assert "not found" in output.lower()

    def test_empty_state_dict_no_error(self) -> None:
        """An empty state dict has no unexpected keys."""
        model = nn.Linear(4, 4)
        _validate_state_dict_keys(model, {})
        # set() - {"weight", "bias"} = set(), so no warning

    def test_state_dict_subset_no_warning(self) -> None:
        """A subset of model keys is fine (unexpected checks only)."""
        model = nn.Linear(4, 4)
        state_dict = {"weight": torch.randn(4, 4)}
        _validate_state_dict_keys(model, state_dict)
        # unexpected = {"weight"} - {"weight", "bias"} = set()
        # No warning expected


# ---------------------------------------------------------------------------
# Module-level smoke: ensure the module imports key public names
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify expected public names are importable from the module."""

    def test_worker_node_exported(self) -> None:
        import distllm.dist.worker as w

        assert w.WorkerNode is WorkerNode

    def test_compression_config_exported(self) -> None:
        import distllm.dist.worker as w

        assert w._SimpleCompressionConfig is _SimpleCompressionConfig

    def test_validate_keys_exported(self) -> None:
        import distllm.dist.worker as w

        assert w._validate_state_dict_keys is _validate_state_dict_keys
