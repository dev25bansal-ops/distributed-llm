"""Tests for AutoPartitioner: hardware profiling, layer estimation, TP groups, memory report.

Tests: LayerInfo, DeviceAssignment, PartitionPlan, AutoPartitioner init,
estimate_layer_memory, build_layers, partition (single/multi-GPU),
TP group building, memory report, and edge cases.

Run: pytest tests/core/test_auto_partitioner.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from distllm.core.auto_partitioner import (
    AutoPartitioner,
    LayerInfo,
    DeviceAssignment,
    PartitionPlan,
)
from distllm.core.gpu_profiler import GPUInfo


def _make_gpu(gpu_id=0, name="TestGPU", total_memory=8 * 1024**3):
    return GPUInfo(
        gpu_id=gpu_id,
        name=name,
        total_memory=total_memory,
        used_memory=0,
        free_memory=total_memory,
        utilization=0.0,
    )


# --- Dataclass tests ---


class TestLayerInfo:
    """Tests for LayerInfo dataclass."""

    def test_defaults(self):
        layer = LayerInfo(name="test", layer_id=0)
        assert layer.memory_bytes == 0
        assert layer.flops_per_token == 0
        assert layer.layer_type == "attention"
        assert layer.is_embedding is False
        assert layer.is_lm_head is False

    def test_custom_values(self):
        layer = LayerInfo(
            name="mlp.0", layer_id=5, memory_bytes=1024,
            flops_per_token=500, layer_type="mlp",
        )
        assert layer.memory_bytes == 1024
        assert layer.flops_per_token == 500
        assert layer.layer_type == "mlp"


class TestDeviceAssignment:
    """Tests for DeviceAssignment dataclass."""

    def test_memory_utilization(self):
        da = DeviceAssignment(device_id=0, device_name="gpu0", total_memory_bytes=4096)
        assert da.memory_utilization == 1.0

    def test_empty_assignment(self):
        da = DeviceAssignment(device_id=0, device_name="cpu")
        assert da.layers == []
        assert da.total_memory_bytes == 0
        assert da.total_flops == 0


class TestPartitionPlan:
    """Tests for PartitionPlan dataclass."""

    def test_summary_empty(self):
        plan = PartitionPlan()
        summary = plan.summary()
        assert "0 devices" in summary
        assert "0 layers" in summary

    def test_summary_with_data(self):
        assignments = [
            DeviceAssignment(device_id=0, device_name="gpu0", layers=[
                LayerInfo(name="layer0", layer_id=0, memory_bytes=100),
            ]),
        ]
        plan = PartitionPlan(assignments=assignments, estimated_throughput=500.0)
        summary = plan.summary()
        assert "1 devices" in summary
        assert "1 layers" in summary
        assert "throughput=500 tok/s" in summary


# --- AutoPartitioner init tests ---


class TestAutoPartitionerInit:
    """Tests for AutoPartitioner initialization."""

    def test_defaults(self):
        ap = AutoPartitioner()
        assert ap._hidden == 4096
        assert ap._num_layers == 32
        assert ap._num_heads == 32
        assert ap._num_kv == 32
        assert ap._intermediate == 11008
        assert ap._vocab == 32000
        assert ap._max_seq == 4096
        assert ap._batch == 1

    def test_custom_params(self):
        ap = AutoPartitioner(
            hidden_size=2048, num_layers=12, num_attention_heads=16,
            num_kv_heads=4, intermediate_size=8192, vocab_size=50000,
            max_seq_len=2048, batch_size=4,
        )
        assert ap._hidden == 2048
        assert ap._num_layers == 12
        assert ap._num_heads == 16
        assert ap._num_kv == 4
        assert ap._intermediate == 8192
        assert ap._vocab == 50000
        assert ap._max_seq == 2048
        assert ap._batch == 4


# --- Memory estimation tests ---


class TestEstimateLayerMemory:
    """Tests for _estimate_layer_memory."""

    def test_attention_memory(self):
        ap = AutoPartitioner(hidden_size=1024)
        mem = ap._estimate_layer_memory("attention")
        # qkv + o_proj = 3*1024*1024 + 1024*1024 = 4*1024^2
        expected = 4 * 1024 * 1024 * 2  # *2 for fp16
        assert mem == expected

    def test_mlp_memory(self):
        ap = AutoPartitioner(hidden_size=1024, intermediate_size=4096)
        mem = ap._estimate_layer_memory("mlp")
        # gate + up + down = 2*1024*4096 + 4096*1024 = 3*1024*4096
        expected = 3 * 1024 * 4096 * 2
        assert mem == expected

    def test_norm_memory(self):
        ap = AutoPartitioner(hidden_size=1024)
        mem = ap._estimate_layer_memory("norm")
        expected = 2 * 1024 * 2
        assert mem == expected

    def test_embed_memory(self):
        ap = AutoPartitioner(hidden_size=512, vocab_size=10000)
        mem = ap._estimate_layer_memory("embed")
        expected = 10000 * 512 * 2
        assert mem == expected

    def test_unknown_type_returns_zero(self):
        ap = AutoPartitioner()
        assert ap._estimate_layer_memory("unknown") == 0


# --- Build layers tests ---


class TestBuildLayers:
    """Tests for _build_layers."""

    def test_correct_number_of_layers(self):
        ap = AutoPartitioner(num_layers=4)
        layers = ap._build_layers()
        # Each layer produces attention + mlp = 2 LayerInfos
        assert len(layers) == 8

    def test_layer_names(self):
        ap = AutoPartitioner(num_layers=2)
        layers = ap._build_layers()
        assert layers[0].name == "model.layers.0.self_attn"
        assert layers[1].name == "model.layers.0.mlp"
        assert layers[2].name == "model.layers.1.self_attn"
        assert layers[3].name == "model.layers.1.mlp"

    def test_layer_types(self):
        ap = AutoPartitioner(num_layers=1)
        layers = ap._build_layers()
        assert layers[0].layer_type == "attention"
        assert layers[1].layer_type == "mlp"

    def test_layer_ids_sequential(self):
        ap = AutoPartitioner(num_layers=2)
        layers = ap._build_layers()
        for i, layer in enumerate(layers):
            assert layer.layer_id == i

    def test_memory_bytes_populated(self):
        ap = AutoPartitioner(num_layers=1)
        layers = ap._build_layers()
        assert layers[0].memory_bytes > 0
        assert layers[1].memory_bytes > 0


# --- Partition tests ---


class TestPartitionNoGpu:
    """Tests for partition() when no GPUs are available."""

    @patch.object(AutoPartitioner, "__init__", lambda self, **kwargs: None)
    def test_no_gpus_returns_single_cpu_plan(self):
        ap = AutoPartitioner()
        ap._num_layers = 4
        ap._hidden = 1024
        ap._num_heads = 8
        ap._num_kv = 8
        ap._intermediate = 4096
        ap._vocab = 10000
        ap._max_seq = 512
        ap._batch = 1
        ap._profiler = MagicMock()
        ap._profiler.enumerate_gpus.return_value = []

        plan = ap.partition()

        assert len(plan.assignments) == 1
        assert plan.assignments[0].device_id == 0
        assert plan.assignments[0].device_name == "cpu"
        assert len(plan.assignments[0].layers) > 0


class TestPartitionSingleGpu:
    """Tests for partition() with a single GPU."""

    @patch.object(AutoPartitioner, "__init__", lambda self, **kwargs: None)
    def test_single_gpu_all_layers_assigned(self):
        ap = AutoPartitioner()
        ap._num_layers = 4
        ap._hidden = 1024
        ap._num_heads = 8
        ap._num_kv = 8
        ap._intermediate = 4096
        ap._vocab = 10000
        ap._max_seq = 512
        ap._batch = 1
        ap._profiler = MagicMock()
        ap._profiler.enumerate_gpus.return_value = [_make_gpu(0)]

        plan = ap.partition()

        assert len(plan.assignments) == 1
        total_layers = sum(len(a.layers) for a in plan.assignments)
        assert total_layers == 8  # 4 layers * 2 (attn + mlp)


class TestPartitionMultiGpu:
    """Tests for partition() with multiple GPUs."""

    @patch.object(AutoPartitioner, "__init__", lambda self, **kwargs: None)
    def test_multi_gpu_balanced_distribution(self):
        ap = AutoPartitioner()
        ap._num_layers = 8
        ap._hidden = 1024
        ap._num_heads = 8
        ap._num_kv = 8
        ap._intermediate = 4096
        ap._vocab = 10000
        ap._max_seq = 512
        ap._batch = 1
        ap._profiler = MagicMock()
        ap._profiler.enumerate_gpus.return_value = [
            _make_gpu(0, "gpu0", 8 * 1024**3),
            _make_gpu(1, "gpu1", 8 * 1024**3),
        ]

        plan = ap.partition()

        assert len(plan.assignments) == 2
        total_layers = sum(len(a.layers) for a in plan.assignments)
        assert total_layers == 16  # 8 * 2
        # Layers should be roughly balanced
        loads = [a.total_memory_bytes for a in plan.assignments]
        ratio = max(loads) / min(loads) if min(loads) > 0 else float('inf')
        assert ratio <= 2.0  # reasonable balance

    @patch.object(AutoPartitioner, "__init__", lambda self, **kwargs: None)
    def test_multi_gpu_tp_groups(self):
        ap = AutoPartitioner()
        ap._num_layers = 4
        ap._hidden = 1024
        ap._num_heads = 8
        ap._num_kv = 8
        ap._intermediate = 4096
        ap._vocab = 10000
        ap._max_seq = 512
        ap._batch = 1
        ap._profiler = MagicMock()
        ap._profiler.enumerate_gpus.return_value = [
            _make_gpu(0), _make_gpu(1), _make_gpu(2), _make_gpu(3),
        ]

        plan = ap.partition()

        assert len(plan.tp_groups) > 0
        # All device IDs should be covered by TP groups
        all_ids = set()
        for group in plan.tp_groups:
            all_ids.update(group)
        assert all_ids == {0, 1, 2, 3}

    @patch.object(AutoPartitioner, "__init__", lambda self, **kwargs: None)
    def test_multi_gpu_pp_stages(self):
        ap = AutoPartitioner()
        ap._num_layers = 4
        ap._hidden = 1024
        ap._num_heads = 8
        ap._num_kv = 8
        ap._intermediate = 4096
        ap._vocab = 10000
        ap._max_seq = 512
        ap._batch = 1
        ap._profiler = MagicMock()
        ap._profiler.enumerate_gpus.return_value = [
            _make_gpu(0), _make_gpu(1),
        ]

        plan = ap.partition()

        assert len(plan.pp_stages) == 2
        assert all(len(stage) == 1 for stage in plan.pp_stages)

    @patch.object(AutoPartitioner, "__init__", lambda self, **kwargs: None)
    def test_throughput_estimated(self):
        ap = AutoPartitioner()
        ap._num_layers = 4
        ap._hidden = 1024
        ap._num_heads = 8
        ap._num_kv = 8
        ap._intermediate = 4096
        ap._vocab = 10000
        ap._max_seq = 512
        ap._batch = 1
        ap._profiler = MagicMock()
        ap._profiler.enumerate_gpus.return_value = [_make_gpu(0)]

        plan = ap.partition()

        assert plan.estimated_throughput > 0


# --- TP group building tests ---


class TestBuildTpGroups:
    """Tests for _build_tp_groups."""

    def test_single_gpu(self):
        ap = AutoPartitioner()
        gpus = [_make_gpu(0)]
        groups = ap._build_tp_groups(gpus)
        assert groups == [[0]]

    def test_two_gpus(self):
        ap = AutoPartitioner()
        gpus = [_make_gpu(0), _make_gpu(1)]
        groups = ap._build_tp_groups(gpus)
        assert groups == [[0, 1]]

    def test_four_gpus_paired(self):
        ap = AutoPartitioner()
        gpus = [_make_gpu(0), _make_gpu(1), _make_gpu(2), _make_gpu(3)]
        groups = ap._build_tp_groups(gpus)
        assert len(groups) == 2
        assert groups[0] == [0, 1]
        assert groups[1] == [2, 3]

    def test_odd_number_gpus(self):
        ap = AutoPartitioner()
        gpus = [_make_gpu(0), _make_gpu(1), _make_gpu(2)]
        groups = ap._build_tp_groups(gpus)
        # Should have pairs, with odd one appended to last group
        all_ids = []
        for g in groups:
            all_ids.extend(g)
        assert sorted(all_ids) == [0, 1, 2]


# --- Memory report tests ---


class TestMemoryReport:
    """Tests for get_memory_report."""

    @patch.object(AutoPartitioner, "__init__", lambda self, **kwargs: None)
    def test_report_structure(self):
        ap = AutoPartitioner()
        ap._num_layers = 4
        ap._hidden = 1024
        ap._num_heads = 8
        ap._num_kv = 8
        ap._intermediate = 4096
        ap._vocab = 10000
        ap._max_seq = 512
        ap._batch = 1
        ap._profiler = MagicMock()
        ap._profiler.enumerate_gpus.return_value = [_make_gpu(0)]

        report = ap.get_memory_report()

        assert "num_gpus" in report
        assert "num_layers" in report
        assert "total_layer_memory_gb" in report
        assert "gpu_memory_gb" in report
        assert "gpu_names" in report
        assert "estimated_attention_memory_gb" in report
        assert "estimated_mlp_memory_gb" in report
        assert report["num_gpus"] == 1
        assert report["num_layers"] == 4

    @patch.object(AutoPartitioner, "__init__", lambda self, **kwargs: None)
    def test_report_no_gpus(self):
        ap = AutoPartitioner()
        ap._num_layers = 4
        ap._hidden = 1024
        ap._num_heads = 8
        ap._num_kv = 8
        ap._intermediate = 4096
        ap._vocab = 10000
        ap._max_seq = 512
        ap._batch = 1
        ap._profiler = MagicMock()
        ap._profiler.enumerate_gpus.return_value = []

        report = ap.get_memory_report()

        assert report["num_gpus"] == 0
        assert report["gpu_memory_gb"] == []
        assert report["gpu_names"] == []
