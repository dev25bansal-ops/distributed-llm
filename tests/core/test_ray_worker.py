"""Tests for RayWorkerNode (without requiring a Ray cluster).

RayWorkerNode is intentionally NOT decorated with @ray.remote so it
can be instantiated directly in tests. The decorator is applied at
runtime in production via ray.remote(num_gpus=1)(RayWorkerNode).
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from distllm.core.ray_worker import RayWorkerNode


class TestRayWorkerNodeInit:
    """Test RayWorkerNode initialization."""

    def test_init_sets_basic_properties(self):
        worker = RayWorkerNode(
            node_id="test-worker",
            model_name="test-model",
            start_layer=0, end_layer=5, total_layers=12,
            device="cpu", dtype="float32",
            _skip_load=True,
        )
        assert worker.node_id == "test-worker"
        assert worker.start_layer == 0
        assert worker.end_layer == 5
        assert worker.total_layers == 12
        assert worker.is_first is True
        assert worker.is_last is False
        assert worker._device == "cpu"

    def test_init_middle_node_roles(self):
        worker = RayWorkerNode(
            node_id="worker-1", model_name="test-model",
            start_layer=4, end_layer=7, total_layers=12,
            device="cpu", dtype="float32", _skip_load=True,
        )
        assert worker.is_first is False
        assert worker.is_last is False

    def test_init_last_node_roles(self):
        worker = RayWorkerNode(
            node_id="worker-last", model_name="test-model",
            start_layer=8, end_layer=11, total_layers=12,
            device="cpu", dtype="float32", _skip_load=True,
        )
        assert worker.is_first is False
        assert worker.is_last is True

    @patch("distllm.core.ray_worker.torch.cuda.is_available", return_value=True)
    def test_resolve_device_cuda(self, mock_cuda):
        worker = RayWorkerNode(
            node_id="gpu-worker", model_name="test-model",
            start_layer=0, end_layer=5, total_layers=12,
            device="auto", dtype="float32", _skip_load=True,
        )
        assert worker._device == "cuda"

    @patch("distllm.core.ray_worker.torch.cuda.is_available", return_value=False)
    def test_resolve_device_cpu_fallback(self, mock_cuda):
        worker = RayWorkerNode(
            node_id="cpu-worker", model_name="test-model",
            start_layer=0, end_layer=5, total_layers=12,
            device="auto", dtype="float32", _skip_load=True,
        )
        assert worker._device == "cpu"


class TestRayWorkerNodeKVCache:
    """Test KV cache management."""

    def test_clear_kv_cache(self):
        worker = RayWorkerNode(
            node_id="test", model_name="test-model",
            start_layer=0, end_layer=5, total_layers=12,
            device="cpu", dtype="float32", _skip_load=True,
        )
        worker._kv_caches["req-1"] = [("mock_kv")]
        assert "req-1" in worker._kv_caches
        worker.clear_kv_cache("req-1")
        assert "req-1" not in worker._kv_caches

    def test_clear_all_kv_caches(self):
        worker = RayWorkerNode(
            node_id="test", model_name="test-model",
            start_layer=0, end_layer=5, total_layers=12,
            device="cpu", dtype="float32", _skip_load=True,
        )
        worker._kv_caches["req-1"] = [("a")]
        worker._kv_caches["req-2"] = [("b")]
        assert len(worker._kv_caches) == 2
        worker.clear_all_kv_caches()
        assert len(worker._kv_caches) == 0

    def test_get_kv_cache(self):
        worker = RayWorkerNode(
            node_id="test", model_name="test-model",
            start_layer=0, end_layer=5, total_layers=12,
            device="cpu", dtype="float32", _skip_load=True,
        )
        worker._kv_caches["req-1"] = [("mock_kv")]
        assert worker.get_kv_cache("req-1") == [("mock_kv")]
        assert worker.get_kv_cache("nonexistent") is None


class TestRayWorkerNodeHealth:
    """Test health and info methods."""

    @patch("distllm.core.ray_worker.torch.cuda.is_available", return_value=False)
    def test_health(self, mock_cuda):
        worker = RayWorkerNode(
            node_id="test", model_name="test-model",
            start_layer=0, end_layer=5, total_layers=12,
            device="cpu", dtype="float32", _skip_load=True,
        )
        result = worker.health()
        assert result["node_id"] == "test"
        assert result["healthy"] is True
        assert result["layers"] == (0, 5)
        assert result["kv_cache_entries"] == 0

    @patch("distllm.core.ray_worker.torch.cuda.is_available", return_value=False)
    def test_get_node_info(self, mock_cuda):
        worker = RayWorkerNode(
            node_id="test", model_name="test-model",
            start_layer=0, end_layer=5, total_layers=12,
            device="cpu", dtype="float32", _skip_load=True,
        )
        info = worker.get_node_info()
        assert info["node_id"] == "test"
        assert info["device_type"] == "cpu"
        assert info["is_first"] is True
        assert info["is_last"] is False


class TestRayWorkerNodeForward:
    """Test RayWorkerNode.forward() validation."""

    def test_forward_requires_input(self):
        worker = RayWorkerNode(
            node_id="test", model_name="test-model",
            start_layer=0, end_layer=1, total_layers=4,
            device="cpu", dtype="float32", _skip_load=True,
        )
        with pytest.raises(ValueError, match="Either input_ids or hidden_states required"):
            worker.forward(hidden_states=None, input_ids=None)
