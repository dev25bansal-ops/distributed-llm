"""Tests for WorkerNode quantization config support."""

from unittest.mock import MagicMock, patch

import pytest

from distllm.core.node import WorkerNode


class TestWorkerNodeQuantization:
    """Test WorkerNode quantization config handling."""

    def test_init_without_quantization(self):
        node = WorkerNode(
            node_id="test-node",
            model_name="test-model",
            start_layer=0,
            end_layer=3,
            total_layers=6,
            port=50051,
        )
        assert node.quantization_config is None

    def test_init_with_quantization(self):
        quant_config = MagicMock(method="bnb_4bit")
        node = WorkerNode(
            node_id="test-node",
            model_name="test-model",
            start_layer=0,
            end_layer=3,
            total_layers=6,
            port=50051,
            quantization_config=quant_config,
        )
        assert node.quantization_config is quant_config

    @patch("distllm.core.node.ModelPartitioner")
    def test_load_model_passes_quantization(self, mock_partitioner_cls):
        mock_partitioner = MagicMock()
        mock_partitioner_cls.return_value = mock_partitioner

        quant_config = MagicMock(method="bnb_8bit")
        node = WorkerNode(
            node_id="test-node",
            model_name="test-model",
            start_layer=0,
            end_layer=2,
            total_layers=6,
            port=50051,
            quantization_config=quant_config,
        )

        with patch.object(node, "_get_device", return_value="cuda"):
            node.load_model()

        mock_partitioner_cls.assert_called_once_with(
            model_name="test-model",
            device="auto",
            dtype="float16",
            quantization_config=quant_config,
            compression_config=None,
        )
        mock_partitioner.load_layer_subset.assert_called_once_with(
            0, 2, 6, device="cuda"
        )

    @patch("distllm.core.node.ModelPartitioner")
    def test_load_model_without_quantization(self, mock_partitioner_cls):
        mock_partitioner = MagicMock()
        mock_partitioner_cls.return_value = mock_partitioner

        node = WorkerNode(
            node_id="test-node",
            model_name="test-model",
            start_layer=0,
            end_layer=2,
            total_layers=6,
            port=50051,
        )

        with patch.object(node, "_get_device", return_value="cpu"):
            node.load_model()

        mock_partitioner_cls.assert_called_once_with(
            model_name="test-model",
            device="auto",
            dtype="float16",
            quantization_config=None,
            compression_config=None,
        )
