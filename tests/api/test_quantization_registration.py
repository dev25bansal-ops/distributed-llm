"""Tests for quantization in gRPC registration flow."""

from unittest.mock import MagicMock, patch

import pytest

try:
    from distllm.communication.grpc import CoordinatorService
except ImportError:
    pytest.skip("distllm.communication.grpc module removed", allow_module_level=True)


class TestCoordinatorServiceQuantization:
    """Test CoordinatorService includes quantization in registration response."""

    def test_register_node_without_quantization(self):
        service = CoordinatorService()
        mock_context = MagicMock()

        node_info = MagicMock()
        node_info.node_id = "test-node"
        node_info.host = "localhost"
        node_info.port = 50051

        request = MagicMock()
        request.node_info = node_info
        request.metadata = []

        with patch("distllm.communication.grpc.grpc.insecure_channel"):
            with patch("distllm.communication.grpc.NodeServiceStub"):
                response = service.RegisterNode(request, mock_context)

        assert response.accepted is True
        assert response.quantization.method == ""

    def test_register_node_with_quantization_4bit(self):
        quant_config = MagicMock()
        quant_config.method = "bnb_4bit"
        quant_config.bnb_4bit_compute_dtype = "float16"
        quant_config.bnb_4bit_quant_type = "nf4"
        quant_config.bnb_4bit_use_double_quant = True
        quant_config.llm_int8_threshold = 6.0

        service = CoordinatorService(quantization_config=quant_config)
        mock_context = MagicMock()

        node_info = MagicMock()
        node_info.node_id = "test-node"
        node_info.host = "localhost"
        node_info.port = 50051

        request = MagicMock()
        request.node_info = node_info
        request.metadata = []

        with patch("distllm.communication.grpc.grpc.insecure_channel"):
            with patch("distllm.communication.grpc.NodeServiceStub"):
                response = service.RegisterNode(request, mock_context)

        assert response.accepted is True
        assert response.quantization.method == "bnb_4bit"
        assert response.quantization.bnb_4bit_compute_dtype == "float16"
        assert response.quantization.bnb_4bit_quant_type == "nf4"
        assert response.quantization.bnb_4bit_use_double_quant is True

    def test_register_node_with_quantization_8bit(self):
        quant_config = MagicMock()
        quant_config.method = "bnb_8bit"
        quant_config.bnb_4bit_compute_dtype = "float16"
        quant_config.bnb_4bit_quant_type = "nf4"
        quant_config.bnb_4bit_use_double_quant = True
        quant_config.llm_int8_threshold = 8.0

        service = CoordinatorService(quantization_config=quant_config)
        mock_context = MagicMock()

        node_info = MagicMock()
        node_info.node_id = "test-node"
        node_info.host = "localhost"
        node_info.port = 50051

        request = MagicMock()
        request.node_info = node_info
        request.metadata = []

        with patch("distllm.communication.grpc.grpc.insecure_channel"):
            with patch("distllm.communication.grpc.NodeServiceStub"):
                response = service.RegisterNode(request, mock_context)

        assert response.accepted is True
        assert response.quantization.method == "bnb_8bit"
        assert response.quantization.llm_int8_threshold == 8.0

    def test_register_node_with_none_method_skips_quantization(self):
        quant_config = MagicMock()
        quant_config.method = "none"

        service = CoordinatorService(quantization_config=quant_config)
        mock_context = MagicMock()

        node_info = MagicMock()
        node_info.node_id = "test-node"
        node_info.host = "localhost"
        node_info.port = 50051

        request = MagicMock()
        request.node_info = node_info
        request.metadata = []

        with patch("distllm.communication.grpc.grpc.insecure_channel"):
            with patch("distllm.communication.grpc.NodeServiceStub"):
                response = service.RegisterNode(request, mock_context)

        assert response.accepted is True
        assert response.quantization.method == ""
