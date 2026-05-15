"""Shared fixtures for integration tests.

Provides in-process gRPC servers and coordinators for real distributed pipeline testing
without requiring GPU or model downloads.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
import torch

from distllm.core.coordinator import Coordinator
from distllm.core.resource_manager import NodeRegistration
from distllm.communication.grpc import NodeClient, AsyncNodeClient


@pytest.fixture
def mock_model_partitioner():
    """Create a mock model partitioner that returns deterministic tensor outputs."""
    partitioner = MagicMock()
    mock_model = MagicMock()

    # Mock model parameters for VRAM estimation
    mock_param = torch.randn(10, 10)
    mock_model.parameters.return_value = iter([mock_param])
    mock_model.config = MagicMock()
    mock_model.config.num_hidden_layers = 12
    mock_model.config.hidden_size = 768
    mock_model.config.num_attention_heads = 12

    partitioner.full_model = mock_model
    partitioner.start_layer = 0
    partitioner.end_layer = 11
    return partitioner


@pytest.fixture
def mock_tensor_serializer():
    """Create a mock tensor serializer/deserializer."""
    from distllm.communication.serializers import tensor_to_proto, proto_to_tensor

    return tensor_to_proto, proto_to_tensor


@pytest.fixture
def integration_coordinator(mock_tokenizer, mock_model_partitioner):
    """Create a Coordinator with real gRPC infrastructure but mock model.

    This sets up a coordinator with:
    - Mock tokenizer and model partitioner (no GPU needed)
    - Real batch scheduler and pipeline orchestrator
    - Mock node clients that simulate gRPC communication
    """
    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
        max_batch_size=4,
        max_tokens_per_batch=512,
    )
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12
    coord.local_partitioner = mock_model_partitioner

    return coord


@pytest.fixture
def integration_coordinator_with_nodes(mock_tokenizer, mock_model_partitioner):
    """Coordinator with 2 mock node registrations using real gRPC client infrastructure."""
    coord = Coordinator(
        model_name="test-model",
        dtype="float32",
        max_batch_size=4,
        max_tokens_per_batch=512,
    )
    coord.tokenizer = mock_tokenizer
    coord.model_info = {"num_layers": 12, "hidden_size": 768, "num_attention_heads": 12}
    coord.total_layers = 12
    coord.local_partitioner = mock_model_partitioner

    # Register mock nodes with real client infrastructure but mocked stubs
    for i in range(2):
        mock_client = MagicMock()
        mock_health = MagicMock()
        mock_health.healthy = True
        mock_health.memory_used = 1024
        mock_health.memory_total = 8192
        mock_health.gpu_utilization = 0.5
        mock_client.health_check.return_value = mock_health

        # Mock forward pass response
        mock_forward = MagicMock()
        mock_forward.success = True
        mock_forward.error_message = ""
        mock_forward.request_id = "test-request"
        mock_client.forward_pass.return_value = mock_forward

        mock_async_client = AsyncMock()
        mock_async_client.health_check.return_value = mock_health
        mock_async_client.forward_pass.return_value = mock_forward

        reg = NodeRegistration(
            node_id=f"node-{i}",
            host="localhost",
            port=50051 + i,
            start_layer=i * 6,
            end_layer=(i + 1) * 6 - 1,
        )
        reg.client = mock_client
        reg.async_client = mock_async_client
        reg.healthy = True
        coord.nodes[f"node-{i}"] = reg
        coord.node_order.append(f"node-{i}")

    return coord
