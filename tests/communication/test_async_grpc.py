"""Tests for async gRPC implementations using grpc.aio.

Tests:
- AsyncNodeService: ForwardPass, HealthCheck, GetNodeInfo
- AsyncCoordinatorService: RegisterNode, Infer
- AsyncNodeClient: health_check, get_info, forward, close
- AsyncGRPCServer: start, stop
- AsyncChannelPool: get_stub, release, close_all
- PipelineOrchestrator.run_pipeline_async

Run: pytest tests/communication/test_async_grpc.py -v
"""

import asyncio
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, patch

from distllm.communication.grpc import (
    AsyncNodeService, AsyncCoordinatorService, AsyncGRPCServer,
    AsyncNodeClient, AsyncChannelPool,
)
from distllm.communication.node_pb2 import (
    ForwardPassRequest, HealthCheckRequest, HealthCheckResponse,
    NodeInfo, ForwardPassResponse, Tensor,
    RegistrationResponse, LogitsResponse,
)
from distllm.core.kv_cache import KVCache
from distllm.communication.serializers import tensor_to_proto


class TestAsyncNodeService:
    """Tests for AsyncNodeService."""

    @pytest.mark.asyncio
    async def test_forward_pass_success(self):
        """ForwardPass should return output tensor."""
        mock_forward = MagicMock(return_value=(
            torch.tensor([[0.1, 0.2, 0.3]]),
            None,
        ))
        service = AsyncNodeService("node-0", mock_forward)

        request = ForwardPassRequest(
            request_id="req-1",
            use_cache=False,
        )
        request.input_ids.extend([1, 2, 3])

        response = await service.ForwardPass(request, MagicMock())

        assert response.success is True
        assert response.request_id == "req-1"

    @pytest.mark.asyncio
    async def test_forward_pass_with_hidden_states(self):
        """ForwardPass should accept hidden_states input."""
        mock_forward = MagicMock(return_value=(
            torch.tensor([[0.1, 0.2]]),
            None,
        ))
        service = AsyncNodeService("node-0", mock_forward)

        request = ForwardPassRequest(request_id="req-2", use_cache=False)
        hs_tensor = torch.tensor([[0.5, 0.6]])
        request.hidden_states.CopyFrom(tensor_to_proto(hs_tensor))

        response = await service.ForwardPass(request, MagicMock())

        assert response.success is True
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    async def test_forward_pass_error_handling(self):
        """ForwardPass should return success=False on error."""
        mock_forward = MagicMock(side_effect=RuntimeError("Model error"))
        service = AsyncNodeService("node-0", mock_forward)

        request = ForwardPassRequest(request_id="req-3", use_cache=False)
        request.input_ids.clear()

        response = await service.ForwardPass(request, MagicMock())

        assert response.success is False

    @pytest.mark.asyncio
    async def test_health_check(self):
        """HealthCheck should return node health."""
        service = AsyncNodeService("node-0", MagicMock())

        response = await service.HealthCheck(HealthCheckRequest(), MagicMock())

        assert response.healthy is True
        assert response.node_id == "node-0"

    @pytest.mark.asyncio
    async def test_get_node_info(self):
        """GetNodeInfo should return hardware info."""
        service = AsyncNodeService("node-0", MagicMock())

        response = await service.GetNodeInfo(HealthCheckRequest(), MagicMock())

        assert response.node_id == "node-0"
        assert response.device_type in ("cuda", "cpu")


class TestAsyncCoordinatorService:
    """Tests for AsyncCoordinatorService."""

    @pytest.mark.asyncio
    async def test_register_node(self):
        """RegisterNode should accept node registration."""
        service = AsyncCoordinatorService()

        request = MagicMock()
        request.metadata = []
        node_info = NodeInfo(
            node_id="worker-1",
            device_type="cuda",
            device_name="Test GPU",
            total_memory=8000000000,
            available_memory=4000000000,
        )
        node_info.host = "localhost"
        node_info.port = 50051
        request.node_info = node_info

        response = await service.RegisterNode(request, MagicMock())

        assert response.accepted is True
        assert "worker-1" in service.nodes

    @pytest.mark.asyncio
    async def test_infer_redirect(self):
        """Infer should return error when no worker nodes registered."""
        service = AsyncCoordinatorService()

        request = MagicMock()
        request.request_id = "req-1"

        response = await service.Infer(request, MagicMock())

        assert response.success is False
        assert "No worker nodes registered" in response.error_message


class TestAsyncNodeClient:
    """Tests for AsyncNodeClient."""

    @pytest.mark.asyncio
    async def test_client_creation(self):
        """AsyncNodeClient should create async channel and stub."""
        with patch('distllm.communication.grpc.grpc.aio.insecure_channel') as mock_channel, \
             patch('distllm.communication.grpc.NodeServiceStub') as mock_stub_class:
            mock_stub = MagicMock()
            mock_stub_class.return_value = mock_stub
            mock_channel.return_value = MagicMock()

            client = AsyncNodeClient("localhost", 50051, use_tls=False)

            assert client.host == "localhost"
            assert client.port == 50051
            assert client.stub == mock_stub
            mock_channel.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_async(self):
        """health_check should call stub.HealthCheck."""
        with patch('distllm.communication.grpc.grpc.aio.insecure_channel') as mock_channel, \
             patch('distllm.communication.grpc.NodeServiceStub') as mock_stub_class:
            mock_stub = AsyncMock()
            mock_stub.HealthCheck = AsyncMock(return_value=HealthCheckResponse(
                node_id="node-0", healthy=True
            ))
            mock_stub_class.return_value = mock_stub
            mock_channel.return_value = MagicMock()

            client = AsyncNodeClient("localhost", 50051, use_tls=False)
            response = await client.health_check()

            assert response.healthy is True
            mock_stub.HealthCheck.assert_called_once()

    @pytest.mark.asyncio
    async def test_forward_async(self):
        """forward should call stub.ForwardPass."""
        with patch('distllm.communication.grpc.grpc.aio.insecure_channel') as mock_channel, \
             patch('distllm.communication.grpc.NodeServiceStub') as mock_stub_class:
            mock_stub = AsyncMock()
            mock_stub.ForwardPass = AsyncMock(return_value=ForwardPassResponse(
                request_id="req-1", success=True
            ))
            mock_stub_class.return_value = mock_stub
            mock_channel.return_value = MagicMock()

            client = AsyncNodeClient("localhost", 50051, use_tls=False)
            request = ForwardPassRequest(request_id="req-1", use_cache=False)
            request.input_ids.extend([1, 2])

            response = await client.forward(request)

            assert response.success is True
            mock_stub.ForwardPass.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_async(self):
        """close should close the async channel."""
        with patch('distllm.communication.grpc.grpc.aio.insecure_channel') as mock_channel, \
             patch('distllm.communication.grpc.NodeServiceStub') as mock_stub_class:
            mock_channel_instance = MagicMock()
            mock_channel_instance.close = AsyncMock()
            mock_channel.return_value = mock_channel_instance
            mock_stub_class.return_value = MagicMock()

            client = AsyncNodeClient("localhost", 50051, use_tls=False)
            await client.close()

            mock_channel_instance.close.assert_called_once()


class TestAsyncChannelPool:
    """Tests for AsyncChannelPool."""

    @pytest.mark.asyncio
    async def test_get_stub_creates_new_channel(self):
        """get_stub should create new channel when pool is empty."""
        pool = AsyncChannelPool()

        with patch('distllm.communication.grpc.grpc.aio.insecure_channel') as mock_channel:
            mock_channel.return_value = MagicMock()
            stub, channel = await pool.get_stub("localhost:50051", MagicMock())

            assert stub is not None
            assert channel is not None
            assert pool.active_connections == 1

    @pytest.mark.asyncio
    async def test_release_returns_to_pool(self):
        """release should return channel to pool if healthy."""
        from unittest.mock import AsyncMock
        import grpc as grpc_module
        pool = AsyncChannelPool()

        with patch('distllm.communication.grpc.grpc.aio.insecure_channel') as mock_channel:
            mock_ch = MagicMock()
            mock_ch.check_connectivity_state.return_value = grpc_module.ChannelConnectivity.READY
            mock_ch.close = AsyncMock()
            mock_channel.return_value = mock_ch

            stub_class = MagicMock()
            mock_stub = MagicMock()
            stub_class.return_value = mock_stub

            stub, channel = await pool.get_stub("localhost:50051", stub_class)
            await pool.release("localhost:50051", channel, stub)

            assert pool.pooled_connections == 1

    @pytest.mark.asyncio
    async def test_close_all(self):
        """close_all should close all pooled channels."""
        import grpc as grpc_module
        from unittest.mock import AsyncMock
        pool = AsyncChannelPool()

        with patch('distllm.communication.grpc.grpc.aio.insecure_channel') as mock_channel:
            mock_ch = MagicMock()
            mock_ch.close = AsyncMock()
            mock_ch.check_connectivity_state.return_value = grpc_module.ChannelConnectivity.READY
            mock_channel.return_value = mock_ch

            stub_class = MagicMock()
            stub_class.return_value = MagicMock()

            stub, channel = await pool.get_stub("localhost:50051", stub_class)
            await pool.release("localhost:50051", channel, stub)

            await pool.close_all()

            mock_ch.close.assert_called_once()
            assert pool.pooled_connections == 0


class TestAsyncGRPCServer:
    """Tests for AsyncGRPCServer."""

    @pytest.mark.asyncio
    async def test_server_start_stop(self):
        """Server should start and stop cleanly."""
        mock_servicer = MagicMock()

        with patch('distllm.communication.grpc.grpc.aio.server') as mock_server_class:
            mock_server = AsyncMock()
            mock_server.start = AsyncMock()
            mock_server.stop = AsyncMock()
            mock_server.add_insecure_port = MagicMock()
            mock_server_class.return_value = mock_server

            server = AsyncGRPCServer(50050, mock_servicer, use_tls=False)
            await server.start()

            assert server.server is not None
            mock_server.start.assert_called_once()

            await server.stop()
            mock_server.stop.assert_called_once()


class TestPipelineOrchestratorAsync:
    """Tests for PipelineOrchestrator.run_pipeline_async."""

    @pytest.mark.asyncio
    async def test_run_pipeline_async_with_async_client(self):
        """run_pipeline_async should use async client when available."""
        from distllm.core.pipeline_orchestrator import PipelineOrchestrator
        from distllm.core.resource_manager import NodeRegistration
        from unittest.mock import MagicMock, patch, AsyncMock

        mock_node = MagicMock()
        mock_node.host = "localhost"
        mock_node.port = 50051
        mock_node.start_layer = 0
        mock_node.end_layer = 11
        mock_node.healthy = True
        mock_node.async_client = MagicMock()
        mock_node.async_client.stub = AsyncMock()

        mock_output = torch.tensor([[0.1, 0.2, 0.3]])
        mock_response = ForwardPassResponse(
            request_id="req-1",
            success=True,
            output=tensor_to_proto(mock_output),
        )
        mock_node.async_client.stub.ForwardPass.return_value = mock_response

        orchestrator = PipelineOrchestrator()
        orchestrator.nodes["node-0"] = mock_node
        orchestrator.node_order = ["node-0"]

        input_ids = torch.tensor([[1, 2, 3]])
        node_kv_caches = {"node-0": None}

        result = await orchestrator.run_pipeline_async(input_ids, node_kv_caches, "req-1")

        assert result is not None
        mock_node.async_client.stub.ForwardPass.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_pipeline_async_fallback_to_sync(self):
        """run_pipeline_async should fallback to sync client via asyncio.to_thread."""
        from distllm.core.pipeline_orchestrator import PipelineOrchestrator
        from unittest.mock import MagicMock, patch

        mock_node = MagicMock()
        mock_node.host = "localhost"
        mock_node.port = 50051
        mock_node.start_layer = 0
        mock_node.end_layer = 11
        mock_node.healthy = True
        mock_node.async_client = None
        mock_node.client = MagicMock()
        mock_node.client.stub = MagicMock()

        mock_output = torch.tensor([[0.1, 0.2, 0.3]])
        mock_response = ForwardPassResponse(
            request_id="req-1",
            success=True,
            output=tensor_to_proto(mock_output),
        )
        mock_node.client.stub.ForwardPass.return_value = mock_response

        orchestrator = PipelineOrchestrator()
        orchestrator.nodes["node-0"] = mock_node
        orchestrator.node_order = ["node-0"]

        input_ids = torch.tensor([[1, 2, 3]])
        node_kv_caches = {"node-0": None}

        result = await orchestrator.run_pipeline_async(input_ids, node_kv_caches, "req-1")

        assert result is not None
        mock_node.client.stub.ForwardPass.assert_called_once()
