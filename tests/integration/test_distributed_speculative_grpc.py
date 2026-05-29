"""gRPC transport tests for RemoteDraftModel."""

from unittest.mock import MagicMock

from distllm.core.distributed_speculative import (
    RemoteDraftConfig,
    RemoteDraftModel,
)


class TestGRPCTransport:
    def test_config_grpc_transport(self):
        cfg = RemoteDraftConfig(
            endpoint_url="grpc://draft-node:50051",
            transport="grpc",
        )
        assert cfg.transport == "grpc"

    def test_grpc_stub_lazy_init(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="localhost:50051",
            transport="grpc",
        ))
        assert model._grpc_stub is None
        model.close()

    def test_grpc_call_returns_draft_result(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="localhost:50051",
            transport="grpc",
        ))

        # Mock the gRPC stub
        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.output.data = [10, 20, 30]
        mock_response.error_message = ""
        mock_stub.ForwardPass.return_value = mock_response
        model._grpc_stub = mock_stub

        result = model.generate_tokens([1, 2, 3], num_tokens=3)
        assert result.ok
        assert result.token_ids == [10, 20, 30]

    def test_grpc_call_failure(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="localhost:50051",
            transport="grpc",
        ))

        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.output = None
        mock_response.error_message = "model not loaded"
        mock_stub.ForwardPass.return_value = mock_response
        model._grpc_stub = mock_stub

        result = model.generate_tokens([1, 2, 3], num_tokens=3)
        assert not result.ok
        assert "model not loaded" in result.error

    def test_grpc_stats_accumulate(self):
        model = RemoteDraftModel(RemoteDraftConfig(
            endpoint_url="localhost:50051",
            transport="grpc",
        ))

        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.output.data = [5, 6]
        mock_stub.ForwardPass.return_value = mock_response
        model._grpc_stub = mock_stub

        model.generate_tokens([1], num_tokens=2)
        model.generate_tokens([1], num_tokens=2)

        s = model.stats
        assert s["total_calls"] == 2
        assert s["total_tokens"] == 4
