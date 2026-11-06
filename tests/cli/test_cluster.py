"""Tests: CLI cluster commands — status, scale, start, join."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from distllm.cli.main import app

runner = CliRunner()


class TestClusterStatus:
    def test_status_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "nodes": [
                {"id": "node_0", "status": "healthy", "gpu_name": "A100",
                 "memory_used": "10GB", "active_requests": 2, "start_layer": 0, "end_layer": 3},
            ],
            "summary": {"total_nodes": 1, "healthy_nodes": 1, "total_gpu_memory": "80GB"},
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("distllm.cli.cluster.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_status
            _cluster_status("localhost", 8000)

            mock_client.get.assert_called_once_with("/v1/cluster/status")

    def test_status_no_nodes(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"nodes": [], "summary": {}}
        mock_resp.raise_for_status = MagicMock()

        with patch("distllm.cli.cluster.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_status
            _cluster_status("localhost", 8000)

    def test_status_connection_error(self):
        with patch("distllm.cli.cluster.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_status
            _cluster_status("localhost", 8000)


class TestClusterScale:
    def test_scale_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": "Scaling to 4 nodes", "job_id": "job-123"}
        mock_resp.raise_for_status = MagicMock()

        with patch("distllm.cli.cluster.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_scale
            _cluster_scale("localhost", 8000, 4, "A100")

            mock_client.post.assert_called_once_with(
                "/v1/cluster/scale",
                json={"target_nodes": 4, "gpu_type": "A100"},
            )

    def test_scale_http_error(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad request"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad", request=MagicMock(), response=mock_resp
        )

        with patch("distllm.cli.cluster.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_scale
            _cluster_scale("localhost", 8000, 4)


class TestClusterListNodes:
    def test_list_nodes_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "nodes": [
                {"id": "node_0", "host": "10.0.0.1", "port": 50051, "status": "healthy"},
                {"id": "node_1", "host": "10.0.0.2", "port": 50052, "status": "healthy"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("distllm.cli.cluster.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_list_nodes
            _cluster_list_nodes("localhost", 8000)
