"""Tests: CLI cluster commands — status, scale, start, join."""

from unittest.mock import MagicMock, patch

import httpx
import pytest
import typer
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

        with patch("distllm.cli.cluster.client_for") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_status
            _cluster_status("localhost", 8000)

            mock_client.get.assert_called_once_with("/v1/health")

    def test_status_no_nodes(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"nodes": [], "summary": {}}
        mock_resp.raise_for_status = MagicMock()

        with patch("distllm.cli.cluster.client_for") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_status
            _cluster_status("localhost", 8000)

    def test_status_connection_error(self):
        with patch("distllm.cli.cluster.client_for") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.ConnectError("refused")
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_status
            _cluster_status("localhost", 8000)


class TestClusterScale:
    """Scaling has no server route; the command must say so, not 404."""

    def test_scale_reports_unsupported(self):
        from distllm.cli.cluster import _cluster_scale

        with patch("distllm.cli.cluster.client_for") as mock_client_cls:
            with pytest.raises(typer.Exit) as exc:
                _cluster_scale("localhost", 8000, 4, "A100")
        assert exc.value.exit_code == 2
        mock_client_cls.assert_not_called()

    def test_scale_message_names_the_alternative(self):
        from distllm.cli.cluster import _cluster_scale

        with patch("distllm.cli.cluster.console") as mock_console:
            with pytest.raises(typer.Exit):
                _cluster_scale("localhost", 8000, 4)
        printed = " ".join(str(c.args[0]) for c in mock_console.print.call_args_list if c.args)
        assert "not supported" in printed.lower()
        assert "cluster join" in printed


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

        with patch("distllm.cli.cluster.client_for") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_client

            from distllm.cli.cluster import _cluster_list_nodes
            _cluster_list_nodes("localhost", 8000)
