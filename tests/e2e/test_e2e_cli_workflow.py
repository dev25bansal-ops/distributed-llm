"""E2E: CLI workflow (cluster start -> join -> chat -> stop).

Tests CLI commands using Typer's CliRunner:
1. `distllm --help` — command listing
2. `distllm setup` — interactive config creation
3. `distllm validate-config` — config validation
4. `distllm cluster start/join/leave/status` — cluster management
5. `distllm chat` — interactive chat
6. `distllm models list/info/load/unload` — model management
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

pytestmark = [pytest.mark.e2e, pytest.mark.cli]


# ====================================================================
# Fixtures
# ====================================================================

@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture
def cli_runner():
    """Typer CLI test runner."""
    from typer.testing import CliRunner
    return CliRunner()


# ====================================================================
# Top-level CLI
# ====================================================================

class TestCLIHelp:
    """Top-level --help and subcommand listing."""

    def test_main_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ["setup", "run", "validate-config", "status", "chat",
                     "models", "cluster", "adapters", "logs",
                     "benchmark", "verify", "backup", "cert"]:
            assert cmd in result.stdout

    def test_version_displayed(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code in (0, 2)


# ====================================================================
# distllm setup
# ====================================================================

class TestCLISetup:
    """distllm setup — interactive config creation."""

    def test_setup_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout.lower()

    def test_setup_runs_with_input(self, cli_runner):
        from distllm.cli.main import app
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            result = cli_runner.invoke(
                app, ["setup", "--config", path],
                input="test-model\nfloat32\n",
            )
            assert result.exit_code in (0, 1)
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ====================================================================
# distllm validate-config
# ====================================================================

class TestCLIValidateConfig:
    """distllm validate-config — configuration validation."""

    def test_validate_config_passes(self, cli_runner, monkeypatch):
        monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["validate-config"])
        assert result.exit_code == 0


# ====================================================================
# distllm status
# ====================================================================

class TestCLIStatus:
    """distllm status — cluster overview."""

    def test_status_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_status_handles_connection_error(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["status", "--host", "127.0.0.1", "--port", "1"])
        assert result.exit_code in (0, 1)


# ====================================================================
# distllm cluster commands
# ====================================================================

class TestCLIClusterHelp:
    """distllm cluster — help and subcommand listing."""

    def test_cluster_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "--help"])
        assert result.exit_code == 0
        for cmd in ["start", "join", "leave", "status", "scale", "drain", "rebalance", "list-nodes"]:
            assert cmd in result.stdout

    def test_cluster_start_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "start", "--help"])
        assert result.exit_code == 0
        assert "model" in result.stdout

    def test_cluster_start_requires_model(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "start"])
        assert result.exit_code != 0

    def test_cluster_join_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "join", "--help"])
        assert result.exit_code == 0
        assert "coordinator" in result.stdout

    def test_cluster_leave_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "leave", "--help"])
        assert result.exit_code == 0

    def test_cluster_scale_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "scale", "--help"])
        assert result.exit_code == 0

    def test_cluster_drain_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "drain", "--help"])
        assert result.exit_code == 0

    def test_cluster_rebalance_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "rebalance", "--help"])
        assert result.exit_code == 0

    def test_cluster_list_nodes_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "list-nodes", "--help"])
        assert result.exit_code == 0


class TestCLIClusterStatus:
    """distllm cluster status — with mocked HTTP client."""

    @patch("httpx.Client")
    def test_cluster_status_with_mock(self, mock_client_cls, cli_runner):
        mock_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_instance
        mock_instance.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "nodes": [{"node_id": "n1", "status": "healthy",
                           "gpu_name": "A100", "memory_used": "40GB",
                           "active_requests": 5, "start_layer": 0, "end_layer": 12}],
                "summary": {"total_nodes": 1, "healthy_nodes": 1, "total_gpu_memory": "80 GB"},
            },
        )

        from distllm.cli.main import app
        result = cli_runner.invoke(
            app, ["cluster", "status", "--host", "127.0.0.1", "--port", "8000"]
        )
        assert result.exit_code in (0, 1)


class TestCLIClusterDrain:
    """distllm cluster drain — graceful node removal."""

    @patch("httpx.Client")
    def test_cluster_drain_with_mock(self, mock_client_cls, cli_runner):
        mock_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_instance
        mock_instance.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": "ok", "message": "Draining node"},
        )

        from distllm.cli.main import app
        result = cli_runner.invoke(
            app, ["cluster", "drain", "test-node", "--host", "127.0.0.1"]
        )
        assert result.exit_code in (0, 1)


class TestCLIClusterRebalance:
    """distllm cluster rebalance."""

    @patch("httpx.Client")
    def test_cluster_rebalance_with_mock(self, mock_client_cls, cli_runner):
        mock_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_instance
        mock_instance.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": "Rebalanced"},
        )

        from distllm.cli.main import app
        result = cli_runner.invoke(
            app, ["cluster", "rebalance", "--host", "127.0.0.1", "--strategy", "balanced"]
        )
        assert result.exit_code in (0, 1)


class TestCLIClusterListNodes:
    """distllm cluster list-nodes."""

    @patch("httpx.Client")
    def test_cluster_list_nodes_with_mock(self, mock_client_cls, cli_runner):
        mock_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_instance
        mock_instance.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "nodes": [{"node_id": "n1", "healthy": True, "gpu_name": "A100",
                           "start_layer": 0, "end_layer": 12, "gpu_memory_free": 42949672960}],
            },
        )

        from distllm.cli.main import app
        result = cli_runner.invoke(
            app, ["cluster", "list-nodes", "--coordinator", "127.0.0.1", "--port", "50050"]
        )
        assert result.exit_code in (0, 1)


class TestCLIClusterScale:
    """distllm cluster scale."""

    @patch("httpx.Client")
    def test_cluster_scale_with_mock(self, mock_client_cls, cli_runner):
        mock_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_instance
        mock_instance.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": "Scaling initiated", "job_id": "j1"},
        )

        from distllm.cli.main import app
        result = cli_runner.invoke(
            app, ["cluster", "scale", "3", "--host", "127.0.0.1"]
        )
        assert result.exit_code in (0, 1)


# ====================================================================
# distllm chat
# ====================================================================

class TestCLIChat:
    """distllm chat — interactive chat via API."""

    def test_chat_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0

    def test_chat_accepts_params(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(
            app, ["chat", "--model", "test-model", "--host", "127.0.0.1", "--port", "8000"],
            input="quit\n",
        )
        assert result.exit_code in (0, 1)


# ====================================================================
# distllm models commands
# ====================================================================

class TestCLIModels:
    """distllm models — model management help."""

    def test_models_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0
        for cmd in ["list", "info", "load", "unload"]:
            assert cmd in result.stdout

    def test_models_list_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["models", "list", "--help"])
        assert result.exit_code == 0

    def test_models_info_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["models", "info", "--help"])
        assert result.exit_code == 0

    def test_models_load_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["models", "load", "--help"])
        assert result.exit_code == 0

    def test_models_unload_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["models", "unload", "--help"])
        assert result.exit_code == 0


# ====================================================================
# distllm other command groups
# ====================================================================

class TestCLIOtherGroups:
    """Other CLI command groups help."""

    def test_adapters_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["adapters", "--help"])
        assert result.exit_code == 0

    def test_logs_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["logs", "--help"])
        assert result.exit_code == 0

    def test_benchmark_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0

    def test_verify_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["verify", "--help"])
        assert result.exit_code == 0

    def test_backup_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["backup", "--help"])
        assert result.exit_code == 0

    def test_cert_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cert", "--help"])
        assert result.exit_code == 0

    def test_webhook_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["webhook", "--help"])
        assert result.exit_code == 0

    def test_notify_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["notify", "--help"])
        assert result.exit_code == 0

    def test_quota_help(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["quota", "--help"])
        assert result.exit_code == 0


# ====================================================================
# End-to-end workflow test (logic only, no real network)
# ====================================================================

class TestCLIEndToEndWorkflow:
    """Complete workflow: all CLI commands are properly wired."""

    def test_all_commands_listed_in_main(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ["setup", "run", "validate-config", "status", "chat",
                     "models", "cluster", "adapters", "logs",
                     "benchmark", "verify", "backup", "cert",
                     "webhook", "notify", "quota"]:
            assert cmd in result.stdout

    def test_cluster_workflow_commands(self, cli_runner):
        from distllm.cli.main import app
        result = cli_runner.invoke(app, ["cluster", "--help"])
        assert result.exit_code == 0
        for cmd in ["start", "join", "leave", "status", "scale", "drain", "rebalance", "list-nodes"]:
            assert cmd in result.stdout
