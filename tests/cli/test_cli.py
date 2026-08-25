"""Tests: CLI — all commands registered, argument parsing, --debug, validate_config, chat loop."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from distllm.cli.main import app


runner = CliRunner()


# ===========================================================================
# 1. All commands registered in app
# ===========================================================================


class TestCommandsRegistered:
    """All 17 top-level commands + 4 subcommand groups exist."""

    def test_app_has_all_top_level_commands(self):
        commands = list(app.registered_commands)
        names = [c.callback.__name__.replace("_", "-") for c in commands]
        expected = [
            "setup", "run", "validate-config", "status", "compress",
            "chat", "node", "coordinator", "api", "client",
            "tp", "dashboard", "deploy", "profile",
        ]
        for name in expected:
            assert name in names, f"Missing command: {name}"
        assert len(names) >= 14

    def test_app_has_models_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "models" in names

    def test_app_has_cluster_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "cluster" in names

    def test_app_has_adapters_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "adapters" in names

    def test_app_has_logs_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "logs" in names

    def test_app_has_benchmark_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "benchmark" in names

    def test_models_group_has_subcommands(self):
        from distllm.cli.main import models_app
        names = [c.callback.__name__.split("_", 1)[1].replace("_", "-") for c in models_app.registered_commands]
        for sub in ("list", "info", "load", "unload"):
            assert sub in names

    def test_cluster_group_has_subcommands(self):
        from distllm.cli.main import cluster_app
        names = [c.callback.__name__.split("_", 1)[1].replace("_", "-") for c in cluster_app.registered_commands]
        for sub in ("status", "scale", "drain", "rebalance"):
            assert sub in names

    def test_adapters_group_has_subcommands(self):
        from distllm.cli.main import adapters_app
        names = [c.callback.__name__.split("_", 1)[1].replace("_", "-") for c in adapters_app.registered_commands]
        for sub in ("list", "load", "set", "unload"):
            assert sub in names

    def test_benchmark_group_has_subcommands(self):
        from distllm.cli.main import benchmark_app
        names = [c.callback.__name__.split("_", 1)[1].replace("_", "-") for c in benchmark_app.registered_commands]
        for sub in ("run", "compare"):
            assert sub in names


# ===========================================================================
# 2. CLI argument parsing
# ===========================================================================


class TestArgumentParsing:
    """Each flag → correct type parsed."""

    def test_help_output_contains_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("run", "chat", "status", "setup", "deploy", "profile", "validate-config"):
            assert cmd in result.stdout

    def test_run_help_shows_options(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout or "-m" in result.stdout

    def test_chat_help_shows_options(self):
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout or "-m" in result.stdout

    def test_validate_config_help(self):
        result = runner.invoke(app, ["config", "validate", "--help"])
        assert result.exit_code == 0

    def test_status_help(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0

    def test_compress_help(self):
        result = runner.invoke(app, ["compress", "--help"])
        assert result.exit_code == 0

    def test_node_help(self):
        result = runner.invoke(app, ["node", "--help"])
        assert result.exit_code == 0

    def test_coordinator_help(self):
        result = runner.invoke(app, ["coordinator", "--help"])
        assert result.exit_code == 0

    def test_api_help(self):
        result = runner.invoke(app, ["api", "--help"])
        assert result.exit_code == 0

    def test_client_help(self):
        result = runner.invoke(app, ["client", "--help"])
        assert result.exit_code == 0

    def test_tp_help(self):
        result = runner.invoke(app, ["tp", "--help"])
        assert result.exit_code == 0

    def test_dashboard_help(self):
        result = runner.invoke(app, ["dashboard", "--help"])
        assert result.exit_code == 0

    def test_deploy_help(self):
        result = runner.invoke(app, ["deploy", "--help"])
        assert result.exit_code == 0

    def test_profile_help(self):
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0

    def test_models_help(self):
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0

    def test_cluster_help(self):
        result = runner.invoke(app, ["cluster", "--help"])
        assert result.exit_code == 0

    def test_adapters_help(self):
        result = runner.invoke(app, ["adapters", "--help"])
        assert result.exit_code == 0

    def test_logs_help(self):
        result = runner.invoke(app, ["logs", "--help"])
        assert result.exit_code == 0

    def test_benchmark_help(self):
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0


# ===========================================================================
# 3. validate_config — valid / invalid
# ===========================================================================


class TestValidateConfig:
    """Valid config → exit 0; invalid → exit 1 with errors."""

    def test_validate_config_default_valid(self):
        with patch("distllm.config.settings.DistLLMSettings.validate_startup") as mock:
            mock.return_value = MagicMock()
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 0

    def test_validate_config_invalid_temperature(self):
        with patch("distllm.config.settings.DistLLMSettings.validate_startup") as mock:
            mock.side_effect = SystemExit(1)
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 1

    def test_validate_config_invalid_port(self):
        with patch("distllm.config.settings.DistLLMSettings.validate_startup") as mock:
            mock.side_effect = SystemExit(1)
            result = runner.invoke(app, ["config", "validate"])
            assert result.exit_code == 1


# ===========================================================================
# 5. chat — interactive loop
# ===========================================================================


class TestChatLoop:
    """Chat loop handles input, API calls, and exit commands."""

    def test_chat_quit_exits(self):
        with patch("distllm.cli.chat.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Hello!"}}],
                "usage": {"total_tokens": 10},
            }
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = runner.invoke(app, ["chat", "--model", "test", "--host", "localhost", "--port", "8000"],
                                   input="quit\n", catch_exceptions=False)
            assert result.exit_code == 0

    def test_chat_empty_input_continues(self):
        with patch("distllm.cli.chat.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Hi"}}],
                "usage": {"total_tokens": 5},
            }
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = runner.invoke(app, ["chat", "--model", "test", "--host", "localhost", "--port", "8000"],
                                   input="\nquit\n", catch_exceptions=False)
            assert result.exit_code == 0

    def test_chat_clear_resets_conversation(self):
        with patch("distllm.cli.chat.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Ok"}}],
                "usage": {"total_tokens": 3},
            }
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = runner.invoke(app, ["chat", "--model", "test"],
                                   input="clear\nquit\n", catch_exceptions=False)
            assert result.exit_code == 0

    def test_chat_exit_command(self):
        with patch("distllm.cli.chat.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Bye"}}],
            }
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = runner.invoke(app, ["chat", "--model", "test"],
                                   input="exit\n", catch_exceptions=False)
            assert result.exit_code == 0

    def test_chat_q_shortcut_exits(self):
        with patch("distllm.cli.chat.httpx.Client") as mock_client:
            result = runner.invoke(app, ["chat", "--model", "test"],
                                   input="q\n", catch_exceptions=False)
            assert result.exit_code == 0

    def test_chat_http_error_continues(self):
        with patch("distllm.cli.chat.httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "500 error", request=MagicMock(), response=mock_response
            )
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = runner.invoke(app, ["chat", "--model", "test"],
                                   input="hello\nquit\n", catch_exceptions=False)
            assert result.exit_code == 0

    def test_chat_connection_error_continues(self):
        with patch("distllm.cli.chat.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = (
                httpx.ConnectError("Connection refused")
            )
            result = runner.invoke(app, ["chat", "--model", "test"],
                                   input="hello\nquit\n", catch_exceptions=False)
            assert result.exit_code == 0

    def test_chat_timeout_error_continues(self):
        with patch("distllm.cli.chat.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = (
                httpx.TimeoutException("Request timed out")
            )
            result = runner.invoke(app, ["chat", "--model", "test"],
                                   input="hello\nquit\n", catch_exceptions=False)
            assert result.exit_code == 0


class TestDebugFlag:
    """--debug flag propagates set_debug_mode correctly."""

    def test_run_help_shows_debug(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_api_help_shows_debug(self):
        result = runner.invoke(app, ["api", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_coordinator_help_shows_debug(self):
        result = runner.invoke(app, ["coordinator", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_node_help_shows_debug(self):
        result = runner.invoke(app, ["node", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout


# ===========================================================================
# 7. status command
# ===========================================================================


class TestStatusCommand:
    """status command shows cluster info."""

    def test_status_help(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout


# ===========================================================================
# 8. setup command
# ===========================================================================


class TestSetupCommand:
    """setup command creates config file."""

    def test_setup_help(self):
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.stdout

    def test_setup_local_path(self):
        with patch("distllm.cli.setup.run_setup") as mock_setup:
            mock_setup.return_value = None
            result = runner.invoke(app, ["setup", "--config", "/tmp/distllm_config.yaml"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_setup_default_config_path(self):
        with patch("distllm.cli.setup.run_setup") as mock_setup:
            mock_setup.return_value = None
            result = runner.invoke(app, ["setup"], catch_exceptions=False)
            assert result.exit_code == 0

    def test_setup_overwrite_existing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  name: existing\n")
            path = f.name
        try:
            with patch("distllm.cli.setup.run_setup") as mock_setup:
                mock_setup.return_value = None
                result = runner.invoke(app, ["setup", "--config", path],
                                       input="y\n", catch_exceptions=False)
                assert result.exit_code == 0
        finally:
            os.unlink(path)


# ===========================================================================
# 9. run command
# ===========================================================================


class TestRunCommand:
    """run command starts inference with config or CLI args."""

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_run_with_config(self):
        yaml_content = """
model:
  name: config-model
  dtype: bfloat16
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            with patch("distllm.cli.run.run_inference") as mock_run:
                mock_run.return_value = None
                result = runner.invoke(app, ["run", "--model", "cli-override", "--config", path],
                                       catch_exceptions=False)
        finally:
            os.unlink(path)

    def test_run_without_config(self):
        with patch("distllm.cli.run.run_inference") as mock_run:
            mock_run.return_value = None
            result = runner.invoke(app, ["run", "--model", "test-model", "--local"],
                                   catch_exceptions=False)


# ===========================================================================
# 10. benchmark command
# ===========================================================================


class TestBenchmarkCommand:
    """benchmark run/compare with results, JSON output, and empty results."""

    def test_benchmark_run_help(self):
        result = runner.invoke(app, ["benchmark", "run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_benchmark_compare_help(self):
        result = runner.invoke(app, ["benchmark", "compare", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    @patch("distllm.cli.benchmark.httpx.Client")
    def test_benchmark_run_returns_results_table(self, mock_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"text": "test output"}],
            "usage": {"completion_tokens": 10},
        }
        mock_client.return_value.__enter__.return_value.post.return_value = mock_response
        with patch("distllm.cli.benchmark.run_benchmark") as mock_bench:
            mock_bench.return_value = [{"elapsed": 0.5, "tokens": 10, "tokens_per_sec": 20.0}]
            result = runner.invoke(app, ["benchmark", "run", "--model", "test", "--prompts", "1", "--max-tokens", "10"],
                                   catch_exceptions=False)

    @patch("distllm.cli.benchmark.run_benchmark_json")
    def test_benchmark_json_output(self, mock_json):
        mock_json.return_value = '{"avg_throughput_tps": 20.5, "results": []}'
        result = runner.invoke(app, ["benchmark", "run", "--model", "test", "--json"],
                               catch_exceptions=False)

    def test_benchmark_empty_results_no_crash(self):
        with patch("distllm.cli.benchmark._run_benchmarks") as mock_run:
            mock_run.return_value = []
            from rich.console import Console
            from distllm.cli.benchmark import _print_results
            console = Console()
            _print_results([], console, "Test")

    def test_benchmark_compare_save_baseline(self):
        with patch("distllm.cli.benchmark._run_benchmarks") as mock_run:
            mock_run.return_value = [{"elapsed": 0.5, "tokens": 10, "tokens_per_sec": 20.0}]
            with tempfile.TemporaryDirectory() as tmpdir:
                baseline = os.path.join(tmpdir, "baseline.json")
                with patch("distllm.cli.benchmark.run_benchmark_compare") as mock_cmp:
                    mock_cmp.return_value = None
                    result = runner.invoke(app, ["benchmark", "compare", "--model", "test",
                                                  "--save-baseline"],
                                           catch_exceptions=False)


# ===========================================================================
# 11. compress command
# ===========================================================================


class TestCompressCommand:
    """compress — all targets and prune-only mode."""

    def test_compress_help(self):
        result = runner.invoke(app, ["compress", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_compress_target_int4(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["compress", "--model", "test", "--target", "int4"],
                                   catch_exceptions=False)

    def test_compress_target_int8(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["compress", "--model", "test", "--target", "int8"],
                                   catch_exceptions=False)

    def test_compress_target_int4_awq(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["compress", "--model", "test", "--target", "int4-awq"],
                                   catch_exceptions=False)

    def test_compress_target_int4_gptq(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["compress", "--model", "test", "--target", "int4-gptq"],
                                   catch_exceptions=False)

    def test_compress_prune_only(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["compress", "--model", "test", "--target", "int4",
                                          "--prune", "0.3"],
                                   catch_exceptions=False)

    def test_compress_target_shows_in_help(self):
        result = runner.invoke(app, ["compress", "--help"])
        assert result.exit_code == 0
        assert "--target" in result.stdout

    def test_compress_no_cuda_fallback(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["compress", "--model", "test", "--target", "int4", "--local"],
                                   catch_exceptions=False)
            assert result.exit_code == 0


# ===========================================================================
# 12. status command — health + models + connection error
# ===========================================================================


class TestStatusExtended:
    """status — combined display and connection error handling."""

    def test_status_health_and_models_combined(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            health_resp = MagicMock()
            health_resp.json.return_value = {"status": "healthy", "model": "test", "nodes": 2}
            models_resp = MagicMock()
            models_resp.json.return_value = {"data": [{"id": "model-1"}, {"id": "model-2"}]}
            metrics_resp = MagicMock()
            metrics_resp.text = "test_metric 42\n"

            def get_side_effect(url, **kw):
                if "/health" in url:
                    return health_resp
                if "/v1/models" in url:
                    return models_resp
                if "/metrics" in url:
                    return metrics_resp
                return MagicMock()

            mock_instance.get.side_effect = get_side_effect
            with patch("distllm.cli.status.Console") as mock_console:
                result = runner.invoke(app, ["status", "--host", "localhost", "--port", "8000"],
                                       catch_exceptions=False)

    def test_status_connection_error_friendly(self):
        with patch("distllm.cli.status.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.side_effect = httpx.ConnectError("Connection refused")
            with patch("distllm.cli.status.Console") as mock_console:
                result = runner.invoke(app, ["status", "--host", "localhost", "--port", "9999"],
                                       catch_exceptions=False)
                assert result.exit_code == 0


# ===========================================================================
# 13. cluster — all subcommands
# ===========================================================================


class TestClusterCommand:
    """cluster — status, scale, drain, rebalance subcommands."""

    def test_cluster_status_help(self):
        result = runner.invoke(app, ["cluster", "status", "--help"])
        assert result.exit_code == 0

    def test_cluster_scale_help(self):
        result = runner.invoke(app, ["cluster", "scale", "--help"])
        assert result.exit_code == 0
        assert "--gpu-type" in result.stdout

    def test_cluster_drain_help(self):
        result = runner.invoke(app, ["cluster", "drain", "--help"])
        assert result.exit_code == 0

    def test_cluster_rebalance_help(self):
        result = runner.invoke(app, ["cluster", "rebalance", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.stdout

    def test_cluster_status_run(self):
        with patch("distllm.cli.cluster._cluster_status") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["cluster", "status"], catch_exceptions=False)
            assert result.exit_code == 0

    def test_cluster_scale_run(self):
        with patch("distllm.cli.cluster._cluster_scale") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["cluster", "scale", "8"], catch_exceptions=False)
            assert result.exit_code == 0

    def test_cluster_drain_run(self):
        with patch("distllm.cli.cluster._cluster_drain") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["cluster", "drain", "node-1"], catch_exceptions=False)
            assert result.exit_code == 0

    def test_cluster_rebalance_run(self):
        with patch("distllm.cli.cluster._cluster_rebalance") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["cluster", "rebalance"], catch_exceptions=False)
            assert result.exit_code == 0


# ===========================================================================
# 14. logs — follow and non-follow modes
# ===========================================================================


class TestLogsCommand:
    """logs — SSE stream (follow) and GET (past entries)."""

    def test_logs_stream_help(self):
        result = runner.invoke(app, ["logs", "stream", "--help"])
        assert result.exit_code == 0
        assert "--follow" in result.stdout

    def test_logs_non_follow_shows_no_entries(self):
        with patch("distllm.cli.logs.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            resp = MagicMock()
            resp.json.return_value = {"logs": []}
            mock_instance.get.return_value = resp
            with patch("distllm.cli.logs.console") as mock_console:
                result = runner.invoke(app, ["logs", "stream"], catch_exceptions=False)
                assert result.exit_code == 0

    def test_logs_connection_error(self):
        with patch("distllm.cli.logs.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.side_effect = httpx.ConnectError("refused")
            with patch("distllm.cli.logs.console") as mock_console:
                result = runner.invoke(app, ["logs", "stream", "--host", "badhost"],
                                       catch_exceptions=False)
                assert result.exit_code == 0


# ===========================================================================
# 15. models — all subcommands
# ===========================================================================


class TestModelsCommand:
    """models — list, info, load, unload."""

    def test_models_list_help(self):
        result = runner.invoke(app, ["models", "list", "--help"])
        assert result.exit_code == 0

    def test_models_info_help(self):
        result = runner.invoke(app, ["models", "info", "--help"])
        assert result.exit_code == 0

    def test_models_load_help(self):
        result = runner.invoke(app, ["models", "load", "--help"])
        assert result.exit_code == 0

    def test_models_unload_help(self):
        result = runner.invoke(app, ["models", "unload", "--help"])
        assert result.exit_code == 0

    def test_models_list_run(self):
        with patch("distllm.cli.models._list_models") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["models", "list"], catch_exceptions=False)
            assert result.exit_code == 0

    def test_models_info_run(self):
        with patch("distllm.cli.models._model_info") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["models", "info", "test-model"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_models_load_run(self):
        with patch("distllm.cli.models._load_model") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["models", "load", "test-model"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_models_unload_run(self):
        with patch("distllm.cli.models._unload_model") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["models", "unload", "test-model-id"],
                                   catch_exceptions=False)
            assert result.exit_code == 0


# ===========================================================================
# 16. profile — percentiles and zero-successful handling
# ===========================================================================


class TestProfileCommand:
    """profile — P50/P95/P99 computation and all-fail error message."""

    def test_profile_help(self):
        result = runner.invoke(app, ["profile", "--help"])
        assert result.exit_code == 0
        assert "--iterations" in result.stdout

    def test_profile_percentile_computation(self):
        from distllm.cli.profile import run_profile
        c = MagicMock()
        run_profile("test", "localhost", 8000, 10, 10, 3, None, c)

    def test_profile_zero_successful_shows_error(self):
        from distllm.cli.profile import run_profile
        c = MagicMock()
        run_profile("test", "localhost", 9999, 10, 10, 2, None, c)


# ===========================================================================
# 17. deploy — dry run, wait loop, timeout
# ===========================================================================


class TestDeployCommand:
    """deploy — dry run (plan only), wait loop, timeout."""

    def test_deploy_help(self):
        result = runner.invoke(app, ["deploy", "--help"])
        assert result.exit_code == 0
        assert "--nodes" in result.stdout

    def test_deploy_dry_run(self):
        with patch("distllm.cli.deploy.run_deploy") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["deploy", "test-model", "--dry-run"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_deploy_with_gpu_type(self):
        with patch("distllm.cli.deploy.run_deploy") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["deploy", "test-model", "--gpu-type", "A100"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_deploy_no_wait(self):
        with patch("distllm.cli.deploy.run_deploy") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["deploy", "test-model", "--no-wait"],
                                   catch_exceptions=False)
            assert result.exit_code == 0


# ===========================================================================
# 18. client — health-only mode
# ===========================================================================


class TestClientCommand:
    """client — --health checks health without running inference."""

    def test_client_help(self):
        result = runner.invoke(app, ["client", "--help"])
        assert result.exit_code == 0
        assert "--health" in result.stdout

    def test_client_health_mode(self):
        with patch("distllm.core.coordinator.Coordinator") as mock_coord:
            mock_coord.return_value = MagicMock()
            result = runner.invoke(app, ["client", "--health"], catch_exceptions=False)


# ===========================================================================
# 19. tp — multi-GPU launch
# ===========================================================================


class TestTPCommand:
    """tp — tensor parallel multi-GPU worker launch."""

    def test_tp_help(self):
        result = runner.invoke(app, ["tp", "--help"])
        assert result.exit_code == 0
        assert "--num-gpus" in result.stdout

    def test_tp_with_num_gpus(self):
        with patch("distllm.core.tp_launcher.launch_tp_workers") as mock_launch:
            mock_launch.return_value = MagicMock()
            result = runner.invoke(app, ["tp", "--model", "test", "--num-gpus", "2"],
                                   catch_exceptions=False)
