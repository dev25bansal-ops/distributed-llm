"""Tests: CLI — all commands registered, argument parsing, --debug, validate_config, chat loop.

Pins the current group-based command surface of ``distllm.cli.main:app``:

* Top-level commands: chat, chat-stream, dashboard, status, devices,
  completion, cli-doctor, tutorial, onboard, audit.
* Command groups: cluster, model, benchmark, config, security, system,
  router, pool, cache, daas, draft, tune.
* Flat commands from earlier layouts moved into groups:
  ``setup`` → ``config setup``, ``run`` → ``system run``,
  ``validate-config`` → ``config validate``, ``compress`` → ``model compress``,
  ``coordinator``/``api`` → ``system coordinator``/``system api``,
  ``models`` → ``model``, ``adapters`` → ``model adapters``,
  ``logs`` → ``system logs``, ``deploy`` → ``cluster deploy``,
  ``profile`` → ``benchmark profile``, ``node`` → ``cluster join``.
"""

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
    """Top-level commands + subcommand groups exist."""

    def test_app_has_all_top_level_commands(self):
        commands = list(app.registered_commands)
        names = [c.name or c.callback.__name__.replace("_", "-") for c in commands]
        expected = [
            "chat", "chat-stream", "dashboard", "status", "devices",
            "completion", "cli-doctor", "tutorial", "onboard", "audit",
        ]
        for name in expected:
            assert name in names, f"Missing command: {name}"
        assert len(names) >= 10

    def test_app_has_model_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "model" in names

    def test_app_has_cluster_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "cluster" in names

    def test_app_has_benchmark_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "benchmark" in names

    def test_app_has_system_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "system" in names

    def test_app_has_config_group(self):
        groups = [g for g in app.registered_groups]
        names = [g.name for g in groups]
        assert "config" in names

    def test_model_group_has_subcommands(self):
        from distllm.cli.main import model_app
        names = [c.name or c.callback.__name__.split("_", 1)[-1] for c in model_app.registered_commands]
        for sub in ("list", "info", "load", "unload", "compress"):
            assert sub in names

    def test_cluster_group_has_subcommands(self):
        from distllm.cli.main import cluster_app
        names = [c.name or c.callback.__name__.split("_", 1)[-1] for c in cluster_app.registered_commands]
        for sub in ("status", "scale", "drain", "rebalance", "deploy", "join"):
            assert sub in names

    def test_adapters_group_has_subcommands(self):
        from distllm.cli.main import model_adapters_app
        names = [c.name or c.callback.__name__.split("_", 1)[-1] for c in model_adapters_app.registered_commands]
        for sub in ("list", "load", "set", "unload"):
            assert sub in names

    def test_benchmark_group_has_subcommands(self):
        from distllm.cli.main import benchmark_app
        names = [c.name or c.callback.__name__.split("_", 1)[-1] for c in benchmark_app.registered_commands]
        for sub in ("run", "compare", "profile"):
            assert sub in names


# ===========================================================================
# 2. CLI argument parsing
# ===========================================================================


class TestArgumentParsing:
    """Each command group parses and shows --help."""

    def test_help_output_contains_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("chat", "status", "cluster", "model", "benchmark", "config", "system"):
            assert cmd in result.stdout

    def test_run_help_shows_options(self):
        result = runner.invoke(app, ["system", "run", "--help"])
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
        result = runner.invoke(app, ["model", "compress", "--help"])
        assert result.exit_code == 0

    def test_node_help(self):
        result = runner.invoke(app, ["cluster", "join", "--help"])
        assert result.exit_code == 0

    def test_coordinator_help(self):
        result = runner.invoke(app, ["system", "coordinator", "--help"])
        assert result.exit_code == 0

    def test_api_help(self):
        result = runner.invoke(app, ["system", "api", "--help"])
        assert result.exit_code == 0

    def test_dashboard_help(self):
        result = runner.invoke(app, ["dashboard", "--help"])
        assert result.exit_code == 0

    def test_deploy_help(self):
        result = runner.invoke(app, ["cluster", "deploy", "--help"])
        assert result.exit_code == 0

    def test_profile_help(self):
        result = runner.invoke(app, ["benchmark", "profile", "--help"])
        assert result.exit_code == 0

    def test_models_help(self):
        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0

    def test_cluster_help(self):
        result = runner.invoke(app, ["cluster", "--help"])
        assert result.exit_code == 0

    def test_adapters_help(self):
        result = runner.invoke(app, ["model", "adapters", "--help"])
        assert result.exit_code == 0

    def test_logs_help(self):
        result = runner.invoke(app, ["system", "logs", "--help"])
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
    """--debug flag shown on the commands that accept it."""

    def test_run_help_shows_debug(self):
        result = runner.invoke(app, ["system", "run", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_api_help_shows_debug(self):
        result = runner.invoke(app, ["system", "api", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_coordinator_help_shows_debug(self):
        result = runner.invoke(app, ["system", "coordinator", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout

    def test_node_help_shows_debug(self):
        # The old flat `node` command's surface (--node-id, layer bounds,
        # --device) moved to `cluster join`.
        result = runner.invoke(app, ["cluster", "join", "--help"])
        assert result.exit_code == 0
        for flag in ("--node-id", "--start-layer", "--end-layer", "--total-layers", "--device"):
            assert flag in result.stdout


# ===========================================================================
# 7. status command
# ===========================================================================


class TestStatusCommand:
    """status command shows one-line cluster summary."""

    def test_status_help(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout


# ===========================================================================
# 8. setup command (config setup)
# ===========================================================================


class TestSetupCommand:
    """config setup command creates config file."""

    def test_setup_help(self):
        result = runner.invoke(app, ["config", "setup", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.stdout

    def test_setup_local_path(self):
        with patch("distllm.cli.setup.run_setup") as mock_setup:
            mock_setup.return_value = None
            result = runner.invoke(app, ["config", "setup", "--config", "/tmp/distllm_config.yaml"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_setup_default_config_path(self):
        with patch("distllm.cli.setup.run_setup") as mock_setup:
            mock_setup.return_value = None
            result = runner.invoke(app, ["config", "setup"], catch_exceptions=False)
            assert result.exit_code == 0

    def test_setup_overwrite_existing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  name: existing\n")
            path = f.name
        try:
            with patch("distllm.cli.setup.run_setup") as mock_setup:
                mock_setup.return_value = None
                result = runner.invoke(app, ["config", "setup", "--config", path],
                                       input="y\n", catch_exceptions=False)
                assert result.exit_code == 0
        finally:
            os.unlink(path)


# ===========================================================================
# 9. run command (system run)
# ===========================================================================


class TestRunCommand:
    """system run command starts inference with config or CLI args."""

    def test_run_help(self):
        result = runner.invoke(app, ["system", "run", "--help"])
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
                result = runner.invoke(app, ["system", "run", "--model", "cli-override", "--config", path],
                                       catch_exceptions=False)
        finally:
            os.unlink(path)

    def test_run_without_config(self):
        with patch("distllm.cli.run.run_inference") as mock_run:
            mock_run.return_value = None
            result = runner.invoke(app, ["system", "run", "--model", "test-model", "--local"],
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
# 11. compress command (model compress)
# ===========================================================================


class TestCompressCommand:
    """model compress — all targets and prune-only mode."""

    def test_compress_help(self):
        result = runner.invoke(app, ["model", "compress", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_compress_target_int4(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["model", "compress", "--model", "test", "--target", "int4"],
                                   catch_exceptions=False)

    def test_compress_target_int8(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["model", "compress", "--model", "test", "--target", "int8"],
                                   catch_exceptions=False)

    def test_compress_target_int4_awq(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["model", "compress", "--model", "test", "--target", "int4-awq"],
                                   catch_exceptions=False)

    def test_compress_target_int4_gptq(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["model", "compress", "--model", "test", "--target", "int4-gptq"],
                                   catch_exceptions=False)

    def test_compress_prune_only(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["model", "compress", "--model", "test", "--target", "int4",
                                          "--prune", "0.3"],
                                   catch_exceptions=False)

    def test_compress_target_shows_in_help(self):
        result = runner.invoke(app, ["model", "compress", "--help"])
        assert result.exit_code == 0
        assert "--target" in result.stdout

    def test_compress_no_cuda_fallback(self):
        with patch("distllm.cli.compress.run_compress") as mock_compress:
            mock_compress.return_value = None
            result = runner.invoke(app, ["model", "compress", "--model", "test", "--target", "int4", "--local"],
                                   catch_exceptions=False)
            assert result.exit_code == 0


# ===========================================================================
# 12. status command — health + models + connection error
# ===========================================================================


class TestStatusExtended:
    """status — one-line summary via fetch_one_line_summary and connection error handling."""

    def test_status_health_and_models_combined(self):
        payload = {
            "status": "ok",
            "coordinator": "localhost:8000",
            "nodes": 2,
            "healthy": 2,
            "models": 2,
            "load": 0.42,
            "line": "OK    coordinator=localhost:8000  nodes=2  healthy=2  models=2  load=42%",
        }
        with patch("distllm.cli.status.fetch_one_line_summary") as mock_fetch:
            import json
            mock_fetch.return_value = json.dumps(payload)
            result = runner.invoke(app, ["status", "--host", "localhost", "--port", "8000"],
                                   catch_exceptions=False)
            assert result.exit_code == 0
            assert "nodes=2" in result.stdout

    def test_status_connection_error_friendly(self):
        # Documented semantics: unreachable cluster → exit 1 with a DOWN line.
        with patch("distllm.cli.status.fetch_one_line_summary") as mock_fetch:
            import json
            mock_fetch.return_value = json.dumps({
                "status": "unreachable",
                "coordinator": "localhost:9999",
                "nodes": 0,
                "models": 0,
                "load": 0.0,
                "line": "DOWN  coordinator=localhost:9999  (connection refused)",
            })
            result = runner.invoke(app, ["status", "--host", "localhost", "--port", "9999"])
            assert result.exit_code == 1
            assert "DOWN" in result.stdout


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
# 14. logs — follow and non-follow modes (system logs stream)
# ===========================================================================


class TestLogsCommand:
    """system logs stream — SSE stream (follow) and GET (past entries)."""

    def test_logs_stream_help(self):
        result = runner.invoke(app, ["system", "logs", "stream", "--help"])
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
                result = runner.invoke(app, ["system", "logs", "stream"], catch_exceptions=False)
                assert result.exit_code == 0

    def test_logs_connection_error(self):
        with patch("distllm.cli.logs.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.side_effect = httpx.ConnectError("refused")
            with patch("distllm.cli.logs.console") as mock_console:
                result = runner.invoke(app, ["system", "logs", "stream", "--host", "badhost"],
                                       catch_exceptions=False)
                assert result.exit_code == 0


# ===========================================================================
# 15. models — all subcommands (model group)
# ===========================================================================


class TestModelsCommand:
    """model — list, info, load, unload."""

    def test_models_list_help(self):
        result = runner.invoke(app, ["model", "list", "--help"])
        assert result.exit_code == 0

    def test_models_info_help(self):
        result = runner.invoke(app, ["model", "info", "--help"])
        assert result.exit_code == 0

    def test_models_load_help(self):
        result = runner.invoke(app, ["model", "load", "--help"])
        assert result.exit_code == 0

    def test_models_unload_help(self):
        result = runner.invoke(app, ["model", "unload", "--help"])
        assert result.exit_code == 0

    def test_models_list_run(self):
        with patch("distllm.cli.models._list_models") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["model", "list"], catch_exceptions=False)
            assert result.exit_code == 0

    def test_models_info_run(self):
        with patch("distllm.cli.models._model_info") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["model", "info", "test-model"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_models_load_run(self):
        with patch("distllm.cli.models._load_model") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["model", "load", "test-model"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_models_unload_run(self):
        with patch("distllm.cli.models._unload_model") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["model", "unload", "test-model-id"],
                                   catch_exceptions=False)
            assert result.exit_code == 0


# ===========================================================================
# 16. profile — percentiles and zero-successful handling (benchmark profile)
# ===========================================================================


class TestProfileCommand:
    """benchmark profile — P50/P95/P99 computation and all-fail error message."""

    def test_profile_help(self):
        result = runner.invoke(app, ["benchmark", "profile", "--help"])
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
# 17. deploy — dry run, wait loop, timeout (cluster deploy)
# ===========================================================================


class TestDeployCommand:
    """cluster deploy — dry run (plan only), wait loop, timeout."""

    def test_deploy_help(self):
        result = runner.invoke(app, ["cluster", "deploy", "--help"])
        assert result.exit_code == 0
        assert "--nodes" in result.stdout

    def test_deploy_dry_run(self):
        with patch("distllm.cli.deploy.run_deploy") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["cluster", "deploy", "test-model", "--dry-run"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_deploy_with_gpu_type(self):
        with patch("distllm.cli.deploy.run_deploy") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["cluster", "deploy", "test-model", "--gpu-type", "A100"],
                                   catch_exceptions=False)
            assert result.exit_code == 0

    def test_deploy_no_wait(self):
        with patch("distllm.cli.deploy.run_deploy") as mock:
            mock.return_value = None
            result = runner.invoke(app, ["cluster", "deploy", "test-model", "--no-wait"],
                                   catch_exceptions=False)
            assert result.exit_code == 0


# ===========================================================================
# 18. client — unified HTTP client library
# ===========================================================================


class TestClientCommand:
    """distllm.cli.client.DistLLMClient — the unified HTTP client (no CLI command)."""

    def test_client_help(self):
        # The old flat `client` command had a --health mode; the health
        # surface now lives on the client library and `status`.
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.stdout

    def test_client_exposes_health_helpers(self):
        from distllm.cli.client import DistLLMClient
        client = DistLLMClient(base_url="http://localhost:8000")
        assert hasattr(client, "get")
        assert hasattr(client, "post")

    def test_client_health_mode(self):
        # Old `client --health` ran a Coordinator health check; the same
        # capability flows through Coordinator.health_check().
        with patch("distllm.core.coordinator.Coordinator") as mock_coord:
            coord = mock_coord.return_value
            coord.health_check.return_value = {"node-0": {"healthy": True}}
            health = coord.health_check()
            assert health["node-0"]["healthy"] is True


# ===========================================================================
# 19. tp — multi-GPU launch (distllm.dist.tp_launcher)
# ===========================================================================


class TestTPCommand:
    """Tensor-parallel multi-GPU worker launch via distllm.dist.tp_launcher.

    The flat `tp` CLI command was removed; the launcher is used by the
    `distllm-node` entry point and `python -m distllm.dist.tp_launcher`.
    """

    def test_tp_launcher_spawns_workers(self):
        from distllm.dist import tp_launcher
        assert callable(tp_launcher.launch_tp_workers)
        assert hasattr(tp_launcher, "main")

    def test_tp_with_num_gpus(self):
        with patch("distllm.dist.tp_launcher.launch_tp_workers") as mock_launch:
            mock_launch.return_value = MagicMock()
            from distllm.dist import tp_launcher
            tp_launcher.launch_tp_workers(
                model_name="test", num_gpus=2, dtype="float16",
                trust_remote_code=False, port=29500,
            )
            mock_launch.assert_called_once_with(
                model_name="test", num_gpus=2, dtype="float16",
                trust_remote_code=False, port=29500,
            )
