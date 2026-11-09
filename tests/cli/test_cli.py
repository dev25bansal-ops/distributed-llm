"""Tests: CLI — command surface, help output, argument parsing, and the
behaviours that can be exercised without a live coordinator.

Rewritten 2026-11-08 against the real ``distllm.cli.main.app``. The previous
version asserted a surface from an earlier generation of the project
(``setup``, ``run``, ``validate-config``, ``compress``, ``node``, ``coordinator``,
``api``, ``client``, ``tp``, ``deploy``, ``profile``, ``models``, ``adapters``,
``logs``) — 54 of its 98 tests failed on a clean tree because those commands
no longer exist.

The command list below is what ``distllm --help`` actually prints; the
subcommand lists come from each group's own ``--help``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from distllm.cli.main import app

runner = CliRunner()

# Verified against `python -m distllm.cli.main --help`, 2026-11-08.
TOP_LEVEL_COMMANDS = {
    "completion",
    "status",
    "cli-doctor",
    "chat",
    "devices",
    "chat-stream",
    "dashboard",
    "tutorial",
    "onboard",
    "audit",
}

# Verified against `python -m distllm.cli.main <group> --help`.
GROUPS = {
    "cluster": {"status", "scale", "drain", "rebalance", "start", "join", "leave", "list-nodes", "deploy", "autopsy"},
    "model": {"list", "info", "load", "unload", "compress", "adapters", "prompt", "federate"},
    "benchmark": {"run", "compare", "live", "eval", "profile", "verify"},
    "config": {"setup", "validate", "reference", "openapi", "webhook", "quota", "backup"},
    "security": {"cert"},
    "system": {
        "schedule-viz", "run", "coordinator", "api", "slo-report", "cost-avoid",
        "doctor", "observe", "logs", "notify", "defrag",
    },
    "pool": {"add", "list", "remove"},
    "cache": {"inspect"},
    "daas": {"serve", "status", "benchmark"},
    "draft": {"fleet-status", "migration-status"},
    "tune": {"quantize", "batch", "cache"},
}


# ===========================================================================
# 1. Command surface
# ===========================================================================


class TestCommandsRegistered:
    """Every top-level command and group registered in the app exists."""

    @pytest.mark.parametrize("name", sorted(TOP_LEVEL_COMMANDS))
    def test_top_level_command_registered(self, name):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert name in result.output, f"command {name} missing from --help"

    @pytest.mark.parametrize("group", sorted(GROUPS))
    def test_group_registered(self, group):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert group in result.output, f"group {group} missing from --help"

    @pytest.mark.parametrize("group,subcommands", sorted(GROUPS.items()))
    def test_group_lists_its_subcommands(self, group, subcommands):
        result = runner.invoke(app, [group, "--help"])
        assert result.exit_code == 0, f"{group} --help exited {result.exit_code}"
        for sub in sorted(subcommands):
            assert sub in result.output, f"{group} {sub} missing from help"


# ===========================================================================
# 2. Help output
# ===========================================================================


class TestHelpOutput:
    def test_top_level_help_succeeds(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Distributed LLM" in result.output

    @pytest.mark.parametrize("group", sorted(GROUPS))
    def test_group_help_succeeds(self, group):
        result = runner.invoke(app, [group, "--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        "command",
        ["status", "cli-doctor", "devices", "completion", "tutorial", "dashboard", "audit"],
    )
    def test_leaf_command_help_succeeds(self, command):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"{command} --help exited {result.exit_code}"

    def test_router_group_help_is_honest_about_missing_subcommands(self):
        """router advertises no subcommands; its help must not promise any."""
        import os

        old = os.environ.get("COLUMNS")
        os.environ["COLUMNS"] = "200"
        try:
            result = runner.invoke(app, ["router", "--help"])
        finally:
            if old is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = old
        assert result.exit_code == 0
        assert "not implemented" in result.output


# ===========================================================================
# 3. Argument parsing
# ===========================================================================


class TestArgumentParsing:
    def test_unknown_command_is_rejected(self):
        result = runner.invoke(app, ["definitely-not-a-command"])
        assert result.exit_code != 0

    def test_status_accepts_host_and_port_options(self):
        result = runner.invoke(app, ["status", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--json" in result.output

    def test_cluster_start_requires_a_command(self):
        """A group invoked bare shows help/usage rather than crashing."""
        result = runner.invoke(app, ["cluster"])
        assert "Usage" in result.output or result.exit_code != 0

    def test_completion_requires_shell_argument(self):
        result = runner.invoke(app, ["completion", "--help"])
        assert result.exit_code == 0
        assert "shell" in result.output.lower()


# ===========================================================================
# 4. Behaviours reachable offline
# ===========================================================================


class TestOfflineBehaviours:
    def test_cli_doctor_runs_and_reports(self):
        """cli-doctor inspects the local environment and must not need a server."""
        result = runner.invoke(app, ["cli-doctor"])
        assert result.exit_code == 0, f"cli-doctor exited {result.exit_code}: {result.output[-500:]}"
        assert "Python" in result.output or "python" in result.output

    def test_devices_reports_topology(self):
        result = runner.invoke(app, ["devices"])
        assert result.exit_code == 0, f"devices exited {result.exit_code}: {result.output[-500:]}"

    def test_status_fails_cleanly_without_a_coordinator(self):
        """status against a dead endpoint must report, not raise."""
        result = runner.invoke(app, ["status", "--host", "127.0.0.1", "--port", "1"])
        assert result.exit_code in (0, 1)
        assert "Traceback" not in result.output