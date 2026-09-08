"""Tests: ``distllm pool`` — stub for the week-5 GPU-pool product.

The product lands in week 5; until then we verify the CLI surface is
stable (commands are registered, options are wired) and that each
subcommand raises ``NotImplementedError`` (translated to a friendly
exit-1 in the CLI).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from distllm.cli.main import app
from distllm.cli.pool import SUPPORTED_PROVIDERS, _not_implemented, register


runner = CliRunner()


class TestPoolRegistration:
    def test_pool_group_registered_on_app(self):
        groups = [g.name for g in app.registered_groups]
        assert "pool" in groups

    def test_pool_register_function_attaches_to_app(self):
        import typer

        fresh_app = typer.Typer()
        register(fresh_app)
        names = [g.name for g in fresh_app.registered_groups]
        assert "pool" in names

    def test_supported_providers_includes_hetzner(self):
        assert "hetzner" in SUPPORTED_PROVIDERS


class TestPoolAdd:
    def test_help(self):
        result = runner.invoke(app, ["pool", "add", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.stdout
        assert "--api-key" in result.stdout

    def test_unsupported_provider_exits_2(self):
        result = runner.invoke(
            app,
            ["pool", "add", "--provider", "imaginary-cloud", "--api-key", "x"],
        )
        assert result.exit_code == 2

    def test_supported_provider_exits_1_with_stub_message(self):
        result = runner.invoke(
            app,
            ["pool", "add", "--provider", "hetzner", "--api-key", "hk_test_123"],
        )
        # Exit code 1 from the stub
        assert result.exit_code == 1


class TestPoolList:
    def test_help(self):
        result = runner.invoke(app, ["pool", "list", "--help"])
        assert result.exit_code == 0

    def test_runs_as_stub(self):
        result = runner.invoke(app, ["pool", "list"])
        assert result.exit_code == 1

    def test_runs_json(self):
        result = runner.invoke(app, ["pool", "list", "--json"])
        assert result.exit_code == 1
        # Stub emits {"pools": [], "stub": true}
        assert "stub" in result.stdout


class TestPoolRemove:
    def test_help(self):
        result = runner.invoke(app, ["pool", "remove", "--help"])
        assert result.exit_code == 0

    def test_runs_as_stub(self):
        result = runner.invoke(app, ["pool", "remove", "pool-abc"])
        assert result.exit_code == 1


class TestNotImplementedHelper:
    def test_raises(self):
        with pytest.raises(NotImplementedError) as exc:
            _not_implemented("distllm pool add")
        assert "week-5" in str(exc.value)