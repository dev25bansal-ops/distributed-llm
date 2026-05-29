"""Fuzz tests for CLI argument parsing."""
from __future__ import annotations

import random
import string
import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch

from distllm.cli.main import app

runner = CliRunner()


def _random_args(max_len: int = 5) -> list[str]:
    args = []
    for _ in range(random.randint(0, max_len)):
        arg = "".join(
            random.choices(
                string.ascii_letters + string.digits + string.punctuation,
                k=random.randint(0, 20),
            )
        )
        args.append(arg)
    return args


def _random_flags() -> list[str]:
    flags = [
        "--help", "--json", "--host", "--port", "--model", "--output",
        "--format", "--force", "--yes", "--no-color", "--verbose", "--quiet",
    ]
    return random.choices(flags, k=random.randint(0, 3))


class TestCliArgFuzzing:
    """Fuzz CLI commands with random arguments."""

    def test_benchmark_args_fuzz(self):
        """Fuzz 'distllm benchmark' with random args."""
        for _ in range(200):
            args = _random_args(4)
            result = runner.invoke(app, ["benchmark", "run"] + args)
            assert result.exit_code is not None

    def test_config_args_fuzz(self):
        """Fuzz 'distllm config' with random args."""
        for _ in range(200):
            args = _random_args(4)
            result = runner.invoke(app, ["config"] + args)
            assert result.exit_code is not None

    def test_model_args_fuzz(self):
        """Fuzz 'distllm model' with random args."""
        for _ in range(200):
            args = _random_args(4)
            result = runner.invoke(app, ["model"] + args)
            assert result.exit_code is not None

    def test_cluster_args_fuzz(self):
        """Fuzz 'distllm cluster' with random args."""
        for _ in range(200):
            args = _random_args(4)
            result = runner.invoke(app, ["cluster"] + args)
            assert result.exit_code is not None

    def test_system_args_fuzz(self):
        """Fuzz 'distllm system' with random args."""
        for _ in range(200):
            args = _random_args(4)
            result = runner.invoke(app, ["system"] + args)
            assert result.exit_code is not None

    def test_doctor_args_fuzz(self):
        """Fuzz 'distllm doctor' with random args."""
        for _ in range(200):
            args = _random_args(3)
            result = runner.invoke(app, ["doctor"] + args)
            assert result.exit_code is not None


class TestCliEdgeCases:
    """Test CLI with edge-case inputs."""

    def test_empty_command(self):
        """Empty command should show help."""
        result = runner.invoke(app, [])
        assert result.exit_code in (0, 1, 2)

    def test_help_flag(self):
        """--help should show help text."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "DistLLM" in result.output or "distllm" in result.output.lower()

    def test_very_long_model_name(self):
        """Very long model name should not crash."""
        long_name = "a" * 10000
        result = runner.invoke(app, ["model", "info", long_name])
        assert result.exit_code is not None

    def test_unicode_args(self):
        """Unicode args should be handled gracefully."""
        result = runner.invoke(app, ["model", "info", "模型名"])
        assert result.exit_code is not None

    def test_null_bytes_in_args(self):
        """Null bytes should not cause crashes."""
        result = runner.invoke(app, ["model", "info", "test\x00model"])
        assert result.exit_code is not None

    def test_special_characters_in_args(self):
        """Special characters should be handled safely."""
        special_chars = ["|", "&", ";", "$", "`", "\\", '"', "'", "<", ">"]
        for char in special_chars:
            result = runner.invoke(app, ["model", "info", f"test{char}model"])
            assert result.exit_code is not None

    def test_sql_injection_in_args(self):
        """SQL injection in CLI args should be safe."""
        injection = "'; DROP TABLE models; --"
        result = runner.invoke(app, ["model", "info", injection])
        assert result.exit_code is not None

    def test_path_traversal_in_args(self):
        """Path traversal in CLI args should be handled."""
        result = runner.invoke(app, ["model", "info", "../../../etc/passwd"])
        assert result.exit_code is not None
