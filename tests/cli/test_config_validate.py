"""End-to-end tests: ``distllm config validate`` reads and validates config.yaml.

Regression coverage for WAVE2 item 42: the command previously called
``DistLLMSettings.validate_startup()`` with no arguments, so it validated only
built-in defaults and environment variables — a deliberately broken YAML file
silently passed. These tests prove the command now resolves the config path,
parses the YAML, applies the full validation chain, and reports path-named
errors.
"""

import os

import pytest
from typer.testing import CliRunner

from distllm.cli.main import app

runner = CliRunner()

VALID_YAML = """\
model:
  name: test-model
  dtype: float16
coordinator:
  host: 127.0.0.1
  port: 50050
  api_port: 8000
generation:
  temperature: 0.7
"""

# Malformed YAML — parser-level failure.
BROKEN_YAML_SYNTAX = "model:\n  name: [unclosed\n"

# Well-formed YAML with a value that violates a field validator.
BROKEN_YAML_VALUE = VALID_YAML.replace("temperature: 0.7", "temperature: 99.0")


@pytest.fixture(autouse=True)
def _clean_distllm_env(monkeypatch):
    """Strip ambient DISTLLM_* env vars so tests exercise file contents."""
    for key in list(os.environ):
        if key.startswith("DISTLLM_"):
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test from an empty temp dir so repo-root config.yaml
    cannot leak into auto-discovery."""
    monkeypatch.chdir(tmp_path)


class TestConfigValidateReadsFile:
    """The validate command must actually load the resolved config file."""

    def test_broken_yaml_syntax_fails_with_path_named_error(self, tmp_path):
        (tmp_path / "config.yaml").write_text(BROKEN_YAML_SYNTAX)
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 1, f"broken YAML passed validation:\n{result.output}"
        assert "Invalid YAML" in result.output

    def test_invalid_field_value_fails(self, tmp_path):
        (tmp_path / "config.yaml").write_text(BROKEN_YAML_VALUE)
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 1, f"invalid temperature passed:\n{result.output}"
        assert "temperature" in result.output

    def test_valid_config_passes_with_summary(self, tmp_path):
        (tmp_path / "config.yaml").write_text(VALID_YAML)
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0, result.output
        assert "passed" in result.output.lower()
        assert "test-model" in result.output


class TestConfigValidatePathHandling:
    """Explicit --config beats discovery; missing files get friendly errors."""

    def test_missing_explicit_file_friendly_error(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        result = runner.invoke(app, ["config", "validate", "--config", str(missing)])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_explicit_path_used_over_discovered_broken_cwd_file(self, tmp_path):
        (tmp_path / "config.yaml").write_text(BROKEN_YAML_SYNTAX)
        good = tmp_path / "good.yaml"
        good.write_text(VALID_YAML)
        result = runner.invoke(app, ["config", "validate", "--config", str(good)])
        assert result.exit_code == 0, result.output

    def test_no_file_found_validates_defaults_only(self, tmp_path):
        # Empty cwd — nothing to discover. Defaults/env validation still runs.
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0, result.output
        assert "No configuration file found" in result.output


class TestConfigValidatePrecedence:
    """Full resolution chain: env vars beat the YAML file."""

    def test_env_override_breaks_despite_valid_yaml(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(VALID_YAML)
        monkeypatch.setenv("DISTLLM_GENERATION__TEMPERATURE", "42")
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 1, (
            "env override over valid YAML did not fail — "
            "command likely never loaded the YAML layer:\n" + result.output
        )
        assert "temperature" in result.output
