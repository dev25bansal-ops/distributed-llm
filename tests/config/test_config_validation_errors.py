"""Tests: config validation errors must identify the offending config file.

Covers the Focus Area 8 item "Config validation errors with file pointers":
- Malformed YAML in ``DistLLMSettings.from_yaml`` raises a ``ValueError``
  whose message contains the config file path.
- Pydantic ``ValidationError`` keeps its type (callers depend on it) but
  carries the path as an exception note on Python 3.11+.
- ``validate_startup`` prints the config path when validation fails.
"""

from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from distllm.config.settings import DistLLMSettings


BAD_PORT_YAML = "coordinator:\n  port: 99999\n"  # port validator rejects >65535
MALFORMED_YAML = "model:\n  name: [unclosed\n"


class TestFromYamlErrorPaths:
    def test_malformed_yaml_error_mentions_path(self, tmp_path):
        cfg = tmp_path / "broken.yaml"
        cfg.write_text(MALFORMED_YAML, encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            DistLLMSettings.from_yaml(config_path=str(cfg))

        message = str(excinfo.value)
        assert str(cfg) in message, (
            f"Expected config path {cfg} in error message, got: {message!r}"
        )
        assert "Invalid YAML" in message

    def test_yaml_error_chains_original_exception(self, tmp_path):
        cfg = tmp_path / "broken.yaml"
        cfg.write_text(MALFORMED_YAML, encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            DistLLMSettings.from_yaml(config_path=str(cfg))

        assert excinfo.value.__cause__ is not None

    def test_validation_error_type_preserved_with_path_note(self, tmp_path):
        """Invalid *values* keep raising ValidationError (callers catch it),
        with the file path attached as an exception note on 3.11+."""
        import yaml  # noqa: F401  (ensure available for from_yaml)

        cfg = tmp_path / "bad-values.yaml"
        cfg.write_text(BAD_PORT_YAML, encoding="utf-8")

        with pytest.raises(ValidationError) as excinfo:
            DistLLMSettings.from_yaml(config_path=str(cfg))

        if sys.version_info >= (3, 11):
            notes = getattr(excinfo.value, "__notes__", [])
            assert any(str(cfg) in n for n in notes), (
                f"Expected path note in {notes!r}"
            )

    def test_missing_file_does_not_raise_yaml_error(self, tmp_path):
        """A nonexistent path is skipped silently (documented behavior)."""
        settings = DistLLMSettings.from_yaml(
            config_path=str(tmp_path / "nope.yaml")
        )
        assert isinstance(settings, DistLLMSettings)


class TestValidateStartupErrorPaths:
    def test_failure_banner_names_config_file(self, tmp_path, capsys):
        cfg = tmp_path / "cluster.yaml"
        cfg.write_text(BAD_PORT_YAML, encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            DistLLMSettings.validate_startup(config_path=str(cfg))

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "Config validation failed" in out
        assert str(cfg) in out

    def test_malformed_yaml_banner_names_config_file(self, tmp_path, capsys):
        cfg = tmp_path / "syntax-error.yaml"
        cfg.write_text(MALFORMED_YAML, encoding="utf-8")

        with pytest.raises(SystemExit):
            DistLLMSettings.validate_startup(config_path=str(cfg))

        out = capsys.readouterr().out
        assert "Config validation failed" in out
        assert str(cfg) in out
