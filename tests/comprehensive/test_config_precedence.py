"""Config precedence tests.

Validates CLI > env > YAML > defaults precedence chain for DistLLMSettings.
"""

import asyncio
import socket
import struct
import threading
import time
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np

try:
    from hypothesis import given, strategies as st, settings as hp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


from tests.comprehensive.conftest import _load_module

# Load clean modules
_config_settings = _load_module("distllm/config/settings.py")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Config Precedence (CLI > env > YAML > defaults)
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigPrecedence:
    """Validates CLI > env > YAML > defaults precedence chain."""

    def test_defaults_used_when_nothing_provided(self):
        settings = _config_settings.DistLLMSettings()
        assert settings.model.name == ""
        assert settings.generation.max_new_tokens == 256

    def test_apply_cli_overrides_flat(self):
        data = {"model": {"name": "base", "dtype": "float16"}}
        overrides = {"model": {"name": "cli-override"}}
        result = _config_settings.DistLLMSettings._apply_cli_overrides(data, overrides)
        assert result["model"]["name"] == "cli-override"
        assert result["model"]["dtype"] == "float16"

    def test_apply_cli_overrides_new_key(self):
        data = {"model": {"name": "base"}}
        overrides = {"model": {"unknown_key": "new-value"}}
        result = _config_settings.DistLLMSettings._apply_cli_overrides(data, overrides)
        assert result["model"]["unknown_key"] == "new-value"

    def test_apply_cli_overrides_non_dict_value(self):
        data = {"generation": {"max_tokens": 512}}
        overrides = {"generation": {"max_tokens": 2048}}
        result = _config_settings.DistLLMSettings._apply_cli_overrides(data, overrides)
        assert result["generation"]["max_tokens"] == 2048

    def test_precedence_cli_overrides_yaml(self, tmp_path):
        yml = tmp_path / "config.yaml"
        yml.write_text("model:\n  name: yaml-model\n  dtype: float16\n")
        settings = _config_settings.DistLLMSettings.from_yaml(
            config_path=str(yml),
            cli_overrides={"model": {"name": "cli-model"}},
        )
        assert settings.model.name == "cli-model"
        assert settings.model.dtype == "float16"

    def test_yaml_overrides_defaults(self, tmp_path):
        yml = tmp_path / "config.yaml"
        yml.write_text("generation:\n  max_new_tokens: 999\n")
        settings = _config_settings.DistLLMSettings.from_yaml(config_path=str(yml))
        assert settings.generation.max_new_tokens == 999

    def test_yaml_missing_file_uses_defaults(self):
        settings = _config_settings.DistLLMSettings.from_yaml(
            config_path="/nonexistent/path/config.yaml"
        )
        assert settings.generation.max_new_tokens == 256

    def test_apply_cli_overrides_nested_dict(self):
        data = {"auto_partition": {"enabled": False, "strategy": "auto"}}
        overrides = {"auto_partition": {"enabled": True}}
        result = _config_settings.DistLLMSettings._apply_cli_overrides(data, overrides)
        assert result["auto_partition"]["enabled"] is True
        assert result["auto_partition"]["strategy"] == "auto"

    def test_apply_cli_overrides_replaces_non_dict_with_dict(self):
        data = {"some_field": "old_value"}
        overrides = {"some_field": {"new": "dict"}}
        result = _config_settings.DistLLMSettings._apply_cli_overrides(data, overrides)
        assert result["some_field"] == {"new": "dict"}

    def test_cli_overrides_create_new_section(self):
        data: dict = {}
        overrides = {"new_section": {"key": "val"}}
        result = _config_settings.DistLLMSettings._apply_cli_overrides(data, overrides)
        assert result["new_section"] == {"key": "val"}

    def test_setting_validation_invalid_str(self):
        with pytest.raises(SystemExit):
            _config_settings.DistLLMSettings.validate_startup(
                config_path=None,
                cli_overrides={"model": {"name": ""}},
            )

    def test_settings_model_validation(self, tmp_path):
        yml = tmp_path / "bad_config.yaml"
        yml.write_text("generation:\n  temperature: invalid\n")
        with pytest.raises((SystemExit, Exception)):
            _config_settings.DistLLMSettings.validate_startup(
                config_path=str(yml)
            )

    @pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
    @hp_settings(max_examples=100)
    @given(
        cli_name=st.text(min_size=0, max_size=20).filter(lambda x: x != "bad_key"),
    )
    def test_precedence_invariant(self, cli_name):
        """CLI override always wins over YAML, regardless of value."""
        from pydantic import ValidationError
        data = {"model": {"name": "yaml-value"}}
        if cli_name:
            override = {"model": {"name": cli_name}}
            result = _config_settings.DistLLMSettings._apply_cli_overrides(
                data, override
            )
            assert result["model"]["name"] == cli_name
        else:
            result = _config_settings.DistLLMSettings._apply_cli_overrides(
                data, {}
            )
            assert result["model"]["name"] == "yaml-value"
