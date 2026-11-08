"""The flat DISTLLM_MODEL form used by every deployment manifest must load.

The Dockerfile, the Helm chart, the kustomize base and the inference deployment
all set ``DISTLLM_MODEL=<id>``. ``model`` is a nested ModelSettings, so
pydantic-settings tried to JSON-parse that string and the server died at
startup with

    pydantic_settings.exceptions.SettingsError:
      error parsing value for field "model" from source "EnvSettingsSource"

The container could therefore never start with its own default environment.

``from_yaml(config_path=None, cli_overrides=None)`` reads the environment
itself, so these tests drive it through the environment and through
``cli_overrides`` rather than passing a data dict as the first argument.
"""

from __future__ import annotations

import os

import pytest

from distllm.config.settings import DistLLMSettings


@pytest.fixture()
def clean_model_env(monkeypatch):
    monkeypatch.delenv("DISTLLM_MODEL", raising=False)
    monkeypatch.delenv("DISTLLM_MODEL__NAME", raising=False)
    return monkeypatch


class TestFlatModelEnv:
    def test_flat_distllm_model_is_read_as_model_name(self, clean_model_env):
        clean_model_env.setenv("DISTLLM_MODEL", "meta-llama/Llama-3.2-1B")
        settings = DistLLMSettings.from_yaml()
        assert settings.model.name == "meta-llama/Llama-3.2-1B"

    def test_nested_form_still_wins(self, clean_model_env):
        clean_model_env.setenv("DISTLLM_MODEL", "flat/Model")
        clean_model_env.setenv("DISTLLM_MODEL__NAME", "nested/Model")
        settings = DistLLMSettings.from_yaml()
        assert settings.model.name == "nested/Model"

    def test_nested_form_alone_works(self, clean_model_env):
        clean_model_env.setenv("DISTLLM_MODEL__NAME", "nested/Only")
        settings = DistLLMSettings.from_yaml()
        assert settings.model.name == "nested/Only"

    def test_environment_is_restored_after_loading(self, clean_model_env):
        """Loading settings must not mutate the process environment."""
        clean_model_env.setenv("DISTLLM_MODEL", "meta-llama/Llama-3.2-1B")
        DistLLMSettings.from_yaml()
        assert os.environ.get("DISTLLM_MODEL") == "meta-llama/Llama-3.2-1B"
        assert "DISTLLM_MODEL__NAME" not in os.environ

    def test_cli_override_beats_the_flat_env(self, clean_model_env):
        clean_model_env.setenv("DISTLLM_MODEL", "flat/Model")
        settings = DistLLMSettings.from_yaml(
            cli_overrides={"model": {"name": "cli/Model"}}
        )
        assert settings.model.name == "cli/Model"
