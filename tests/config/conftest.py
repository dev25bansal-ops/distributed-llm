"""Pytest fixtures for the config test suite."""
from __future__ import annotations
from typing import Any
import pytest
from distllm.config.settings import DistLLMSettings
from distllm.config._model import ModelSettings


@pytest.fixture
def config_fixture() -> DistLLMSettings:
    """Create baseline DistLLMSettings for config tests."""
    return DistLLMSettings(model=ModelSettings(name="test-model"))


@pytest.fixture
def mock_env_vars(monkeypatch: Any) -> None:
    """Set env vars that the config actually reads.

    pydantic-settings is configured with env_prefix='DISTLLM_' + delimiter='__',
    so the working name is DISTLLM_MODEL__NAME (single underscore after the
    prefix). The old DISTLLM__MODEL__NAME double form is silently ignored.
    """
    monkeypatch.setenv("DISTLLM_MODEL__NAME", "env-test-model")
    monkeypatch.setenv("DISTLLM_MODEL__DTYPE", "bfloat16")
