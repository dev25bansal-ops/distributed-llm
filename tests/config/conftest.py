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
    """Set DISTLLM__ prefixed env vars for config tests."""
    monkeypatch.setenv("DISTLLM__MODEL__NAME", "env-test-model")
    monkeypatch.setenv("DISTLLM__MODEL__DTYPE", "bfloat16")
