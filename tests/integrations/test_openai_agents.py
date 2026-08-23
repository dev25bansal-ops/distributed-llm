"""Tests for the distllm-openai-agents integration package."""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Test DistLLMAgentModel creation
# ---------------------------------------------------------------------------


class TestDistLLMAgentModel:
    """Verify the model wrapper constructs correctly."""

    def test_import(self) -> None:
        """The module should be importable."""
        from distllm.integrations.openai_agents import DistLLMAgentModel

        assert DistLLMAgentModel is not None

    def test_create_with_defaults(self) -> None:
        """Creating a model should return an instance when SDK is present."""
        from distllm.integrations.openai_agents.model import DistLLMAgentModel

        with patch(
            "distllm.integrations.openai_agents.model.DistLLMAgentModel.__new__"
        ) as mock_new:
            mock_new.return_value = object()
            instance = DistLLMAgentModel()
            assert instance is not None

    def test_resolve_base_url_default(self) -> None:
        """Should use default base_url when none provided."""
        from distllm.integrations.openai_agents.model import _resolve_base_url

        url = _resolve_base_url(None)
        assert url == "http://localhost:8000"

    def test_resolve_base_url_env(self) -> None:
        """Should read base_url from env var."""
        import os

        from distllm.integrations.openai_agents.model import _resolve_base_url

        os.environ["DISTLLM_API_BASE"] = "http://test-cluster:9000"
        try:
            url = _resolve_base_url(None)
            assert url == "http://test-cluster:9000"
        finally:
            del os.environ["DISTLLM_API_BASE"]

    def test_resolve_api_key_default(self) -> None:
        """Should use default api_key when none provided."""
        from distllm.integrations.openai_agents.model import _resolve_api_key

        key = _resolve_api_key(None)
        assert key == "not-needed"

    def test_resolve_api_key_env(self) -> None:
        """Should read api_key from env var."""
        import os

        from distllm.integrations.openai_agents.model import _resolve_api_key

        os.environ["DISTLLM_API_KEY"] = "test-key-123"
        try:
            key = _resolve_api_key(None)
            assert key == "test-key-123"
        finally:
            del os.environ["DISTLLM_API_KEY"]


# ---------------------------------------------------------------------------
# Test graceful degradation when SDK missing
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Verify ImportError is raised when openai-agents is not installed."""

    def test_model_import_error(self) -> None:
        """DistLLMAgentModel should raise ImportError if SDK is missing."""
        import builtins

        from distllm.integrations.openai_agents.model import DistLLMAgentModel

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if "agents" in name:
                raise ImportError(f"No module named {name}")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", _mock_import):
            with pytest.raises(ImportError, match="openai-agents"):
                DistLLMAgentModel()

    def test_runner_import_error(self) -> None:
        """DistLLMAgentRunner should raise ImportError if SDK is missing."""
        import builtins

        from distllm.integrations.openai_agents.runner import DistLLMAgentRunner

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if "agents" in name:
                raise ImportError(f"No module named {name}")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", _mock_import):
            with pytest.raises(ImportError, match="openai-agents"):
                DistLLMAgentRunner()


# ---------------------------------------------------------------------------
# Test DistLLMModelProvider configuration
# ---------------------------------------------------------------------------


class TestDistLLMModelProvider:
    """Verify the provider configures correctly."""

    def test_import(self) -> None:
        """The provider module should be importable."""
        from distllm.integrations.openai_agents import DistLLMModelProvider

        assert DistLLMModelProvider is not None

    def test_create_with_defaults(self) -> None:
        """Creating a provider should store defaults."""
        from distllm.integrations.openai_agents.provider import DistLLMModelProvider

        provider = DistLLMModelProvider()
        assert provider._default_model == "distributed-llm"
        assert provider._base_url is None
        assert provider._api_key is None

    def test_create_with_explicit_values(self) -> None:
        """Creating a provider with explicit values."""
        from distllm.integrations.openai_agents.provider import DistLLMModelProvider

        provider = DistLLMModelProvider(
            model="my-model",
            base_url="http://localhost:9000",
            api_key="sk-test",
        )
        assert provider._default_model == "my-model"
        assert provider._base_url == "http://localhost:9000"
        assert provider._api_key == "sk-test"

    def test_get_model(self) -> None:
        """get_model should call DistLLMAgentModel with correct args."""
        from distllm.integrations.openai_agents.provider import DistLLMModelProvider

        provider = DistLLMModelProvider(
            model="test-model",
            base_url="http://localhost:9000",
            api_key="sk-test",
        )

        with patch(
            "distllm.integrations.openai_agents.provider.DistLLMAgentModel"
        ) as mock_model_cls:
            mock_model_cls.return_value = object()
            result = provider.get_model("custom-model")
            mock_model_cls.assert_called_once_with(
                model="custom-model",
                base_url="http://localhost:9000",
                api_key="sk-test",
            )
            assert result is not None

    def test_get_model_default(self) -> None:
        """get_model should use default model when none specified."""
        from distllm.integrations.openai_agents.provider import DistLLMModelProvider

        provider = DistLLMModelProvider(model="fallback-model")

        with patch(
            "distllm.integrations.openai_agents.provider.DistLLMAgentModel"
        ) as mock_model_cls:
            mock_model_cls.return_value = object()
            provider.get_model()
            mock_model_cls.assert_called_once_with(
                model="fallback-model",
                base_url=None,
                api_key=None,
            )

    def test_from_env(self) -> None:
        """from_env should read environment variables."""
        import os

        from distllm.integrations.openai_agents.provider import DistLLMModelProvider

        os.environ["DISTLLM_API_BASE"] = "http://env-cluster:7000"
        os.environ["DISTLLM_API_KEY"] = "env-key"
        try:
            provider = DistLLMModelProvider.from_env()
            assert provider._base_url == "http://env-cluster:7000"
            assert provider._api_key == "env-key"
        finally:
            del os.environ["DISTLLM_API_BASE"]
            del os.environ["DISTLLM_API_KEY"]


# ---------------------------------------------------------------------------
# Test module __init__
# ---------------------------------------------------------------------------


class TestPackageInit:
    """Verify the package exports match expectations."""

    def test_all_exports(self) -> None:
        """The package should export all expected names."""
        import distllm.integrations.openai_agents as pkg

        assert hasattr(pkg, "DistLLMAgentModel")
        assert hasattr(pkg, "DistLLMModelProvider")
        assert hasattr(pkg, "DistLLMAgentRunner")
        assert pkg.__all__ == [
            "DistLLMAgentModel",
            "DistLLMModelProvider",
            "DistLLMAgentRunner",
        ]
