"""DistLLM configuration adapter for AutoGPT."""

from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_API_KEY = "not-needed"


def _resolve_base_url(base_url: Optional[str] = None) -> str:
    return base_url or os.getenv("DISTLLM_API_BASE", _DEFAULT_BASE_URL)


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    return api_key or os.getenv("DISTLLM_API_KEY", _DEFAULT_API_KEY)


class DistLLMAutoGPTConfig:
    """AutoGPT configuration that routes through DistLLM.

    Produces the configuration dict that AutoGPT's ``AgentConfig`` expects,
    with the ``openai`` section pointing at a DistLLM cluster.

    Graceful degradation: if the ``autogpt`` package is not installed,
    instantiating this class emits a clear ``ImportError``.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create an AutoGPT configuration.

        Parameters
        ----------
        base_url : str, optional
            DistLLM coordinator URL.
        api_key : str, optional
            API key.
        **kwargs
            Additional configuration keys forwarded to the AutoGPT config.
        """
        self._base_url = _resolve_base_url(base_url)
        self._api_key = _resolve_api_key(api_key)
        self._extra = kwargs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def openai_config(self) -> dict[str, Any]:
        """Return the ``openai`` section of the AutoGPT config."""
        return {
            "api_type": "openai",
            "api_base": f"{self._base_url.rstrip('/')}/v1",
            "api_key": self._api_key,
            "api_version": "2024-02-01",
        }

    @property
    def scheduling_headers(self) -> dict[str, str]:
        """Return scheduling headers for DistLLM."""
        return {
            "X-DistLLM-Source": "autogpt-integration",
            "X-DistLLM-Priority": "default",
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full configuration as a dictionary."""
        config: dict[str, Any] = {
            "openai": self.openai_config,
            "scheduling_headers": self.scheduling_headers,
        }
        config.update(self._extra)
        return config

    # ------------------------------------------------------------------
    # AutoGPT integration
    # ------------------------------------------------------------------

    def configure(self) -> Any:
        """Apply the configuration to the AutoGPT runtime environment.

        Returns the ``AgentConfig`` instance if successful, or ``None``
        if the ``autogpt`` package is unavailable.
        """
        try:
            from autogpt.config import AgentConfig
        except ImportError:
            raise ImportError(
                "The autogpt package is required. "
                "Install it with: pip install autogpt"
            ) from None

        return AgentConfig(**self.to_dict())
