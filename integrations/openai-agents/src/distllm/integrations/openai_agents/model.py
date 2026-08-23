"""DistLLM model wrapper for the OpenAI Agents SDK."""

from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_API_KEY = "not-needed"


def _resolve_base_url(base_url: Optional[str] = None) -> str:
    return base_url or os.getenv("DISTLLM_API_BASE", _DEFAULT_BASE_URL)


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    return api_key or os.getenv("DISTLLM_API_KEY", _DEFAULT_API_KEY)


class DistLLMAgentModel:
    """Drop-in model wrapper for the OpenAI Agents SDK.

    Wraps ``agents.OpenAIModel`` with a DistLLM base URL, so that agent
    inference is routed through your DistLLM cluster.

    Graceful degradation: if the ``openai-agents`` SDK is not installed,
    instantiating this class emits a clear ``ImportError`` message.
    """

    def __new__(
        cls,
        model: str = "distributed-llm",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create and return an ``agents.OpenAIModel`` instance.

        Parameters
        ----------
        model : str
            Model identifier to use on the DistLLM cluster.
        base_url : str, optional
            DistLLM coordinator URL.
        api_key : str, optional
            API key.
        **kwargs
            Additional arguments forwarded to ``agents.OpenAIModel``.

        Returns
        -------
        agents.OpenAIModel
            An OpenAIModel configured to talk through DistLLM.
        """
        try:
            from agents.models.openai import OpenAIModel
        except ImportError:
            raise ImportError(
                "The openai-agents SDK is required. "
                "Install it with: pip install openai-agents"
            ) from None

        resolved_base = _resolve_base_url(base_url)
        resolved_key = _resolve_api_key(api_key)

        # Create an OpenAI client pointed at DistLLM
        import openai

        client = openai.AsyncOpenAI(
            base_url=f"{resolved_base.rstrip('/')}/v1",
            api_key=resolved_key,
            default_headers={
                "X-DistLLM-Source": "openai-agents-sdk",
                "X-DistLLM-Priority": "default",
            },
        )

        instance = OpenAIModel(
            model=model,
            openai_client=client,
            **kwargs,
        )
        return instance
