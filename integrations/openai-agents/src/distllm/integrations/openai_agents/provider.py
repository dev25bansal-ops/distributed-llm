"""DistLLM model provider for the OpenAI Agents SDK."""

from __future__ import annotations

from typing import Any, Optional

from distllm.integrations.openai_agents.model import DistLLMAgentModel


class DistLLMModelProvider:
    """Provider that returns ``DistLLMAgentModel`` instances.

    Implements the ``agents.ModelProvider`` protocol so the Agents SDK
    can resolve model names through DistLLM.

    Usage::

        provider = DistLLMModelProvider(base_url="http://localhost:8000")
        agent = Agent(name="Assistant", model_provider=provider)
    """

    def __init__(
        self,
        model: str = "distributed-llm",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._default_model = model
        self._base_url = base_url
        self._api_key = api_key

    # ------------------------------------------------------------------
    # ModelProvider protocol
    # ------------------------------------------------------------------

    def get_model(self, model_name: Optional[str] = None) -> Any:
        """Return an ``OpenAIModel`` for *model_name* (or the default)."""
        resolved = model_name or self._default_model
        return DistLLMAgentModel(
            model=resolved,
            base_url=self._base_url,
            api_key=self._api_key,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> DistLLMModelProvider:
        """Create a provider from environment variables.

        Reads ``DISTLLM_API_BASE`` and ``DISTLLM_API_KEY``.
        """
        import os

        return cls(
            base_url=os.getenv("DISTLLM_API_BASE"),
            api_key=os.getenv("DISTLLM_API_KEY"),
        )
