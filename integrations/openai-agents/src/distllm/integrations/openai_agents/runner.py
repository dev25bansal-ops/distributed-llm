"""DistLLM runner for the OpenAI Agents SDK."""

from __future__ import annotations

from typing import Any, Optional


class DistLLMAgentRunner:
    """Thin wrapper around ``agents.Runner`` with DistLLM defaults.

    Provides convenient ``run()`` and ``run_streamed()`` methods that
    automatically configure the model provider to point at DistLLM.

    Graceful degradation: if the ``openai-agents`` SDK is not installed
    the class cannot be instantiated.
    """

    def __new__(
        cls,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Return a callable-like wrapper around ``agents.Runner``.

        The returned object provides ``run()`` and ``run_streamed()``
        that delegate to ``agents.Runner`` with a DistLLM-aware
        ``RunnerContext``.

        Parameters
        ----------
        base_url : str, optional
            DistLLM coordinator URL.
        api_key : str, optional
            API key.
        **kwargs
            Additional arguments forwarded to ``agents.Runner.run()``.

        Returns
        -------
        _RunnerProxy
        """
        try:
            from agents import Runner as AgentsRunner
        except ImportError:
            raise ImportError(
                "The openai-agents SDK is required. "
                "Install it with: pip install openai-agents"
            ) from None

        from distllm.integrations.openai_agents.provider import (
            DistLLMModelProvider,
        )

        provider = DistLLMModelProvider(
            base_url=base_url,
            api_key=api_key,
        )

        return _RunnerProxy(
            runner_cls=AgentsRunner,
            model_provider=provider,
            **kwargs,
        )


class _RunnerProxy:
    """Internal proxy that delegates to ``agents.Runner`` with a DistLLM provider."""

    def __init__(
        self,
        runner_cls: Any,
        model_provider: Any,
        **kwargs: Any,
    ) -> None:
        self._runner_cls = runner_cls
        self._model_provider = model_provider
        self._defaults = kwargs

    async def run(self, agent: Any, input: Any, **kwargs: Any) -> Any:
        """Run an agent synchronously (async) via DistLLM."""
        from agents import Runner

        merged = dict(self._defaults)
        merged.update(kwargs)
        return await Runner.run(
            agent,
            input,
            model_provider=self._model_provider,
            **merged,
        )

    async def run_streamed(self, agent: Any, input: Any, **kwargs: Any) -> Any:
        """Run an agent with streaming via DistLLM."""
        from agents import Runner

        merged = dict(self._defaults)
        merged.update(kwargs)
        return Runner.run_streamed(
            agent,
            input,
            model_provider=self._model_provider,
            **merged,
        )
