"""AutoGen LLM config helper for DistLLM.

Generates AutoGen-compatible ``llm_config`` dicts pointing at a DistLLM
cluster.  Since DistLLM is OpenAI-compatible, AutoGen can use it via
its standard OpenAI client.

Usage::

    from distllm_autogen import DistLLMConfig
    import autogen

    config = DistLLMConfig(base_url="http://localhost:8000/v1")

    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=config.assistant_config(),
    )
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="TERMINATE",
        llm_config=config.proxy_config(),
    )
    user_proxy.initiate_chat(assistant, message="Hello!")
"""

from __future__ import annotations

from typing import Any, Optional


class DistLLMConfig:
    """Generate AutoGen-compatible LLM configs for DistLLM.

    Parameters
    ----------
    base_url : str
        DistLLM API base URL (e.g. ``http://localhost:8000/v1``).
    model : str
        Model identifier on the DistLLM cluster.
    api_key : str, optional
        API key (DistLLM doesn't require one; any non-empty string works).
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens per response.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "distributed-llm",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or "distllm"
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _base_config(self) -> dict[str, Any]:
        return {
            "config_list": [
                {
                    "model": self.model,
                    "api_key": self.api_key,
                    "base_url": self.base_url,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_seed": None,  # Disable caching for distributed inference
        }

    def assistant_config(self, **overrides: Any) -> dict[str, Any]:
        """Config for ``autogen.AssistantAgent``."""
        cfg = self._base_config()
        cfg.update(overrides)
        return cfg

    def proxy_config(self, **overrides: Any) -> dict[str, Any]:
        """Config for ``autogen.UserProxyAgent``."""
        cfg = self._base_config()
        cfg.update(overrides)
        return cfg

    def group_chat_config(
        self,
        agents: list[Any],
        max_round: int = 10,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Config for ``autogen.GroupChat``."""
        return {
            "agents": agents,
            "max_round": max_round,
            "speaker_selection_method": "auto",
            **overrides,
        }

    def llm_config_for(self, agent_type: str = "assistant", **overrides: Any) -> dict[str, Any]:
        """Return config for a given agent type name."""
        method = getattr(self, f"{agent_type}_config", self.assistant_config)
        return method(**overrides)

    def __repr__(self) -> str:
        return (
            f"DistLLMConfig(base_url={self.base_url!r}, "
            f"model={self.model!r}, temperature={self.temperature})"
        )
