"""Launch an AutoGPT agent with DistLLM as the inference backend."""

from __future__ import annotations

from typing import Any, Optional


async def launch_agent(
    config: Any,
    task: str,
    *,
    continuous_mode: bool = False,
    allow_code_execution: bool = False,
    **kwargs: Any,
) -> Any:
    """Launch an AutoGPT agent backed by a DistLLM cluster.

    Graceful degradation: if the ``autogpt`` package is not installed
    this function raises a clear ``ImportError``.

    Parameters
    ----------
    config : DistLLMAutoGPTConfig or dict
        DistLLM-aware configuration.  If a ``DistLLMAutoGPTConfig``
        instance is passed its ``to_dict()`` is called automatically.
    task : str
        The task or goal for the agent.
    continuous_mode : bool
        Whether to run in continuous mode (no user confirmation).
    allow_code_execution : bool
        Whether to allow code execution.
    **kwargs
        Additional arguments forwarded to the agent constructor.

    Returns
    -------
    autogpt.agent.Agent
        The running agent instance.

    Raises
    ------
    ImportError
        If the ``autogpt`` package is not installed.
    """
    try:
        from autogpt.agent import Agent
        from autogpt.config import AgentConfig
    except ImportError:
        raise ImportError(
            "The autogpt package is required. "
            "Install it with: pip install autogpt"
        ) from None

    # Resolve configuration
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise TypeError(
            f"Expected DistLLMAutoGPTConfig or dict, got {type(config).__name__}"
        )

    agent_config = AgentConfig(**config_dict)

    agent = Agent(
        config=agent_config,
        task=task,
        continuous_mode=continuous_mode,
        allow_code_execution=allow_code_execution,
        **kwargs,
    )

    return agent
