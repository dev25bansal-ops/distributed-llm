"""DistLLM integration for AutoGPT.

Provides ``DistLLMAutoGPTConfig`` and ``launch_agent()`` so you can run
AutoGPT agents backed by a DistLLM cluster.

Usage::

    from distllm.integrations.autogpt import DistLLMAutoGPTConfig, launch_agent

    config = DistLLMAutoGPTConfig(base_url="http://localhost:8000")
    await launch_agent(config, "Research the latest AI trends")
"""

from __future__ import annotations

from distllm.integrations.autogpt.config import DistLLMAutoGPTConfig
from distllm.integrations.autogpt.launcher import launch_agent

__all__ = [
    "DistLLMAutoGPTConfig",
    "launch_agent",
]
