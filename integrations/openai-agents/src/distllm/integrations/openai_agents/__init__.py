"""DistLLM integration for the OpenAI Agents SDK.

Provides ``DistLLMAgentModel``, ``DistLLMModelProvider``, and
``DistLLMAgentRunner`` so you can use DistLLM as the inference backend
for the ``openai-agents`` SDK.

Usage::

    from distllm.integrations.openai_agents import DistLLMAgentModel, DistLLMModelProvider

    model = DistLLMAgentModel(base_url="http://localhost:8000")
    provider = DistLLMModelProvider(model=model)

    from agents import Agent, Runner
    agent = Agent(name="Assistant", instructions="You are helpful.", model=model)
"""

from __future__ import annotations

from distllm.integrations.openai_agents.model import DistLLMAgentModel
from distllm.integrations.openai_agents.provider import DistLLMModelProvider
from distllm.integrations.openai_agents.runner import DistLLMAgentRunner

__all__ = [
    "DistLLMAgentModel",
    "DistLLMModelProvider",
    "DistLLMAgentRunner",
]
