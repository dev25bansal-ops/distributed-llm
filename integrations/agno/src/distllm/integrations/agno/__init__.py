"""DistLLM integration for Agno.

Provides ``DistLLM`` (an OpenAIChat subclass), ``DistLLMEmbedder``, and
``DistLLMToolProvider`` so you can use DistLLM as the inference backend
for Agno agents.

Usage::

    from distllm.integrations.agno import DistLLM

    llm = DistLLM(base_url="http://localhost:8000")
    agent = Agent(model=llm)
"""

from __future__ import annotations

from distllm.integrations.agno.llm import DistLLM
from distllm.integrations.agno.embedder import DistLLMEmbedder
from distllm.integrations.agno.tools import DistLLMToolProvider

__all__ = [
    "DistLLM",
    "DistLLMEmbedder",
    "DistLLMToolProvider",
]
