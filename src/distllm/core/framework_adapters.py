"""Framework Integration Adapters — unified interface for LLM frameworks.

Provides drop-in adapters for popular frameworks:
- LangChain (ChatOpenAI wrapper)
- LlamaIndex (OpenAI LLM wrapper)
- CrewAI (LangChain-based)
- Dify (custom provider plugin)
- Haystack (OpenAI generator)
- AutoGPT / Agency Swarm (OpenAI client)

Usage::

    # LangChain
    from distllm.integrations import get_langchain_llm
    llm = get_langchain_llm(base_url="http://localhost:8000")
    response = llm.invoke("Hello!")

    # LlamaIndex
    from distllm.integrations import get_llamaindex_llm
    llm = get_llamaindex_llm(base_url="http://localhost:8000")

    # Generic OpenAI-compatible
    from distllm.integrations import get_openai_client
    client = get_openai_client(base_url="http://localhost:8000")
"""

from __future__ import annotations

import os
from typing import Any


def get_openai_client(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    **kwargs,
) -> Any:
    """Get an OpenAI-compatible client pointing to DistLLM.

    Works with the openai Python package.
    """
    from openai import OpenAI
    return OpenAI(
        base_url=f"{base_url}/v1",
        api_key=api_key,
        **kwargs,
    )


def get_async_openai_client(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    **kwargs,
) -> Any:
    """Get an async OpenAI-compatible client pointing to DistLLM."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key=api_key,
        **kwargs,
    )


def get_langchain_llm(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    model: str = "distributed-llm",
    **kwargs,
) -> Any:
    """Get a LangChain ChatOpenAI instance pointing to DistLLM.

    Requires: pip install langchain-openai
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        openai_api_base=f"{base_url}/v1",
        openai_api_key=api_key,
        **kwargs,
    )


def get_llamaindex_llm(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    model: str = "distributed-llm",
    **kwargs,
) -> Any:
    """Get a LlamaIndex OpenAI LLM instance pointing to DistLLM.

    Requires: pip install llama-index-llms-openai
    """
    from llama_index.llms.openai import OpenAI
    return OpenAI(
        model=model,
        api_base=f"{base_url}/v1",
        api_key=api_key,
        **kwargs,
    )


def get_haystack_generator(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    model: str = "distributed-llm",
    **kwargs,
) -> Any:
    """Get a Haystack OpenAIGenerator instance pointing to DistLLM.

    Requires: pip install haystack-ai
    """
    from haystack.components.generators import OpenAIGenerator
    return OpenAIGenerator(
        api_base_url=f"{base_url}/v1",
        api_key=api_key,
        model=model,
        **kwargs,
    )


# Framework compatibility matrix
FRAMEWORK_COMPAT = {
    "langchain": {
        "package": "langchain-openai",
        "class": "ChatOpenAI",
        "adapter": get_langchain_llm,
        "features": ["chat", "streaming", "tools", "structured_output"],
        "example": "examples/langchain_example.py",
    },
    "llamaindex": {
        "package": "llama-index-llms-openai",
        "class": "OpenAI",
        "adapter": get_llamaindex_llm,
        "features": ["chat", "completion", "streaming", "rag"],
        "example": "examples/llamaindex_example.py",
    },
    "crewai": {
        "package": "crewai",
        "class": "Agent (via langchain-openai)",
        "adapter": get_langchain_llm,
        "features": ["multi-agent", "tasks", "sequential/parallel"],
        "example": "examples/crewai_example.py",
    },
    "haystack": {
        "package": "haystack-ai",
        "class": "OpenAIGenerator",
        "adapter": get_haystack_generator,
        "features": ["chat", "completion", "rag", "pipelines"],
        "example": "examples/haystack_example.py",
    },
    "dify": {
        "package": "dify (self-hosted)",
        "class": "CustomProvider",
        "adapter": None,
        "features": ["chat", "completion", "streaming", "workflow"],
        "example": "integrations/dify/distllm_provider.py",
    },
    "autogpt": {
        "package": "openai",
        "class": "OpenAI",
        "adapter": get_openai_client,
        "features": ["agents", "tools", "memory"],
        "example": "examples/openai_agents_example.py",
    },
    "agency_swarm": {
        "package": "agency-swarm",
        "class": "Agent (via openai)",
        "adapter": get_openai_client,
        "features": ["multi-agent", "delegation", "tools"],
        "example": "examples/agency_swarm_example.py",
    },
}


def list_frameworks() -> list[dict]:
    """List all supported framework integrations."""
    return [
        {
            "name": name,
            "package": info["package"],
            "class": info["class"],
            "features": info["features"],
            "example": info["example"],
        }
        for name, info in FRAMEWORK_COMPAT.items()
    ]


def get_framework_adapter(framework: str, **kwargs) -> Any:
    """Get an adapter for a specific framework.

    Args:
        framework: Framework name (langchain, llamaindex, crewai, haystack, etc.)
        **kwargs: Arguments passed to the adapter constructor.

    Returns:
        Configured LLM instance for the framework.
    """
    info = FRAMEWORK_COMPAT.get(framework)
    if info is None:
        raise ValueError(f"Unknown framework: {framework}. Supported: {list(FRAMEWORK_COMPAT.keys())}")

    adapter = info.get("adapter")
    if adapter is None:
        raise ValueError(f"Framework '{framework}' does not support direct adapter creation. Use {info['example']}")

    return adapter(**kwargs)
