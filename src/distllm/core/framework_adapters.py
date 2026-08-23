"""Framework Integration Adapters — unified interface for LLM frameworks.

Provides drop-in adapters for popular frameworks:
- LangChain (ChatOpenAI wrapper)
- LlamaIndex (OpenAI LLM wrapper)
- CrewAI (LangChain-based)
- Dify (custom provider plugin)
- Haystack (OpenAI generator)
- OpenAI Agents SDK
- Agno
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
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Scheduling headers injected into every request
# ---------------------------------------------------------------------------

_SCHEDULING_HEADERS: dict[str, str] = {
    "X-DistLLM-Source": "framework-adapter",
    "X-DistLLM-Priority": "default",
}

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_API_KEY = "not-needed"


def _resolve_base_url(base_url: Optional[str] = None) -> str:
    return base_url or os.getenv("DISTLLM_API_BASE", _DEFAULT_BASE_URL)


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    return api_key or os.getenv("DISTLLM_API_KEY", _DEFAULT_API_KEY)


# ---------------------------------------------------------------------------
# OpenAI-compatible clients
# ---------------------------------------------------------------------------


def get_openai_client(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    **kwargs: Any,
) -> Any:
    """Get a sync OpenAI-compatible client pointing to DistLLM.

    Includes scheduling headers for the DistLLM coordinator.

    Requires: pip install openai
    """
    from openai import OpenAI

    resolved_base = _resolve_base_url(base_url)
    resolved_key = _resolve_api_key(api_key)

    return OpenAI(
        base_url=f"{resolved_base.rstrip('/')}/v1",
        api_key=resolved_key,
        default_headers=_SCHEDULING_HEADERS,
        **kwargs,
    )


def get_async_openai_client(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    **kwargs: Any,
) -> Any:
    """Get an async OpenAI-compatible client pointing to DistLLM.

    Includes scheduling headers for the DistLLM coordinator.

    Requires: pip install openai
    """
    from openai import AsyncOpenAI

    resolved_base = _resolve_base_url(base_url)
    resolved_key = _resolve_api_key(api_key)

    return AsyncOpenAI(
        base_url=f"{resolved_base.rstrip('/')}/v1",
        api_key=resolved_key,
        default_headers=_SCHEDULING_HEADERS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


def get_langchain_llm(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    model: str = "distributed-llm",
    **kwargs: Any,
) -> Any:
    """Get a LangChain ChatOpenAI instance pointing to DistLLM.

    Requires: pip install langchain-openai
    """
    from langchain_openai import ChatOpenAI

    resolved_base = _resolve_base_url(base_url)
    resolved_key = _resolve_api_key(api_key)

    return ChatOpenAI(
        model=model,
        openai_api_base=f"{resolved_base.rstrip('/')}/v1",
        openai_api_key=resolved_key,
        default_headers=_SCHEDULING_HEADERS,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


def get_llamaindex_llm(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    model: str = "distributed-llm",
    **kwargs: Any,
) -> Any:
    """Get a LlamaIndex OpenAI LLM instance pointing to DistLLM.

    Requires: pip install llama-index-llms-openai
    """
    from llama_index.llms.openai import OpenAI

    resolved_base = _resolve_base_url(base_url)
    resolved_key = _resolve_api_key(api_key)

    return OpenAI(
        model=model,
        api_base=f"{resolved_base.rstrip('/')}/v1",
        api_key=resolved_key,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Haystack
# ---------------------------------------------------------------------------


def get_haystack_generator(
    base_url: str = "http://localhost:8000",
    api_key: str = "not-needed",
    model: str = "distributed-llm",
    **kwargs: Any,
) -> Any:
    """Get a Haystack OpenAIGenerator instance pointing to DistLLM.

    Requires: pip install haystack-ai
    """
    from haystack.components.generators import OpenAIGenerator

    resolved_base = _resolve_base_url(base_url)
    resolved_key = _resolve_api_key(api_key)

    return OpenAIGenerator(
        api_base_url=f"{resolved_base.rstrip('/')}/v1",
        api_key=resolved_key,
        model=model,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Agno
# ---------------------------------------------------------------------------


def get_agno_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Return an Agno ``OpenAIChat`` subclass pointed at a DistLLM endpoint.

    The returned instance is a subclass of ``agno.models.openai.OpenAIChat``
    that overrides the base URL and injects scheduling headers.

    Parameters
    ----------
    base_url : str, optional
        DistLLM base URL.  Defaults to ``DISTLLM_API_BASE`` env var or
        ``http://localhost:8000``.
    api_key : str, optional
        API key.  Defaults to ``DISTLLM_API_KEY`` env var or ``"not-needed"``.
    **kwargs
        Additional arguments forwarded to ``OpenAIChat``.

    Returns
    -------
    agno.models.openai.OpenAIChat
    """
    from agno.models.openai import OpenAIChat

    resolved_base = _resolve_base_url(base_url)
    resolved_key = _resolve_api_key(api_key)

    return OpenAIChat(
        id=kwargs.pop("id", "distributed-llm"),
        api_key=resolved_key,
        base_url=resolved_base,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# AutoGPT configuration
# ---------------------------------------------------------------------------


def get_autogpt_config(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a configuration dictionary for an AutoGPT agent.

    The returned dict provides the ``openai`` section needed to point an
    AutoGPT instance at a DistLLM cluster.

    Parameters
    ----------
    base_url : str, optional
        DistLLM base URL.
    api_key : str, optional
        API key.
    **kwargs
        Additional configuration keys (e.g. ``continuous_mode``).

    Returns
    -------
    dict
        Configuration dict with ``openai`` keys pointing at DistLLM.
    """
    resolved_base = _resolve_base_url(base_url)
    resolved_key = _resolve_api_key(api_key)

    config: dict[str, Any] = {
        "openai": {
            "api_type": "openai",
            "api_base": f"{resolved_base.rstrip('/')}/v1",
            "api_key": resolved_key,
            "api_version": "2024-02-01",
        },
        "scheduling_headers": dict(_SCHEDULING_HEADERS),
    }
    config.update(kwargs)
    return config


# ---------------------------------------------------------------------------
# Framework compatibility matrix
# ---------------------------------------------------------------------------


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
    "openai_agents": {
        "package": "openai-agents",
        "class": "DistLLMAgentModel",
        "adapter": get_openai_client,
        "features": ["agents", "tools", "handoffs", "streaming"],
        "example": "examples/openai_agents_example.py",
    },
    "agno": {
        "package": "agno",
        "class": "OpenAIChat",
        "adapter": get_agno_client,
        "features": ["agents", "tools", "rag", "multi-modal"],
        "example": "examples/agno_example.py",
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


def get_framework_adapter(framework: str, **kwargs: Any) -> Any:
    """Get an adapter for a specific framework.

    Args:
        framework: Framework name (langchain, llamaindex, crewai, haystack, etc.)
        **kwargs: Arguments passed to the adapter constructor.

    Returns:
        Configured LLM instance for the framework.
    """
    info = FRAMEWORK_COMPAT.get(framework)
    if info is None:
        raise ValueError(
            f"Unknown framework: {framework}. "
            f"Supported: {list(FRAMEWORK_COMPAT.keys())}"
        )

    adapter = info.get("adapter")
    if adapter is None:
        raise ValueError(
            f"Framework '{framework}' does not support direct adapter creation. "
            f"Use {info['example']}"
        )

    return adapter(**kwargs)
