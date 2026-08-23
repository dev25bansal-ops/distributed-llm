"""DistLLM chat model for Agno."""

from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_API_KEY = "not-needed"

_SCHEDULING_HEADERS: dict[str, str] = {
    "X-DistLLM-Source": "agno-integration",
    "X-DistLLM-Priority": "default",
}


def _resolve_base_url(base_url: Optional[str] = None) -> str:
    return base_url or os.getenv("DISTLLM_API_BASE", _DEFAULT_BASE_URL)


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    return api_key or os.getenv("DISTLLM_API_KEY", _DEFAULT_API_KEY)


class DistLLM:
    """Agno-compatible chat model backed by a DistLLM cluster.

    Wraps ``agno.models.openai.OpenAIChat`` with a DistLLM base URL.

    Graceful degradation: if ``agno`` is not installed, instantiating
    this class emits a clear ``ImportError``.
    """

    def __new__(
        cls,
        model: str = "distributed-llm",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create and return an ``agno.models.openai.OpenAIChat`` instance.

        Parameters
        ----------
        model : str
            Model identifier to use on the DistLLM cluster.
        base_url : str, optional
            DistLLM coordinator URL.
        api_key : str, optional
            API key.
        **kwargs
            Additional arguments forwarded to ``OpenAIChat``.

        Returns
        -------
        agno.models.openai.OpenAIChat
        """
        try:
            from agno.models.openai import OpenAIChat
        except ImportError:
            raise ImportError(
                "The agno package is required. "
                "Install it with: pip install agno"
            ) from None

        resolved_base = _resolve_base_url(base_url)
        resolved_key = _resolve_api_key(api_key)

        # Merge scheduling headers into client_kwargs
        client_kwargs = kwargs.pop("client_kwargs", {})
        existing_headers = client_kwargs.get("default_headers", {})
        client_kwargs["default_headers"] = {**existing_headers, **_SCHEDULING_HEADERS}

        return OpenAIChat(
            id=model,
            api_key=resolved_key,
            base_url=resolved_base,
            client_kwargs=client_kwargs,
            **kwargs,
        )
