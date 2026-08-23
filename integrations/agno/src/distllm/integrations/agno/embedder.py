"""DistLLM embedder for Agno."""

from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_API_KEY = "not-needed"


def _resolve_base_url(base_url: Optional[str] = None) -> str:
    return base_url or os.getenv("DISTLLM_API_BASE", _DEFAULT_BASE_URL)


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    return api_key or os.getenv("DISTLLM_API_KEY", _DEFAULT_API_KEY)


class DistLLMEmbedder:
    """Agno-compatible embedder backed by a DistLLM cluster.

    Wraps ``agno.embedder.openai.OpenAIEmbedder`` with a DistLLM base URL.

    Graceful degradation: if ``agno`` is not installed, instantiating
    this class emits a clear ``ImportError``.
    """

    def __new__(
        cls,
        model: str = "text-embedding-ada-002",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create and return an ``agno.embedder.openai.OpenAIEmbedder`` instance.

        Parameters
        ----------
        model : str
            Embedding model identifier.
        base_url : str, optional
            DistLLM coordinator URL.
        api_key : str, optional
            API key.
        **kwargs
            Additional arguments forwarded to ``OpenAIEmbedder``.

        Returns
        -------
        agno.embedder.openai.OpenAIEmbedder
        """
        try:
            from agno.embedder.openai import OpenAIEmbedder
        except ImportError:
            raise ImportError(
                "The agno package is required. "
                "Install it with: pip install agno"
            ) from None

        resolved_base = _resolve_base_url(base_url)
        resolved_key = _resolve_api_key(api_key)

        return OpenAIEmbedder(
            id=model,
            api_key=resolved_key,
            base_url=resolved_base,
            **kwargs,
        )
