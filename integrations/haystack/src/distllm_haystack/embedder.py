"""Haystack Text Embedder component backed by DistLLM."""

from __future__ import annotations

from typing import Any, Optional

from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder


class DistLLMTextEmbedder(OpenAITextEmbedder):
    """Haystack Text Embedder that uses DistLLM's OpenAI-compatible API.

    Usage::

        from distllm_haystack import DistLLMTextEmbedder

        embedder = DistLLMTextEmbedder(
            model="bge-large",
            api_base_url="http://localhost:8000/v1",
        )
        result = embedder.run("Hello world")
        print(len(result["embedding"]))  # embedding dimension
    """

    def __init__(
        self,
        model: str = "distributed-llm",
        api_base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_base_url=api_base_url,
            api_key=api_key or "distllm",
            dimensions=dimensions,
            timeout=timeout,
            **kwargs,
        )


class DistLLMDocumentEmbedder(OpenAIDocumentEmbedder):
    """Haystack Document Embedder that uses DistLLM's OpenAI-compatible API.

    Usage::

        from distllm_haystack import DistLLMDocumentEmbedder
        from haystack import Document

        embedder = DistLLMDocumentEmbedder(
            model="bge-large",
            api_base_url="http://localhost:8000/v1",
        )
        result = embedder.run([Document(content="Hello"), Document(content="World")])
    """

    def __init__(
        self,
        model: str = "distributed-llm",
        api_base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_base_url=api_base_url,
            api_key=api_key or "distllm",
            dimensions=dimensions,
            timeout=timeout,
            **kwargs,
        )
