"""Haystack Generator component backed by DistLLM."""

from __future__ import annotations

from typing import Any, Optional

from haystack.components.generators import OpenAIGenerator


class DistLLMGenerator(OpenAIGenerator):
    """Haystack Generator that uses DistLLM's OpenAI-compatible API.

    Drop-in replacement for ``OpenAIGenerator`` — just change the
    ``api_base_url`` parameter.

    Usage::

        from distllm_haystack import DistLLMGenerator

        generator = DistLLMGenerator(
            model="distributed-llm",
            api_base_url="http://localhost:8000/v1",
        )
        result = generator.run("What is distributed inference?")
        print(result["replies"])
    """

    def __init__(
        self,
        model: str = "distributed-llm",
        api_base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        # Haystack's OpenAIGenerator accepts api_base_url directly
        super().__init__(
            model=model,
            api_base_url=api_base_url,
            api_key=api_key or "distllm",
            generation_kwargs=generation_kwargs or {},
            timeout=timeout,
            **kwargs,
        )
