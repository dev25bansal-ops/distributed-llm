"""Semantic Kernel embedding service backed by DistLLM."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import EmbeddingGeneratorBase


class DistLLMEmbeddingService(EmbeddingGeneratorBase):
    """Semantic Kernel embedding service backed by DistLLM's OpenAI-compatible API.

    Usage::

        import semantic_kernel as sk
        from distllm_sk import DistLLMEmbeddingService

        kernel = sk.Kernel()
        embedder = DistLLMEmbeddingService(
            model_id="bge-large",
            base_url="http://localhost:8000/v1",
        )
        kernel.add_service(embedder)
    """

    def __init__(
        self,
        model_id: str = "distributed-llm",
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        super().__init__(ai_model_id=model_id, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or "distllm"
        self._timeout = timeout

    async def generate_embeddings(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                json={"model": self.ai_model_id, "input": texts},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return [item["embedding"] for item in data.get("data", [])]
