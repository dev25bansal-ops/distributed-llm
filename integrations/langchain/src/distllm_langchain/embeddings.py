from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings

from distllm.sdk import DistLLMClient, DistLLMClientSync
from distllm.sdk.types import EmbeddingResponse


class DistLLMEmbeddings(Embeddings):
    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 120.0
    batch_size: int = 32

    def __init__(self, **kwargs: Any) -> None:
        self.model = kwargs.pop("model", "distributed-llm")
        self.base_url = kwargs.pop("base_url", "http://localhost:8000")
        self.api_key = kwargs.pop("api_key", None)
        self.timeout = kwargs.pop("timeout", 120.0)
        self.batch_size = kwargs.pop("batch_size", 32)
        super().__init__()
        self._client = DistLLMClientSync(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )
        self._async_client = DistLLMClient(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._client.embeddings(input=batch, model=self.model)
            results.extend(self._extract_embeddings(resp))
        return results

    def embed_query(self, text: str) -> List[float]:
        resp = self._client.embeddings(input=[text], model=self.model)
        embeddings = self._extract_embeddings(resp)
        return embeddings[0] if embeddings else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = await self._async_client.embeddings(input=batch, model=self.model)
            results.extend(self._extract_embeddings(resp))
        return results

    async def aembed_query(self, text: str) -> List[float]:
        resp = await self._async_client.embeddings(input=[text], model=self.model)
        embeddings = self._extract_embeddings(resp)
        return embeddings[0] if embeddings else []

    @staticmethod
    def _extract_embeddings(resp: Any) -> List[List[float]]:
        if isinstance(resp, EmbeddingResponse):
            return [e.embedding for e in resp.data]
        if isinstance(resp, dict):
            return [e["embedding"] for e in resp.get("data", [])]
        return []
