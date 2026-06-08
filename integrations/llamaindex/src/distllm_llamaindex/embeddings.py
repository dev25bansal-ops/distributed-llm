from typing import Any, List, Optional

from pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from distllm.sdk import DistLLMClient, DistLLMClientSync
from distllm.sdk.types import EmbeddingResponse


class DistLLMEmbeddings(BaseEmbedding):
    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 120.0

    _client: DistLLMClientSync = PrivateAttr(default=None)
    _async_client: DistLLMClient = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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

    @classmethod
    def class_name(cls) -> str:
        return "distllm_embeddings"

    def _get_text_embedding(self, text: str) -> List[float]:
        resp = self._client.embeddings(input=[text], model=self.model)
        return self._extract_single(resp)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings(input=texts, model=self.model)
        return self._extract_all(resp)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        resp = await self._async_client.embeddings(input=[text], model=self.model)
        return self._extract_single(resp)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        resp = await self._async_client.embeddings(input=texts, model=self.model)
        return self._extract_all(resp)

    @staticmethod
    def _extract_single(resp: Any) -> List[float]:
        if isinstance(resp, EmbeddingResponse):
            return resp.data[0].embedding if resp.data else []
        if isinstance(resp, dict):
            data = resp.get("data", [])
            return data[0]["embedding"] if data else []
        return []

    @staticmethod
    def _extract_all(resp: Any) -> List[List[float]]:
        if isinstance(resp, EmbeddingResponse):
            return [e.embedding for e in resp.data]
        if isinstance(resp, dict):
            return [e["embedding"] for e in resp.get("data", [])]
        return []
