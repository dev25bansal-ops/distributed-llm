"""DistLLM Embeddings — LlamaIndex Embeddings implementation.

Wraps the DistLLM SDK client to provide native LlamaIndex ``BaseEmbedding``
for document and query embedding.

Usage::

    from distllm_llamaindex import DistLLMEmbeddings

    embeddings = DistLLMEmbeddings(base_url="http://localhost:8000")
    vectors = embeddings.get_text_embedding("Hello world")
    query_vec = embeddings.get_query_embedding("What is distributed inference?")
"""

from typing import Any, List, Optional

from llama_index.core.embeddings import BaseEmbedding

from distllm.sdk import DistLLMClient, DistLLMClientSync


class DistLLMEmbeddings(BaseEmbedding):
    """LlamaIndex Embeddings backed by DistLLM's embedding endpoint.

    Wraps the DistLLM SDK client to provide sync/async embedding
    of texts and queries.

    .. rubric:: Example

    .. code-block:: python

        from distllm_llamaindex import DistLLMEmbeddings

        embeddings = DistLLMEmbeddings(
            base_url="http://localhost:8000",
            model="distributed-llm",
        )
        docs = embeddings.get_text_embedding_batch(["Hello", "World"])
    """

    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 120.0

    _client: DistLLMClientSync = None
    _async_client: DistLLMClient = None

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
        return self._extract_embedding(resp)

    def _get_text_embeddings(
        self, texts: List[str]
    ) -> List[List[float]]:
        resp = self._client.embeddings(input=texts, model=self.model)
        if isinstance(resp, dict):
            return [e["embedding"] for e in resp.get("data", [])]
        return [e.embedding for e in resp.data]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        resp = await self._async_client.embeddings(input=[text], model=self.model)
        return self._extract_embedding(resp)

    async def _aget_text_embeddings(
        self, texts: List[str]
    ) -> List[List[float]]:
        resp = await self._async_client.embeddings(input=texts, model=self.model)
        if isinstance(resp, dict):
            return [e["embedding"] for e in resp.get("data", [])]
        return [e.embedding for e in resp.data]

    @staticmethod
    def _extract_embedding(resp: Any) -> List[float]:
        if isinstance(resp, dict):
            data = resp.get("data", [])
            if data:
                return data[0]["embedding"]
            return []
        if resp.data:
            return resp.data[0].embedding
        return []
