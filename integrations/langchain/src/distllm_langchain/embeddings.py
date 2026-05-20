"""DistLLM Embeddings — LangChain Embeddings implementation.

Wraps the DistLLM SDK client to provide native LangChain ``Embeddings``
for document and query embedding.

Usage::

    from distllm_langchain import DistLLMEmbeddings

    embeddings = DistLLMEmbeddings(base_url="http://localhost:8000")
    vectors = embeddings.embed_documents(["Hello world", "Goodbye"])
    query_vec = embeddings.embed_query("What is distributed inference?")
"""

from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings

from distllm.sdk import DistLLMClient, DistLLMClientSync


class DistLLMEmbeddings(Embeddings):
    """LangChain Embeddings backed by DistLLM's embedding endpoint.

    Wraps the DistLLM SDK client to provide synchronous embedding
    of documents and queries.

    .. rubric:: Example

    .. code-block:: python

        from distllm_langchain import DistLLMEmbeddings

        embeddings = DistLLMEmbeddings(
            base_url="http://localhost:8000",
            model="distributed-llm",
        )
        docs = embeddings.embed_documents(["Hello", "World"])
    """

    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 120.0

    _client: DistLLMClientSync = None  # type: ignore[assignment]
    _async_client: DistLLMClient = None  # type: ignore[assignment]

    def __init__(self, **kwargs: Any) -> None:
        self.model = kwargs.pop("model", "distributed-llm")
        self.base_url = kwargs.pop("base_url", "http://localhost:8000")
        self.api_key = kwargs.pop("api_key", None)
        self.timeout = kwargs.pop("timeout", 120.0)
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
        """Embed a list of documents.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors, one per document.
        """
        resp = self._client.embeddings(input=texts, model=self.model)
        if isinstance(resp, dict):
            return [e["embedding"] for e in resp.get("data", [])]
        return [e.embedding for e in resp.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector.
        """
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents asynchronously.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors.
        """
        resp = await self._async_client.embeddings(input=texts, model=self.model)
        if isinstance(resp, dict):
            return [e["embedding"] for e in resp.get("data", [])]
        return [e.embedding for e in resp.data]

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query asynchronously.

        Args:
            text: Query string to embed.

        Returns:
            Embedding vector.
        """
        vectors = await self.aembed_documents([text])
        return vectors[0] if vectors else []
