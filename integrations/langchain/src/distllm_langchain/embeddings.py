from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings

from distllm.sdk import DistLLMClient, DistLLMClientSync


class DistLLMEmbeddings(Embeddings):
    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 120.0
    batch_size: int = 32

    def __init__(self, **kwargs: Any) -> None:
        # ``langchain_core.embeddings.Embeddings`` is a plain ABC whose
        # ``__init__`` is ``object.__init__``, so kwargs must be applied here
        # rather than forwarded to super().
        for field in ("model", "base_url", "api_key", "timeout", "batch_size"):
            if field in kwargs:
                setattr(self, field, kwargs.pop(field))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")
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
        if isinstance(resp, dict):
            data = resp.get("data")
        else:
            data = getattr(resp, "data", None)
        if not isinstance(data, (list, tuple)):
            return []
        out: List[List[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
            out.append(list(embedding) if embedding else [])
        return out
