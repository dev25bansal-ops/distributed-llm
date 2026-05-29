from typing import Any, List, Optional

from distllm.sdk import DistLLMClientSync


class DistLLMCrewEmbedder:
    model: str = "distributed-llm"
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 120.0

    def __init__(self, **kwargs: Any):
        self.model = kwargs.pop("model", "distributed-llm")
        self.base_url = kwargs.pop("base_url", "http://localhost:8000")
        self.api_key = kwargs.pop("api_key", None)
        self.timeout = kwargs.pop("timeout", 120.0)
        self._client = DistLLMClientSync(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )

    def embed_text(self, text: str) -> List[float]:
        resp = self._client.embeddings(input=[text], model=self.model)
        if isinstance(resp, dict):
            data = resp.get("data", [])
            return data[0]["embedding"] if data else []
        return resp.data[0].embedding if resp.data else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings(input=texts, model=self.model)
        if isinstance(resp, dict):
            return [e["embedding"] for e in resp.get("data", [])]
        return [e.embedding for e in resp.data]
