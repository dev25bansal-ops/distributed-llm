from typing import Any, List, Optional

from distllm.sdk import DistLLMClientSync


class DistLLMCrewEmbedder:
    """CrewAI-compatible embedder backed by a DistLLM cluster.

    Implements the CrewAI ``Embedder`` protocol (``embed_text``,
    ``embed_batch``, ``get_embedding_model``).
    """

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

    # ------------------------------------------------------------------
    # CrewAI Embedder protocol
    # ------------------------------------------------------------------

    def get_embedding_model(self) -> str:
        """Return the embedding model identifier.

        CrewAI uses this to introspect which model is powering embeddings.
        """
        return self.model

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        resp = self._client.embeddings(input=[text], model=self.model)
        if isinstance(resp, dict):
            data = resp.get("data", [])
            return data[0]["embedding"] if data else []
        return resp.data[0].embedding if resp.data else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        resp = self._client.embeddings(input=texts, model=self.model)
        if isinstance(resp, dict):
            return [e["embedding"] for e in resp.get("data", [])]
        return [e.embedding for e in resp.data]
