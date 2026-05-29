from typing import Any, Optional

from distllm.sdk import DistLLMClientSync


class DistLLMKnowledgeSource:
    source_id: str
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 120.0

    def __init__(self, source_id: str, **kwargs: Any):
        self.source_id = source_id
        self.base_url = kwargs.pop("base_url", "http://localhost:8000")
        self.api_key = kwargs.pop("api_key", None)
        self.timeout = kwargs.pop("timeout", 120.0)
        self._client = DistLLMClientSync(
            base_url=self.base_url,
            api_key=self.api_key or None,
            timeout=self.timeout,
        )

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        try:
            import httpx
            resp = httpx.post(
                f"{self._client.base_url}/knowledge/{self.source_id}/query",
                json={"query": query_text, "top_k": top_k},
                headers={"Content-Type": "application/json"},
                timeout=self._client._timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("results", [])
            return []
        except Exception:
            return []
