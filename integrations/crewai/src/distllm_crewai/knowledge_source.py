import logging
from typing import Any, Optional

from distllm.sdk import DistLLMClientSync

logger = logging.getLogger("distllm_crewai")


class DistLLMKnowledgeSource:
    """CrewAI-compatible knowledge source backed by a DistLLM cluster.

    Implements the CrewAI ``KnowledgeSource`` protocol (``load_content``,
    ``query``).
    """

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
        self._content: list[dict] = []

    # ------------------------------------------------------------------
    # CrewAI KnowledgeSource protocol
    # ------------------------------------------------------------------

    def load_content(self) -> None:
        """Load content from the DistLLM knowledge store.

        CrewAI calls this before agents start working. Results are cached
        in ``self._content`` for subsequent ``query`` calls.
        """
        try:
            import httpx

            resp = httpx.get(
                f"{self.base_url}/knowledge/{self.source_id}/documents",
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                self._content = resp.json().get("documents", [])
            else:
                logger.warning(
                    "Knowledge load returned HTTP %d for source %s",
                    resp.status_code,
                    self.source_id,
                )
        except Exception as e:
            logger.warning("Failed to load knowledge source %s: %s", self.source_id, e)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Query the knowledge source for relevant documents."""
        try:
            import httpx

            resp = httpx.post(
                f"{self.base_url}/knowledge/{self.source_id}/query",
                json={"query": query_text, "top_k": top_k},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("results", [])
            logger.warning(
                "Knowledge query returned HTTP %d for source %s",
                resp.status_code,
                self.source_id,
            )
            return []
        except Exception as e:
            logger.warning(
                "Knowledge query failed for source %s: %s", self.source_id, e
            )
            return []

    @property
    def content(self) -> list[dict]:
        """Return cached content (populated by ``load_content``)."""
        return self._content
