"""Multi-model router for DistLLM.

Routes logical task names to specific models on the DistLLM cluster.
Each integration can use this to serve different models for different
use-cases (code, chat, embeddings, etc.) through a single entry point.

Usage::

    from _common.model_router import DistLLMModelRouter

    router = DistLLMModelRouter(base_url="http://localhost:8000")
    router.route("code", model="deepseek-coder-33b")
    router.route("chat", model="llama-70b")
    router.route("embed", model="bge-large")

    # Use in LangChain
    from distllm_langchain import DistLLMChat
    llm = DistLLMChat(base_url="http://localhost:8000", model=router.model_for("chat"))
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("distllm")


class DistLLMModelRouter:
    """Route logical task names to model identifiers.

    Parameters
    ----------
    base_url : str
        DistLLM coordinator URL.
    default_model : str
        Fallback model when no route matches.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        default_model: str = "distributed-llm",
    ):
        self.base_url = base_url
        self.default_model = default_model
        self._routes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Route management
    # ------------------------------------------------------------------

    def route(self, task: str, model: str) -> None:
        """Register a model for a given task name."""
        self._routes[task] = model
        logger.info("Routed task %r -> model %r", task, model)

    def unroute(self, task: str) -> None:
        """Remove a route."""
        self._routes.pop(task, None)

    def model_for(self, task: str) -> str:
        """Return the model for *task*, falling back to ``default_model``."""
        return self._routes.get(task, self.default_model)

    def list_routes(self) -> dict[str, str]:
        """Return a copy of all registered routes."""
        return dict(self._routes)

    # ------------------------------------------------------------------
    # Dynamic discovery
    # ------------------------------------------------------------------

    def discover_models(self) -> list[dict[str, Any]]:
        """Fetch available models from the DistLLM API."""
        try:
            import httpx

            resp = httpx.get(
                f"{self.base_url}/v1/models",
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.warning("Failed to discover models: %s", e)
        return []

    def auto_route(
        self,
        task: str,
        *,
        prefer_tags: Optional[list[str]] = None,
        min_context_window: int = 0,
    ) -> Optional[str]:
        """Auto-select a model for *task* based on available models' metadata.

        Parameters
        ----------
        prefer_tags : list[str], optional
            Tags to prefer (e.g. ``["code"]``, ``["instruct"]``).
        min_context_window : int
            Minimum required context window.

        Returns
        -------
        str or None
            The selected model id, or ``None`` if no suitable model found.
        """
        models = self.discover_models()
        candidates = []
        for m in models:
            if m.get("context_window", 0) < min_context_window:
                continue
            tags = m.get("tags", [])
            if prefer_tags and not any(t in tags for t in prefer_tags):
                continue
            candidates.append(m)

        if not candidates:
            logger.warning("No suitable model found for task %r", task)
            return None

        # Pick the first candidate (could be smarter: sort by context, cost, etc.)
        selected = candidates[0]["id"]
        self.route(task, selected)
        return selected

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def langchain_chat(self, task: str, **kwargs: Any) -> Any:
        """Return a ``DistLLMChat`` for the given task."""
        from distllm_langchain import DistLLMChat

        return DistLLMChat(model=self.model_for(task), base_url=self.base_url, **kwargs)

    def llamaindex_llm(self, task: str, **kwargs: Any) -> Any:
        """Return a LlamaIndex ``DistLLM`` for the given task."""
        from distllm_llamaindex import DistLLM

        return DistLLM(model=self.model_for(task), base_url=self.base_url, **kwargs)

    def __repr__(self) -> str:
        routes = ", ".join(f"{k}={v}" for k, v in self._routes.items())
        return f"DistLLMModelRouter(default={self.default_model!r}, routes=[{routes}])"
