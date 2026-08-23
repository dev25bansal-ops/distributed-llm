"""Base tool provider shared across LangChain, LlamaIndex, and CrewAI integrations."""

import json
import logging
import time
from typing import Any, Callable, Optional, TypeVar

from distllm.sdk import DistLLMClientSync

logger = logging.getLogger("distllm")

T = TypeVar("T")


def _retry(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs: Any,
) -> T:
    """Retry a callable with exponential backoff.

    Retries on any exception up to ``max_retries`` times.  Delay between
    attempts doubles each time, capped at ``max_delay`` seconds.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "Attempt %d/%d failed (%s), retrying in %.1fs…",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


class BaseToolProvider:
    """Shared tool discovery and API-calling logic for all framework integrations.

    Subclasses override ``get_tools()`` to wrap ``_discover_tools()`` results
    in framework-native tool objects (StructuredTool, FunctionTool, BaseTool, etc.).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._client = DistLLMClientSync(
            base_url=base_url,
            api_key=api_key or None,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def discover_tools(self) -> list[dict]:
        """Discover tools from the DistLLM API, falling back to defaults."""
        try:
            import httpx

            resp = _retry(
                httpx.get,
                f"{self.base_url}/v1/tools",
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.warning("Failed to discover tools from API, using defaults: %s", e)
        return self.default_tools()

    def discover_tools_from_openapi(self) -> list[dict]:
        """Discover tools by parsing the OpenAPI spec (LangChain-style)."""
        try:
            import httpx

            resp = _retry(
                httpx.get,
                f"{self.base_url}/openapi.json",
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                spec = resp.json()
                return self._parse_openapi_tools(spec)
        except Exception as e:
            logger.warning(
                "Failed to discover tools from OpenAPI spec, using defaults: %s", e
            )
        return self.default_tools()

    @staticmethod
    def default_tools() -> list[dict]:
        """Return the built-in default tool definitions."""
        return [
            {
                "name": "distllm_chat",
                "description": "Generate a chat completion. Input: messages (list of {role, content})",
            },
            {
                "name": "distllm_complete",
                "description": "Generate a text completion. Input: prompt (string)",
            },
            {
                "name": "distllm_embed",
                "description": "Generate embeddings. Input: input (string or list of strings)",
            },
        ]

    # ------------------------------------------------------------------
    # API calling
    # ------------------------------------------------------------------

    def call_tool(self, name: str, **kwargs: Any) -> str:
        """Call a tool on the DistLLM API and return the JSON-encoded result."""
        try:
            import httpx

            payload = {"tool": name, "parameters": kwargs}

            def _do_post():
                return httpx.post(
                    f"{self.base_url}/v1/tools/{name}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )

            resp = _retry(_do_post)
            if resp.status_code == 200:
                data = resp.json()
                return json.dumps(data.get("result", data))
            return json.dumps({"error": f"Tool call failed: HTTP {resp.status_code}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def make_callable(self, tool_def: dict) -> callable:
        """Create a callable closure for a tool definition."""
        name = tool_def.get("name", "unknown")

        def _call(**kwargs: Any) -> str:
            return self.call_tool(name, **kwargs)

        _call.__name__ = name
        _call.__doc__ = tool_def.get("description", "")
        return _call

    # ------------------------------------------------------------------
    # OpenAPI parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_openapi_tools(spec: dict) -> list[dict]:
        tools = []
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                tags = details.get("tags", [])
                if "tools" in tags or (
                    method.lower() == "post" and "/tools" in path
                ):
                    tools.append(
                        {
                            "name": details.get(
                                "operationId", path.strip("/").replace("/", "_")
                            ),
                            "description": details.get("description")
                            or details.get("summary", ""),
                            "parameters": details.get("parameters", []),
                        }
                    )
        return tools
