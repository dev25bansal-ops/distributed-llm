"""DistLLM tool provider for Agno."""

from __future__ import annotations

import json
from typing import Any, Optional


class DistLLMToolProvider:
    """Agno-compatible tool provider backed by the DistLLM API.

    Discovers tools from the DistLLM coordinator and wraps them for
    use with Agno agents.

    Graceful degradation: if ``agno`` is not installed, tool discovery
    still works but wrapping into Agno tools is skipped.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def discover_tools(self) -> list[dict]:
        """Discover tools from the DistLLM API, falling back to defaults."""
        try:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = httpx.get(
                f"{self.base_url}/v1/tools",
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception:
            pass
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
    # Tool calling
    # ------------------------------------------------------------------

    def call_tool(self, name: str, **kwargs: Any) -> str:
        """Call a tool on the DistLLM API and return the JSON-encoded result."""
        try:
            import httpx

            payload = {"tool": name, "parameters": kwargs}
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = httpx.post(
                f"{self.base_url}/v1/tools/{name}",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return json.dumps(data.get("result", data))
            return json.dumps({"error": f"Tool call failed: HTTP {resp.status_code}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # Agno tool functions
    # ------------------------------------------------------------------

    def get_agno_tools(self) -> list[Any]:
        """Return a list of Agno ``Tool`` instances for the registered tools."""
        try:
            from agno.tools import Tool
        except ImportError:
            return []

        registered = self.discover_tools()
        agno_tools: list[Any] = []
        for tool_def in registered:
            name = tool_def.get("name", "unknown")
            desc = tool_def.get("description", "")
            provider = self

            def _make_fn(tool_name: str) -> Any:
                def _run(**fwargs: Any) -> str:
                    return provider.call_tool(tool_name, **fwargs)

                _run.__name__ = tool_name
                _run.__doc__ = desc
                return _run

            agno_tools.append(
                Tool(
                    name=name,
                    description=desc,
                    func=_make_fn(name),
                )
            )
        return agno_tools
