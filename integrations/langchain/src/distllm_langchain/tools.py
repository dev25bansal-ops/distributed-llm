"""DistLLM Tool provider — LangChain Tool integration.

Provides DistLLM as a ``Tool`` provider for use with LangChain agents.
Exposes the API server's available tools and allows custom tool definitions
to be executed by the DistLLM backend.
"""

from typing import Any, Optional

from distllm.sdk import DistLLMClientSync


class DistLLMToolProvider:
    """Tool provider for the DistLLM API.

    Discovers available tools from the DistLLM API server and creates
    LangChain-compatible ``Tool`` instances.

    Usage::

        from distllm_langchain import DistLLMToolProvider

        provider = DistLLMToolProvider(base_url="http://localhost:8000")
        tools = provider.get_tools()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self._client = DistLLMClientSync(
            base_url=base_url,
            api_key=api_key or None,
            timeout=timeout,
        )

    def get_tools(self) -> list[Any]:
        """Discover and return LangChain-compatible tools from the API.

        Returns:
            List of ``StructuredTool`` instances.
        """
        tools: list[Any] = []
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            return tools

        registered = self._list_api_tools()
        for tool_def in registered:
            tool = StructuredTool.from_function(
                name=tool_def.get("name", "unknown"),
                description=tool_def.get("description", ""),
                func=self._make_api_call(tool_def),
                args_schema=None,
            )
            tools.append(tool)
        return tools

    def _list_api_tools(self) -> list[dict]:
        """Query the DistLLM API for available tools.

        Returns:
            List of tool definition dicts with ``name`` and ``description``.
        """
        try:
            import httpx
            resp = httpx.get(
                f"{self._client.base_url}/v1/tools",
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception:
            pass
        return self._default_tools()

    def _default_tools(self) -> list[dict]:
        """Return default tool definitions when the API doesn't have a tools endpoint."""
        return [
            {
                "name": "distllm_chat",
                "description": "Generate a chat completion using DistLLM. "
                               "Input is a JSON object with 'messages' (list of {role, content} dicts).",
            },
            {
                "name": "distllm_complete",
                "description": "Generate a text completion using DistLLM. "
                               "Input is a JSON object with 'prompt' (string).",
            },
            {
                "name": "distllm_embed",
                "description": "Generate embeddings for text input. "
                               "Input is a JSON object with 'input' (string or list of strings).",
            },
        ]

    def _make_api_call(self, tool_def: dict) -> callable:
        """Create a callable that invokes the tool via the DistLLM API.

        Args:
            tool_def: Tool definition dict.

        Returns:
            Callable that takes kwargs and returns the API response.
        """
        name = tool_def.get("name", "unknown")

        def _call(**kwargs: Any) -> str:
            import json
            try:
                import httpx
                payload = {
                    "tool": name,
                    "parameters": kwargs,
                }
                resp = httpx.post(
                    f"{self._client.base_url}/v1/tools/{name}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self._client._timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return json.dumps(data.get("result", data))
                return json.dumps({"error": f"Tool call failed: HTTP {resp.status_code}"})
            except Exception as e:
                return json.dumps({"error": str(e)})

        _call.__name__ = name
        _call.__doc__ = tool_def.get("description", "")
        return _call
