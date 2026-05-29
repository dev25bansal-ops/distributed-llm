from typing import Any, Optional

from distllm.sdk import DistLLMClientSync


class DistLLMToolProvider:
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
        tools: list[Any] = []
        try:
            from crewai.tools import BaseTool
            from pydantic import BaseModel, Field
        except ImportError:
            return tools

        registered = self._list_api_tools()
        for tool_def in registered:
            tool = self._build_crew_tool(tool_def, BaseTool, BaseModel, Field)
            if tool:
                tools.append(tool)
        return tools

    def _build_crew_tool(self, tool_def: dict, BaseTool, BaseModel, Field) -> Any:
        name = tool_def.get("name", "unknown")
        desc = tool_def.get("description", "")

        class _DynamicTool(BaseTool):
            name: str = name
            description: str = desc

            def _run(self, **kwargs: Any) -> str:
                import json
                try:
                    import httpx
                    payload = {"tool": name, "parameters": kwargs}
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

        return _DynamicTool(_client=self._client)

    def _list_api_tools(self) -> list[dict]:
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
        return [
            {"name": "distllm_chat", "description": "Generate a chat completion using DistLLM"},
            {"name": "distllm_complete", "description": "Generate a text completion using DistLLM"},
            {"name": "distllm_embed", "description": "Generate embeddings for text input"},
        ]
