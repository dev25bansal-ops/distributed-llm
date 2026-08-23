from typing import Any, Optional

try:
    from distllm.integrations._common.base_tool_provider import BaseToolProvider
except ImportError:
    from _common.base_tool_provider import BaseToolProvider


class DistLLMToolProvider(BaseToolProvider):
    """CrewAI tool provider backed by the DistLLM API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(base_url=base_url, api_key=api_key, timeout=timeout)

    def get_tools(self) -> list[Any]:
        tools: list[Any] = []
        try:
            from crewai.tools import BaseTool
            from pydantic import BaseModel, Field
        except ImportError:
            return tools

        registered = self.discover_tools()
        for tool_def in registered:
            tool = self._build_crew_tool(tool_def, BaseTool)
            if tool:
                tools.append(tool)
        return tools

    def _build_crew_tool(self, tool_def: dict, BaseTool) -> Any:
        name = tool_def.get("name", "unknown")
        desc = tool_def.get("description", "")
        provider = self

        class _DynamicTool(BaseTool):
            name: str = name
            description: str = desc

            def _run(self_inner, **kwargs: Any) -> str:
                return provider.call_tool(name, **kwargs)

        return _DynamicTool()
