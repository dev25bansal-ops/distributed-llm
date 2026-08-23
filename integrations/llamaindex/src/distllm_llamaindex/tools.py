from typing import Any, Optional

try:
    from distllm.integrations._common.base_tool_provider import BaseToolProvider
except ImportError:
    from _common.base_tool_provider import BaseToolProvider


class DistLLMToolProvider(BaseToolProvider):
    """LlamaIndex tool provider backed by the DistLLM API."""

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
            from llama_index.core.tools import FunctionTool
        except ImportError:
            return tools

        registered = self.discover_tools()
        for tool_def in registered:
            tool = FunctionTool.from_defaults(
                fn=self.make_callable(tool_def),
                name=tool_def.get("name", "unknown"),
                description=tool_def.get("description", ""),
            )
            tools.append(tool)
        return tools
