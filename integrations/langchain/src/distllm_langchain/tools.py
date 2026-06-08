from typing import Any, Optional

from _common.base_tool_provider import BaseToolProvider


class DistLLMToolProvider(BaseToolProvider):
    """LangChain tool provider backed by the DistLLM API."""

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
            from langchain_core.tools import StructuredTool
            from pydantic import create_model
        except ImportError:
            return tools

        registered = self.discover_tools_from_openapi()
        for tool_def in registered:
            params = tool_def.get("parameters", {})
            schema = self._build_args_schema(
                tool_def.get("name", "unknown"), params, create_model
            )
            tool = StructuredTool.from_function(
                name=tool_def.get("name", "unknown"),
                description=tool_def.get("description", ""),
                func=self.make_callable(tool_def),
                args_schema=schema,
            )
            tools.append(tool)
        return tools

    @staticmethod
    def _build_args_schema(name: str, params: list, create_model) -> Any:
        fields = {}
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for p in params:
            p_name = p.get("name", "arg")
            p_type = p.get("schema", {}).get("type", "string")
            p_required = p.get("required", True)
            python_type = type_map.get(p_type, str)
            if p_required:
                fields[p_name] = (python_type, ...)
            else:
                fields[p_name] = (Optional[python_type], None)
        if not fields:
            fields["query"] = (str, "Default query parameter")
        return create_model(f"{name}Args", **fields)
