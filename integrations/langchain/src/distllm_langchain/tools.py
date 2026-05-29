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
            from langchain_core.tools import StructuredTool
            from pydantic import create_model
        except ImportError:
            return tools

        registered = self._discover_tools()
        for tool_def in registered:
            params = tool_def.get("parameters", {})
            schema = self._build_args_schema(tool_def.get("name", "unknown"), params, create_model)
            tool = StructuredTool.from_function(
                name=tool_def.get("name", "unknown"),
                description=tool_def.get("description", ""),
                func=self._make_api_call(tool_def),
                args_schema=schema,
            )
            tools.append(tool)
        return tools

    def _discover_tools(self) -> list[dict]:
        try:
            import httpx
            resp = httpx.get(
                f"{self._client.base_url}/openapi.json",
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                spec = resp.json()
                return self._parse_openapi_tools(spec)
        except Exception:
            pass
        return self._default_tools()

    def _parse_openapi_tools(self, spec: dict) -> list[dict]:
        tools = []
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                tags = details.get("tags", [])
                if "tools" in tags or method.lower() == "post" and "/tools" in path:
                    tools.append({
                        "name": details.get("operationId", path.strip("/").replace("/", "_")),
                        "description": details.get("description") or details.get("summary", ""),
                        "parameters": details.get("parameters", []),
                    })
        return tools

    @staticmethod
    def _build_args_schema(name: str, params: list, create_model) -> Any:
        fields = {}
        for p in params:
            p_name = p.get("name", "arg")
            p_type = p.get("schema", {}).get("type", "string")
            type_map = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}
            p_required = p.get("required", True)
            python_type = type_map.get(p_type, str)
            if p_required:
                fields[p_name] = (python_type, ...)
            else:
                fields[p_name] = (Optional[python_type], None)
        if not fields:
            fields["query"] = (str, "Default query parameter")
        return create_model(f"{name}Args", **fields)

    def _default_tools(self) -> list[dict]:
        return [
            {"name": "distllm_chat", "description": "Generate a chat completion. Input: messages (list of {role, content})", "parameters": []},
            {"name": "distllm_complete", "description": "Generate a text completion. Input: prompt (string)", "parameters": []},
            {"name": "distllm_embed", "description": "Generate embeddings. Input: input (string or list of strings)", "parameters": []},
        ]

    def _make_api_call(self, tool_def: dict) -> callable:
        name = tool_def.get("name", "unknown")

        def _call(**kwargs: Any) -> str:
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

        _call.__name__ = name
        _call.__doc__ = tool_def.get("description", "")
        return _call
