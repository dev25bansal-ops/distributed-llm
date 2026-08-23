"""Tool-calling engine that parses and executes OpenAI-format tool calls.

Supports two formats:
1. JSON block: ``{"tool_calls": [...]}``
2. XML-style: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``

Tool execution:
- Register callable handlers via ``register_tool(name, func)``
- ``execute_tool_calls()`` dispatches to registered handlers
- Unregistered tools return a diagnostic message
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any


class ToolCallingEngine:
    """Parse, validate, execute, and inject tool calls for the chat endpoint.

    Extracted from ``routes/chat.py`` so it can be imported by both
    ``chat.py`` and ``streaming.py`` without circular dependencies.
    """

    def __init__(self):
        self._registered_tools: dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a callable handler for a tool name."""
        self._registered_tools[name] = func

    def register_tools_from_schemas(self, tools: list[dict]) -> None:
        """Register tools from OpenAI-format tool schemas.

        Only registers tools that have a 'handler' key in their metadata.
        """
        for tool in tools:
            func_def = tool.get("function", tool)
            name = func_def.get("name", "")
            handler = tool.get("handler") or func_def.get("handler")
            if handler and callable(handler):
                self._registered_tools[name] = handler

    def parse_schemas(self, tools):
        return list(tools) if tools else []

    def build_tool_prompt(self, schemas, messages, tool_choice="auto"):
        """Build a system prompt that includes tool descriptions."""
        prompts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompts.append(f"{role}: {content}")

        tool_descriptions = []
        for tool in schemas:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            tool_descriptions.append(f"- {name}: {desc}\n  Parameters: {json.dumps(params)}")

        tool_section = "Available tools:\n" + "\n".join(tool_descriptions)
        tool_section += (
            '\n\nTo call a tool, respond with: '
            '{"tool_calls": [{"id": "call_<id>", "type": "function", '
            '"function": {"name": "<name>", "arguments": "<json>"}}]}'
        )
        return "\n".join(prompts) + "\n\n" + tool_section, None

    def has_tool_calls(self, text: str | None) -> bool:
        if not text:
            return False
        if '"tool_calls"' in text and '"function"' in text:
            return True
        if "<tool_call>" in text and "</tool_call>" in text:
            return True
        return False

    def extract_tool_calls(self, text: str) -> list[dict]:
        """Extract tool calls from assistant response text."""
        if not text:
            return []
        calls: list[dict] = []
        try:

            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            for match in re.finditer(json_pattern, text):
                try:
                    obj = json.loads(match.group())
                    raw_calls = obj.get("tool_calls", [])
                    if not raw_calls:
                        continue
                    for tc in raw_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        args_str = func.get("arguments", "{}")
                        try:
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except json.JSONDecodeError:
                            args = {"raw": args_str}
                        calls.append({
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        })
                    break
                except json.JSONDecodeError:
                    continue
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

        if not calls:

            xml_pattern = r'<tool_call>(.*?)</tool_call>'
            matches = re.findall(xml_pattern, text, re.DOTALL)
            for match in matches:
                try:
                    obj = json.loads(match.strip())
                    name = obj.get("name", "")
                    args = obj.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    calls.append({
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                except (json.JSONDecodeError, ValueError):
                    continue

        return calls

    def enforce_tool_choice(self, choice, calls, *, parallel: bool = True):
        """Filter *calls* by *choice* and, when *parallel* is False, return
        at most one call to force sequential tool execution.

        OpenAI's ``parallel_tool_calls`` parameter controls whether the model
        may issue multiple tool calls in a single response turn.
        """
        if not calls:
            return calls
        if choice == "none":
            return []
        if choice == "required":
            return calls if parallel else calls[:1]
        if isinstance(choice, dict) and choice.get("type") == "function":
            func_name = choice.get("function", {}).get("name", "")
            filtered = [c for c in calls if c.get("function", {}).get("name") == func_name]
            return filtered if parallel else filtered[:1]
        return calls if parallel else calls[:1]

    def execute_tool_calls(self, calls, timeout_s: float = 30.0):
        """Execute tool calls using registered handlers."""
        import concurrent.futures as _cf

        results = []
        for call in calls:
            func = call.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
            tool_call_id = call.get("id", f"call_{uuid.uuid4().hex[:24]}")

            handler = self._registered_tools.get(name)
            if handler:
                try:
                    if isinstance(args, dict):
                        invoke = lambda: handler(**args)
                    elif isinstance(args, str):
                        parsed_args = json.loads(args)
                        invoke = lambda: handler(**parsed_args)
                    else:
                        invoke = lambda: handler(args)

                    with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                        fut = _pool.submit(invoke)
                        result = fut.result(timeout=timeout_s)

                    results.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": str(result),
                    })
                except _cf.TimeoutError:
                    results.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": f"Error executing tool '{name}': timed out after {timeout_s}s",
                    })
                except Exception as e:
                    results.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": f"Error executing tool '{name}': {e}",
                    })
            else:
                results.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": f"Tool '{name}' is not registered. Available tools: {list(self._registered_tools.keys())}",
                })
        return results

    def should_continue_after_tool_calls(self, calls, results) -> bool:
        return bool(results) and any(r.get("content") for r in results)

    def inject_tool_results(self, messages, result, calls, results):
        """Build updated message list with assistant tool_calls + tool results."""
        new_messages = list(messages)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": result or None}
        if result:
            assistant_msg["content"] = result
        if calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                }
                for i, tc in enumerate(calls)
            ]
        new_messages.append(assistant_msg)
        for i, tc in enumerate(calls):
            tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
            result_content = results[i].get("content", str(results)) if i < len(results) else str(results)
            new_messages.append({
                "tool_call_id": tc_id,
                "role": "tool",
                "content": result_content,
            })
        return new_messages
