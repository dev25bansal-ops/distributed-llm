"""Tool calling engine for OpenAI-compatible function/tool calling.

Handles:
- Parsing tool schemas from OpenAI format
- Detecting tool calls in generated text
- Executing tool calls and injecting results
- Parallel tool call support
"""

import json
import re
import uuid
from typing import Any, Callable
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ToolCall:
    """Represents a single tool/function call."""
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_openai_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class ToolResult:
    """Result from executing a tool call."""
    tool_call_id: str
    content: str
    role: str = "tool"

    def to_message_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }


class ToolSchema:
    """Parsed tool/function schema for OpenAI-compatible tool calling."""

    def __init__(self, tool_def: dict):
        """Parse an OpenAI-style tool definition.

        Args:
            tool_def: Dict with "type", "function" keys (OpenAI format).
        """
        self.type = tool_def.get("type", "function")
        func_def = tool_def.get("function", {})
        self.name = func_def.get("name", "")
        self.description = func_def.get("description", "")
        self.parameters = func_def.get("parameters", {})
        self.required = self.parameters.get("required", [])

    def to_prompt_text(self) -> str:
        """Convert schema to text for inclusion in the system prompt."""
        params_str = json.dumps(self.parameters, indent=2) if self.parameters else "{}"
        required_str = ", ".join(self.required) if self.required else "none"
        return (
            f"Function: {self.name}\n"
            f"Description: {self.description}\n"
            f"Parameters: {params_str}\n"
            f"Required: {required_str}\n"
        )

    def __repr__(self):
        return f"ToolSchema(name='{self.name}')"


class ToolCallingEngine:
    """Engine for OpenAI-compatible tool/function calling.

    Workflow:
    1. Parse tool schemas from request
    2. Build tool-augmented prompt
    3. Generate text (model produces tool call JSON)
    4. Parse tool calls from generated text
    5. Execute tools (if handlers provided)
    6. Inject results into conversation
    7. Continue generation if needed
    """

    # Regex patterns for extracting tool calls from generated text
    TOOL_CALL_PATTERNS = [
        # Pattern 1: JSON array of tool calls
        re.compile(r'\[\s*\{.*?"name"\s*:\s*".*?".*?\}\s*\]', re.DOTALL),
        # Pattern 2: Single JSON object with name field
        re.compile(r'\{\s*"name"\s*:\s*".*?"[\s\S]*?"arguments"\s*:\s*\{[\s\S]*?\}\s*\}', re.DOTALL),
        # Pattern 3:  name: "func_name"  args: {...}
        re.compile(r'(?i)name[:\s]+["\']([^"\']+)["\'][\s\S]*?(?:arguments|args)[\s:]+(\{[\s\S]*?\})', re.DOTALL),
    ]

    def __init__(self):
        self._tool_handlers: dict[str, Callable] = {}

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a tool handler function.

        Args:
            name: Tool name (must match schema name).
            handler: Callable that takes kwargs and returns a result.
        """
        self._tool_handlers[name] = handler

    def parse_schemas(self, tools: list[dict]) -> list[ToolSchema]:
        """Parse OpenAI-style tool definitions into ToolSchema objects.

        Args:
            tools: List of tool dicts with "type" and "function" keys.

        Returns:
            List of parsed ToolSchema objects.
        """
        schemas = []
        for tool_def in tools:
            try:
                schema = ToolSchema(tool_def)
                if schema.name:
                    schemas.append(schema)
            except Exception as e:
                logger.warning(f"Failed to parse tool schema: {e}")
        return schemas

    def build_tool_prompt(
        self,
        schemas: list[ToolSchema],
        messages: list[dict],
        tool_choice: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Build a tool-augmented prompt from messages and tool schemas.

        Adds a system message describing available tools, then formats
        the conversation history.

        Args:
            schemas: Parsed tool schemas.
            messages: Original conversation messages.
            tool_choice: "none", "auto", "required", or specific tool name.

        Returns:
            (prompt_text, tool_choice_directive)
        """
        # Build tool description
        tool_text = "You have access to the following tools:\n\n"
        for schema in schemas:
            tool_text += schema.to_prompt_text() + "\n"

        # Add tool usage instructions based on tool_choice
        if tool_choice == "none":
            tool_text += "\nDo NOT call any tools. Respond directly."
        elif tool_choice == "required":
            tool_text += "\nYou MUST call a tool. Do NOT respond without a tool call."
        elif tool_choice and tool_choice != "auto":
            tool_text += f"\nYou MUST call the tool '{tool_choice}'."
        else:
            tool_text += (
                "\nTo call a tool, respond with a JSON object:\n"
                '{\n  "name": "function_name",\n  "arguments": {"arg1": "value1"}\n}\n'
                "For multiple parallel calls, respond with a JSON array of such objects.\n"
                "If no tool call is needed, respond normally."
            )

        # Prepend tool description as system message
        prompt_messages = [{"role": "system", "content": tool_text}]
        prompt_messages.extend(messages)

        # Build plain text prompt
        prompt = "\n".join([
            f"{msg['role']}: {msg.get('content', '')}"
            for msg in prompt_messages
            if msg.get("content")
        ])

        return prompt, tool_choice

    def extract_tool_calls(self, generated_text: str) -> list[ToolCall]:
        """Extract tool calls from generated text.

        Supports:
        - JSON array of tool calls (parallel)
        - Single JSON object (single call)
        - Text-based function call format

        Args:
            generated_text: Raw text from model generation.

        Returns:
            List of extracted ToolCall objects.
        """
        # Try to find JSON content in the text
        text = generated_text.strip()

        # Find JSON boundaries
        json_candidates = []

        # Try full text as JSON
        try:
            data = json.loads(text)
            json_candidates.append(data)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            for pattern in self.TOOL_CALL_PATTERNS:
                matches = pattern.findall(text)
                for match in matches:
                    try:
                        if isinstance(match, tuple):
                            # Pattern 3: (name, args_str)
                            name, args_str = match
                            try:
                                args = json.loads(args_str)
                            except json.JSONDecodeError:
                                # Try to fix common JSON issues
                                args_str = args_str.replace("'", '"')
                                args = json.loads(args_str)
                            json_candidates.append({"name": name, "arguments": args})
                        else:
                            data = json.loads(match)
                            json_candidates.append(data)
                    except json.JSONDecodeError:
                        continue

        if not json_candidates:
            return []

        # Parse candidates into ToolCall objects
        tool_calls = []
        for candidate in json_candidates:
            if isinstance(candidate, list):
                # Array of tool calls
                for item in candidate:
                    tc = self._parse_tool_call_dict(item)
                    if tc:
                        tool_calls.append(tc)
            elif isinstance(candidate, dict):
                # Single tool call
                tc = self._parse_tool_call_dict(candidate)
                if tc:
                    tool_calls.append(tc)

        return tool_calls

    def _parse_tool_call_dict(self, data: dict) -> ToolCall | None:
        """Parse a single tool call from a dict."""
        # Handle different field naming conventions
        name = data.get("name") or data.get("function_name") or data.get("tool_name")
        if not name:
            return None

        # Arguments might be in various fields
        args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=args if isinstance(args, dict) else {},
        )

    def execute_tool_calls(
        self,
        tool_calls: list[ToolCall],
        handlers: dict[str, Callable] | None = None,
    ) -> list[ToolResult]:
        """Execute tool calls and return results.

        Args:
            tool_calls: List of tool calls to execute.
            handlers: Optional override dict of name -> handler.

        Returns:
            List of ToolResult objects.
        """
        active_handlers = handlers or self._tool_handlers
        results = []

        for tc in tool_calls:
            handler = active_handlers.get(tc.name)
            if handler is None:
                # No handler registered; simulate execution
                logger.warning(f"No handler for tool '{tc.name}', simulating")
                result = ToolResult(
                    tool_call_id=tc.id,
                    content=f"Error: Tool '{tc.name}' not found",
                )
            else:
                try:
                    output = handler(**tc.arguments)
                    result = ToolResult(
                        tool_call_id=tc.id,
                        content=str(output) if output is not None else "",
                    )
                except Exception as e:
                    logger.error(f"Tool '{tc.name}' execution failed: {e}")
                    result = ToolResult(
                        tool_call_id=tc.id,
                        content=f"Error: {str(e)}",
                    )
            results.append(result)

        return results

    def inject_tool_results(
        self,
        messages: list[dict],
        assistant_content: str,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
    ) -> list[dict]:
        """Inject tool results into the conversation for continuation.

        Args:
            messages: Original messages list.
            assistant_content: The assistant's generated text (with tool calls).
            tool_calls: The parsed tool calls.
            tool_results: Results from executing the tool calls.

        Returns:
            Updated messages list with tool calls and results.
        """
        new_messages = list(messages)

        # Add assistant message with tool_calls field
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [tc.to_openai_dict() for tc in tool_calls],
        }
        new_messages.append(assistant_msg)

        # Add tool result messages
        for result in tool_results:
            new_messages.append(result.to_message_dict())

        return new_messages

    def should_continue_after_tool_calls(
        self,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
    ) -> bool:
        """Determine if generation should continue after tool calls.

        Continues if any tool returned an error or if there are results
        that need to be incorporated into the final response.

        Args:
            tool_calls: Executed tool calls.
            tool_results: Results from execution.

        Returns:
            True if another generation pass is needed.
        """
        # Continue if there are any results to incorporate
        return len(tool_results) > 0

    def has_tool_calls(self, generated_text: str) -> bool:
        """Check if the generated text contains tool calls."""
        return len(self.extract_tool_calls(generated_text)) > 0

    def enforce_tool_choice(
        self,
        tool_choice: str | None,
        tool_calls: list[ToolCall],
    ) -> list[ToolCall]:
        """Enforce tool_choice constraint on extracted tool calls.

        Args:
            tool_choice: "none", "auto", "required", or specific name.
            tool_calls: Extracted tool calls.

        Returns:
            Filtered list of tool calls.
        """
        if tool_choice == "none":
            return []
        elif tool_choice and tool_choice not in ("auto", "required"):
            # Only allow the specified tool
            return [tc for tc in tool_calls if tc.name == tool_choice]
        return tool_calls
