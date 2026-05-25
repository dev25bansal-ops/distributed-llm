"""Unit tests for ToolCallingEngine: parameter parsing, execution, error handling.

Tests ToolCallingEngine directly with mock handlers, no API setup.
"""

import json
from distllm.core.tool_engine import (
    ToolCallingEngine,
    ToolSchema,
    ToolCall,
    ToolResult,
)


# ─── Tool Schema Definitions (OpenAI format) ─────────────────────────────

TOOL_GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

TOOL_CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform arithmetic",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]},
            },
            "required": ["a", "b", "op"],
        },
    },
}

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}


# ─── ToolSchema tests (parameter parsing) ────────────────────────────────


class TestToolSchema:
    """ToolSchema: function definition → parsed schema."""

    def test_parse_name(self):
        schema = ToolSchema(TOOL_GET_WEATHER)
        assert schema.name == "get_weather"

    def test_parse_description(self):
        schema = ToolSchema(TOOL_GET_WEATHER)
        assert "weather" in schema.description

    def test_parse_parameters(self):
        schema = ToolSchema(TOOL_GET_WEATHER)
        assert "properties" in schema.parameters
        assert schema.parameters["properties"]["location"]["type"] == "string"

    def test_parse_required(self):
        schema = ToolSchema(TOOL_GET_WEATHER)
        assert schema.required == ["location"]

    def test_parse_multiple_required(self):
        schema = ToolSchema(TOOL_CALCULATOR)
        assert "a" in schema.required
        assert "b" in schema.required
        assert "op" in schema.required

    def test_to_prompt_text(self):
        schema = ToolSchema(TOOL_GET_WEATHER)
        text = schema.to_prompt_text()
        assert "get_weather" in text
        assert "location" in text
        assert "Required: location" in text


# ─── Parameter parsing: ToolCallingEngine.parse_schemas ──────────────────


class TestParseSchemas:
    """parse_schemas: list of tool defs → list of ToolSchema."""

    def test_parse_single_tool(self):
        engine = ToolCallingEngine()
        schemas = engine.parse_schemas([TOOL_GET_WEATHER])
        assert len(schemas) == 1
        assert schemas[0].name == "get_weather"

    def test_parse_multiple_tools(self):
        engine = ToolCallingEngine()
        schemas = engine.parse_schemas([TOOL_GET_WEATHER, TOOL_CALCULATOR])
        assert len(schemas) == 2
        assert schemas[0].name == "get_weather"
        assert schemas[1].name == "calculator"

    def test_parse_empty_list(self):
        engine = ToolCallingEngine()
        schemas = engine.parse_schemas([])
        assert schemas == []

    def test_parse_skips_malformed(self):
        engine = ToolCallingEngine()
        schemas = engine.parse_schemas([{"type": "function", "function": {}}])
        assert len(schemas) == 0


# ─── Parameter parsing: extract_tool_calls ─────────────────────────────


class TestExtractToolCalls:
    """extract_tool_calls: LLM output → parsed ToolCall objects."""

    def test_json_object_format(self):
        engine = ToolCallingEngine()
        text = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
        calls = engine.extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"location": "Tokyo"}

    def test_json_array_format(self):
        engine = ToolCallingEngine()
        text = (
            '[{"name": "get_weather", "arguments": {"location": "Tokyo"}},'
            '{"name": "get_weather", "arguments": {"location": "Osaka"}}]'
        )
        calls = engine.extract_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].arguments["location"] == "Tokyo"
        assert calls[1].arguments["location"] == "Osaka"

    def test_text_format(self):
        engine = ToolCallingEngine()
        text = 'name: "calculator"  arguments: {"a": 2, "b": 3, "op": "add"}'
        calls = engine.extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "calculator"
        assert calls[0].arguments == {"a": 2, "b": 3, "op": "add"}

    def test_no_tool_call_returns_empty(self):
        engine = ToolCallingEngine()
        calls = engine.extract_tool_calls("Just a normal response.")
        assert calls == []

    def test_alternate_field_names(self):
        engine = ToolCallingEngine()
        text = '{"function_name": "search", "args": {"query": "weather"}}'
        calls = engine.extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments == {"query": "weather"}

    def test_arguments_as_string(self):
        engine = ToolCallingEngine()
        text = '{"name": "echo", "arguments": "{\\"msg\\": \\"hello\\"}"}'
        calls = engine.extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments == {"msg": "hello"}

    def test_has_tool_calls_true(self):
        engine = ToolCallingEngine()
        text = '{"name": "get_weather", "arguments": {}}'
        assert engine.has_tool_calls(text) is True

    def test_has_tool_calls_false(self):
        engine = ToolCallingEngine()
        assert engine.has_tool_calls("Hello world") is False


# ─── Execute: tool function → result ───────────────────────────────────


class TestExecuteToolCalls:
    """execute_tool_calls: ToolCall objects → ToolResult objects."""

    def test_execute_with_handler(self):
        engine = ToolCallingEngine()
        engine.register_handler("get_weather", lambda location: {"temp": 25})
        calls = [ToolCall(id="call_1", name="get_weather", arguments={"location": "Tokyo"})]
        results = engine.execute_tool_calls(calls)
        assert len(results) == 1
        assert "temp" in results[0].content
        assert results[0].tool_call_id == "call_1"
        assert results[0].role == "tool"

    def test_execute_with_handler_overrides(self):
        engine = ToolCallingEngine()
        calls = [ToolCall(id="call_1", name="add", arguments={"x": 1, "y": 2})]
        handlers = {"add": lambda x, y: x + y}
        results = engine.execute_tool_calls(calls, handlers=handlers)
        assert len(results) == 1
        assert results[0].content == "3"

    def test_execute_multiple_calls(self):
        engine = ToolCallingEngine()
        engine.register_handler("mul", lambda a, b: a * b)
        calls = [
            ToolCall(id="c1", name="mul", arguments={"a": 3, "b": 4}),
            ToolCall(id="c2", name="mul", arguments={"a": 5, "b": 6}),
        ]
        results = engine.execute_tool_calls(calls)
        assert len(results) == 2
        assert results[0].content == "12"
        assert results[1].content == "30"

    def test_execute_returns_none(self):
        engine = ToolCallingEngine()
        engine.register_handler("noop", lambda: None)
        calls = [ToolCall(id="c1", name="noop", arguments={})]
        results = engine.execute_tool_calls(calls)
        assert results[0].content == ""


# ─── Built-in tools ────────────────────────────────────────────────────


class TestBuiltinTools:
    """Simulated built-in tools: calculator, search, etc."""

    def test_calculator_add(self):
        engine = ToolCallingEngine()
        engine.register_handler("calculator", lambda a, b, op: {
            "add": a + b,
            "sub": a - b,
            "mul": a * b,
            "div": a / b,
        }.get(op, "unknown"))
        calls = [ToolCall(id="c1", name="calculator", arguments={"a": 10, "b": 5, "op": "add"})]
        results = engine.execute_tool_calls(calls)
        assert results[0].content == "15"
        assert "15" in results[0].content

    def test_calculator_mul(self):
        engine = ToolCallingEngine()
        engine.register_handler("calculator", lambda a, b, op: {
            "add": a + b,
            "sub": a - b,
            "mul": a * b,
            "div": a / b,
        }.get(op, "unknown"))
        calls = [ToolCall(id="c1", name="calculator", arguments={"a": 6, "b": 7, "op": "mul"})]
        results = engine.execute_tool_calls(calls)
        assert results[0].content == "42"

    def test_calculator_div(self):
        engine = ToolCallingEngine()
        engine.register_handler("calculator", lambda a, b, op: {
            "add": a + b,
            "sub": a - b,
            "mul": a * b,
            "div": a / b,
        }.get(op, "unknown"))
        calls = [ToolCall(id="c1", name="calculator", arguments={"a": 10, "b": 2, "op": "div"})]
        results = engine.execute_tool_calls(calls)
        assert results[0].content == "5.0"

    def test_search_returns_result(self):
        engine = ToolCallingEngine()
        engine.register_handler("web_search", lambda query: f"Results for: {query}")
        calls = [ToolCall(id="c1", name="web_search", arguments={"query": "weather in Paris"})]
        results = engine.execute_tool_calls(calls)
        assert "Paris" in results[0].content

    def test_builtin_tools_use_register_handler(self):
        engine = ToolCallingEngine()
        assert engine._tool_handlers == {}
        engine.register_handler("calculator", lambda a, b: a + b)
        assert "calculator" in engine._tool_handlers


# ─── Error handling ─────────────────────────────────────────────────────


class TestToolErrorHandling:
    """Error handling: unknown tool, bad args, handler exception."""

    def test_unknown_tool_returns_error(self):
        engine = ToolCallingEngine()
        calls = [ToolCall(id="c1", name="nonexistent_tool", arguments={})]
        results = engine.execute_tool_calls(calls)
        assert len(results) == 1
        assert "Error" in results[0].content
        assert "nonexistent_tool" in results[0].content

    def test_unknown_tool_not_found_message(self):
        engine = ToolCallingEngine()
        calls = [ToolCall(id="c1", name="unknown_func", arguments={"x": 1})]
        results = engine.execute_tool_calls(calls)
        assert "not found" in results[0].content

    def test_handler_exception_returns_error(self):
        engine = ToolCallingEngine()
        def failing_handler(**kwargs):
            raise ValueError("Something went wrong")
        engine.register_handler("failing_tool", failing_handler)
        calls = [ToolCall(id="c1", name="failing_tool", arguments={})]
        results = engine.execute_tool_calls(calls)
        assert len(results) == 1
        assert "Error" in results[0].content
        assert "Something went wrong" in results[0].content

    def test_no_handler_still_returns_result(self):
        engine = ToolCallingEngine()
        calls = [ToolCall(id="c1", name="unknown", arguments={})]
        results = engine.execute_tool_calls(calls)
        assert len(results) == 1
        assert results[0].tool_call_id == "c1"
        assert results[0].role == "tool"

    def test_multiple_calls_partial_failure(self):
        engine = ToolCallingEngine()
        engine.register_handler("good_tool", lambda: "success")
        def bad_handler(**kwargs):
            raise RuntimeError("fail")
        engine.register_handler("bad_tool", bad_handler)
        calls = [
            ToolCall(id="c1", name="good_tool", arguments={}),
            ToolCall(id="c2", name="bad_tool", arguments={}),
            ToolCall(id="c3", name="unknown_tool", arguments={}),
        ]
        results = engine.execute_tool_calls(calls)
        assert len(results) == 3
        assert results[0].content == "success"
        assert "Error" in results[1].content
        assert "Error" in results[2].content
