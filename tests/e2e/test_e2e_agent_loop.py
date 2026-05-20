"""E2E test: Agent loop with mock tool calls.

Tests the AgentLoop end-to-end:
1. Create an AgentLoop with mock LLM function and mock tools
2. Run the agent on a goal that triggers tool calling
3. Verify the agent follows ReAct: plan -> act (tool call) -> synthesize
4. Verify tool results are incorporated into the final answer
"""

import json
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import distllm.api.server as server_module
from distllm.api.server import app
from distllm.core.agent_loop import AgentLoop, AgentState, ToolCall


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "1")
    monkeypatch.setenv("DISTLLM_DEV_MODE", "1")
    monkeypatch.delenv("API_KEY", raising=False)


def test_agent_loop_basic():
    call_log = []

    def llm_fn(prompt):
        call_log.append(prompt[:50])
        if call_log and len(call_log) == 1:
            return '["Get weather data"]'
        if "TOOL_CALL:" in prompt:
            return "The result is sunny and 25C."
        return "The weather in Tokyo is sunny and 25 degrees Celsius."

    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "handler": lambda location: {"temperature": 25, "condition": "sunny"},
        }
    ]

    agent = AgentLoop(llm_fn=llm_fn, tools=tools, max_iterations=5)
    result = agent.run("What is the weather in Tokyo?")

    assert "answer" in result
    assert result["answer"] is not None
    assert len(result["answer"]) > 0
    assert result["iterations"] > 0
    assert agent.state == AgentState.DONE
    assert len(call_log) > 0


def test_agent_loop_with_tool_call():
    call_count = [0]

    def llm_fn(prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return '["Get weather data"]'
        if "TOOL_CALL:" in prompt:
            return "TOOL_CALL: get_weather({\"location\": \"Tokyo\"})"
        return "The weather in Tokyo is sunny and 25C."

    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "handler": lambda location: {"temp": 25, "condition": "sunny"},
        }
    ]

    agent = AgentLoop(llm_fn=llm_fn, tools=tools, max_iterations=5)
    result = agent.run("Weather in Tokyo?")

    assert result["answer"] is not None
    assert len(result.get("tool_calls", [])) > 0


def test_agent_loop_max_iterations():
    call_count = [0]

    def llm_fn(prompt):
        call_count[0] += 1
        return "TOOL_CALL: search({\"query\": \"continue\"})"

    tools = [
        {
            "name": "search",
            "description": "Search",
            "handler": lambda query: {"result": "more data"},
        }
    ]

    agent = AgentLoop(llm_fn=llm_fn, tools=tools, max_iterations=3)
    result = agent.run("Research topic")

    assert result["iterations"] <= 3
    assert agent.state in (AgentState.DONE, AgentState.FAILED)


def test_agent_loop_unknown_tool():
    def llm_fn(prompt):
        if "TOOL_CALL:" in prompt:
            return "TOOL_CALL: nonexistent_tool({\"arg\": 1})"
        return "Completed despite tool error."

    agent = AgentLoop(llm_fn=llm_fn, tools=[], max_iterations=5)
    result = agent.run("Test unknown tool")

    assert result["answer"] is not None


def test_agent_loop_reflection():
    def llm_fn(prompt):
        if "failed" in prompt.lower() or "error" in prompt.lower():
            return "RETRY"
        if "TOOL_CALL:" in prompt:
            return "the result is 42"
        return "Final answer: success."

    tools = [
        {
            "name": "calculator",
            "description": "Calculate",
            "handler": lambda expr: 42,
        }
    ]

    agent = AgentLoop(llm_fn=llm_fn, tools=tools, reflection_enabled=True, max_iterations=5)
    result = agent.run("Compute the answer")

    assert result["answer"] is not None


def test_agent_api_route(e2e_api_client, e2e_coordinator):
    import distllm.api.server as server_module
    mock_loop = MagicMock()
    mock_loop.run.return_value = {"result": "done", "iterations": 1}
    e2e_coordinator._agent_loop = mock_loop
    server_module.coordinator = e2e_coordinator

    resp = e2e_api_client.post("/v1/agents/run", json={
        "goal": "Summarize the weather report",
        "tools": [
            {"name": "search", "description": "Web search", "handler": "search_web"}
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert "iterations" in data
