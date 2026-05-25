"""Unit tests for AgentLoop: tool calls, multi-turn, max iterations, error handling.

Tests the AgentLoop class directly with mock LLM functions, no e2e setup.
"""

from distllm.core.agent_loop import AgentLoop, AgentState


def test_single_tool_call():
    """Request tool → model calls tool → result."""
    call_count = [0]

    def llm_fn(prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return '["Get weather"]'
        if "TOOL_CALL:" in prompt:
            return 'TOOL_CALL: get_weather({"location": "Tokyo"})'
        return "The weather in Tokyo is sunny and 25 degrees."

    tools = [{
        "name": "get_weather",
        "description": "Get weather for a location",
        "handler": lambda location: {"temp": 25, "condition": "sunny"},
    }]

    agent = AgentLoop(llm_fn=llm_fn, tools=tools, max_iterations=5)
    result = agent.run("What is the weather in Tokyo?")

    assert result["answer"] is not None
    assert len(result["answer"]) > 0
    assert result["answer"] == "The weather in Tokyo is sunny and 25 degrees."
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "get_weather"
    assert result["tool_calls"][0]["result"] == {"temp": 25, "condition": "sunny"}
    assert result["iterations"] >= 1
    assert agent.state == AgentState.DONE


def test_multi_turn_tool_calls():
    """Multiple tool calls across plan steps → accumulates context."""
    call_count = [0]

    def llm_fn(prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return '["Search population", "Search area", "Synthesize"]'

        if call_count[0] == 2:
            return 'TOOL_CALL: search_population({"city": "Tokyo"})'

        if call_count[0] == 3:
            return 'TOOL_CALL: search_area({"city": "Tokyo"})'

        return "Tokyo has 14 million people and 2,194 sq km."

    tools = [
        {
            "name": "search_population",
            "description": "Search population",
            "handler": lambda city: {"population": "14M"},
        },
        {
            "name": "search_area",
            "description": "Search area",
            "handler": lambda city: {"area": "2,194 sq km"},
        },
    ]

    agent = AgentLoop(llm_fn=llm_fn, tools=tools, max_iterations=5)
    result = agent.run("Get Tokyo info")

    assert result["answer"] is not None
    tool_results = result["tool_calls"]
    assert len(tool_results) >= 2
    tool_names = [t["tool"] for t in tool_results]
    assert tool_names[0] == "search_population"
    assert tool_names[1] == "search_area"
    assert agent.state == AgentState.DONE


def test_max_iterations_exceeded():
    """Exceed max_iterations → return partial result."""
    call_count = [0]

    def llm_fn(prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return '["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]'
        if "TOOL_CALL:" in prompt:
            return 'TOOL_CALL: search({"query": "data"})'
        return "Partial result."

    tools = [{
        "name": "search",
        "description": "Search",
        "handler": lambda query: {"result": "some data"},
    }]

    agent = AgentLoop(llm_fn=llm_fn, tools=tools, max_iterations=3)
    result = agent.run("Research")

    assert result["iterations"] <= 3
    assert "answer" in result
    assert agent.state in (AgentState.DONE, AgentState.FAILED)


def test_tool_error_handled():
    """Tool throws → model handles error and continues."""
    call_count = [0]

    def llm_fn(prompt):
        call_count[0] += 1
        if call_count[0] == 1:
            return '["Process data"]'

        if "Failed" in prompt or "error" in prompt.lower():
            return "RETRY"

        if call_count[0] == 2:
            return 'TOOL_CALL: failing_tool({"input": "bad"})'

        return "Completed despite error."

    def failing_handler(input):
        raise ValueError("Invalid input: bad")

    agent = AgentLoop(
        llm_fn=llm_fn,
        tools=[{"name": "failing_tool", "description": "May fail", "handler": failing_handler}],
        max_iterations=5,
        reflection_enabled=True,
    )
    result = agent.run("Test error handling")

    assert result["answer"] is not None
    assert len(result["answer"]) > 0
    tool_results = result["tool_calls"]
    assert len(tool_results) >= 1
    assert "Error:" in str(tool_results[-1]["result"])
    assert agent.state == AgentState.DONE
