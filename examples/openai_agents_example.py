"""OpenAI Agents SDK + DistLLM — agents, tools, handoffs, and streaming.

This example demonstrates how to use the ``openai-agents`` SDK with
DistLLM as a drop-in inference backend.

Requirements::

    pip install openai-agents openai
    pip install -e integrations/openai-agents  # or: pip install distllm-openai-agents

Usage::

    # Start the DistLLM API server first:
    distllm-api --model meta-llama/Llama-2-70b-hf --local

    # Then run this example:
    python examples/openai_agents_example.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from distllm.integrations.openai_agents import DistLLMAgentModel, DistLLMModelProvider

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("DISTLLM_API_BASE", "http://localhost:8000")
API_KEY = os.getenv("DISTLLM_API_KEY", "not-needed")

# Create the DistLLM-wrapped model via the agent model wrapper
model = DistLLMAgentModel(
    model="distributed-llm",
    base_url=BASE_URL,
    api_key=API_KEY,
)

# Create a provider so the SDK can resolve model names
provider = DistLLMModelProvider(
    model="distributed-llm",
    base_url=BASE_URL,
    api_key=API_KEY,
)

# ---------------------------------------------------------------------------
# Tools (plain functions decorated with @function_tool)
# ---------------------------------------------------------------------------


def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    weather_data = {
        "san francisco": "62°F, Partly Cloudy",
        "new york": "45°F, Clear",
        "london": "55°F, Rainy",
    }
    return weather_data.get(location.lower(), f"Weather data unavailable for {location}")


def calculate(expression: str) -> str:
    """Perform a safe mathematical calculation."""
    import ast
    import operator

    _SAFE_OPS: dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError(f"Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool-calling agent using the Agents SDK
# ---------------------------------------------------------------------------


async def run_tool_agent() -> None:
    """Demonstrate an agent with tool calling via the SDK."""
    print("=== Agent with Tool Calling ===\n")
    try:
        from agents import Agent, Runner
        from agents import function_tool
    except ImportError:
        print("ERROR: openai-agents SDK not installed. Run: pip install openai-agents")
        return

    @function_tool
    def get_weather_tool(location: str) -> str:
        """Get the current weather for a location."""
        return get_weather(location)

    @function_tool
    def calculate_tool(expression: str) -> str:
        """Perform a safe mathematical calculation."""
        return calculate(expression)

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant with tools. Use them when appropriate.",
        model=model,
        tools=[get_weather_tool, calculate_tool],
    )

    result = await Runner.run(
        agent,
        input="What's the weather in San Francisco? Also, what is 15 * 23?",
    )

    print(f"User: What's the weather in San Francisco? Also, what is 15 * 23?\n")
    print(f"Assistant: {result.final_output}\n")
    print(f"--- Turn history ---")
    for i, turn in enumerate(result.new_items):
        print(f"  [{i}] {type(turn).__name__}: {turn}")


# ---------------------------------------------------------------------------
# Agent with handoffs
# ---------------------------------------------------------------------------


async def run_handoff_agent() -> None:
    """Demonstrate multi-agent handoffs."""
    print("\n=== Agent with Handoffs ===\n")
    try:
        from agents import Agent, Runner
    except ImportError:
        print("ERROR: openai-agents SDK not installed.")
        return

    # Sub-agents
    research_agent = Agent(
        name="Researcher",
        instructions="You are a research specialist. Provide concise research summaries.",
        model=model,
    )

    writer_agent = Agent(
        name="Writer",
        instructions="You are a writing specialist. Turn research into polished text.",
        model=model,
    )

    # Triage agent that hands off
    triage_agent = Agent(
        name="Triage",
        instructions=(
            "You route tasks: hand off to Researcher for research questions "
            "and to Writer for writing/editing tasks."
        ),
        model=model,
        handoffs=[research_agent, writer_agent],
    )

    result = await Runner.run(
        triage_agent,
        input="Research the key benefits of distributed computing and write a short paragraph about it.",
    )

    print(f"User: Research the key benefits of distributed computing and write a short paragraph.\n")
    print(f"Final output: {result.final_output}\n")
    for i, turn in enumerate(result.new_items):
        print(f"  [{i}] {type(turn).__name__}")


# ---------------------------------------------------------------------------
# Streaming agent
# ---------------------------------------------------------------------------


async def run_streaming_agent() -> None:
    """Demonstrate streaming output from an agent."""
    print("\n=== Streaming Agent ===\n")
    try:
        from agents import Agent, Runner
    except ImportError:
        print("ERROR: openai-agents SDK not installed.")
        return

    agent = Agent(
        name="Poet",
        instructions="You are a poet who writes haikus.",
        model=model,
    )

    print("User: Write a haiku about distributed computing.\n")
    print("Assistant: ", end="", flush=True)

    result = Runner.run_streamed(agent, input="Write a haiku about distributed computing.")
    async for chunk in result.stream_events():
        if hasattr(chunk, "delta") and chunk.delta:
            print(chunk.delta, end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# Provider-based agent
# ---------------------------------------------------------------------------


async def run_provider_agent() -> None:
    """Demonstrate using a DistLLMModelProvider directly."""
    print("\n=== Provider-Based Agent ===\n")
    try:
        from agents import Agent, Runner
    except ImportError:
        print("ERROR: openai-agents SDK not installed.")
        return

    agent = Agent(
        name="Assistant",
        instructions="You are a concise assistant. Keep responses under three sentences.",
        model_provider=provider,
    )

    result = await Runner.run(agent, input="What is 2 + 2?")

    print(f"User: What is 2 + 2?\n")
    print(f"Assistant: {result.final_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all agent examples."""
    print("OpenAI Agents SDK + DistLLM Examples\n")
    print("=" * 60 + "\n")

    try:
        await run_tool_agent()
        await run_handoff_agent()
        await run_streaming_agent()
        await run_provider_agent()
    except ImportError as e:
        print(f"\nDependency error: {e}")
        print(
            "Install required packages:\n"
            "  pip install openai-agents openai\n"
            "  pip install -e integrations/openai-agents"
        )
    except Exception as e:
        print(f"\nRuntime error: {e}")
        print(
            "Make sure the DistLLM API server is running:\n"
            "  distllm-api --model <model> --local"
        )


if __name__ == "__main__":
    asyncio.run(main())
