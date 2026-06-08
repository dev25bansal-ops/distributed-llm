"""OpenAI Agents SDK compatibility with Distributed LLM.

This example shows how to use the OpenAI Agents SDK (openai-agents)
with DistLLM as a drop-in replacement for OpenAI's API.

The Agents SDK provides a framework for building agentic AI applications
with tool use, handoffs, and guardrails.

Requirements:
    pip install openai-agents openai

Usage:
    # Start the API server first:
    distllm-api --model meta-llama/Llama-2-70b-hf --local

    # Then run this example:
    python examples/openai_agents_example.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI


# Configure OpenAI client to point to DistLLM
client = AsyncOpenAI(
    base_url=os.getenv("DISTLLM_API_BASE", "http://localhost:8000/v1"),
    api_key=os.getenv("DISTLLM_API_KEY", "not-needed"),
)


async def simple_agent_example():
    """Simple agent with tool use via function calling."""
    print("=== Simple Agent with Tools ===\n")

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name (e.g., 'San Francisco, CA')",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate (e.g., '2 + 2')",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    # Tool handlers
    def get_weather(location: str) -> str:
        # Mock weather data
        weather_data = {
            "san francisco": "62°F, Partly Cloudy",
            "new york": "45°F, Clear",
            "london": "55°F, Rainy",
        }
        return weather_data.get(location.lower(), f"Weather data unavailable for {location}")

    def calculate(expression: str) -> str:
        import ast
        import operator

        _SAFE_OPS = {
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

        def _eval(node):
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
            tree = ast.parse(expression, mode='eval')
            result = _eval(tree)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error: {e}"

    # Agent loop
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when appropriate."},
        {"role": "user", "content": "What's the weather in San Francisco? Also, what's 15 * 23?"},
    ]

    print(f"User: {messages[-1]['content']}\n")

    for turn in range(5):  # Max 5 tool-use turns
        response = await client.chat.completions.create(
            model="distributed-llm",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=512,
        )

        assistant_msg = response.choices[0].message
        messages.append(assistant_msg.model_dump(exclude_none=True))

        # Check for tool calls
        if assistant_msg.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                func_name = tool_call.function.name
                import json
                args = json.loads(tool_call.function.arguments)

                print(f"  Tool call: {func_name}({args})")

                # Execute tool
                if func_name == "get_weather":
                    result = get_weather(**args)
                elif func_name == "calculate":
                    result = calculate(**args)
                else:
                    result = f"Unknown tool: {func_name}"

                print(f"  Tool result: {result}\n")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            # No more tool calls — final response
            print(f"Assistant: {assistant_msg.content}")
            break


async def streaming_agent_example():
    """Streaming agent with real-time token output."""
    print("\n=== Streaming Agent ===\n")

    messages = [
        {"role": "user", "content": "Write a haiku about distributed computing."},
    ]

    print("User: Write a haiku about distributed computing.\n")
    print("Assistant: ", end="", flush=True)

    stream = await client.chat.completions.create(
        model="distributed-llm",
        messages=messages,
        stream=True,
        temperature=0.9,
        max_tokens=100,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)

    print("\n")


async def structured_output_example():
    """Agent with structured JSON output."""
    print("\n=== Structured Output Agent ===\n")

    messages = [
        {"role": "system", "content": "You are a data extraction assistant. Extract information as JSON."},
        {"role": "user", "content": "John Smith is 35 years old and works as a software engineer at Google in Mountain View, California."},
    ]

    response = await client.chat.completions.create(
        model="distributed-llm",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=256,
    )

    print(f"User: {messages[-1]['content']}\n")
    print(f"Structured Response:\n{response.choices[0].content}")


async def multi_turn_conversation():
    """Multi-turn conversation with context."""
    print("\n=== Multi-Turn Conversation ===\n")

    conversation = [
        {"role": "system", "content": "You are a helpful coding assistant."},
    ]

    turns = [
        "Write a Python function to calculate fibonacci numbers.",
        "Now add memoization to it.",
        "What's the time complexity of your solution?",
    ]

    for user_msg in turns:
        conversation.append({"role": "user", "content": user_msg})
        print(f"User: {user_msg}\n")

        response = await client.chat.completions.create(
            model="distributed-llm",
            messages=conversation,
            temperature=0.3,
            max_tokens=512,
        )

        assistant_msg = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": assistant_msg})
        print(f"Assistant: {assistant_msg}\n")
        print("-" * 60 + "\n")


async def main():
    """Run all agent examples."""
    print("OpenAI Agents SDK + DistLLM Examples\n")
    print("=" * 60 + "\n")

    try:
        await simple_agent_example()
        await streaming_agent_example()
        await structured_output_example()
        await multi_turn_conversation()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure DistLLM API is running: distllm-api --model <model> --local")


if __name__ == "__main__":
    asyncio.run(main())
