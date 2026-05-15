"""Direct SDK usage example with DistLLM.

This example shows how to use the DistLLM Python SDK directly.

Requirements:
    pip install httpx  # or: pip install distllm[sdk]

Usage:
    # Start the API server first:
    distllm-api --model roneneldan/TinyStories-1M --local

    # Then run this example:
    python examples/sdk_example.py
"""

import asyncio
from distllm.sdk import DistLLMClient, DistLLMClientSync


def sync_example():
    """Synchronous SDK example."""
    print("=== Synchronous SDK Example ===\n")

    with DistLLMClientSync(base_url="http://localhost:8000") as client:
        # List models
        models = client.list_models()
        print(f"Available models: {[m.id for m in models.data]}")

        # Chat completion
        response = client.chat_completions(
            messages=[
                {"role": "user", "content": "What is machine learning?"}
            ],
            max_tokens=100,
        )

        print(f"\nResponse: {response.choices[0].message.content}")
        if response.generation_time:
            print(f"Time: {response.generation_time:.2f}s")


async def async_example():
    """Asynchronous SDK example."""
    print("\n=== Asynchronous SDK Example ===\n")

    async with DistLLMClient(base_url="http://localhost:8000") as client:
        # Health check
        health = await client.health_check()
        print(f"Health: {health['status']}")

        # Chat completion with streaming
        print("\nStreaming response:")
        async for token in client.chat_completions_stream(
            messages=[
                {"role": "user", "content": "Tell me a short story about AI."}
            ],
            max_tokens=100,
        ):
            print(token, end="", flush=True)
        print()


if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
