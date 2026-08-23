"""DistLLM Python SDK — programmatic interface to distributed inference clusters.

Usage::

    import asyncio
    from distllm.client import DistLLMClient

    async def main():
        client = await DistLLMClient.connect(
            coordinator_url="http://10.0.0.1:8000",
            api_key="sk-...",
        )

        # Generate text
        result = await client.generate(
            prompt="What is the capital of France?",
            max_tokens=128,
        )
        print(result.text)

        # Stream tokens
        async for chunk in client.stream_generate(prompt="Tell me a story"):
            print(chunk.text, end="", flush=True)

        # Cluster info
        nodes = await client.list_nodes()
        metrics = await client.get_metrics()

        await client.close()

    asyncio.run(main())
"""

from __future__ import annotations

from distllm.client.client import DistLLMClient

__all__ = ["DistLLMClient"]
