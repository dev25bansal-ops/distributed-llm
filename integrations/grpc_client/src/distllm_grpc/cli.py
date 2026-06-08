"""CLI for testing DistLLM gRPC connectivity."""

import argparse
import asyncio

from distllm_grpc.client import DistLLMGrpcClient


async def _test(host: str, port: int) -> None:
    async with DistLLMGrpcClient(host=host, port=port) as client:
        print(f"Connected to {host}:{port}")

        # Test chat
        resp = await client.chat_completion(
            messages=[{"role": "user", "content": "Say hello in one word."}]
        )
        print(f"Chat response: {resp}")

        # Test streaming
        print("Stream: ", end="")
        async for chunk in client.chat_completion_stream(
            messages=[{"role": "user", "content": "Count to 5."}]
        ):
            print(chunk, end="", flush=True)
        print()


def main():
    parser = argparse.ArgumentParser(description="Test DistLLM gRPC client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    asyncio.run(_test(args.host, args.port))


if __name__ == "__main__":
    main()
