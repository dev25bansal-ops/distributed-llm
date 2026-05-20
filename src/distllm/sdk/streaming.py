"""SSE stream parser for DistLLM SDK."""

import json
from typing import AsyncIterator, Iterator


async def parse_sse_stream_async(response) -> AsyncIterator[dict]:
    """Parse Server-Sent Events stream from an async httpx response.

    Args:
        response: httpx Response object with streaming enabled

    Yields:
        Parsed JSON data from each SSE event
    """
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def parse_sse_stream_sync(response) -> Iterator[dict]:
    """Parse Server-Sent Events stream from a sync httpx response.

    Args:
        response: httpx Response object with streaming enabled

    Yields:
        Parsed JSON data from each SSE event
    """
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


parse_sse_stream = parse_sse_stream_async
