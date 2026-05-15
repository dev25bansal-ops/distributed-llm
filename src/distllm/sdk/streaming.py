"""SSE stream parser for DistLLM SDK."""

import json
from typing import AsyncIterator


async def parse_sse_stream(response) -> AsyncIterator[dict]:
    """Parse Server-Sent Events stream from the API.

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
