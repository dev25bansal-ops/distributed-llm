import json
from typing import AsyncIterator, Iterator


async def parse_sse_stream_async(response) -> AsyncIterator[dict]:
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
