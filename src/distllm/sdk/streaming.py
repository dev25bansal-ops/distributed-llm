"""SSE stream parser and WebSocket streaming client for DistLLM SDK.

Provides:
- parse_sse_stream_async / parse_sse_stream_sync (existing)
- WebSocketStreamClient: WebSocket-native client with backpressure,
  server-initiated events, and lower latency than HTTP SSE.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Iterator

from loguru import logger


# ── SSE parsing (existing) ───────────────────────────────────────────────

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


# ── WebSocket streaming client ──────────────────────────────────────────

class WebSocketStreamClient:
    """WebSocket-native streaming client with backpressure.

    Provides proper flow control (WINDOW_UPDATE frames), server-initiated
    events (model switch, cache hit notification), and lower latency by
    avoiding HTTP overhead for each token.

    Binary protocol:
        [varint-encoded token length][token bytes][4-byte flags]

    Flags:
        0x01 = done
        0x02 = error
        0x04 = model_switched
        0x08 = cached

    Usage::

        client = WebSocketStreamClient("ws://localhost:8000/ws/v1/chat")
        await client.connect()

        # Stream with backpressure
        async for chunk in client.stream(
            messages=[{"role": "user", "content": "Hello"}],
        ):
            print(chunk)

        await client.close()
    """

    def __init__(
        self,
        url: str,
        api_key: str = "",
        max_buffer_size: int = 4096,
        reconnect_delay_s: float = 1.0,
    ):
        self._url = url
        self._api_key = api_key
        self._max_buffer = max_buffer_size
        self._reconnect_delay = reconnect_delay_s
        self._ws: Any = None
        self._receive_buffer: asyncio.Queue[dict] = asyncio.Queue()
        self._running = False

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        import websockets
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._ws = await websockets.connect(
            self._url, additional_headers=headers,
        )
        self._running = True
        logger.info(f"WebSocket connected: {self._url}")

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "distributed-llm",
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> AsyncIterator[dict]:
        """Send a chat request and stream the response with backpressure.

        Client-side backpressure: sends WINDOW_UPDATE when the receive
        buffer drops below half capacity.

        Yields:
            Response chunks (same format as SSE chunks).
        """
        request = {
            "type": "chat",
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        await self._ws.send(json.dumps(request))

        # Start background receive loop
        self._receive_buffer = asyncio.Queue(maxsize=self._max_buffer)
        reader = asyncio.create_task(self._read_loop())

        try:
            while True:
                chunk = await self._receive_buffer.get()
                flags = chunk.get("_flags", 0)

                if flags & 0x02:  # error
                    error_msg = chunk.get("error", "WebSocket error")
                    raise RuntimeError(error_msg)

                yield chunk

                # Backpressure: signal server when buffer is below half
                if self._receive_buffer.qsize() < self._max_buffer // 2:
                    await self._ws.send(json.dumps({
                        "type": "window_update",
                        "available": self._max_buffer - self._receive_buffer.qsize(),
                    }))

                if flags & 0x01:  # done
                    break
        finally:
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass

    async def _read_loop(self) -> None:
        """Background task: receive messages into the buffer."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    chunk = self._parse_binary(message)
                else:
                    chunk = json.loads(message)
                await self._receive_buffer.put(chunk)
        except Exception as e:
            logger.debug(f"WebSocket read loop ended: {e}")

    @staticmethod
    def _parse_binary(data: bytes) -> dict:
        """Parse a binary WebSocket frame.

        Format: [varint token_len][token bytes][flags:4 bytes]
        """
        i = 0
        # Varint decode
        token_len = 0
        shift = 0
        while i < len(data):
            byte = data[i]
            token_len |= (byte & 0x7F) << shift
            shift += 7
            i += 1
            if not (byte & 0x80):
                break

        if i + token_len + 4 > len(data):
            return {"text": data.decode("utf-8", errors="replace"), "_flags": 0}

        token_bytes = data[i:i + token_len]
        flags = int.from_bytes(data[i + token_len:i + token_len + 4], "little")
        return {
            "text": token_bytes.decode("utf-8"),
            "_flags": flags,
        }

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

