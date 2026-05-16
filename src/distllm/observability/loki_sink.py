"""Loguru sink that pushes logs to Grafana Loki in batches.

Uses httpx for async HTTP POST to Loki's push API.
Logs are batched by count or time interval, whichever comes first.
"""

import asyncio
import json
import time
from collections import deque
from typing import Callable


def loki_sink(
    url: str,
    service_name: str,
    batch_size: int = 50,
    flush_interval: float = 5.0,
) -> Callable:
    """Create a loguru sink that batches logs and pushes to Loki.

    Args:
        url: Loki base URL (e.g. "http://localhost:3100").
        service_name: Service label for Loki stream.
        batch_size: Number of log entries before forced flush.
        flush_interval: Seconds between periodic flushes.
    """
    import httpx

    buffer: deque = deque()
    labels = {"service": service_name}
    push_url = f"{url.rstrip('/')}/loki/api/v1/push"

    async def _push_batch() -> None:
        entries = []
        while buffer:
            ts_ns, line = buffer.popleft()
            entries.append([str(ts_ns), line])

        if not entries:
            return

        payload = {
            "streams": [
                {
                    "stream": dict(labels),
                    "values": entries,
                }
            ]
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    push_url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=10.0,
                )
        except httpx.HTTPError:
            # Drop logs on push failure to avoid blocking the logger
            pass

    async def _periodic_flush() -> None:
        while True:
            await asyncio.sleep(flush_interval)
            await _push_batch()

    # Start the periodic flush task
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_periodic_flush())
    except RuntimeError:
        # No running event loop — flush will run on first log call
        _loop_task = None

    def _sink(message) -> None:
        record = message.record
        ts_ns = int(time.time() * 1e9)
        line = json.dumps(
            {
                "level": record["level"].name,
                "module": record["name"],
                "function": record["function"],
                "message": record["message"],
                **{k: v for k, v in record["extra"].items()},
            },
            default=str,
        )
        buffer.append((ts_ns, line))

        if len(buffer) >= batch_size:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_push_batch())
            except RuntimeError:
                pass  # No event loop yet; will flush periodically

    return _sink
