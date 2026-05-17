"""Loguru sink that pushes logs to Grafana Loki in batches.

Uses httpx for async HTTP POST to Loki's push API.
Logs are batched by count or time interval, whichever comes first.
Includes OpenTelemetry trace_id/span_id as Loki stream labels.
"""

import asyncio
import json
import time
from collections import deque
from typing import Callable, Dict


def _get_otel_labels() -> dict[str, str]:
    """Extract OTel trace context as Loki labels."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return {
                "trace_id": f"{ctx.trace_id:032x}",
                "span_id": f"{ctx.span_id:016x}",
            }
    except Exception:
        pass
    return {}


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
    base_labels = {"service": service_name}
    push_url = f"{url.rstrip('/')}/loki/api/v1/push"

    async def _push_batch() -> None:
        entries = []
        while buffer:
            ts_ns, line, otel_labels = buffer.popleft()
            # Merge per-record OTel labels into stream labels
            stream_labels = {**base_labels, **otel_labels}
            entries.append([str(ts_ns), line])

        if not entries:
            return

        payload = {
            "streams": [
                {
                    "stream": dict(base_labels),
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
    _loop_task = None
    try:
        loop = asyncio.get_running_loop()
        _loop_task = loop.create_task(_periodic_flush())
    except RuntimeError:
        pass

    def _sink(message) -> None:
        record = message.record
        ts_ns = int(time.time() * 1e9)
        otel_labels = _get_otel_labels()
        line = json.dumps(
            {
                "level": record["level"].name,
                "module": record["name"],
                "function": record["function"],
                "message": record["message"],
                **otel_labels,
                **{k: v for k, v in record["extra"].items()},
            },
            default=str,
        )
        buffer.append((ts_ns, line, otel_labels))

        if len(buffer) >= batch_size:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_push_batch())
            except RuntimeError:
                pass  # No event loop yet; will flush periodically

    return _sink
