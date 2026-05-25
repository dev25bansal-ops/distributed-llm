"""Loguru sink that pushes logs to Grafana Loki in batches.

from loguru import logger
Uses httpx for async HTTP POST to Loki's push API.
Logs are batched by count or time interval, whichever comes first.
Includes OpenTelemetry trace_id/span_id as Loki stream labels.
"""

import asyncio
import json
import threading
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
                logger.debug("Loki sink batch failed")
    except Exception:
        logger.debug("Loki sink write failed")

    async def _periodic_flush() -> None:
        while True:
            await asyncio.sleep(flush_interval)
            await _push_batch()

    # Start the periodic flush task
    _loop_task = None
    _loop_thread = None

    def _start_background_loop() -> None:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(_periodic_flush())

    try:
        loop = asyncio.get_running_loop()
        _loop_task = loop.create_task(_periodic_flush())
    except RuntimeError:
        _loop_thread = threading.Thread(target=_start_background_loop, daemon=True)
        _loop_thread.start()

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
            except RuntimeError:
                loop = _ensure_background_loop()
            if loop is not None:
                task = loop.create_task(_push_batch())
                task.add_done_callback(lambda t: t.exception() if t.done() and not t.cancelled() else None)

    return _sink
