"""Loguru sink that pushes logs to Grafana Loki in batches.

Uses httpx for async HTTP POST to Loki's push API.
Logs are batched by count or time interval, whichever comes first.
Includes OpenTelemetry trace_id/span_id as Loki stream labels.
"""

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any, Callable

from loguru import logger


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


def make_loki_sink(
    url: str,
    labels: dict[str, str] | str | None = None,
    batch_size: int = 100,
    flush_interval: float = 5.0,
    timeout: float = 10.0,
) -> Callable:
    """Create a loguru-compatible sink that pushes logs to Grafana Loki.

    Args:
        url: Loki push API URL (e.g. ``http://localhost:3100/loki/api/v1/push``).
        labels: Extra labels to attach to every log line. If a string is passed,
            it is treated as ``{"service": <string>}`` for backward compatibility.
        batch_size: Number of log lines to buffer before flushing.
        flush_interval: Seconds between periodic flushes.
        timeout: HTTP request timeout in seconds.

    Returns:
        A callable sink function compatible with ``logger.add()``.
    """
    if isinstance(labels, str):
        default_labels = {"service": labels}
    else:
        default_labels = labels or {}
    buffer: deque[tuple[int, str, dict[str, str]]] = deque(maxlen=batch_size * 10)
    _loop_task: asyncio.Task | None = None
    _loop_thread: threading.Thread | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _loop_lock = threading.Lock()
    _http_client = None

    def _get_http_client():
        nonlocal _http_client
        if _http_client is None:
            import httpx
            _http_client = httpx.AsyncClient(timeout=timeout)
        return _http_client

    async def _push_batch() -> None:
        """Push buffered log lines to Loki."""
        if not buffer:
            return

        batch: list[tuple[int, str, dict[str, str]]] = []
        while buffer and len(batch) < batch_size:
            batch.append(buffer.popleft())

        if not batch:
            return

        streams: dict[str, list] = {}
        for ts_ns, line, otel_labels in batch:
            merged_labels = {**default_labels, **otel_labels}
            label_key = json.dumps(merged_labels, sort_keys=True)
            if label_key not in streams:
                streams[label_key] = {
                    "stream": merged_labels,
                    "values": [],
                }
            streams[label_key]["values"].append([str(ts_ns), line])

        payload = {"streams": list(streams.values())}

        try:
            client = _get_http_client()
            resp = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code not in (200, 204):
                logger.debug(f"Loki push returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.debug(f"Loki sink push failed: {e}")

    async def _periodic_flush() -> None:
        """Periodically flush buffered logs."""
        while True:
            await asyncio.sleep(flush_interval)
            try:
                await _push_batch()
            except Exception as e:
                logger.debug(f"Loki sink periodic flush failed: {e}")

    def _ensure_background_loop() -> asyncio.AbstractEventLoop:
        """Get or create a background event loop for flushing."""
        nonlocal _loop, _loop_thread
        with _loop_lock:
            if _loop is not None and _loop.is_running():
                return _loop
            new_loop = asyncio.new_event_loop()
            _loop = new_loop

            def _run_loop() -> None:
                asyncio.set_event_loop(new_loop)
                new_loop.run_forever()

            _loop_thread = threading.Thread(target=_run_loop, daemon=True)
            _loop_thread.start()
            return new_loop

    def _start_periodic_flush() -> None:
        """Start the periodic flush task."""
        nonlocal _loop_task
        try:
            loop = asyncio.get_running_loop()
            _loop_task = loop.create_task(_periodic_flush())
        except RuntimeError:
            bg_loop = _ensure_background_loop()
            asyncio.run_coroutine_threadsafe(_periodic_flush(), bg_loop)

    def _sink(message) -> None:
        """Loguru sink function: buffer a log line for batched push."""
        record = message.record
        ts_ns = int(time.time() * 1e9)
        otel_labels = _get_otel_labels()
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
        buffer.append((ts_ns, line, otel_labels))

        if len(buffer) >= batch_size:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = _ensure_background_loop()
            if loop is not None:
                task = asyncio.run_coroutine_threadsafe(_push_batch(), loop)
                task.add_done_callback(
                    lambda t: t.exception() if t.done() and not t.cancelled() else None
                )

    _start_periodic_flush()
    return _sink


# Backward-compatible alias
loki_sink = make_loki_sink
