"""Shared bounded executor for tool-call handlers.

Both ``distllm.api.tool_calling.ToolCallingEngine`` and
``distllm.api.routes.chat._ToolCallingEngine`` used to construct a
single-worker ``ThreadPoolExecutor`` *per tool call*. At high request rates
that meant one short-lived pool per call (thread churn) and, because the pools
were created and waited on sequentially, no parallelism across the tool calls
of one assistant turn.

This module owns one process-wide bounded pool instead. Size comes from
``DISTLLM_TOOL_EXECUTORS`` (default ``min(32, cpu_count + 4)``); the pool is
created lazily on first use so importing this module stays cheap.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

_ENV_VAR = "DISTLLM_TOOL_EXECUTORS"

_executor: ThreadPoolExecutor | None = None
_executor_lock = Lock()
_max_workers: int | None = None


def _default_max_workers() -> int:
    return min(32, (os.cpu_count() or 1) + 4)


def _configured_max_workers() -> int:
    raw = os.environ.get(_ENV_VAR)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return _default_max_workers()
        if value > 0:
            return value
    return _default_max_workers()


def get_tool_executor() -> ThreadPoolExecutor:
    """Return the shared bounded executor, creating it once.

    The executor is deliberately never shut down at interpreter exit
    (``atexit`` is not registered) so in-flight tool calls keep working while
    the process unwinds; daemon-free threads would otherwise risk hangs at
    exit, and non-daemon threads are the safe default here because the pool is
    bounded and tool calls carry their own per-call timeout.
    """
    global _executor, _max_workers
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            _max_workers = _configured_max_workers()
            _executor = ThreadPoolExecutor(
                max_workers=_max_workers,
                thread_name_prefix="distllm-tool",
            )
    return _executor


def get_tool_executor_max_workers() -> int:
    """Effective max_workers of the shared executor (created on demand)."""
    get_tool_executor()
    return _max_workers if _max_workers is not None else _default_max_workers()


def reset_tool_executor() -> None:
    """Test-only: drop the cached executor so the next call rebuilds it.

    The abandoned pool is left to be GC'd; its idle threads exit on their own
    once the executor is collected. Used by tests that patch
    ``DISTLLM_TOOL_EXECUTORS``.
    """
    global _executor, _max_workers
    with _executor_lock:
        _executor = None
        _max_workers = None
