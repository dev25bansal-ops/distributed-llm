"""Regression tests for the shared bounded tool-call executor (PERF-04-EXECUTOR).

Covers ``distllm.api.tool_executor`` plus both tool-calling engines that use it
(``distllm.api.tool_calling.ToolCallingEngine`` and the route-local duplicate
``distllm.api.routes.chat._ToolCallingEngine``).
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from distllm.api import tool_executor
from distllm.api.routes.chat import _ToolCallingEngine  # alias of ToolCallingEngine
from distllm.api.tool_calling import ToolCallingEngine

ENGINE_FACTORIES = [
    pytest.param(lambda: ToolCallingEngine(), id="canonical"),
    pytest.param(lambda: _ToolCallingEngine(), id="routes_chat_alias"),
]


@pytest.fixture(autouse=True)
def _fresh_executor():
    """Rebuild the shared executor before/after each test (env patches take effect)."""
    tool_executor.reset_tool_executor()
    yield
    tool_executor.reset_tool_executor()


class _ConcurrencyTracker:
    """Records in-flight count so tests can assert on peak concurrency."""

    def __init__(self):
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def make_task(self, tag, delay):
        def task():
            with self._lock:
                self.current += 1
                if self.current > self.peak:
                    self.peak = self.current
            try:
                time.sleep(delay)
                return f"done{tag}"
            finally:
                with self._lock:
                    self.current -= 1

        return task


# ---------------------------------------------------------------------------
# Shared executor construction
# ---------------------------------------------------------------------------

class TestSharedExecutor:
    def test_default_max_workers(self, monkeypatch):
        monkeypatch.delenv("DISTLLM_TOOL_EXECUTORS", raising=False)
        executor = tool_executor.get_tool_executor()
        expected = min(32, (os.cpu_count() or 1) + 4)
        assert executor._max_workers == expected
        assert tool_executor.get_tool_executor_max_workers() == expected

    def test_env_var_configures_max_workers(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_TOOL_EXECUTORS", "3")
        executor = tool_executor.get_tool_executor()
        assert executor._max_workers == 3
        assert tool_executor.get_tool_executor_max_workers() == 3

    def test_env_var_ignored_when_invalid(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_TOOL_EXECUTORS", "not-a-number")
        executor = tool_executor.get_tool_executor()
        assert executor._max_workers == min(32, (os.cpu_count() or 1) + 4)

    def test_env_var_zero_falls_back(self, monkeypatch):
        monkeypatch.setenv("DISTLLM_TOOL_EXECUTORS", "0")
        executor = tool_executor.get_tool_executor()
        assert executor._max_workers == min(32, (os.cpu_count() or 1) + 4)

    def test_single_shared_instance(self):
        assert tool_executor.get_tool_executor() is tool_executor.get_tool_executor()


# ---------------------------------------------------------------------------
# Engine behaviour against the shared executor
# ---------------------------------------------------------------------------

class TestToolCallExecution:
    @pytest.mark.parametrize("make_engine", ENGINE_FACTORIES)
    def test_twenty_calls_bounded_concurrency_and_speedup(self, make_engine, monkeypatch):
        """20 parallel tool calls: peak concurrency <= executor max, wall < sequential."""
        monkeypatch.setenv("DISTLLM_TOOL_EXECUTORS", "4")
        tool_executor.reset_tool_executor()

        eng = make_engine()
        tracker = _ConcurrencyTracker()
        for i in range(20):
            eng.register_tool(f"t{i}", tracker.make_task(i, delay=0.1))

        calls = [
            {"id": f"c{i}", "function": {"name": f"t{i}", "arguments": {}}}
            for i in range(20)
        ]

        t0 = time.perf_counter()
        results = eng.execute_tool_calls(calls, timeout_s=10.0)
        wall_parallel = time.perf_counter() - t0

        assert len(results) == 20
        assert [r["tool_call_id"] for r in results] == [f"c{i}" for i in range(20)]
        assert [r["content"] for r in results] == [f"done{i}" for i in range(20)]

        max_workers = tool_executor.get_tool_executor_max_workers()
        assert tracker.peak <= max_workers, (
            f"peak concurrency {tracker.peak} exceeded executor max {max_workers}"
        )
        assert tracker.peak > 1, "calls did not run concurrently at all"

        # Sequential equivalent: 20 x 0.1 s on one worker.
        assert wall_parallel < 20 * 0.1, (
            f"parallel wall {wall_parallel:.2f}s not faster than sequential 2.0s"
        )

    @pytest.mark.parametrize("make_engine", ENGINE_FACTORIES)
    def test_unregistered_tool_diagnostic(self, make_engine):
        eng = make_engine()
        results = eng.execute_tool_calls(
            [{"id": "c0", "function": {"name": "nope", "arguments": {}}}]
        )
        assert len(results) == 1
        assert "not registered" in results[0]["content"]
        assert results[0]["tool_call_id"] == "c0"

    @pytest.mark.parametrize("make_engine", ENGINE_FACTORIES)
    def test_handler_error_is_captured(self, make_engine):
        eng = make_engine()

        def boom():
            raise ValueError("kaboom")

        eng.register_tool("boom", boom)
        results = eng.execute_tool_calls(
            [{"id": "c0", "function": {"name": "boom", "arguments": {}}}]
        )
        assert "Error executing tool 'boom'" in results[0]["content"]
        assert "kaboom" in results[0]["content"]

    @pytest.mark.parametrize("make_engine", ENGINE_FACTORIES)
    def test_string_arguments_parsed(self, make_engine):
        eng = make_engine()
        eng.register_tool("add", lambda a, b: a + b)
        results = eng.execute_tool_calls(
            [{"id": "c0", "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}]
        )
        assert results[0]["content"] == "5"

    @pytest.mark.parametrize("make_engine", ENGINE_FACTORIES)
    def test_empty_calls_returns_empty_list(self, make_engine):
        assert make_engine().execute_tool_calls([]) == []

    @pytest.mark.parametrize("make_engine", ENGINE_FACTORIES)
    def test_timeout_error_message(self, make_engine, monkeypatch):
        monkeypatch.setenv("DISTLLM_TOOL_EXECUTORS", "2")
        eng = make_engine()
        eng.register_tool("slow", lambda: time.sleep(1.5))
        results = eng.execute_tool_calls(
            [{"id": "c0", "function": {"name": "slow", "arguments": {}}}],
            timeout_s=0.1,
        )
        assert "timed out" in results[0]["content"]
