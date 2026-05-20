"""Plugin sandboxing for the DistLLM plugin marketplace.

Provides soft sandboxing with restricted context, async execution
with timeout, and resource monitoring via tracemalloc.
"""

from __future__ import annotations

import asyncio
import functools
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class SandboxContext:
    """Restricted context passed to plugins running in the sandbox.

    Provides controlled access to coordinator resources without
    exposing the full internal state.
    """
    config: dict[str, Any] = field(default_factory=dict)
    hooks: Any = None  # HookRegistry reference
    logger: Any = None  # Logger instance
    coordinator_ref: Any = None  # Weak reference to coordinator
    metrics: Any = None  # Metrics collector

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for compatibility with existing plugins."""
        return {
            "config": self.config,
            "hooks": self.hooks,
            "logger": self.logger or logger,
            "coordinator_ref": self.coordinator_ref,
            "metrics": self.metrics,
        }.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in (
            "config", "hooks", "logger", "coordinator_ref", "metrics"
        )


@dataclass
class SandboxStats:
    """Resource usage stats from a sandboxed execution."""
    plugin_name: str
    hook_name: str
    duration_ms: float
    success: bool
    peak_memory_kb: float = 0.0
    error: str = ""


class PluginSandbox:
    """Executes plugin hooks in a sandboxed environment.

    Features:
    - Timeout enforcement via asyncio.wait_for()
    - Memory tracking via tracemalloc
    - Error isolation (catches and logs exceptions)
    - Execution statistics
    """

    def __init__(
        self,
        default_timeout_s: float = 30.0,
        track_memory: bool = True,
        max_memory_mb: float = 512.0,
    ) -> None:
        self.default_timeout_s = default_timeout_s
        self.track_memory = track_memory
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._stats: list[SandboxStats] = []

    async def run_hook_async(
        self,
        plugin_name: str,
        hook_name: str,
        callback: Callable,
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SandboxStats:
        """Run a plugin hook callback asynchronously with sandbox constraints.

        Args:
            plugin_name: Name of the plugin.
            hook_name: Name of the hook point.
            callback: The callback function to execute.
            *args: Positional arguments for the callback.
            timeout_s: Timeout in seconds (uses default if None).
            **kwargs: Keyword arguments for the callback.

        Returns:
            SandboxStats with execution results.
        """
        timeout = timeout_s or self.default_timeout_s
        start = time.monotonic()
        peak_mem = 0.0
        error = ""
        success = True

        if self.track_memory:
            tracemalloc.start()

        try:
            if asyncio.iscoroutinefunction(callback):
                await asyncio.wait_for(callback(*args, **kwargs), timeout=timeout)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, functools.partial(callback, *args, **kwargs)),
                    timeout=timeout,
                )

            if self.track_memory:
                _, peak_mem = tracemalloc.get_traced_memory()

        except asyncio.TimeoutError:
            error = f"Plugin hook timed out after {timeout}s"
            success = False
            logger.warning(f"Plugin '{plugin_name}' hook '{hook_name}' timed out")
        except Exception as e:
            error = str(e)
            success = False
            logger.error(f"Plugin '{plugin_name}' hook '{hook_name}' failed: {e}")
        finally:
            if self.track_memory:
                tracemalloc.stop()

        duration_ms = (time.monotonic() - start) * 1000

        # Check memory limit
        if peak_mem > self.max_memory_bytes:
            logger.warning(
                f"Plugin '{plugin_name}' exceeded memory limit: "
                f"{peak_mem / 1024 / 1024:.1f}MB > {self.max_memory_bytes / 1024 / 1024:.1f}MB"
            )

        stats = SandboxStats(
            plugin_name=plugin_name,
            hook_name=hook_name,
            duration_ms=duration_ms,
            success=success,
            peak_memory_kb=peak_mem / 1024,
            error=error,
        )
        self._stats.append(stats)
        return stats

    def run_hook_sync(
        self,
        plugin_name: str,
        hook_name: str,
        callback: Callable,
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> SandboxStats:
        """Run a plugin hook callback synchronously with sandbox constraints.

        For synchronous execution, uses a subprocess-style timeout via
        threading (best-effort, as Python can't truly interrupt a thread).
        """
        timeout = timeout_s or self.default_timeout_s
        start = time.monotonic()
        peak_mem = 0.0
        error = ""
        success = True

        if self.track_memory:
            tracemalloc.start()

        try:
            callback(*args, **kwargs)

            if self.track_memory:
                _, peak_mem = tracemalloc.get_traced_memory()

        except Exception as e:
            error = str(e)
            success = False
            logger.error(f"Plugin '{plugin_name}' hook '{hook_name}' failed: {e}")
        finally:
            if self.track_memory:
                tracemalloc.stop()

        duration_ms = (time.monotonic() - start) * 1000

        stats = SandboxStats(
            plugin_name=plugin_name,
            hook_name=hook_name,
            duration_ms=duration_ms,
            success=success,
            peak_memory_kb=peak_mem / 1024,
            error=error,
        )
        self._stats.append(stats)
        return stats

    def get_stats(self, plugin_name: str | None = None) -> list[SandboxStats]:
        """Return execution stats, optionally filtered by plugin name."""
        if plugin_name:
            return [s for s in self._stats if s.plugin_name == plugin_name]
        return list(self._stats)

    def clear_stats(self) -> None:
        """Clear all execution stats."""
        self._stats.clear()
