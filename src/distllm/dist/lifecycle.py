"""Component lifecycle manager for distributed system orchestration.

Provides a dependency-aware startup/shutdown system with topological
ordering, parallel independent startup, graceful degradation, and
status reporting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class ComponentState(str, Enum):
    """Enumeration of possible component lifecycle states.

    The normal progression is::

        UNINITIALIZED -> INITIALIZING -> RUNNING
        RUNNING -> STOPPING -> STOPPED
        Any state -> FAILED (on error)
    """

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class Component:
    """Descriptor for a managed component in the lifecycle system.

    Attributes:
        name: Unique identifier for this component.
        start: Async or sync callable that starts the component.
        stop: Async or sync callable that stops the component.
        dependencies: Names of components that must start before this one.
        state: Current lifecycle state of the component.
        timeout: Maximum seconds to wait for start/stop operations.
    """

    name: str
    start: Callable[[], Any]
    stop: Callable[[], Any]
    dependencies: list[str] = field(default_factory=list)
    state: ComponentState = ComponentState.UNINITIALIZED
    timeout: float = 30.0


class LifecycleManager:
    """Manages component lifecycle with dependency ordering and parallel startup.

    Provides topological-sort-based startup, reverse-order shutdown,
    graceful degradation on failure, and status reporting.

    Usage::

        mgr = LifecycleManager()
        mgr.register(Component(name="db", start=db_start, stop=db_stop))
        mgr.register(Component(
            name="api", start=api_start, stop=api_stop,
            dependencies=["db"],
        ))
        await mgr.start()
        await mgr.stop()
    """

    def __init__(self) -> None:
        self._components: dict[str, Component] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, component: Component) -> None:
        """Register a component with the lifecycle manager.

        Args:
            component: The component descriptor to register.

        Raises:
            ValueError: If a component with the same name already exists.
        """
        if component.name in self._components:
            raise ValueError(
                f"Component '{component.name}' is already registered"
            )
        self._components[component.name] = component

    def get_component(self, name: str) -> Component | None:
        """Look up a registered component by name."""
        return self._components.get(name)

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_sort(self) -> list[str]:
        """Return component names in topological order (dependencies first).

        Uses Kahn's algorithm. Detects cycles and raises.

        Returns:
            List of component names in dependency order.

        Raises:
            ValueError: If a dependency references an unknown component.
            ValueError: If a circular dependency is detected.
        """
        in_degree: dict[str, int] = {}
        adjacency: dict[str, list[str]] = {n: [] for n in self._components}

        for name, comp in self._components.items():
            in_degree.setdefault(name, 0)
            for dep in comp.dependencies:
                if dep not in self._components:
                    raise ValueError(
                        f"Component '{name}' depends on unknown "
                        f"component '{dep}'"
                    )
                adjacency.setdefault(dep, []).append(name)
                in_degree[name] = in_degree.get(name, 0) + 1

        ready = [n for n, deg in in_degree.items() if deg == 0]
        sorted_names: list[str] = []

        while ready:
            name = ready.pop()
            sorted_names.append(name)
            for dependent in adjacency.get(name, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)

        if len(sorted_names) != len(self._components):
            raise ValueError("Circular dependency detected among components")

        return sorted_names

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all components in dependency order.

        Independent components (same topological depth) are started in
        parallel.  If a component fails to start, all of its transitive
        dependents are skipped and remain in their current state.

        Raises:
            RuntimeError: If no components are registered.
        """
        if not self._components:
            raise RuntimeError("No components registered")

        order = self._topological_sort()

        # Assign each component a depth (longest dependency chain).
        depth: dict[str, int] = {}
        for name in order:
            comp = self._components[name]
            if not comp.dependencies:
                depth[name] = 0
            else:
                depth[name] = 1 + max(depth[d] for d in comp.dependencies)

        # Group by depth for parallel startup.
        groups: dict[int, list[str]] = {}
        for name, d in depth.items():
            groups.setdefault(d, []).append(name)

        failed: set[str] = set()

        for level in sorted(groups.keys()):
            batch: list[str] = []
            for name in groups[level]:
                if any(dep in failed for dep in self._components[name].dependencies):
                    logger.warning(
                        "Skipping '{}' due to failed dependency", name
                    )
                    continue
                batch.append(name)

            if not batch:
                continue

            tasks = [
                asyncio.create_task(self._start_component(n))
                for n in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(batch, results):
                if isinstance(result, BaseException):
                    failed.add(name)
                    logger.error(
                        "Component '{}' failed to start: {}", name, result
                    )

        logger.info(
            "Lifecycle start complete: {} running, {} failed",
            sum(
                1 for c in self._components.values()
                if c.state is ComponentState.RUNNING
            ),
            len(failed),
        )

    async def _start_component(self, name: str) -> None:
        """Start a single component and update its state.

        Handles both async and sync callables.  Sync callables are
        offloaded to a thread executor to avoid blocking the event loop.
        """
        comp = self._components[name]
        if comp.state is not ComponentState.UNINITIALIZED:
            logger.info(
                "Component '{}' is already {} (skipping start)",
                name, comp.state.value,
            )
            return

        logger.info("Starting component '{}'", name)

        async with self._lock:
            comp.state = ComponentState.INITIALIZING

        try:
            result = comp.start()
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=comp.timeout)
            else:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, comp.start),
                    timeout=comp.timeout,
                )

            async with self._lock:
                comp.state = ComponentState.RUNNING
            logger.info("Component '{}' is RUNNING", name)

        except Exception:
            async with self._lock:
                comp.state = ComponentState.FAILED
            logger.exception("Component '{}' failed to start", name)
            raise

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Stop all components in reverse dependency order.

        Components at the same depth are stopped in parallel.
        Components that are not RUNNING or INITIALIZING are skipped.
        Individual stop failures are logged but do not block other
        components from stopping.
        """
        if not self._components:
            return

        order = self._topological_sort()

        # Recompute depth (same logic as in start).
        depth: dict[str, int] = {}
        for name in order:
            comp = self._components[name]
            if not comp.dependencies:
                depth[name] = 0
            else:
                depth[name] = 1 + max(depth[d] for d in comp.dependencies)

        max_depth = max(depth.values()) if depth else 0

        for level in range(max_depth, -1, -1):
            batch = [
                name for name in order
                if depth[name] == level
                and self._components[name].state
                in (ComponentState.RUNNING, ComponentState.INITIALIZING)
            ]
            if not batch:
                continue

            tasks = [
                asyncio.create_task(self._stop_component(n))
                for n in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(batch, results):
                if isinstance(result, BaseException):
                    logger.error(
                        "Component '{}' failed to stop: {}", name, result
                    )

        logger.info("Lifecycle stop complete")

    async def _stop_component(self, name: str) -> None:
        """Stop a single component and update its state.

        Handles both async and sync callables.  Sync callables are
        offloaded to a thread executor.
        """
        comp = self._components[name]
        if comp.state not in (ComponentState.RUNNING, ComponentState.INITIALIZING):
            logger.debug(
                "Component '{}' is {} (skipping stop)", name, comp.state.value
            )
            return

        logger.info("Stopping component '{}'", name)

        async with self._lock:
            comp.state = ComponentState.STOPPING

        try:
            result = comp.stop()
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=comp.timeout)
            else:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, comp.stop),
                    timeout=comp.timeout,
                )

            async with self._lock:
                comp.state = ComponentState.STOPPED
            logger.info("Component '{}' is STOPPED", name)

        except Exception:
            async with self._lock:
                comp.state = ComponentState.FAILED
            logger.exception("Component '{}' failed to stop", name)
            raise

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, str]:
        """Return a snapshot of all component states.

        Returns:
            Dict mapping component name to its current state string.
        """
        return {name: comp.state.value for name, comp in self._components.items()}

    def get_full_status(self) -> dict[str, dict[str, Any]]:
        """Return detailed status for every component.

        Returns:
            Dict mapping component name to a dict with ``state``,
            ``dependencies``, and ``timeout`` keys.
        """
        return {
            name: {
                "state": comp.state.value,
                "dependencies": list(comp.dependencies),
                "timeout": comp.timeout,
            }
            for name, comp in self._components.items()
        }

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------

    async def wait_for(self, component_name: str, timeout: float | None = None) -> None:
        """Wait for a component to reach the RUNNING state.

        Polls the component's state at a high frequency.  If the component
        transitions to FAILED the wait terminates immediately with an error.

        Args:
            component_name: Name of the component to wait for.
            timeout: Maximum seconds to wait.  Defaults to the component's
                configured *timeout*.

        Raises:
            ValueError: If the component is not registered.
            TimeoutError: If the component does not reach RUNNING within
                the timeout period.
            RuntimeError: If the component enters the FAILED state.
        """
        comp = self._components.get(component_name)
        if comp is None:
            raise ValueError(f"Unknown component '{component_name}'")

        effective_timeout = timeout if timeout is not None else comp.timeout

        async def _poll() -> None:
            while True:
                async with self._lock:
                    if comp.state is ComponentState.RUNNING:
                        return
                    if comp.state is ComponentState.FAILED:
                        raise RuntimeError(
                            f"Component '{component_name}' has FAILED"
                        )
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(_poll(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timed out waiting for component '{component_name}' "
                f"to reach RUNNING state (timeout={effective_timeout}s)"
            ) from None


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


async def start_with_deps(manager: LifecycleManager, name: str) -> None:
    """Start a component and all of its transitive dependencies.

    This is a convenience wrapper around :meth:`LifecycleManager.start`
    when you only need a specific component and everything it depends on.
    Components that are not in the dependency tree of *name* are left
    untouched in their current state.

    Args:
        manager: The lifecycle manager instance.
        name: The target component to start (with its deps).

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in manager._components:
        raise ValueError(f"Unknown component '{name}'")

    # Collect transitive dependencies via DFS.
    needed: set[str] = set()

    def _collect(n: str) -> None:
        if n in needed:
            return
        needed.add(n)
        for dep in manager._components[n].dependencies:
            _collect(dep)

    _collect(name)

    # Temporarily scope the manager's registry to the needed subset and start.
    saved = manager._components.copy()
    manager._components = {n: saved[n] for n in needed}

    try:
        await manager.start()
    finally:
        # Merge state changes back into the full registry.
        for n in needed:
            saved[n] = manager._components[n]
        manager._components = saved
