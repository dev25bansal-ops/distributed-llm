"""Plugin registry — discover, load, and manage community extensions.

Uses ``importlib.metadata.entry_points`` to discover plugins registered
by third-party packages, plus a local registry for in-process plugins.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Optional

from loguru import logger


class PluginRegistry:
    """Registry for DistLLM plugins.

    Supports:
    1. Entry-point discovery (``importlib.metadata.entry_points``)
    2. Direct registration via ``register()``
    3. ``@cli_plugin`` decorator for ad-hoc commands
    """

    def __init__(self):
        self._plugins: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def discover(self, entry_point_group: str = "distllm.plugins") -> int:
        """Discover plugins via importlib entry_points.

        Returns the number of newly discovered plugins.
        """
        import importlib.metadata
        discovered = 0
        for ep in importlib.metadata.entry_points(group=entry_point_group):
            try:
                plugin_fn = ep.load()
                name = ep.name
                metadata = plugin_fn() if callable(plugin_fn) else plugin_fn
                self.register(name, metadata)
                discovered += 1
            except Exception as e:
                logger.warning(f"Failed to load plugin {ep.name}: {e}")
        return discovered

    def register(self, name: str, metadata: dict[str, Any]) -> None:
        """Register a plugin manually."""
        with self._lock:
            self._plugins[name] = {
                "name": name,
                "description": metadata.get("description", ""),
                "version": metadata.get("version", "0.1.0"),
                "entry_point": metadata.get("entry_point"),
                "cli_commands": metadata.get("cli_commands", []),
            }
            logger.info(f"Plugin registered: {name} v{self._plugins[name]['version']}")

    def get(self, name: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._plugins.get(name)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._plugins.values())

    def install(self, package_name: str) -> bool:
        """Attempt to pip-install a plugin package and discover it.

        Delegates to ``pip install <package_name>`` and then
        re-discovers entry points.
        """
        import subprocess
        import sys
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name],
            )
            self.discover()
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install plugin {package_name}: {e}")
            return False

    def stats(self) -> dict[str, Any]:
        return {
            "total_plugins": len(self._plugins),
            "names": list(self._plugins.keys()),
        }


# Global singleton
_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def get_plugin(name: str) -> Optional[dict[str, Any]]:
    return get_registry().get(name)


def cli_plugin(
    name: Optional[str] = None,
    description: str = "",
    version: str = "0.1.0",
):
    """Decorator that registers a function as a CLI plugin.

    Usage::

        @cli_plugin(name="my-exporter", description="Export to S3")
        def my_exporter(args):
            ...

        # Later, discover and run:
        from distllm.plugins import get_plugin
    """
    def decorator(func: Callable) -> Callable:
        plugin_name = name or func.__name__
        get_registry().register(plugin_name, {
            "description": description or func.__doc__ or "",
            "version": version,
            "entry_point": func,
            "cli_commands": [plugin_name],
        })
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator
