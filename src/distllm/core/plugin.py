"""Plugin system for DistLLM extensibility.

Provides a hook-based plugin architecture with lifecycle management,
entry point discovery, and built-in plugins.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from loguru import logger


# --- Hook Points ---

class HookPoint(str, Enum):
    """Available hook points in the DistLLM lifecycle."""
    ON_INIT = "on_init"
    ON_START = "on_start"
    ON_STOP = "on_stop"
    ON_REQUEST = "on_request"
    ON_RESPONSE = "on_response"
    ON_ERROR = "on_error"
    ON_NODE_REGISTER = "on_node_register"
    ON_NODE_UNREGISTER = "on_node_unregister"
    ON_MODEL_LOAD = "on_model_load"
    ON_MODEL_UNLOAD = "on_model_unload"


# --- Plugin Protocol ---

@runtime_checkable
class IPlugin(Protocol):
    """Base interface for DistLLM plugins."""

    name: str
    version: str = "0.1.0"
    description: str = ""

    def initialize(self, context: dict[str, Any]) -> None:
        """Called when the plugin is loaded.

        Args:
            context: Shared context dict with coordinator, config, etc.
        """
        ...

    def shutdown(self) -> None:
        """Called when the plugin is being unloaded."""
        ...


class HookRegistry:
    """Registry for hook callbacks."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable]] = {}

    def register(self, hook: str, callback: Callable) -> None:
        """Register a callback for a hook point."""
        if hook not in self._hooks:
            self._hooks[hook] = []
        self._hooks[hook].append(callback)
        logger.debug(f"Registered callback for hook: {hook}")

    def unregister(self, hook: str, callback: Callable) -> None:
        """Remove a callback from a hook point."""
        if hook in self._hooks:
            self._hooks[hook] = [c for c in self._hooks[hook] if c != callback]

    def emit(self, hook: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Fire all callbacks for a hook point."""
        results = []
        callbacks = self._hooks.get(hook, [])
        for callback in callbacks:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook '{hook}' callback failed: {e}")
        return results

    def list_hooks(self) -> dict[str, int]:
        """Return count of callbacks per hook point."""
        return {hook: len(cbs) for hook, cbs in self._hooks.items()}


class PluginManager:
    """Manages plugin lifecycle and discovery."""

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        self.hooks = HookRegistry()
        self._plugins: dict[str, IPlugin] = {}
        self._context = context or {}

    def register_plugin(self, plugin: IPlugin) -> None:
        """Register and initialize a plugin."""
        name = plugin.name
        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already registered, replacing")
            self._plugins[name].shutdown()

        self._plugins[name] = plugin
        plugin.initialize(self._context)
        logger.info(f"Plugin '{name}' v{plugin.version} loaded")

    def unregister_plugin(self, name: str) -> None:
        """Unregister and shutdown a plugin."""
        plugin = self._plugins.pop(name, None)
        if plugin:
            try:
                plugin.shutdown()
                logger.info(f"Plugin '{name}' unloaded")
            except Exception as e:
                logger.error(f"Plugin '{name}' shutdown failed: {e}")

    def get_plugin(self, name: str) -> IPlugin | None:
        """Get a registered plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict[str, str]]:
        """Return information about all loaded plugins."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
            }
            for p in self._plugins.values()
        ]

    def discover_entry_points(self) -> list[IPlugin]:
        """Discover plugins via Python entry points."""
        plugins = []
        try:
            entry_points = importlib.metadata.entry_points()
            if hasattr(entry_points, "select"):
                eps = entry_points.select(group="distllm.plugins")
            else:
                eps = entry_points.get("distllm.plugins", [])

            for ep in eps:
                try:
                    plugin_cls = ep.load()
                    plugin = plugin_cls()
                    plugins.append(plugin)
                    logger.info(f"Discovered plugin via entry point: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load entry point {ep.name}: {e}")
        except Exception as e:
            logger.debug(f"No entry point plugins found: {e}")
        return plugins

    def discover_from_config(self, config: dict[str, Any]) -> list[IPlugin]:
        """Load plugins specified in configuration.

        Security: Only allows imports from trusted plugin modules.
        Prevents arbitrary code execution via malicious plugin paths.
        """
        plugins = []
        plugin_configs = config.get("plugins", [])

        # Security: Allowlist of safe plugin module prefixes
        SAFE_PREFIXES = ("distllm.", "distllm_plugins.", "plugins.")

        for plugin_cfg in plugin_configs:
            try:
                if isinstance(plugin_cfg, str):
                    module_path = plugin_cfg
                    plugin_config = {}
                else:
                    module_path = plugin_cfg["module"]
                    plugin_config = plugin_cfg.get("config", {})

                # Security: Validate module path against allowlist
                if not any(module_path.startswith(prefix) for prefix in SAFE_PREFIXES):
                    logger.error(
                        f"Plugin '{module_path}' blocked: not in allowed module prefixes {SAFE_PREFIXES}"
                    )
                    continue

                # Security: Prevent path traversal and special characters
                if ".." in module_path or "#" in module_path or ";" in module_path:
                    logger.error(f"Plugin '{module_path}' blocked: invalid characters in module path")
                    continue

                module_name, class_name = module_path.rsplit(".", 1)

                # Security: Validate class name is a valid identifier
                if not class_name.isidentifier():
                    logger.error(f"Plugin '{class_name}' blocked: invalid class name")
                    continue

                module = importlib.import_module(module_name)
                plugin_cls = getattr(module, class_name)

                # Security: Verify it's a class before instantiating
                if not isinstance(plugin_cls, type):
                    logger.error(f"Plugin '{module_path}' blocked: not a class")
                    continue

                plugin = plugin_cls(**plugin_config)
                plugins.append(plugin)
            except Exception as e:
                logger.error(f"Failed to load plugin {module_path}: {e}")
        return plugins

    def load_all(self, config: dict[str, Any] | None = None) -> None:
        """Discover and load all plugins."""
        # Load from entry points
        for plugin in self.discover_entry_points():
            self.register_plugin(plugin)

        # Load from config
        if config:
            for plugin in self.discover_from_config(config):
                self.register_plugin(plugin)

    def shutdown_all(self) -> None:
        """Shutdown all plugins."""
        for name in list(self._plugins.keys()):
            self.unregister_plugin(name)

    def emit_hook(self, hook: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Emit a hook to all registered plugins."""
        return self.hooks.emit(hook, *args, **kwargs)


# --- Built-in Plugins ---

class RequestLoggingPlugin:
    """Logs all requests and responses."""

    name = "request_logger"
    version = "0.1.0"
    description = "Logs request/response cycles for debugging"

    def __init__(self, log_level: str = "INFO") -> None:
        self.log_level = log_level

    def initialize(self, context: dict[str, Any]) -> None:
        hooks = context.get("hooks")
        if hooks and isinstance(hooks, HookRegistry):
            hooks.register(HookPoint.ON_REQUEST, self._on_request)
            hooks.register(HookPoint.ON_RESPONSE, self._on_response)

    def shutdown(self) -> None:
        pass

    def _on_request(self, request: Any) -> None:
        logger.log(self.log_level, f"Request: {request}")

    def _on_response(self, response: Any) -> None:
        logger.log(self.log_level, f"Response: {response}")


class MetricsPlugin:
    """Collects plugin-level metrics."""

    name = "metrics_collector"
    version = "0.1.0"
    description = "Collects and exports plugin metrics"

    def __init__(self) -> None:
        self._metrics: dict[str, Any] = {}

    def initialize(self, context: dict[str, Any]) -> None:
        hooks = context.get("hooks")
        if hooks and isinstance(hooks, HookRegistry):
            hooks.register(HookPoint.ON_REQUEST, self._count_request)

    def shutdown(self) -> None:
        self._metrics.clear()

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def _count_request(self, request: Any) -> None:
        self._metrics["total_requests"] = self._metrics.get("total_requests", 0) + 1


class HealthCheckPlugin:
    """Provides plugin health status."""

    name = "health_check"
    version = "0.1.0"
    description = "Reports plugin health for monitoring"

    def __init__(self) -> None:
        self._healthy = True

    def initialize(self, context: dict[str, Any]) -> None:
        hooks = context.get("hooks")
        if hooks and isinstance(hooks, HookRegistry):
            hooks.register(HookPoint.ON_ERROR, self._on_error)

    def shutdown(self) -> None:
        pass

    def is_healthy(self) -> bool:
        return self._healthy

    def _on_error(self, error: Exception) -> None:
        self._healthy = False
        logger.error(f"Plugin health degraded: {error}")


BUILTIN_PLUGINS = [
    RequestLoggingPlugin,
    MetricsPlugin,
    HealthCheckPlugin,
]
