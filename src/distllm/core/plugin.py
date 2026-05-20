"""Plugin system for DistLLM extensibility.

Provides a hook-based plugin architecture with lifecycle management,
entry point discovery, built-in plugins, and marketplace support.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, field
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


# --- Plugin Context ---

@dataclass
class PluginContext:
    """Restricted context passed to plugins.

    Replaces the raw dict with a typed, documented interface.
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
        return key in ("config", "hooks", "logger", "coordinator_ref", "metrics")


# --- Plugin Protocol ---

@runtime_checkable
class IPlugin(Protocol):
    """Base interface for DistLLM plugins."""

    name: str
    version: str = "0.1.0"
    description: str = ""

    @property
    def metadata(self) -> Any | None:
        """Optional PluginMetadata from the marketplace system."""
        return None

    def initialize(self, context: PluginContext | dict[str, Any]) -> None:
        """Called when the plugin is loaded.

        Args:
            context: PluginContext (preferred) or dict for backward compatibility.
        """
        ...

    def shutdown(self) -> None:
        """Called when the plugin is being unloaded."""
        ...

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Validate plugin-specific configuration.

        Returns a list of validation errors (empty if valid).
        """
        return []


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

    def __init__(
        self,
        context: PluginContext | dict[str, Any] | None = None,
        host_version: str = "0.1.0",
    ) -> None:
        self.hooks = HookRegistry()
        self._plugins: dict[str, IPlugin] = {}
        self._context = context or PluginContext()
        self._host_version = host_version

        # Marketplace subsystems
        self._metadata_registry: dict[str, Any] = {}
        self._compatibility_checker: Any = None
        self._sandbox: Any = None
        self._telemetry: Any = None
        self._config_validator: Any = None
        self._installer: Any = None

    def set_marketplace_subsystems(
        self,
        *,
        compatibility_checker: Any = None,
        sandbox: Any = None,
        telemetry: Any = None,
        config_validator: Any = None,
        installer: Any = None,
    ) -> None:
        """Wire marketplace subsystems into the PluginManager."""
        self._compatibility_checker = compatibility_checker
        self._sandbox = sandbox
        self._telemetry = telemetry
        self._config_validator = config_validator
        self._installer = installer

    def register_plugin(self, plugin: IPlugin, config: dict[str, Any] | None = None) -> None:
        """Register and initialize a plugin.

        Args:
            plugin: The plugin instance.
            config: Optional plugin-specific configuration (validated if schema exists).
        """
        name = plugin.name

        # Validate config if schema registered
        if config and self._config_validator:
            errors = self._config_validator.validate_config(name, config)
            if errors:
                logger.error(f"Plugin '{name}' config validation failed: {errors}")
                raise ValueError(f"Invalid plugin config: {errors}")

        # Check compatibility if checker available
        if self._compatibility_checker and hasattr(plugin, "metadata") and plugin.metadata:
            meta = plugin.metadata
            result = self._compatibility_checker.check_compatibility(
                min_host_version=getattr(meta, "min_host_version", None),
                max_host_version=getattr(meta, "max_host_version", None),
                dependencies=getattr(meta, "dependencies", []),
            )
            if not result.can_load:
                logger.error(f"Plugin '{name}' incompatible: {result.errors}")
                raise ValueError(f"Plugin incompatible: {result.errors}")
            for warning in result.warnings:
                logger.warning(f"Plugin '{name}' warning: {warning}")

        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already registered, replacing")
            self._plugins[name].shutdown()

        self._plugins[name] = plugin

        # Merge plugin config into context if provided
        if isinstance(self._context, PluginContext) and config:
            self._context.config.update(config)

        plugin.initialize(self._context)
        logger.info(f"Plugin '{name}' v{plugin.version} loaded")

        # Store metadata if available
        if hasattr(plugin, "metadata") and plugin.metadata:
            self._metadata_registry[name] = plugin.metadata

    def unregister_plugin(self, name: str) -> None:
        """Unregister and shutdown a plugin."""
        plugin = self._plugins.pop(name, None)
        if plugin:
            try:
                plugin.shutdown()
                logger.info(f"Plugin '{name}' unloaded")
            except Exception as e:
                logger.error(f"Plugin '{name}' shutdown failed: {e}")
        self._metadata_registry.pop(name, None)

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

    def get_plugin_metadata(self, name: str) -> Any | None:
        """Get marketplace metadata for a plugin."""
        return self._metadata_registry.get(name)

    def list_all_metadata(self) -> dict[str, Any]:
        """Return metadata for all registered plugins."""
        return dict(self._metadata_registry)

    def check_compatibility(self, metadata: Any) -> Any:
        """Check plugin compatibility with current host.

        Returns CompatibilityResult.
        """
        if not self._compatibility_checker:
            raise RuntimeError("Compatibility checker not configured")
        return self._compatibility_checker.check_compatibility(
            min_host_version=getattr(metadata, "min_host_version", None),
            max_host_version=getattr(metadata, "max_host_version", None),
            dependencies=getattr(metadata, "dependencies", []),
        )

    def install_plugin(
        self,
        plugin_name: str,
        version: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Any:
        """Install a plugin from the marketplace.

        Returns PluginInstallResult.
        """
        if not self._installer:
            raise RuntimeError("Plugin installer not configured")
        result = self._installer.install(plugin_name, version)
        if result.success and result.metadata:
            self._metadata_registry[plugin_name] = result.metadata
        return result

    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin from the marketplace."""
        if not self._installer:
            raise RuntimeError("Plugin installer not configured")
        self.unregister_plugin(plugin_name)
        return self._installer.uninstall(plugin_name)

    def get_plugin_telemetry(self, plugin_name: str | None = None) -> Any:
        """Get telemetry data for plugins."""
        if not self._telemetry:
            raise RuntimeError("Telemetry not configured")
        if plugin_name:
            return self._telemetry.get_plugin_stats(plugin_name)
        return self._telemetry.get_all_stats()

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

    async def emit_hook_sandboxed(
        self, hook: str, *args: Any, timeout_s: float | None = None, **kwargs: Any
    ) -> list[Any]:
        """Emit a hook to all plugins with sandboxed execution.

        Runs each callback through the PluginSandbox for timeout/memory protection.
        """
        if not self._sandbox:
            return self.emit_hook(hook, *args, **kwargs)

        results = []
        for name, plugin in self._plugins.items():
            # Find callbacks registered for this hook
            callbacks = self.hooks._hooks.get(hook, [])
            for callback in callbacks:
                stats = await self._sandbox.run_hook_async(
                    plugin_name=name,
                    hook_name=hook,
                    callback=callback,
                    *args,
                    timeout_s=timeout_s,
                    **kwargs,
                )
                # Record telemetry
                if self._telemetry:
                    self._telemetry.record_usage(
                        plugin_name=name,
                        hook_name=hook,
                        duration_ms=stats.duration_ms,
                        success=stats.success,
                        error=stats.error,
                    )
                if stats.success:
                    results.append(None)  # Hook callbacks don't return meaningful values
        return results


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
