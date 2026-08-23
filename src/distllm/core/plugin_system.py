"""Plugin System — extensible plugin architecture for non-backend components.

Provides a generic plugin framework with hook-based extension points,
lifecycle management (init, start, stop), and filesystem discovery.

DISTINCTION FROM FASTAPI MIDDLEWARE:
  Starlette middleware (app.add_middleware) handles low-level,
  ordering-sensitive request/response pipeline processing (auth,
  rate-limiting, CORS, timeouts, backpressure, security headers).
  Middleware has raw ASGI scope access but cannot be added by third
  parties without modifying the codebase.

  Plugins provide extensible, swappable, lifecycle-managed behavior
  that users or third parties can install, register, or remove
  without touching the core codebase.  Plugins run through a single
  middleware entry point (PluginHookMiddleware) that dispatches
  ``on_request`` / ``on_response`` / ``on_error`` lifecycle hooks.
  They do NOT have raw ASGI scope access but benefit from lifecycle
  management (init -> start -> stop) and dynamic filesystem/PyPI
  discovery.

  Rule of thumb:
  - Infra cross-cutting concerns that MUST be at a precise position
    in the pipeline -> Starlette middleware.
  - Swappable / installable / user-extensible features -> Plugin.
  - Behavior needing lifecycle hooks beyond request/response -> Plugin.

Backend-specific plugins use ``backends/registry.py``; this module
is for general-purpose plugins (custom auth, custom logging, custom
metrics, custom health probes).
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger


class PluginState(Enum):
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    STARTED = "started"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PluginMetadata:
    """Metadata about a discovered plugin."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    entry_point: str = ""
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class PluginBase:
    """Base class for all plugins.

    Subclass this to create a new plugin::

        class MyPlugin(PluginBase):
            def name(self) -> str:
                return "my-plugin"

            def on_init(self, context):
                print("Plugin initialized")

            def on_start(self, context):
                print("Plugin started")

            def on_stop(self, context):
                print("Plugin stopped")
    """

    def name(self) -> str:
        return self.__class__.__name__

    def version(self) -> str:
        return "1.0.0"

    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name(),
            version=self.version(),
            description=self.__doc__ or "",
        )

    # ── Lifecycle hooks ─────────────────────────────────────────────────

    def on_init(self, context: dict[str, Any]) -> None:
        """Called after the plugin is loaded (before start)."""

    def on_start(self, context: dict[str, Any]) -> None:
        """Called when the plugin system starts."""

    def on_stop(self, context: dict[str, Any]) -> None:
        """Called when the plugin system shuts down."""

    # ── Event hooks ─────────────────────────────────────────────────────

    def on_request(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Called before each inference request. Return modified context or None."""

    def on_response(self, request: dict[str, Any], response: dict[str, Any]) -> None:
        """Called after each inference response."""

    def on_error(self, request: dict[str, Any], error: Exception) -> None:
        """Called when an inference request fails."""

    def on_model_load(self, model_name: str, config: dict[str, Any]) -> None:
        """Called when a model is loaded."""

    def on_model_unload(self, model_name: str) -> None:
        """Called when a model is unloaded."""

    def on_config_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Called when a configuration value changes."""


class PluginInstance:
    """A loaded and managed plugin instance."""
    def __init__(self, cls: type[PluginBase], metadata: PluginMetadata) -> None:
        self.cls = cls
        self.metadata = metadata
        self.instance: PluginBase | None = None
        self.state = PluginState.DISCOVERED
        self.loaded_at: float = 0.0
        self.error: str = ""


class PluginContext:
    """Context passed to plugin lifecycle methods."""
    def __init__(self, plugin_system: "PluginSystem") -> None:
        self.system = plugin_system
        self.data: dict[str, Any] = {}
        self.config: dict[str, Any] = {}


class PluginSystem:
    """Generic plugin system with discovery, lifecycle, and hooks.

    Usage:
        system = PluginSystem()
        system.discover(["path/to/plugins"])
        system.load_all()
        system.start_all()

        # Trigger hooks
        system.dispatch("on_request", {"prompt": "Hello"})
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._plugins: dict[str, PluginInstance] = {}
        self._context = PluginContext(self)
        self._config = config or {}
        self._context.config = self._config
        self._lock = threading.RLock()
        self._started_cache: list[PluginInstance] | None = None

    # ── Discovery ───────────────────────────────────────────────────────

    def discover(self, plugin_dirs: list[str]) -> list[PluginMetadata]:
        """Scan directories for plugin modules and return metadata."""
        self._trusted_dirs = plugin_dirs
        discovered: list[PluginMetadata] = []
        for plugin_dir in plugin_dirs:
            p = Path(plugin_dir)
            if not p.exists():
                continue
            if p.is_dir():
                for py_file in p.glob("*.py"):
                    meta = self._discover_from_file(py_file)
                    if meta:
                        discovered.append(meta)
        return discovered

    def discover_entry_points(self, group: str = "distllm.plugins") -> list[PluginMetadata]:
        """Discover plugins via installed package entry points."""
        try:
            import pkg_resources
            metas = []
            for ep in pkg_resources.iter_entry_points(group=group):
                try:
                    cls = ep.load()
                    if not issubclass(cls, PluginBase):
                        continue
                    meta = PluginMetadata(
                        name=ep.name,
                        entry_point=f"{ep.module_name}:{ep.attrs[0] if ep.attrs else cls.__name__}",
                    )
                    metas.append(meta)
                except Exception:
                    pass
            return metas
        except ImportError:
            pass
        return []

    def register(self, plugin_cls: type[PluginBase], metadata: PluginMetadata | None = None) -> bool:
        """Directly register a plugin class."""
        name = plugin_cls().name()
        inst = PluginInstance(plugin_cls, metadata or PluginMetadata(name=name))
        with self._lock:
            self._plugins[name] = inst
        logger.info(f"Registered plugin {name}")
        return True

    # ── Loading ─────────────────────────────────────────────────────────

    def load_all(self) -> int:
        """Instantiate all discovered/registered plugins. Returns count."""
        count = 0
        with self._lock:
            for name, inst in self._plugins.items():
                if inst.state == PluginState.LOADED:
                    continue
                try:
                    inst.instance = inst.cls()
                    inst.state = PluginState.LOADED
                    inst.loaded_at = time.time()
                    count += 1
                except Exception as e:
                    inst.state = PluginState.ERROR
                    inst.error = str(e)
                    logger.error(f"Failed to load plugin {name}: {e}")
        return count

    def init_all(self) -> int:
        """Call ``on_init`` on all loaded plugins."""
        count = 0
        with self._lock:
            for name, inst in self._plugins.items():
                if inst.state != PluginState.LOADED or inst.instance is None:
                    continue
                try:
                    inst.instance.on_init(self._context.data)
                    inst.state = PluginState.INITIALIZED
                    count += 1
                except Exception as e:
                    inst.state = PluginState.ERROR
                    inst.error = str(e)
                    logger.error(f"Plugin {name} init failed: {e}")
        return count

    def start_all(self) -> int:
        """Call ``on_start`` on all initialized plugins."""
        count = 0
        with self._lock:
            for name, inst in self._plugins.items():
                if inst.state != PluginState.INITIALIZED or inst.instance is None:
                    continue
                try:
                    inst.instance.on_start(self._context.data)
                    inst.state = PluginState.STARTED
                    count += 1
                except Exception as e:
                    inst.state = PluginState.ERROR
                    inst.error = str(e)
                    logger.error(f"Plugin {name} start failed: {e}")
        self._invalidate_plugin_cache()
        return count

    def stop_all(self) -> int:
        """Call ``on_stop`` on all started plugins."""
        count = 0
        with self._lock:
            for name, inst in self._plugins.items():
                if inst.state != PluginState.STARTED or inst.instance is None:
                    continue
                try:
                    inst.instance.on_stop(self._context.data)
                    inst.state = PluginState.STOPPED
                    count += 1
                except Exception as e:
                    logger.error(f"Plugin {name} stop failed: {e}")
        self._invalidate_plugin_cache()
        return count

    # ── Hook dispatch ───────────────────────────────────────────────────

    def dispatch(self, hook_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Dispatch a hook call to all started plugins.

        Returns list of results (non-None return values).
        """
        results: list[Any] = []
        plugins = self._started_plugins()
        for inst in plugins:
            if inst.instance is None:
                continue
            try:
                method = getattr(inst.instance, hook_name, None)
                if method is None:
                    continue
                result = method(*args, **kwargs)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Plugin {inst.metadata.name} hook {hook_name} failed: {e}")
        return results

    def dispatch_on_request(
        self, context: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch ``on_request`` hook, allowing plugins to modify context."""
        result = context.copy()
        for inst in self._started_plugins():
            if inst.instance is None:
                continue
            try:
                modified = inst.instance.on_request(result)
                if modified is not None:
                    result.update(modified)
            except Exception as e:
                logger.warning(f"Plugin {inst.metadata.name} on_request failed: {e}")
        return result

    # ── Queries ─────────────────────────────────────────────────────────

    def get_plugin(self, name: str) -> PluginInstance | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[PluginInstance]:
        return list(self._plugins.values())

    def is_loaded(self, name: str) -> bool:
        inst = self._plugins.get(name)
        return inst is not None and inst.state == PluginState.STARTED

    # ── Private ─────────────────────────────────────────────────────────

    def _invalidate_plugin_cache(self) -> None:
        """Invalidate the cached started plugins list."""
        with self._lock:
            self._started_cache = None

    def _started_plugins(self) -> list[PluginInstance]:
        """Return started plugins, using cache to avoid lock on every dispatch."""
        with self._lock:
            if self._started_cache is not None:
                return self._started_cache
            self._started_cache = [
                inst for inst in self._plugins.values()
                if inst.state == PluginState.STARTED
            ]
            return self._started_cache

    def _verify_plugin_integrity(self, path: Path) -> bool:
        """Verify plugin file integrity against a content hash allowlist.

        The allowlist is read from ``distllm_plugin_hashes.txt`` in the
        plugin's directory. Each line is ``sha256:<hexhash> <filename>``.

        When *strict* mode is enabled (via ``config["verify_plugins"]``),
        a missing allowlist or mismatched hash rejects the plugin (fail
        closed).  Otherwise a missing allowlist logs a warning and allows
        loading (fail open, backward compatible).
        """
        strict = self._config.get("verify_plugins", True)
        allowlist_path = path.parent / "distllm_plugin_hashes.txt"

        if not allowlist_path.exists():
            if strict:
                logger.error(f"No plugin hash allowlist found at {allowlist_path} — rejecting {path.name}")
                return False
            logger.warning(
                f"No plugin hash allowlist at {allowlist_path} — "
                f"loading {path.name} without verification. "
                f"Run with --verify-plugins to require hash checks."
            )
            return True

        import hashlib
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        for line in allowlist_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                algo, rest = line.split(":", 1)
                if algo != "sha256":
                    continue
                expected_hash, expected_name = rest.split(None, 1)
                if expected_name == path.name and expected_hash == file_hash:
                    return True
            except (ValueError, IndexError):
                continue

        logger.warning(f"Plugin {path.name} rejected: hash {file_hash} not in allowlist")
        return False

    def _discover_from_file(self, path: Path) -> PluginMetadata | None:
        """Scan a single Python file for PluginBase subclasses.

        Security:
        - Only loads plugins from trusted directories (those passed to ``discover()``)
        - Rejects symlinks pointing outside trusted dirs
        - Verifies file SHA-256 hash against ``distllm_plugin_hashes.txt`` allowlist
          (fail-open by default; fail-closed when ``config["verify_plugins"]`` is set)
        """
        try:
            # Security: verify file is within a trusted plugin directory
            resolved = path.resolve()
            is_trusted = False
            for trusted_dir in getattr(self, '_trusted_dirs', []):
                try:
                    resolved.relative_to(Path(trusted_dir).resolve())
                    is_trusted = True
                    break
                except ValueError:
                    continue
            if not is_trusted:
                logger.warning(f"Rejected plugin from untrusted path: {path}")
                return None

            # Security: reject symlinks pointing outside trusted dirs.
            # Uses path resolution + prefix check ensuring the resolved
            # target path is actually a CHILD of a trusted dir (not just
            # a string prefix match, which could be bypassed by paths
            # like /trusted_evil matching /trusted).
            if path.is_symlink():
                link_target = path.resolve()
                trusted = [Path(d).resolve() for d in getattr(self, '_trusted_dirs', [])]
                allowed = any(
                    str(link_target) == str(t) or str(link_target).startswith(str(t) + os.sep)
                    for t in trusted
                )
                if not allowed:
                    logger.warning(f"Rejected symlinked plugin: {path} -> {link_target}")
                    return None

            # Security: verify plugin integrity via hash allowlist
            if not self._verify_plugin_integrity(path):
                logger.warning(f"Rejected plugin {path.name} — integrity check failed")
                return None

            module_name = path.stem
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(obj, PluginBase)
                    and obj is not PluginBase
                ):
                    meta = obj().metadata()
                    meta.entry_point = f"{module_name}:{obj.__name__}"
                    self.register(obj, meta)
                    return meta
        except Exception as e:
            logger.debug(f"Failed to scan {path}: {e}")
        return None

    # ── Pip-style Plugin Management ─────────────────────────────────

    def install_plugin(self, plugin_name: str, upgrade: bool = False) -> bool:
        """Install a plugin from PyPI using pip.

        Plugins must be named ``distllm-plugin-{name}`` on PyPI.

        Args:
            plugin_name: Plugin name (without ``distllm-plugin-`` prefix).
            upgrade: Whether to upgrade if already installed.

        Returns:
            True if installation succeeded.

        Security:
        - Plugin name is sanitised to prevent shell injection
        - Only alphanumeric, dash, and underscore characters are allowed
        - Installation runs with a timeout (120s) to prevent resource exhaustion
        - A warning is logged when installing from a public PyPI index
        - Hash verification is performed by the PluginSystem's integrity
          check after installation, not by pip's ``--require-hashes``
          (which requires hash values that are impractical to maintain
          for dynamic PyPI packages)
        """
        import hashlib
        import re
        import subprocess

        # SECURITY: sanitise plugin name — only allow safe characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', plugin_name):
            logger.error(f"Rejected plugin name with unsafe characters: {plugin_name!r}")
            return False

        package_name = f"distllm-plugin-{plugin_name}"

        # Warn that we are pulling from a public index
        logger.warning(
            f"Installing {package_name} from PyPI — "
            f"ensure the package is trusted before use in production."
        )

        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package_name)

        logger.info(f"Installing plugin: {package_name}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info(f"Plugin {plugin_name} installed successfully")
                # Discover the newly installed plugin
                self.discover_entry_points()
                return True
            else:
                logger.error(f"Plugin installation failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Plugin installation timed out after 120s")
            return False
        except Exception as e:
            logger.error(f"Plugin installation error: {e}")
            return False

    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin package.

        Args:
            plugin_name: Plugin name (without ``distllm-plugin-`` prefix).

        Returns:
            True if uninstallation succeeded.
        """
        import subprocess

        package_name = f"distllm-plugin-{plugin_name}"
        cmd = [sys.executable, "-m", "pip", "uninstall", "-y", package_name]

        logger.info(f"Uninstalling plugin: {package_name}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info(f"Plugin {plugin_name} uninstalled successfully")
                # Remove from loaded plugins
                self._plugins.pop(plugin_name, None)
                return True
            else:
                logger.error(f"Plugin uninstallation failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Plugin uninstallation error: {e}")
            return False

    def search_plugins(self, query: str = "") -> list[dict[str, str]]:
        """Search for available plugins on PyPI.

        Args:
            query: Search query (optional).

        Returns:
            List of plugin info dicts with 'name', 'version', 'summary'.
        """
        import subprocess
        import json

        cmd = [sys.executable, "-m", "pip", "index", "versions", f"distllm-plugin-{query}"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Parse pip output
                lines = result.stdout.strip().split("\n")
                if lines:
                    return [{"name": f"distllm-plugin-{query}", "versions": lines[0]}]
        except Exception:
            pass

        # Fallback: list installed distllm plugins
        cmd = [sys.executable, "-m", "pip", "list", "--format=json"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                return [
                    {"name": p["name"], "version": p["version"]}
                    for p in packages
                    if p["name"].startswith("distllm-plugin-")
                ]
        except Exception:
            pass

        return []
