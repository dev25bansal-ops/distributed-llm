"""Plugin Marketplace — discover, install, and manage DistLLM plugins.

Extends the plugin system with:
- Custom backend plugins (inference engines)
- Custom routing strategies (model selection algorithms)
- Custom sampling algorithms (token sampling strategies)
- Plugin discovery via PyPI and local directories
- Plugin registry with metadata and dependencies

Usage::

    marketplace = PluginMarketplace()
    plugins = marketplace.discover()
    marketplace.install("custom-backend-vllm")
    marketplace.register_sampling_strategy("top-a", top_a_sampling)
"""

from __future__ import annotations

import importlib
import inspect
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger


class PluginCategory(str, Enum):
    """Plugin category types."""
    BACKEND = "backend"           # Inference backend (vLLM, TRT-LLM, etc.)
    ROUTING = "routing"           # Model routing strategy
    SAMPLING = "sampling"         # Token sampling algorithm
    MIDDLEWARE = "middleware"     # Request/response middleware
    TRANSPORT = "transport"      # Communication transport
    CACHE = "cache"              # Caching strategy
    MONITORING = "monitoring"    # Observability plugin
    AUTH = "auth"                # Authentication plugin
    OTHER = "other"


@dataclass
class PluginEntry:
    """A registered plugin in the marketplace."""
    name: str
    category: PluginCategory
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    entry_point: str = ""        # module:ClassName
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    installed: bool = False
    enabled: bool = True
    source: str = ""             # "local", "pypi", "git"
    config_schema: dict = field(default_factory=dict)


@dataclass
class SamplingStrategy:
    """A custom sampling algorithm."""
    name: str
    description: str
    fn: Callable  # (logits, temperature, top_k, top_p, **kwargs) -> token_id
    requires_params: list[str] = field(default_factory=list)


@dataclass
class RoutingStrategy:
    """A custom routing strategy."""
    name: str
    description: str
    fn: Callable  # (request, available_models, **kwargs) -> model_name
    requires_params: list[str] = field(default_factory=list)


class PluginMarketplace:
    """Plugin marketplace with discovery, installation, and registry.

    Supports:
    - Local plugin discovery (Python files in directories)
    - PyPI plugin discovery (distllm-plugin-* packages)
    - Custom backend registration
    - Custom sampling strategy registration
    - Custom routing strategy registration
    """

    def __init__(
        self,
        plugin_dirs: list[str] | None = None,
        enable_pypi: bool = True,
    ):
        self._plugin_dirs = plugin_dirs or []
        self._enable_pypi = enable_pypi
        self._registry: dict[str, PluginEntry] = {}
        self._sampling_strategies: dict[str, SamplingStrategy] = {}
        self._routing_strategies: dict[str, RoutingStrategy] = {}
        self._backend_factories: dict[str, Callable] = {}
        self._lock = threading.Lock()

        # Register built-in strategies
        self._register_builtin_strategies()

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover(self) -> list[PluginEntry]:
        """Discover all available plugins from local dirs and PyPI.

        Returns:
            List of discovered PluginEntry objects.
        """
        discovered = []

        # Local discovery
        for plugin_dir in self._plugin_dirs:
            p = Path(plugin_dir)
            if not p.exists():
                continue
            for py_file in p.glob("*.py"):
                entry = self._discover_file(py_file)
                if entry:
                    discovered.append(entry)

        # PyPI discovery
        if self._enable_pypi:
            pypi_plugins = self._discover_pypi()
            discovered.extend(pypi_plugins)

        return discovered

    def _discover_file(self, path: Path) -> PluginEntry | None:
        """Discover a plugin from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(path.stem, str(path))
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Look for plugin metadata
            for name, obj in inspect.getmembers(mod):
                if name == "PLUGIN_METADATA" and isinstance(obj, dict):
                    entry = PluginEntry(
                        name=obj.get("name", path.stem),
                        category=PluginCategory(obj.get("category", "other")),
                        version=obj.get("version", "1.0.0"),
                        description=obj.get("description", ""),
                        author=obj.get("author", ""),
                        entry_point=f"{path.stem}:{obj.get('class_name', '')}",
                        tags=obj.get("tags", []),
                        source="local",
                    )
                    with self._lock:
                        self._registry[entry.name] = entry
                    return entry
        except Exception as e:
            logger.debug(f"Failed to discover plugin {path}: {e}")
        return None

    def _discover_pypi(self) -> list[PluginEntry]:
        """Discover plugins from PyPI (distllm-plugin-* packages)."""
        import subprocess
        discovered = []

        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                for pkg in packages:
                    if pkg["name"].startswith("distllm-plugin-"):
                        name = pkg["name"].replace("distllm-plugin-", "")
                        entry = PluginEntry(
                            name=name,
                            category=PluginCategory.OTHER,
                            version=pkg["version"],
                            installed=True,
                            source="pypi",
                        )
                        with self._lock:
                            self._registry[name] = entry
                        discovered.append(entry)
        except Exception:
            pass

        return discovered

    # ── Installation ──────────────────────────────────────────────────────

    def install(self, plugin_name: str, upgrade: bool = False) -> bool:
        """Install a plugin from PyPI.

        Args:
            plugin_name: Plugin name (without distllm-plugin- prefix).
            upgrade: Whether to upgrade if already installed.

        Returns:
            True if installation succeeded.
        """
        import subprocess
        import sys

        package = f"distllm-plugin-{plugin_name}"
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)

        logger.info(f"Installing plugin: {package}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                logger.info(f"Plugin {plugin_name} installed")
                with self._lock:
                    if plugin_name in self._registry:
                        self._registry[plugin_name].installed = True
                return True
            else:
                logger.error(f"Installation failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Installation error: {e}")
            return False

    def uninstall(self, plugin_name: str) -> bool:
        """Uninstall a plugin."""
        import subprocess
        import sys

        package = f"distllm-plugin-{plugin_name}"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", package],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                with self._lock:
                    self._registry.pop(plugin_name, None)
                return True
            return False
        except Exception:
            return False

    # ── Custom Backend Registration ───────────────────────────────────────

    def register_backend(self, name: str, factory: Callable, description: str = "") -> None:
        """Register a custom inference backend.

        Args:
            name: Backend name (e.g., "my-custom-backend").
            factory: Callable that returns a BackendAdapter instance.
            description: Human-readable description.
        """
        with self._lock:
            self._backend_factories[name] = factory
            self._registry[name] = PluginEntry(
                name=name,
                category=PluginCategory.BACKEND,
                description=description,
                source="runtime",
                installed=True,
            )
        logger.info(f"Registered custom backend: {name}")

    def get_backend_factory(self, name: str) -> Callable | None:
        """Get a registered backend factory."""
        with self._lock:
            return self._backend_factories.get(name)

    def list_backends(self) -> list[str]:
        """List all registered backend names."""
        with self._lock:
            return list(self._backend_factories.keys())

    # ── Custom Sampling Strategies ────────────────────────────────────────

    def register_sampling_strategy(
        self,
        name: str,
        fn: Callable,
        description: str = "",
        requires_params: list[str] | None = None,
    ) -> None:
        """Register a custom sampling strategy.

        Args:
            name: Strategy name (e.g., "top-a", "min-p").
            fn: Sampling function with signature:
                (logits: Tensor, temperature: float, top_k: int, top_p: float, **kwargs) -> int
            description: Human-readable description.
            requires_params: List of required parameter names.
        """
        with self._lock:
            self._sampling_strategies[name] = SamplingStrategy(
                name=name,
                description=description,
                fn=fn,
                requires_params=requires_params or [],
            )
        logger.info(f"Registered sampling strategy: {name}")

    def get_sampling_strategy(self, name: str) -> SamplingStrategy | None:
        """Get a registered sampling strategy."""
        with self._lock:
            return self._sampling_strategies.get(name)

    def list_sampling_strategies(self) -> list[str]:
        """List all registered sampling strategy names."""
        with self._lock:
            return list(self._sampling_strategies.keys())

    # ── Custom Routing Strategies ─────────────────────────────────────────

    def register_routing_strategy(
        self,
        name: str,
        fn: Callable,
        description: str = "",
        requires_params: list[str] | None = None,
    ) -> None:
        """Register a custom routing strategy.

        Args:
            name: Strategy name (e.g., "cost-optimal", "latency-first").
            fn: Routing function with signature:
                (request: dict, available_models: list[str], **kwargs) -> str
            description: Human-readable description.
            requires_params: List of required parameter names.
        """
        with self._lock:
            self._routing_strategies[name] = RoutingStrategy(
                name=name,
                description=description,
                fn=fn,
                requires_params=requires_params or [],
            )
        logger.info(f"Registered routing strategy: {name}")

    def get_routing_strategy(self, name: str) -> RoutingStrategy | None:
        """Get a registered routing strategy."""
        with self._lock:
            return self._routing_strategies.get(name)

    def list_routing_strategies(self) -> list[str]:
        """List all registered routing strategy names."""
        with self._lock:
            return list(self._routing_strategies.keys())

    # ── Registry ──────────────────────────────────────────────────────────

    def list_plugins(self, category: PluginCategory | None = None) -> list[PluginEntry]:
        """List all registered plugins, optionally filtered by category."""
        with self._lock:
            plugins = list(self._registry.values())
        if category:
            plugins = [p for p in plugins if p.category == category]
        return plugins

    def get_plugin(self, name: str) -> PluginEntry | None:
        """Get a specific plugin by name."""
        with self._lock:
            return self._registry.get(name)

    def enable_plugin(self, name: str) -> bool:
        with self._lock:
            entry = self._registry.get(name)
            if entry:
                entry.enabled = True
                return True
            return False

    def disable_plugin(self, name: str) -> bool:
        with self._lock:
            entry = self._registry.get(name)
            if entry:
                entry.enabled = False
                return True
            return False

    # ── Built-in Strategies ───────────────────────────────────────────────

    def _register_builtin_strategies(self) -> None:
        """Register built-in sampling and routing strategies."""

        # Sampling strategies
        self.register_sampling_strategy(
            "greedy",
            lambda logits, **kw: int(logits.argmax().item()),
            description="Greedy decoding — always pick the highest probability token",
        )

        self.register_sampling_strategy(
            "top-p",
            self._top_p_sampling,
            description="Nucleus sampling — sample from top-p probability mass",
            requires_params=["temperature", "top_p"],
        )

        self.register_sampling_strategy(
            "min-p",
            self._min_p_sampling,
            description="Min-P sampling — filter tokens below min_p * max_prob",
            requires_params=["temperature", "min_p"],
        )

        # Routing strategies
        self.register_routing_strategy(
            "round-robin",
            self._round_robin_routing,
            description="Round-robin across available models",
        )

        self.register_routing_strategy(
            "least-loaded",
            self._least_loaded_routing,
            description="Route to the model with fewest active requests",
        )

    @staticmethod
    def _top_p_sampling(logits, temperature=1.0, top_p=0.9, **kwargs):
        import torch
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs /= sorted_probs.sum()
            idx = torch.multinomial(sorted_probs, 1)
            return sorted_idx[idx].item()
        return int(logits.argmax().item())

    @staticmethod
    def _min_p_sampling(logits, temperature=1.0, min_p=0.05, **kwargs):
        import torch
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            max_prob = probs.max()
            mask = probs < max_prob * min_p
            probs[mask] = 0
            probs /= probs.sum()
            return torch.multinomial(probs, 1).item()
        return int(logits.argmax().item())

    _rr_counter = 0

    @classmethod
    def _round_robin_routing(cls, request, available_models, **kwargs):
        if not available_models:
            return ""
        model = available_models[cls._rr_counter % len(available_models)]
        cls._rr_counter += 1
        return model

    @staticmethod
    def _least_loaded_routing(request, available_models, **kwargs):
        loads = kwargs.get("loads", {})
        if not available_models:
            return ""
        return min(available_models, key=lambda m: loads.get(m, 0))

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_plugins": len(self._registry),
                "installed": sum(1 for p in self._registry.values() if p.installed),
                "enabled": sum(1 for p in self._registry.values() if p.enabled),
                "sampling_strategies": len(self._sampling_strategies),
                "routing_strategies": len(self._routing_strategies),
                "custom_backends": len(self._backend_factories),
                "categories": {
                    cat.value: sum(1 for p in self._registry.values() if p.category == cat)
                    for cat in PluginCategory
                },
            }
