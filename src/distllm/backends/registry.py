"""Backend plugin registry — discover, register, and select inference backends.

Provides:
  - ``BackendPlugin`` dataclass: holds a reference to a ``BackendAdapter`` subclass
  - ``BackendRegistry``: singleton that manages all known backends
  - ``get_backend()`` / ``select_backend()`` convenience functions

Built-in backends are registered automatically when the ``distllm.backends``
package is imported. Third-party backends can register via ``pip install``
of a ``distllm_backend_*`` package that calls ``BackendRegistry.register()``
at import time, or via the YAML configuration's ``[backends.plugins]`` section.

Usage:
    from distllm.backends.registry import select_backend

    # Best backend for this machine
    Backend = select_backend()
    adapter = Backend(model_name="...")
    adapter.load_model()
    logits, kv = adapter.forward(input_ids=input_ids)

    # Named backend
    from distllm.backends.registry import get_backend
    Backend = get_backend("vllm")
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
import threading
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from distllm.backends.protocol import BackendAdapter


# ── Module-level state ─────────────────────────────────────────────────

_registry: dict[str, "BackendPlugin"] = {}
"""Global registry: backend_name -> BackendPlugin."""

_plugin_by_class: dict[type, str] = {}
"""Reverse mapping: adapter_class -> backend_name for O(1) plugin lookup."""


# ── Plugin descriptor ──────────────────────────────────────────────────


@dataclass
class BackendPlugin:
    """Descriptor for a registered inference backend.

    Attributes:
        adapter_class: The ``BackendAdapter`` subclass.
        name: Short identifier (e.g. ``"pytorch"``, ``"vllm"``).
        description: One-line description.
        version: Plugin version string.
        config_schema: Optional JSON schema for backend-specific settings.
    """

    adapter_class: type[BackendAdapter]
    name: str
    description: str = ""
    version: str = "1.0.0"
    config_schema: dict[str, Any] = field(default_factory=dict)


# ── Registry ────────────────────────────────────────────────────────────


class BackendRegistry:
    """Singleton registry for inference backend plugins.

    Usage:
        BackendRegistry.register(MyBackend)
        cls = BackendRegistry.get("mybackend")
        cls = BackendRegistry.select("cuda")
    """

    _instance: BackendRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> BackendRegistry:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    # ── Registration ───────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        adapter_class: type[BackendAdapter],
        *,
        name: str | None = None,
        force: bool = False,
    ) -> None:
        """Register a ``BackendAdapter`` subclass in the global registry.

        Args:
            adapter_class: A class that implements ``BackendAdapter``.
            name: Backend identifier (defaults to ``adapter_class.__name__``).
            force: Overwrite an existing registration with the same name.

        Raises:
            ValueError: If ``adapter_class`` does not implement the interface.
            KeyError: If a backend with the same name is already registered
                and ``force`` is ``False``.
        """
        from distllm.backends.protocol import BackendAdapter

        if not issubclass(adapter_class, BackendAdapter):
            raise ValueError(
                f"{adapter_class.__name__} does not implement BackendAdapter. "
                "Ensure it inherits from BackendAdapter."
            )

        backend_name = name or _default_name(adapter_class)

        if not force and backend_name in _registry:
            raise KeyError(
                f"Backend '{backend_name}' is already registered "
                f"(existing: {_registry[backend_name].adapter_class.__name__}). "
                f"Use force=True to overwrite."
            )

        plugin = BackendPlugin(
            adapter_class=adapter_class,
            name=backend_name,
            description=adapter_class.description(),
            version=adapter_class.version(),
        )
        with cls._lock:
            _registry[backend_name] = plugin
            _plugin_by_class[adapter_class] = backend_name
        logger.debug(f"Registered backend '{backend_name}': {adapter_class.__name__}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a backend from the registry."""
        with cls._lock:
            plugin = _registry.pop(name, None)
            if plugin:
                _plugin_by_class.pop(plugin.adapter_class, None)

    @classmethod
    def reset(cls) -> None:
        """Clear the registry and reset the singleton instance.

        This is primarily useful in test teardown to prevent state
        leaking between test cases.  After calling ``reset()`` the
        registry is in the same state as before any backends were
        registered — callers must re-run autodiscovery or re-register
        backends before using the registry again.
        """
        with cls._lock:
            _registry.clear()
            _plugin_by_class.clear()
            cls._instance = None

    # ── Query ──────────────────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> type[BackendAdapter] | None:
        """Look up a backend adapter class by name.

        Returns:
            The ``BackendAdapter`` subclass, or ``None`` if not found.
        """
        plugin = _registry.get(name)
        return plugin.adapter_class if plugin else None

    @classmethod
    def get_plugin(cls, name: str) -> BackendPlugin | None:
        """Look up the full ``BackendPlugin`` descriptor by name."""
        return _registry.get(name)

    @classmethod
    def list_backends(cls) -> list[BackendPlugin]:
        """Return all registered backends."""
        return list(_registry.values())

    @classmethod
    def list_available(cls) -> list[BackendPlugin]:
        """Return only backends whose dependencies are installed."""
        return [
            p
            for p in _registry.values()
            if p.adapter_class.is_available()
        ]

    # ── Auto-selection ─────────────────────────────────────────────────

    @classmethod
    def select(
        cls,
        device_type: str | None = None,
        preferred_backend: str | None = None,
    ) -> type[BackendAdapter] | None:
        """Select the best backend for the given device.

        Args:
            device_type: ``"cuda"``, ``"cpu"``, ``"mps"``, etc.
                ``None`` = auto-detect.
            preferred_backend: If set, return this backend if available.

        Returns:
            A ``BackendAdapter`` subclass, or ``None`` if no backend is
            available for the device.
        """
        if preferred_backend:
            cls_ = cls.get(preferred_backend)
            if cls_ and cls_.is_available():
                return cls_
            logger.warning(
                f"Preferred backend '{preferred_backend}' not available; "
                f"falling back to auto-select"
            )

        if device_type is None:
            device_type = _detect_device()

        available = cls.list_available()
        if not available:
            return None

        # Health-aware filtering: skip backends that report unhealthy.
        healthy = [
            p for p in available
            if _check_health(p.adapter_class)
        ]
        if not healthy:
            logger.warning("All available backends are unhealthy; ignoring health status")
            healthy = available

        # Sort by priority (descending), then by load (ascending) as tiebreaker.
        healthy.sort(
            key=lambda p: (
                -p.adapter_class.priority_for(device_type),
                _get_load(p.adapter_class),
            ),
        )
        best = healthy[0]
        priority = best.adapter_class.priority_for(device_type)
        if priority <= 0:
            return None

        logger.debug(
            f"Auto-selected backend '{best.name}' "
            f"(priority={priority}) for device '{device_type}'"
        )
        return best.adapter_class

    @classmethod
    def select_plugin(
        cls,
        device_type: str | None = None,
        preferred_backend: str | None = None,
    ) -> BackendPlugin | None:
        """Like ``select()`` but returns the full ``BackendPlugin``.

        .. deprecated::
            Use ``select()`` to get the adapter class, then
            ``get_plugin()`` by name if the plugin descriptor is needed.
        """
        warnings.warn(
            "BackendRegistry.select_plugin() is deprecated; "
            "use select() + get_plugin() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        adapter = cls.select(
            device_type=device_type, preferred_backend=preferred_backend
        )
        if adapter is None:
            return None
        name = _plugin_by_class.get(adapter)
        return _registry.get(name) if name else None

    # ── Entry-point discovery ───────────────────────────────────────────

    @classmethod
    def autodiscover(cls) -> int:
        """Scan ``distllm_backend.*`` entry points and register backends.

        Any installed package that declares an entry point under the
        ``distllm_backend`` group is imported and registered.  This is
        the preferred way for third-party backends to plug in — no
        manual ``register()`` call required.

        Returns:
            The number of newly registered backends.
        """
        count = 0
        eps = importlib.metadata.entry_points()
        # Python 3.12+ returns a SelectableGroups; older returns a dict.
        # Filter for the distllm_backend group.
        group_eps = eps.select(group="distllm_backend") if hasattr(eps, "select") else eps.get("distllm_backend", [])
        for ep in group_eps:
            try:
                adapter_cls = ep.load()
                cls.register(adapter_cls, name=ep.name, force=True)
                count += 1
            except Exception:
                logger.opt(exception=True).warning(
                    f"Failed to load backend entry point '{ep.name}'"
                )
        return count


# ── Convenience functions ──────────────────────────────────────────────


def get_backend(name: str) -> type[BackendAdapter] | None:
    """Shortcut for ``BackendRegistry.get(name)``."""
    return BackendRegistry.get(name)


def select_backend(
    device_type: str | None = None,
    preferred_backend: str | None = None,
) -> type[BackendAdapter] | None:
    """Shortcut for ``BackendRegistry.select(device_type, preferred_backend)``."""
    return BackendRegistry.select(
        device_type=device_type, preferred_backend=preferred_backend
    )


def list_backends() -> list[BackendPlugin]:
    """Shortcut for ``BackendRegistry.list_backends()``."""
    return BackendRegistry.list_backends()


def list_available_backends() -> list[BackendPlugin]:
    """Shortcut for ``BackendRegistry.list_available()``."""
    return BackendRegistry.list_available()


# ── Internal helpers ───────────────────────────────────────────────────


def _default_name(cls: type) -> str:
    """Derive a backend name from the class name.

    ``VLLMNodeAdapter`` → ``"vllm"``
    ``PyTorchNodeAdapter`` → ``"pytorch"``
    """
    name = cls.__name__
    for suffix in ("NodeAdapter", "Adapter", "Backend"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.lower()


def _detect_device() -> str:
    """Detect the best available compute device (cross-platform)."""
    try:
        from distllm.core.device_registry import detect_platform
        return detect_platform()
    except ImportError:
        pass
    if _module_available("torch"):
        import torch
        if torch.cuda.is_available():
            import torch.version as tv
            if hasattr(tv, "hip") and tv.hip is not None:
                return "rocm"
            return "cuda"
        if hasattr(torch, "mps") and torch.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    return "cpu"


def _module_available(name: str) -> bool:
    """Check whether a Python module can be imported without actually importing it."""
    if name in sys.modules:
        return True
    return importlib.util.find_spec(name) is not None


def _check_health(adapter_class: type) -> bool:
    """Probe backend health.

    Uses a classmethod-style check: calls ``probe_health()`` on the
    class if defined, otherwise falls back to a simple instance-based
    health check.  Returns ``False`` on any error so a broken probe
    does not silently pass as healthy (fail-closed).
    """
    try:
        # If the class defines a ``probe_health`` classmethod, prefer it
        # (no __init__ needed).
        if hasattr(adapter_class, "probe_health"):
            probe_fn = getattr(adapter_class, "probe_health")
            if callable(probe_fn):
                result = probe_fn()
                return bool(result)
        # Fallback for subclasses that only override ``health_check``:
        # try a lightweight instance and call it.
        from distllm.backends.protocol import BackendAdapter as _BA
        if adapter_class.health_check is not _BA.health_check:
            probe = object.__new__(adapter_class)
            return bool(probe.health_check())
        return True
    except Exception:
        return False


def _get_load(adapter_class: type) -> float:
    """Return the current load of an adapter, falling back to ``0.0``."""
    try:
        from distllm.backends.protocol import BackendAdapter as _BA
        if adapter_class.current_load is not _BA.current_load:
            probe = object.__new__(adapter_class)
            return float(probe.current_load())
        return 0.0
    except Exception:
        return 0.0
