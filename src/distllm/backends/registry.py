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

import base64
import importlib.metadata
import importlib.util
import os
import sys
import threading
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from distllm.backends.protocol import BackendAdapter


# ── Trust model for entry-point discovery ──────────────────────────────
#
# B-02: autodiscovery must NOT let an arbitrary installed package overwrite
# a built-in backend (e.g. a fake ``vllm``) via ``force=True`` — that is
# supply-chain code execution through the plugin API.  Third-party backends
# are therefore only loaded when they are *trusted*:
#
# 1. The entry-point name is on the allowlist (``DISTLLM_ALLOWED_BACKENDS``,
#    comma-separated, or the built-in allowlist below), AND
# 2. Either its distribution is signed with an Ed25519 key trusted via
#    ``DISTLLM_BACKEND_TRUSTED_KEYS`` (base64-Ed25519 public keys, or a PEM
#    public key), or it is an explicitly operator-allowed name — matching
#    the allow-by-default for built-ins, deny-by-default otherwise.
#
# Operators can disable third-party discovery entirely with
# ``DISTLLM_DISABLE_BACKEND_AUTODISCOVERY=1``.  Built-in backends registered
# via ``backends/__init__._register_builtins()`` are unaffected.

_BUILTIN_BACKEND_NAMES = frozenset(
    {
        "pytorch",
        "vllm",
        "llamacpp",
        "exllama",
        "onnx",
        "tensorrt",
        "triton",
        "nim",
        "mlx",
        "webgpu",
    }
)
"""Names of backends shipped with distllm itself; never overwritable."""

_DEFAULT_ALLOWED_BACKENDS = _BUILTIN_BACKEND_NAMES


def _get_allowed_backends() -> frozenset[str]:
    """Return the set of entry-point names allowed for registration."""
    raw = os.environ.get("DISTLLM_ALLOWED_BACKENDS", "").strip()
    if not raw:
        return frozenset(_DEFAULT_ALLOWED_BACKENDS)
    return frozenset(n.strip() for n in raw.split(",") if n.strip())


def _get_trusted_public_keys() -> list[Any]:
    """Decode trusted Ed25519 public keys from the environment.

    ``DISTLLM_BACKEND_TRUSTED_KEYS`` is a comma-separated list of
    base64-encoded raw Ed25519 public keys or PEM-encoded public keys.
    """
    keys = []
    raw = os.environ.get("DISTLLM_BACKEND_TRUSTED_KEYS", "").strip()
    if not raw:
        return keys
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except Exception:  # cryptography not installed
        return keys
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            if "PUBLIC KEY" in entry or entry.startswith("-----"):
                keys.append(
                    serialization.load_pem_public_key(entry.encode("utf-8"))
                )
            else:
                keys.append(
                    Ed25519PublicKey.from_public_bytes(
                        base64.b64decode(entry.encode("ascii"))
                    )
                )
        except Exception:
            logger.warning(
                "Ignoring malformed trusted key entry (not a valid "
                "Ed25519 public key)"
            )
    return keys


def _verify_entry_point_signature(ep: importlib.metadata.EntryPoint) -> bool:
    """Check whether *ep*'s distribution carries a valid trusted signature.

    A distribution is trusted when it ships a ``DISTLLM_SIGNATURE`` file in
    its metadata (or a ``distllm_backend_signed.txt`` marker in its file
    list) whose base64 Ed25519 signature verifies the canonical
    ``name:version:entry_point_value`` payload under one of the trusted keys.

    Returns ``True`` only on a positive verification; any error or absence
    of signature metadata fails closed (``False``).
    """
    try:
        from cryptography.exceptions import InvalidSignature
    except Exception:  # pragma: no cover
        return False
    trusted_keys = _get_trusted_public_keys()
    if not trusted_keys:
        return False
    dist = ep.dist
    if dist is None:
        return False
    signature_b64: str | None = None
    try:
        # Preferred: signature stored in distribution metadata.
        raw = dist.read_text("DISTLLM_SIGNATURE")
        if raw:
            signature_b64 = raw.strip()
    except Exception:
        # CQ-08: best-effort: failure is non-fatal (optional capability / telemetry / cleanup).
        pass
    if not signature_b64:
        # Fallback: a marker file listing the signature.
        try:
            sig_file = dist.locate_file("distllm_backend_signed.txt")
            if sig_file.is_file():
                signature_b64 = sig_file.read_text().strip()
        except Exception:
            # CQ-08: best-effort: failure is non-fatal (optional capability / telemetry / cleanup).
            pass
    if not signature_b64:
        return False
    try:
        signature = base64.b64decode(signature_b64.encode("ascii"))
    except Exception:
        # CQ-08: intentional fail-open: return False on failure; callers treat False as 'not available/enabled'.
        return False
    payload = f"{ep.name}:{dist.version}:{ep.value}".encode("utf-8")
    for key in trusted_keys:
        try:
            key.verify(signature, payload)
            return True
        except InvalidSignature:
            continue
        except Exception:
            # CQ-08: skip this item and continue with the rest (partial results are acceptable).
            continue
    return False


def _is_entry_point_allowed(ep: importlib.metadata.EntryPoint) -> bool:
    """Return whether *ep* may register its backend.

    Deny-by-default: an entry point is allowed iff it is on the allowlist
    AND it either (a) carries a valid signature under a trusted key, or
    (b) the operator has opted into allowlisted-but-unsigned entry points
    via ``DISTLLM_BACKEND_ALLOW_UNSIGNED=1``.  Without any trusted key or
    opt-in, non-built-in entry points are rejected — in particular, an
    entry point named like a built-in backend can never overwrite it.
    """
    allowed = _get_allowed_backends()
    if ep.name not in allowed:
        logger.warning(
            f"Backend entry point '{ep.name}' rejected: not on the "
            f"allowlist ({sorted(allowed)})"
        )
        return False
    if _verify_entry_point_signature(ep):
        return True
    if os.environ.get("DISTLLM_BACKEND_ALLOW_UNSIGNED", "").strip() in {
        "1",
        "true",
        "True",
        "TRUE",
    }:
        logger.warning(
            f"Backend entry point '{ep.name}' allowed without a signature "
            f"(DISTLLM_BACKEND_ALLOW_UNSIGNED=1)"
        )
        return True
    logger.warning(
        f"Backend entry point '{ep.name}' rejected: no valid Ed25519 "
        f"signature from a trusted key (set DISTLLM_BACKEND_TRUSTED_KEYS "
        f"or DISTLLM_BACKEND_ALLOW_UNSIGNED=1)"
    )
    return False


# ── Module-level state ─────────────────────────────────────────────────

_registry: dict[str, "BackendPlugin"] = {}
"""Global registry: backend_name -> BackendPlugin."""

_plugin_by_class: dict[type, str] = {}
"""Reverse mapping: adapter_class -> backend_name for O(1) plugin lookup."""

# Built-in backend names are protected: only the built-in registration path
# in ``backends/__init__._register_builtins()`` may set them, and never via
# entry points or ``force=True``.
_PROTECTED_BACKEND_NAMES = frozenset(_BUILTIN_BACKEND_NAMES)

# Set to True only during built-in registration (see
# ``backends/__init__._register_builtins()``).  Tests may set it via the
# helper below; production code never sets it.
_allow_builtin_registration: bool = False


def _set_allow_builtin_registration(value: bool) -> None:
    """Temporarily allow registration of protected built-in names.

    Only ``backends/__init__._register_builtins()`` should call this, and
    only for the duration of built-in registration.  Never set it from
    entry-point or third-party code.
    """
    global _allow_builtin_registration
    _allow_builtin_registration = value


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

        Built-in backend names (``vllm``, ``pytorch``, …) are protected:
        they may only be registered from the built-in path and can never be
        overwritten — not even with ``force=True`` — to prevent a malicious
        entry point from substituting a built-in backend.

        Args:
            adapter_class: A class that implements ``BackendAdapter``.
            name: Backend identifier (defaults to ``adapter_class.__name__``).
            force: Overwrite an existing (non-built-in) registration with
                the same name.

        Raises:
            ValueError: If ``adapter_class`` does not implement the interface.
                If the name matches a protected built-in name and the registry
                is not in built-in registration mode.
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

        # Protected built-in names: reject any non-built-in attempt to set
        # or overwrite them, even with force=True.  Built-in registration
        # sets the module flag below before calling register().
        if backend_name in _PROTECTED_BACKEND_NAMES and not _allow_builtin_registration:
            raise ValueError(
                f"Backend name '{backend_name}' is reserved for a built-in "
                f"backend and cannot be (over)written from this path."
            )

        if backend_name in _PROTECTED_BACKEND_NAMES and _allow_builtin_registration:
            # Built-in registration: force=True is implied for safety,
            # but a duplicate is still an error so tests can catch
            # accidental double-registration.
            if backend_name in _registry and not force:
                raise KeyError(
                    f"Backend '{backend_name}' is already registered "
                    f"(existing: {_registry[backend_name].adapter_class.__name__}). "
                    f"Use force=True to overwrite."
                )
        elif not force and backend_name in _registry:
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
        *,
        require_forward: bool = True,
    ) -> type[BackendAdapter] | None:
        """Select the best backend for the given device.

        Args:
            device_type: ``"cuda"``, ``"cpu"``, ``"mps"``, etc.
                ``None`` = auto-detect.
            preferred_backend: If set, return this backend if available
                (and, when ``require_forward`` is set, if it supports a
                layer-level ``forward()``).
            require_forward: When ``True`` (the default — the pipeline
                ``forward()`` path), only backends whose
                ``supports_forward()`` is ``True`` are candidates.  This
                keeps generate-only backends (Ollama/TGI/NIM-style) out of
                pipeline-parallel selection; pass ``False`` for single-shot
                generation-only callers.

        Returns:
            A ``BackendAdapter`` subclass, or ``None`` if no backend is
            available for the device.
        """
        if preferred_backend:
            cls_ = cls.get(preferred_backend)
            if cls_ and cls_.is_available():
                if require_forward and not _supports_forward(cls_):
                    logger.warning(
                        f"Preferred backend '{preferred_backend}' is "
                        f"generate-only (supports_forward() is False); "
                        f"falling back to auto-select"
                    )
                else:
                    return cls_
            else:
                logger.warning(
                    f"Preferred backend '{preferred_backend}' not available; "
                    f"falling back to auto-select"
                )

        if device_type is None:
            device_type = _detect_device()

        available = cls.list_available()
        if require_forward:
            available = [p for p in available if _supports_forward(p.adapter_class)]
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

        Third-party backends are only loaded when they pass the trust check
        in :func:`_is_entry_point_allowed`: the entry-point name must be on
        the allowlist (``DISTLLM_ALLOWED_BACKENDS``, default built-ins only)
        AND must carry a valid Ed25519 signature from a trusted key
        (``DISTLLM_BACKEND_TRUSTED_KEYS``) or an explicit operator opt-in
        (``DISTLLM_BACKEND_ALLOW_UNSIGNED=1``).  Built-in names are never
        overwritten — discovery registers with ``force=False`` so a
        duplicate name fails instead of replacing the built-in.

        Set ``DISTLLM_DISABLE_BACKEND_AUTODISCOVERY=1`` to skip discovery
        entirely.

        Returns:
            The number of newly registered backends.
        """
        if os.environ.get("DISTLLM_DISABLE_BACKEND_AUTODISCOVERY", "").strip() in {
            "1",
            "true",
            "True",
            "TRUE",
        }:
            logger.debug("Backend autodiscovery disabled via DISTLLM_DISABLE_BACKEND_AUTODISCOVERY")
            return 0
        count = 0
        eps = importlib.metadata.entry_points()
        # Python 3.12+ returns a SelectableGroups; older returns a dict.
        # Filter for the distllm_backend group.
        group_eps = eps.select(group="distllm_backend") if hasattr(eps, "select") else eps.get("distllm_backend", [])
        for ep in group_eps:
            try:
                if not _is_entry_point_allowed(ep):
                    continue
                adapter_cls = ep.load()
                # Never force-overwrite: a name that already exists is
                # refused, so a fake entry point can never replace a
                # built-in (and any re-registration is explicit).
                cls.register(adapter_cls, name=ep.name, force=False)
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
    *,
    require_forward: bool = True,
) -> type[BackendAdapter] | None:
    """Shortcut for ``BackendRegistry.select(...)``.

    ``require_forward=True`` (default) filters candidates to backends that
    support a layer-level ``forward()``; pass ``False`` for generate-only
    callers that accept Ollama/TGI-style adapters.
    """
    return BackendRegistry.select(
        device_type=device_type,
        preferred_backend=preferred_backend,
        require_forward=require_forward,
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
        # CQ-08: intentional fail-open: return False on failure; callers treat False as 'not available/enabled'.
        return False


def _supports_forward(adapter_class: type) -> bool:
    """Return whether *adapter_class* can serve a layer-level ``forward()``.

    Fail-closed: any error (e.g. an adapter that raises inside
    ``supports_forward()``) is treated as "does not support forward" so a
    broken adapter cannot be handed to the pipeline path.
    """
    try:
        fn = getattr(adapter_class, "supports_forward", None)
        if fn is None:
            # Pre-split third-party adapters: assume forward-capable.
            return True
        return bool(fn())
    except Exception:
        # CQ-08: intentional fail-open: return False on failure; callers treat False as 'not available/enabled'.
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
        # CQ-08: intentional fail-open: numeric/estimate fallback on failure.
        return 0.0
