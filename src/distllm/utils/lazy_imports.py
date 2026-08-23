"""Shared lazy-import machinery used by multiple top-level package __init__.py files.

Replaces 4 copies of the same ``_register()`` + ``__getattr__`` pattern
(``core/__init__.py``, ``dist/__init__.py``, ``models/__init__.py``,
``backends/__init__.py``) with a single canonical implementation.

Usage::

    from distllm.utils.lazy_imports import LazyImporter

    _importer = LazyImporter(__name__)

    # Register symbols
    _importer.register("distllm.core.batch_scheduler", "BatchScheduler", "Sequence", "SequenceStatus")
    _importer.register("distllm.core.coordinator", "Coordinator")

    # Expose in module scope
    __getattr__ = _importer.__getattr__
    __dir__ = _importer.__dir__
    __all__ = _importer.all_symbols()
"""

from __future__ import annotations

import importlib
from typing import Any


class LazyImporter:
    """Lazy module importer that defers symbol resolution until access time.

    This breaks circular import chains caused by eager ``from X import Y``
    at module scope.  Symbols are resolved only when first accessed, by
    which time all modules in the cycle have finished loading.

    Each top-level package (``core``, ``dist``, ``models``, ``backends``)
    should create one instance with its ``__name__`` and register all
    publicly-exported symbols.
    """

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._table: dict[str, str] = {}  # symbol_name -> module_path

    def register(self, module_path: str, *symbols: str) -> None:
        """Register *symbols* as lazy exports from *module_path*."""
        for sym in symbols:
            self._table[sym] = module_path

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                f"module {self._module_name!r} has no attribute {name!r}"
            )
        entry = self._table.get(name)
        if entry is not None:
            mod = importlib.import_module(entry)
            value = getattr(mod, name)
            # Cache in the caller's module globals for fast subsequent access
            import sys
            caller_module = sys.modules.get(self._module_name)
            if caller_module is not None:
                setattr(caller_module, name, value)
            return value
        raise AttributeError(
            f"module {self._module_name!r} has no attribute {name!r}"
        )

    def __dir__(self) -> list[str]:
        return sorted(self._table.keys())

    def all_symbols(self) -> list[str]:
        """Return the full list of registered symbol names."""
        return list(self._table.keys())

    def verify(self) -> list[str]:
        """Smoke-test that every registered symbol resolves.

        Returns a list of **missing** symbols — symbols registered but
        not findable in their declared module.
        """
        missing: list[str] = []
        for name, module_path in self._table.items():
            try:
                mod = importlib.import_module(module_path)
                if not hasattr(mod, name):
                    missing.append(f"{name} (not found in {module_path})")
            except Exception as exc:
                missing.append(f"{name} (import error: {exc})")
        return missing
