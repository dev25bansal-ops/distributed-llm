"""Shared test helpers to reduce duplication across conftest files.

Contains:
- ``_make_fake_package`` — create a temporary Python package for testing
- ``_load_module`` — load a module from a file path
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def make_fake_package(
    name: str = "fake_package",
    files: dict[str, str] | None = None,
    base_dir: str | None = None,
) -> Generator[str, None, None]:
    """Create a temporary Python package with the given files.

    Args:
        name: Package name (must be a valid Python identifier).
        files: Mapping of relative file paths to their source content.
        base_dir: If provided, create the package under this directory
                  instead of a temporary directory.

    Yields:
        The absolute path to the package root (parent of the package dir).

    Example::

        with _make_fake_package("mypkg", {"module.py": "VAR = 1"}) as path:
            from mypkg.module import VAR
            assert VAR == 1
    """
    if base_dir:
        pkg_root = Path(base_dir)
        pkg_root.mkdir(parents=True, exist_ok=True)
        pkg_dir = pkg_root / name
        pkg_dir.mkdir(exist_ok=True)
        (pkg_dir / "__init__.py").touch()
        if files:
            for rel_path, content in files.items():
                target = pkg_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        sys.path.insert(0, str(pkg_root))
        try:
            yield str(pkg_root)
        finally:
            sys.path[:] = [p for p in sys.path if p != str(pkg_root)]
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_dir = Path(tmpdir) / name
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").touch()
        if files:
            for rel_path, content in files.items():
                target = pkg_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
        sys.path.insert(0, tmpdir)
        try:
            yield tmpdir
        finally:
            sys.path[:] = [p for p in sys.path if p != tmpdir]


def load_module(file_path: str, module_name: str | None = None) -> Any:
    """Load a Python module from its file path.

    Args:
        file_path: Absolute path to the .py file.
        module_name: Optional module name (defaults to file stem).

    Returns:
        The loaded module object.

    Example::

        mod = load_module("/path/to/my_module.py")
        mod.some_function()
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")
    name = module_name or path.stem

    # Avoid reloading if already imported
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {path}")

    module = importlib.util.module_from_spec(spec)
    # Mark as a test module to prevent side effects
    module.__test__ = True
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unload_module(name: str) -> None:
    """Remove a module from sys.modules to allow clean reimport."""
    sys.modules.pop(name, None)
    # Also remove submodules
    for key in list(sys.modules.keys()):
        if key.startswith(f"{name}."):
            sys.modules.pop(key, None)
