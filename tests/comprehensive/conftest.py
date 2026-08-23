"""Shared helpers for comprehensive test suite.

Provides module-loading utilities that bypass distllm/__init__.py to avoid
circular import issues during isolated unit testing.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _make_fake_package(name: str, path: Path):
    """Create a fake package in sys.modules to avoid __init__.py loading."""
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules.setdefault(name, mod)
    return mod


def _load_module(rel_path: str):
    """Load a module directly from file, bypassing distllm/__init__.py."""
    filepath = SRC_DIR / rel_path
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found")

    rel = filepath.relative_to(SRC_DIR)
    parts = list(rel.parent.parts) + [filepath.stem]
    if parts[0] == "distllm":
        dotted = ".".join(parts)
    else:
        dotted = "distllm." + ".".join(parts)

    if dotted in sys.modules:
        return sys.modules[dotted]

    spec = importlib.util.spec_from_file_location(dotted, filepath,
                                                   submodule_search_locations=[])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # F-007: drop a partial module on exec failure so it can't poison later
        # real imports (e.g. half-loaded distllm.config.settings).
        sys.modules.pop(dotted, None)
        raise
    return mod


# Inject fake packages so _load_module can resolve cross-package imports
_make_fake_package("distllm", SRC_DIR / "distllm")
_make_fake_package("distllm.core", SRC_DIR / "distllm/core")
_make_fake_package("distllm.dist", SRC_DIR / "distllm/dist")
_make_fake_package("distllm.dist.partition", SRC_DIR / "distllm/dist/partition")
_make_fake_package("distllm.dist.backends", SRC_DIR / "distllm/dist/backends")
_make_fake_package("distllm.dist.p2p", SRC_DIR / "distllm/dist/p2p")
_make_fake_package("distllm.dist.scheduling", SRC_DIR / "distllm/dist/scheduling")
_make_fake_package("distllm.backends", SRC_DIR / "distllm/backends")
_make_fake_package("distllm.errors", SRC_DIR / "distllm/errors")
_make_fake_package("distllm.config", SRC_DIR / "distllm/config")
_make_fake_package("distllm.api", SRC_DIR / "distllm/api")
