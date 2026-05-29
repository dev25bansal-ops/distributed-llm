"""Helper functions for benchmark tests.

Provides fake package injection to avoid circular imports in distllm/__init__.py.
"""

import importlib.util
import sys
import types
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def make_fake_package(name: str, path: Path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules.setdefault(name, mod)
    return mod


def load_module(rel_path: str):
    filepath = SRC_DIR / rel_path
    rel = filepath.relative_to(SRC_DIR)
    parts = list(rel.parent.parts) + [filepath.stem]
    if parts[0] == "distllm":
        dotted = ".".join(parts)
    else:
        dotted = "distllm." + ".".join(parts)
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, filepath, submodule_search_locations=[])
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filepath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# Bootstrap fake packages once at import time.
make_fake_package("distllm", SRC_DIR / "distllm")
make_fake_package("distllm.core", SRC_DIR / "distllm/core")
make_fake_package("distllm.dist", SRC_DIR / "distllm/dist")
make_fake_package("distllm.dist.partition", SRC_DIR / "distllm/dist/partition")
make_fake_package("distllm.backends", SRC_DIR / "distllm/backends")

# Pre-load modules used across multiple tests.
_coord = load_module("distllm/core/coordinator.py")
_rm = load_module("distllm/core/resource_manager.py")
_dr = load_module("distllm/core/device_registry.py")
_hs = load_module("distllm/core/heterogeneous_scheduler.py")

Coordinator = _coord.Coordinator
NodeRegistration = _rm.NodeRegistration
DeviceInfo = _dr.DeviceInfo
HeterogeneousCluster = _hs.HeterogeneousCluster
HeterogeneousNode = _hs.HeterogeneousNode
assign_layers_proportional = _hs.assign_layers_proportional
estimate_heterogeneous_throughput = _hs.estimate_heterogeneous_throughput
get_device_compatibility_map = _hs.get_device_compatibility_map
