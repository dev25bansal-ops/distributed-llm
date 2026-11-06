"""Backend adapters for distributed inference.

Each backend adapter wraps a specific inference runtime
(vLLM, llama.cpp, Ray) so that the distributed pipeline
layer can drive any backend uniformly.

Health monitoring, optimization profiles, and graceful
degradation are provided by the sibling modules.

Uses lazy ``__getattr__`` imports so that importing a light backend
module (e.g. ``distllm.dist.backends.ray``) does not force heavyweight
optional dependencies (torch, grpc) to be installed.
"""

from __future__ import annotations

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}


def _register(module: str, *symbols: str) -> None:
    for sym in symbols:
        _LAZY_IMPORTS[sym] = (module, sym)


_register("distllm.dist.backends.llamacpp", "LlamacppPipelineEngine")
_register("distllm.dist.backends.backend_profiles", "BackendProfileManager")
_register("distllm.dist.backends.graceful_degradation", "GracefulDegradationHandler")
_register("distllm.dist.backends.health_monitor", "BackendHealthMonitor")
_register("distllm.dist.backends.ray", "RayPipelineEngine")
_register("distllm.dist.backends.vllm", "VLLMPipelineEngine")

__all__ = [
    "BackendHealthMonitor",
    "BackendProfileManager",
    "GracefulDegradationHandler",
    "LlamacppPipelineEngine",
    "RayPipelineEngine",
    "VLLMPipelineEngine",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module_path, symbol = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, symbol)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'distllm.dist.backends' has no attribute {name!r}")


def __dir__():
    return list(__all__)
