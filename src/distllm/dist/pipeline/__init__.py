"""Pipeline package for distributed inference.

Re-exports all classes from submodules for backward compatibility.

Uses lazy ``__getattr__`` imports so that importing a light submodule
(e.g. ``distllm.dist.pipeline.context``) does not force heavyweight
optional dependencies (torch, grpc) to be installed.  The eager imports
previously pulled in ``core.debug`` (torch) and ``serialization``
(grpc protos) at package-import time.
"""

from __future__ import annotations

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {}


def _register(module: str, *symbols: str) -> None:
    for sym in symbols:
        _LAZY_IMPORTS[sym] = (module, sym)


_register(
    "distllm.dist.pipeline.context",
    "NodeForwardContext", "NodeCheckpoint",
)
_register("distllm.dist.pipeline.profiler", "PipelineProfiler")
_register("distllm.dist.pipeline.simulator", "PipelineSimulator")
_register(
    "distllm.dist.pipeline.strategy",
    "PipelineStrategy", "StrategySelector",
)
_register(
    "distllm.dist.pipeline.transport",
    "TransportBackend", "TensorTransport",
)
_register("distllm.dist.pipeline.orchestrator", "PipelineOrchestrator")
_register(
    "distllm.dist.pipeline.pipeline_reconfig",
    "TopologyVersion",
    "NodeAssignment",
    "PipelineCheckpoint",
    "PipelineCheckpointer",
    "ReconfigState",
    "ReconfigurationPlan",
    "PipelineReconfigurator",
    "RequestRoute",
    "RequestPipelineSelector",
)
_register(
    "distllm.dist.pipeline.serialization",
    "cleanup_tensor_copy_streams",
    "get_tensor_copy_stream",
    "forward_request_to_proto",
    "to_proto_tensor",
    "from_proto_tensor",
    "process_forward_response_pb",
    "set_kv_cache_proto",
    "tensor_quantize",
    "tensor_dequantize",
)

# Backward-compatibility aliases for tests that import underscore-prefixed names.
_LAZY_ALIASES: dict[str, str] = {
    "_process_forward_response_pb": "process_forward_response_pb",
    "_set_kv_cache_proto": "set_kv_cache_proto",
    "_tensor_quantize": "tensor_quantize",
    "_tensor_dequantize": "tensor_dequantize",
    "_to_proto_tensor": "to_proto_tensor",
}

__all__ = [
    "NodeForwardContext",
    "NodeCheckpoint",
    "PipelineProfiler",
    "PipelineSimulator",
    "PipelineStrategy",
    "StrategySelector",
    "TransportBackend",
    "TensorTransport",
    "PipelineOrchestrator",
    "TopologyVersion",
    "NodeAssignment",
    "PipelineCheckpoint",
    "PipelineCheckpointer",
    "ReconfigState",
    "ReconfigurationPlan",
    "PipelineReconfigurator",
    "RequestRoute",
    "RequestPipelineSelector",
    "cleanup_tensor_copy_streams",
    "get_tensor_copy_stream",
    "forward_request_to_proto",
    "to_proto_tensor",
    "from_proto_tensor",
    "process_forward_response_pb",
    "set_kv_cache_proto",
    "tensor_quantize",
    "tensor_dequantize",
]


def __getattr__(name: str):
    target = name
    if name in _LAZY_ALIASES:
        target = _LAZY_ALIASES[name]
    if target in _LAZY_IMPORTS:
        import importlib
        module_path, symbol = _LAZY_IMPORTS[target]
        module = importlib.import_module(module_path)
        value = getattr(module, symbol)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'distllm.dist.pipeline' has no attribute {name!r}")


def __dir__():
    return list(__all__)
