"""Pipeline package for distributed inference.

Re-exports all classes from submodules for backward compatibility.
"""

from __future__ import annotations
from distllm.dist.pipeline.context import NodeForwardContext, NodeCheckpoint
from distllm.dist.pipeline.profiler import PipelineProfiler
from distllm.dist.pipeline.simulator import PipelineSimulator
from distllm.dist.pipeline.strategy import PipelineStrategy, StrategySelector
from distllm.dist.pipeline.transport import TransportBackend, TensorTransport
from distllm.dist.pipeline.orchestrator import PipelineOrchestrator
from distllm.dist.pipeline.serialization import (
    cleanup_tensor_copy_streams,
    get_tensor_copy_stream,
    forward_request_to_proto,
    to_proto_tensor,
    from_proto_tensor,
    process_forward_response_pb,
    set_kv_cache_proto,
    tensor_quantize,
    tensor_dequantize,
)

# Backward-compatibility aliases for tests that import underscore-prefixed names.
_process_forward_response_pb = process_forward_response_pb
_set_kv_cache_proto = set_kv_cache_proto
_tensor_quantize = tensor_quantize
_tensor_dequantize = tensor_dequantize
_to_proto_tensor = to_proto_tensor

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
