"""gRPC communication layer for distributed LLM inference.

This module is a compatibility shim that re-exports all symbols from
the modular gRPC implementation. Import from this file as before —
all existing code continues to work unchanged.

Modular files:
- distllm.communication.tensor_transport - Debug config, tensor parse/build helpers
- distllm.communication.node_service - NodeService, AsyncNodeService
- distllm.communication.coordinator_service - CoordinatorService, AsyncCoordinatorService
- distllm.communication.grpc_client - GRPCServer, AsyncGRPCServer, NodeClient, AsyncNodeClient
"""

import grpc

# Tensor transport utilities
from distllm.communication.tensor_transport import (
    DebugConfig,
    set_debug_mode,
    is_debug_mode,
    _parse_forward_request,
    _build_forward_response,
    _log_forward_debug,
)

# Node service implementations
from distllm.communication.node_service import (
    NodeService,
    AsyncNodeService,
)

# Coordinator service implementations
from distllm.communication.coordinator_service import (
    CoordinatorService,
    AsyncCoordinatorService,
)

# Server and client implementations
from distllm.communication.grpc_client import (
    GRPCServer,
    AsyncGRPCServer,
    NodeClient,
    AsyncNodeClient,
)
from distllm.communication.node_pb2_grpc import NodeServiceStub

__all__ = [
    "grpc",
    "NodeServiceStub",
    # Tensor transport
    "DebugConfig",
    "set_debug_mode",
    "is_debug_mode",
    "_parse_forward_request",
    "_build_forward_response",
    "_log_forward_debug",
    # Node service
    "NodeService",
    "AsyncNodeService",
    # Coordinator service
    "CoordinatorService",
    "AsyncCoordinatorService",
    # Server and client
    "GRPCServer",
    "AsyncGRPCServer",
    "NodeClient",
    "AsyncNodeClient",
]
