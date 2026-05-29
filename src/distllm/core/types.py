"""Shared enum types for inter-node communication.

Protobuf types (``node_pb2``) are now used directly instead of the
dataclass equivalents that were previously defined here.
"""

from enum import IntEnum


class ErrorCode(IntEnum):
    UNKNOWN = 0
    MODEL_ERROR = 1
    OOM = 2
    TIMEOUT = 3
    INVALID_INPUT = 4
    NODE_UNREACHABLE = 5
    CIRCUIT_BREAKER_OPEN = 6
