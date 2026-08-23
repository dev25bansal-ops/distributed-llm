"""Standardized error hierarchy for distributed LLM."""

from distllm.errors.types import (
    APIError,
    AuthError,
    AuthenticationError,
    AuthorizationError,
    BatchCapacityError,
    BatchError,
    CircuitBreakerError,
    CommunicationError,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConnectionLostError,
    ConstraintError,
    ConstraintViolationError,
    DistLLMError,
    GRPCTimeoutError,
    GatewayError,
    InputValidationError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    ModelOOMError,
    NetworkError,
    NetworkTimeoutError,
    NodeError,
    NodeOOMError,
    NodeTimeoutError,
    NodeUnreachableError,
    NotLeaderError,
    OOMError,
    ProtoError,
    ProviderTimeoutError,
    QuotaExceededError,
    RateLimitError,
    SerializationError,
)
from distllm.errors.retry import RetryPolicy, with_retry, with_retry_async
from distllm.errors.policies import (
    ERROR_RETRY_POLICIES,
    get_retry_delay,
    get_retry_policy,
    should_retry,
)

__all__ = [
    # Base
    "DistLLMError",
    # Node
    "NodeError",
    "NodeUnreachableError",
    "NodeTimeoutError",
    "NodeOOMError",
    "CircuitBreakerError",
    # Model
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelOOMError",
    "OOMError",  # backward compat alias
    # Config
    "ConfigError",
    "ConfigValidationError",
    "ConfigFileNotFoundError",
    # Network
    "NetworkError",
    "NetworkTimeoutError",
    "ConnectionLostError",
    # Auth
    "AuthError",
    "AuthenticationError",
    "AuthorizationError",
    # API
    "APIError",
    "RateLimitError",
    "QuotaExceededError",
    # Gateway
    "GatewayError",
    "ProviderTimeoutError",
    # Communication
    "CommunicationError",
    "SerializationError",
    "GRPCTimeoutError",
    "ProtoError",
    # Batch
    "BatchError",
    "BatchCapacityError",
    # HA / leadership
    "NotLeaderError",
    # Constraint
    "ConstraintError",
    "ConstraintViolationError",
    # Input
    "InputValidationError",
    # Retry utilities
    "RetryPolicy",
    "with_retry",
    "with_retry_async",
    "ERROR_RETRY_POLICIES",
    "get_retry_policy",
    "should_retry",
    "get_retry_delay",
]
