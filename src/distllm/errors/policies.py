"""Retry policy registry for distributed LLM error types.

Maps each error type to a default RetryPolicy, enabling consistent
retry behavior across the codebase.
"""

from distllm.errors.retry import RetryPolicy
from distllm.errors.types import (
    DistLLMError,
    NodeUnreachableError,
    CircuitBreakerError,
    CommunicationError,
    SerializationError,
    GRPCTimeoutError,
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    ConfigError,
    ConfigValidationError,
    BatchError,
    BatchCapacityError,
    ConstraintError,
    ConstraintViolationError,
)

# Default retry policies per error category
ERROR_RETRY_POLICIES: dict[type[DistLLMError], RetryPolicy] = {
    # Node errors
    NodeUnreachableError: RetryPolicy(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        retryable=(NodeUnreachableError,),
    ),
    CircuitBreakerError: RetryPolicy(
        max_retries=0,
        retryable=(),
    ),
    # Communication errors
    GRPCTimeoutError: RetryPolicy(
        max_retries=5,
        base_delay=0.5,
        max_delay=30.0,
        retryable=(GRPCTimeoutError,),
    ),
    SerializationError: RetryPolicy(
        max_retries=2,
        base_delay=0.1,
        max_delay=1.0,
        retryable=(SerializationError,),
    ),
    # Model errors - generally not retryable
    ModelNotFoundError: RetryPolicy(
        max_retries=0,
        retryable=(),
    ),
    ModelLoadError: RetryPolicy(
        max_retries=1,
        base_delay=1.0,
        max_delay=5.0,
        retryable=(ModelLoadError,),
    ),
    # Config errors - not retryable
    ConfigValidationError: RetryPolicy(
        max_retries=0,
        retryable=(),
    ),
    # Batch errors - not retryable
    BatchCapacityError: RetryPolicy(
        max_retries=0,
        retryable=(),
    ),
    # Constraint errors - not retryable
    ConstraintViolationError: RetryPolicy(
        max_retries=0,
        retryable=(),
    ),
}


def get_retry_policy(error: DistLLMError) -> RetryPolicy:
    """Get the retry policy for a given error.

    Walks the MRO to find the most specific policy.
    Falls back to a default policy if none is registered.
    """
    for cls in type(error).__mro__:
        if cls in ERROR_RETRY_POLICIES:
            return ERROR_RETRY_POLICIES[cls]
    # Default fallback
    return RetryPolicy(max_retries=2, base_delay=1.0, max_delay=10.0, retryable=(type(error),))


def should_retry(error: DistLLMError, attempt: int) -> bool:
    """Check if an error should be retried given the current attempt count."""
    policy = get_retry_policy(error)
    return attempt < policy.max_retries and type(error) in policy.retryable


def get_retry_delay(error: DistLLMError, attempt: int) -> float:
    """Calculate the retry delay for a given error and attempt."""
    policy = get_retry_policy(error)
    return min(
        policy.base_delay * (policy.backoff_multiplier ** attempt),
        policy.max_delay,
    )
