"""Retry utilities for distributed LLM inference.

Provides retry policies with exponential backoff for both synchronous
and asynchronous operations.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Optional, Type

from loguru import logger

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        retryable: Tuple of exception types that trigger retry.
        backoff_multiplier: Multiplier for exponential backoff (default 2).
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    retryable: tuple = (Exception,)
    backoff_multiplier: float = 2.0


def with_retry(policy: RetryPolicy):
    """Decorator that wraps a function with retry logic and exponential backoff.

    Args:
        policy: RetryPolicy configuring retry behavior.

    Returns:
        Decorated function with retry support.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            for attempt in range(policy.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except policy.retryable as e:
                    last_exception = e
                    if attempt == policy.max_retries:
                        raise
                    delay = min(
                        policy.base_delay * (policy.backoff_multiplier ** attempt),
                        policy.max_delay,
                    )
                    logger.warning(
                        f"{fn.__name__} failed (attempt {attempt + 1}/{policy.max_retries + 1}): "
                        f"{e}, retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator


def with_retry_async(policy: RetryPolicy):
    """Async decorator that wraps an async function with retry logic.

    Args:
        policy: RetryPolicy configuring retry behavior.

    Returns:
        Decorated async function with retry support.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            for attempt in range(policy.max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except policy.retryable as e:
                    last_exception = e
                    if attempt == policy.max_retries:
                        raise
                    delay = min(
                        policy.base_delay * (policy.backoff_multiplier ** attempt),
                        policy.max_delay,
                    )
                    logger.warning(
                        f"{fn.__name__} failed (attempt {attempt + 1}/{policy.max_retries + 1}): "
                        f"{e}, retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator


def retry_grpc_call(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[tuple] = None,
) -> T:
    """Retry a gRPC call with exponential backoff.

    Args:
        fn: Callable that performs the gRPC call.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Tuple of exception types that trigger retry.
            Defaults to grpc.RpcError.

    Returns:
        Result from fn().

    Raises:
        The last exception if all retries exhausted.
    """
    import grpc

    if retryable_exceptions is None:
        retryable_exceptions = (grpc.RpcError,)

    policy = RetryPolicy(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable=retryable_exceptions,
    )

    last_exception = None
    for attempt in range(policy.max_retries + 1):
        try:
            return fn()
        except policy.retryable as e:
            last_exception = e
            if attempt == policy.max_retries:
                raise

            delay = min(policy.base_delay * (policy.backoff_multiplier ** attempt), policy.max_delay)
            logger.warning(
                f"gRPC call failed (attempt {attempt + 1}/{policy.max_retries + 1}): "
                f"{e}, retrying in {delay:.1f}s"
            )
            time.sleep(delay)

    raise last_exception  # type: ignore[misc]
