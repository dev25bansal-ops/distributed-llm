"""Token accumulator for batching decode tokens before WAN transfer.

Accumulates decode tokens into configurable batches to amortize WAN
round-trip latency. Supports time-based flushing for interactive
latency constraints and adaptive batch sizing based on measured RTT.

Usage::

    acc = TokenAccumulator(min_batch_size=32, max_tokens=512, flush_interval_s=0.1)
    acc.add(tokens)
    if acc.should_flush():
        batch = acc.flush()
        await transport.send(batch)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class AccumulatorMetrics:
    """Snapshot of accumulator metrics."""

    batch_count: int = 0
    avg_batch_size: float = 0.0
    total_tokens_accumulated: int = 0
    current_buffer_size: int = 0
    last_flush_ts: float = 0.0


class TokenAccumulator:
    """Batches decode tokens before WAN transfer.

    Accumulates tokens until either the minimum batch count is reached
    or the flush interval expires. The accumulator is adaptive: if the
    caller provides an *adaptive_min_batch_size*, the effective minimum
    will be clamped to ``min(adaptive_min_batch_size, max_tokens)``
    each time ``should_flush()`` is called.

    Args:
        min_batch_size: Minimum number of tokens to accumulate before
            flushing (subject to adaptive override).
        max_tokens: Hard upper bound on accumulated tokens. Flush is
            forced when the buffer reaches this size.
        flush_interval_s: Maximum wall-clock time (seconds) before an
            automatic flush is triggered, even if the batch size hasn't
            been reached.

    Metrics:
        - ``batch_count``: Number of completed flush operations.
        - ``avg_batch_size``: Running average of tokens per flush.
        - ``total_tokens_accumulated``: Sum of all tokens flushed.
    """

    def __init__(
        self,
        min_batch_size: int = 32,
        max_tokens: int = 512,
        flush_interval_s: float = 0.1,
    ) -> None:
        if min_batch_size < 1:
            raise ValueError(f"min_batch_size must be >= 1, got {min_batch_size}")
        if max_tokens < min_batch_size:
            raise ValueError(
                f"max_tokens ({max_tokens}) must be >= min_batch_size ({min_batch_size})"
            )
        if flush_interval_s < 0:
            raise ValueError(f"flush_interval_s must be >= 0, got {flush_interval_s}")

        self._min_batch_size = min_batch_size
        self._max_tokens = max_tokens
        self._flush_interval_s = flush_interval_s

        self._buffer: list[int] = []
        self._last_flush_ts: float = 0.0
        self._batch_count: int = 0
        self._avg_batch_size: float = 0.0
        self._total_tokens_accumulated: int = 0

    # -- Public API -----------------------------------------------------------

    def add(self, tokens: int | Sequence[int]) -> None:
        """Add token(s) to the accumulator buffer.

        Args:
            tokens: A single token ID or a sequence of token IDs.

        Raises:
            ValueError: If any token exceeds the maximum buffer capacity
                after addition.
        """
        if isinstance(tokens, int):
            self._buffer.append(tokens)
        else:
            self._buffer.extend(tokens)

        if len(self._buffer) > self._max_tokens:
            raise ValueError(
                f"Buffer size {len(self._buffer)} exceeds max_tokens {self._max_tokens}"
            )

    def flush(self) -> list[int]:
        """Return all buffered tokens and reset the accumulator.

        Updates internal metrics (batch count, running average, total).

        Returns:
            A copy of the accumulated token IDs. The internal buffer is
            cleared after this call.
        """
        batch = list(self._buffer)
        self._buffer.clear()

        if batch:
            self._batch_count += 1
            n = self._batch_count
            self._avg_batch_size = (
                (self._avg_batch_size * (n - 1) + len(batch)) / n
            )
            self._total_tokens_accumulated += len(batch)

        self._last_flush_ts = time.monotonic()
        return batch

    def should_flush(self, adaptive_min: int | None = None) -> bool:
        """Determine whether the buffer should be flushed now.

        Checks three conditions:

        1. **Empty buffer** -- returns ``False`` immediately.
        2. **Count trigger** -- buffer size >= ``adaptive_min`` (or
           ``min_batch_size`` if adaptive_min is ``None``).
        3. **Time trigger** -- elapsed wall-clock time since last flush
           exceeds ``flush_interval_s`` (only checked once the count
           condition is **not** met).

        Args:
            adaptive_min: Optional adaptive minimum batch size. When
                provided, the effective min is
                ``min(adaptive_min, self._max_tokens)``.

        Returns:
            ``True`` if the buffer should be flushed.
        """
        if not self._buffer:
            return False

        effective_min = self._min_batch_size
        if adaptive_min is not None:
            effective_min = min(adaptive_min, self._max_tokens)

        if len(self._buffer) >= effective_min:
            return True

        if self._flush_interval_s > 0 and self._last_flush_ts > 0:
            elapsed = time.monotonic() - self._last_flush_ts
            if elapsed >= self._flush_interval_s:
                return True

        return False

    def force_flush(self) -> list[int]:
        """Immediately flush the buffer regardless of size or time.

        Equivalent to calling ``flush()`` but makes the intent explicit.
        """
        return self.flush()

    def reset(self) -> None:
        """Clear the buffer and reset all metrics to their initial state."""
        self._buffer.clear()
        self._last_flush_ts = 0.0
        self._batch_count = 0
        self._avg_batch_size = 0.0
        self._total_tokens_accumulated = 0

    # -- Properties -----------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Number of tokens currently buffered."""
        return len(self._buffer)

    @property
    def min_batch_size(self) -> int:
        """Configured minimum batch size."""
        return self._min_batch_size

    @property
    def max_tokens(self) -> int:
        """Hard upper bound on buffer capacity."""
        return self._max_tokens

    @property
    def flush_interval_s(self) -> float:
        """Configured flush interval in seconds."""
        return self._flush_interval_s

    @property
    def metrics(self) -> AccumulatorMetrics:
        """Current snapshot of accumulator metrics."""
        return AccumulatorMetrics(
            batch_count=self._batch_count,
            avg_batch_size=self._avg_batch_size,
            total_tokens_accumulated=self._total_tokens_accumulated,
            current_buffer_size=len(self._buffer),
            last_flush_ts=self._last_flush_ts,
        )
