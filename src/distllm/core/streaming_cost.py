"""Streaming Token Cost Dashboard — real-time cost tracking during streaming.

Extends the streaming response to include real-time cost accumulation,
providing users with live visibility into per-token costs as they
are generated.

Features:
- Real-time cost per token in SSE stream events
- Cumulative cost tracking in stream metadata
- Savings vs cloud API shown in real-time
- Final cost summary in the usage chunk
- WebSocket broadcast for dashboard visualization
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class StreamingCostState:
    """Tracks cost accumulation during a streaming response."""
    request_id: str = ""
    model_name: str = ""
    gpu_type: str = ""

    # Token counts (updated as tokens stream)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # C5: Separate input/output cost rates
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    cumulative_input_cost: float = 0.0
    cumulative_output_cost: float = 0.0
    cumulative_cost: float = 0.0

    # Cloud comparison (also separate rates)
    cloud_input_cost_per_token: float = 0.0
    cloud_output_cost_per_token: float = 0.0
    cumulative_cloud_cost: float = 0.0
    cumulative_savings: float = 0.0

    # Timing
    start_time: float = 0.0
    first_token_time: float = 0.0
    last_token_time: float = 0.0
    ttft_ms: float = 0.0

    # Throughput
    tokens_per_second: float = 0.0
    avg_tokens_per_second: float = 0.0

    def record_input(self, token_count: int) -> None:
        """Record input token count (called once at stream start)."""
        self.input_tokens = token_count
        self.total_tokens = self.input_tokens + self.output_tokens
        self._update_cost()

    def record_output_token(self) -> None:
        """Record a single output token (called per token in stream)."""
        now = time.time()
        self.output_tokens += 1
        self.total_tokens = self.input_tokens + self.output_tokens

        if self.first_token_time == 0:
            self.first_token_time = now
            self.ttft_ms = (now - self.start_time) * 1000

        self.last_token_time = now
        self._update_throughput()
        self._update_cost()

    def _update_cost(self) -> None:
        """C5: Update cumulative cost using separate input/output rates."""
        self.cumulative_input_cost = self.input_tokens * self.input_cost_per_token
        self.cumulative_output_cost = self.output_tokens * self.output_cost_per_token
        self.cumulative_cost = self.cumulative_input_cost + self.cumulative_output_cost

        cloud_input = self.input_tokens * self.cloud_input_cost_per_token
        cloud_output = self.output_tokens * self.cloud_output_cost_per_token
        self.cumulative_cloud_cost = cloud_input + cloud_output
        # Clamp tiny negative savings from float error when no cloud rate is
        # configured (cloud cost is 0); real negative savings still require
        # an actual cloud rate to be set.
        savings = self.cumulative_cloud_cost - self.cumulative_cost
        has_cloud_rates = (
            self.cloud_input_cost_per_token > 0 or self.cloud_output_cost_per_token > 0
        )
        if not has_cloud_rates:
            savings = max(savings, 0.0)
        self.cumulative_savings = savings

    def _update_throughput(self) -> None:
        """Update tokens/second metrics."""
        if self.output_tokens > 1 and self.first_token_time > 0:
            elapsed = self.last_token_time - self.first_token_time
            if elapsed <= 0:
                # Tokens arrived faster than clock resolution; fall back to
                # the start-time window so throughput stays meaningful.
                elapsed = (
                    self.last_token_time - self.start_time
                    if self.start_time > 0 else 0.0
                )
            if elapsed > 0:
                self.tokens_per_second = (self.output_tokens - 1) / elapsed
                self.avg_tokens_per_second = self.output_tokens / elapsed

    def to_token_event(self) -> dict[str, Any]:
        """Generate a cost event for inclusion in SSE stream.

        Returns a dict that can be included in the stream chunk's
        metadata for real-time cost visibility.
        """
        return {
            "cost": {
                "cumulative_usd": round(self.cumulative_cost, 8),
                "per_token_usd": round(self.output_cost_per_token, 10),
                "tokens": self.total_tokens,
                "tps": round(self.tokens_per_second, 1),
            },
            "savings": {
                "cumulative_usd": round(self.cumulative_savings, 8),
                "cloud_cost_usd": round(self.cumulative_cloud_cost, 8),
            },
            "timing": {
                "ttft_ms": round(self.ttft_ms, 1),
                "elapsed_ms": round(
                    (self.last_token_time - self.start_time) * 1000, 1
                ) if self.start_time > 0 else 0,
            },
        }

    def to_final_summary(self) -> dict[str, Any]:
        """Generate the final cost summary for the usage chunk."""
        elapsed = (self.last_token_time - self.start_time) if self.start_time > 0 else 0
        return {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cumulative_cost, 8),
            "cost_per_token_usd": round(
                self.cumulative_cost / max(self.total_tokens, 1), 10
            ),
            "cloud_cost_usd": round(self.cumulative_cloud_cost, 8),
            "savings_usd": round(self.cumulative_savings, 8),
            "ttft_ms": round(self.ttft_ms, 1),
            "total_duration_ms": round(elapsed * 1000, 1),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 1),
        }


class StreamingCostTracker:
    """Manages streaming cost state for concurrent requests.

    Maintains per-request cost state and provides cost data
    for inclusion in streaming responses.
    """

    def __init__(self, default_cost_per_token: float = 0.0):
        self._default_cost_per_token = default_cost_per_token
        self._active: dict[str, StreamingCostState] = {}
        self._completed: list[StreamingCostState] = []
        self._lock = __import__("threading").Lock()

    def start_tracking(
        self,
        request_id: str,
        input_tokens: int,
        model_name: str = "",
        gpu_type: str = "",
        cost_per_token: float = 0.0,
        cloud_cost_per_token: float = 0.0,
        input_cost_per_token: float = 0.0,
        cloud_input_cost_per_token: float = 0.0,
    ) -> StreamingCostState:
        """Start tracking cost for a streaming request.

        C5: Now accepts separate input/output cost rates.
        """
        state = StreamingCostState(
            request_id=request_id,
            model_name=model_name,
            gpu_type=gpu_type,
            input_tokens=input_tokens,
            # C5: Separate input/output rates
            input_cost_per_token=input_cost_per_token or cost_per_token or self._default_cost_per_token,
            output_cost_per_token=cost_per_token or self._default_cost_per_token,
            cloud_input_cost_per_token=cloud_input_cost_per_token,
            cloud_output_cost_per_token=cloud_cost_per_token,
            start_time=time.time(),
        )
        state._update_cost()

        with self._lock:
            self._active[request_id] = state

        return state

    def record_token(self, request_id: str) -> dict[str, Any] | None:
        """Record an output token and return cost event data.

        Args:
            request_id: The request that generated a token.

        Returns:
            Cost event dict for inclusion in SSE stream, or None.
        """
        with self._lock:
            state = self._active.get(request_id)
            if not state:
                return None
            state.record_output_token()
            return state.to_token_event()

    def finish_tracking(self, request_id: str) -> dict[str, Any] | None:
        """Finish tracking and return the final cost summary.

        Args:
            request_id: The completed request.

        Returns:
            Final cost summary dict, or None.
        """
        with self._lock:
            state = self._active.pop(request_id, None)
            if not state:
                return None
            self._completed.append(state)
            # E7: Keep only recent completed states
            if len(self._completed) > 1000:
                self._completed = self._completed[-500:]
            return state.to_final_summary()

    def get_active_state(self, request_id: str) -> StreamingCostState | None:
        """Get the current cost state for an active request."""
        with self._lock:
            return self._active.get(request_id)

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate streaming cost statistics."""
        with self._lock:
            active = list(self._active.values())
            completed = self._completed[-100:]  # Last 100

            return {
                "active_streams": len(active),
                "active_cost_per_second": sum(
                    s.cumulative_cost / max(time.time() - s.start_time, 1)
                    for s in active
                ),
                "completed_count": len(self._completed),
                "avg_cost_per_stream": (
                    sum(s.cumulative_cost for s in completed) / max(len(completed), 1)
                ),
                "avg_savings_per_stream": (
                    sum(s.cumulative_savings for s in completed) / max(len(completed), 1)
                ),
                "total_tokens_tracked": sum(s.total_tokens for s in completed),
                "total_cost_tracked": sum(s.cumulative_cost for s in completed),
                "total_savings_tracked": sum(s.cumulative_savings for s in completed),
            }


# ── Module-level singleton ──────────────────────────────────────────────────

_tracker: StreamingCostTracker | None = None


def get_streaming_cost_tracker() -> StreamingCostTracker:
    """Get or create the module-level StreamingCostTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = StreamingCostTracker()
    return _tracker


def reset_streaming_cost_tracker() -> None:
    """Reset the singleton for testing."""
    global _tracker
    _tracker = None
