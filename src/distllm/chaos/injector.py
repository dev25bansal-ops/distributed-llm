"""Chaos injector for fault injection scenarios."""

import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


@dataclass
class ChaosEvent:
    """Records a single chaos injection event."""
    event_type: str
    node_id: str
    timestamp: float
    params: dict[str, Any]
    result: str  # "success" | "failed" | "recovered"
    duration_s: float = 0.0


class ChaosInjector:
    """Injects faults into the distributed system for chaos engineering.

    Methods:
        kill_node: Simulate node failure via resource manager.
        add_latency: Add artificial delay to node RPC calls.
        drop_message: Drop messages matching a pattern.
        corrupt_data: Introduce bit-flip errors in tensor payloads.
    """

    def __init__(self, resource_manager, max_latency_ms: int = 5000):
        self.resource_manager = resource_manager
        self.max_latency_ms = max_latency_ms
        self._events: list[ChaosEvent] = []
        self._latency_delays: dict[str, float] = {}  # node_id -> delay_ms
        self._drop_patterns: dict[str, str] = {}  # node_id -> pattern
        self._corruption_rates: dict[str, float] = {}  # node_id -> rate

    @property
    def events(self) -> list[ChaosEvent]:
        return list(self._events)

    def _record_event(self, event_type: str, node_id: str, params: dict[str, Any], result: str, duration_s: float = 0.0):
        self._events.append(ChaosEvent(
            event_type=event_type,
            node_id=node_id,
            timestamp=time.time(),
            params=params,
            result=result,
            duration_s=duration_s,
        ))

    def kill_node(self, node_id: str) -> ChaosEvent:
        """Simulate a node failure.

        Triggers the circuit breaker open without killing the actual process.

        Args:
            node_id: The node to kill.

        Returns:
            The recorded chaos event.
        """
        start = time.time()
        logger.warning(f"[Chaos] Killing node {node_id}")
        try:
            self.resource_manager.simulate_node_failure(node_id)
            duration = time.time() - start
            event = ChaosEvent(
                event_type="kill_node",
                node_id=node_id,
                timestamp=start,
                params={},
                result="success",
                duration_s=duration,
            )
        except Exception as e:
            duration = time.time() - start
            event = ChaosEvent(
                event_type="kill_node",
                node_id=node_id,
                timestamp=start,
                params={},
                result=f"failed: {e}",
                duration_s=duration,
            )
        self._events.append(event)
        return event

    def add_latency(self, node_id: str, delay_ms: int, duration_s: float = 0.0) -> ChaosEvent:
        """Add artificial latency to a node's RPC calls.

        Args:
            node_id: The node to add latency to.
            delay_ms: Additional delay in milliseconds.
            duration_s: How long to maintain the latency (0 = until cleared).

        Returns:
            The recorded chaos event.
        """
        delay_ms = min(delay_ms, self.max_latency_ms)
        start = time.time()
        logger.warning(f"[Chaos] Adding {delay_ms}ms latency to node {node_id}")
        self._latency_delays[node_id] = delay_ms
        event = ChaosEvent(
            event_type="add_latency",
            node_id=node_id,
            timestamp=start,
            params={"delay_ms": delay_ms, "duration_s": duration_s},
            result="success",
        )
        self._events.append(event)
        return event

    def clear_latency(self, node_id: str) -> None:
        """Remove artificial latency from a node."""
        self._latency_delays.pop(node_id, None)

    def get_latency_for_node(self, node_id: str) -> float:
        """Get the current artificial latency for a node in ms."""
        return self._latency_delays.get(node_id, 0.0)

    def drop_message(self, node_id: str, message_pattern: str, duration_s: float = 0.0) -> ChaosEvent:
        """Mark messages matching a pattern to be dropped for a node.

        Args:
            node_id: The node to drop messages for.
            message_pattern: Regex pattern to match messages.
            duration_s: How long to drop messages (0 = until cleared).

        Returns:
            The recorded chaos event.
        """
        start = time.time()
        logger.warning(f"[Chaos] Dropping messages matching '{message_pattern}' for node {node_id}")
        self._drop_patterns[node_id] = message_pattern
        event = ChaosEvent(
            event_type="drop_message",
            node_id=node_id,
            timestamp=start,
            params={"pattern": message_pattern, "duration_s": duration_s},
            result="success",
        )
        self._events.append(event)
        return event

    def clear_drop_pattern(self, node_id: str) -> None:
        """Remove drop pattern for a node."""
        self._drop_patterns.pop(node_id, None)

    def should_drop_message(self, node_id: str, message: str) -> bool:
        """Check if a message should be dropped based on drop patterns."""
        import re
        pattern = self._drop_patterns.get(node_id)
        if pattern is None:
            return False
        compiled = re.compile(pattern, re.DOTALL)
        try:
            return bool(compiled.search(message, timeout=1.0))
        except Exception:
            return False

    def corrupt_data(self, node_id: str, corruption_rate: float, duration_s: float = 0.0) -> ChaosEvent:
        """Set a corruption rate for a node's data payloads.

        Args:
            node_id: The node to corrupt data for.
            corruption_rate: Probability of corruption (0.0-1.0).
            duration_s: How long to corrupt data (0 = until cleared).

        Returns:
            The recorded chaos event.
        """
        start = time.time()
        logger.warning(f"[Chaos] Setting corruption rate {corruption_rate} for node {node_id}")
        self._corruption_rates[node_id] = corruption_rate
        event = ChaosEvent(
            event_type="corrupt_data",
            node_id=node_id,
            timestamp=start,
            params={"corruption_rate": corruption_rate, "duration_s": duration_s},
            result="success",
        )
        self._events.append(event)
        return event

    def clear_corruption_rate(self, node_id: str) -> None:
        """Remove corruption rate for a node."""
        self._corruption_rates.pop(node_id, None)

    def should_corrupt(self, node_id: str) -> bool:
        """Check if data should be corrupted based on the corruption rate."""
        rate = self._corruption_rates.get(node_id, 0.0)
        if rate <= 0:
            return False
        return random.random() < rate

    def corrupt_tensor(self, data: bytes) -> bytes:
        """Introduce bit-flip errors in byte data."""
        if not data:
            return data
        data = bytearray(data)
        num_flips = max(1, len(data) // 100)
        for _ in range(num_flips):
            idx = random.randint(0, len(data) - 1)
            data[idx] ^= 0x01  # Flip one bit
        return bytes(data)

    def reset(self) -> None:
        """Clear all injection state and events."""
        self._events.clear()
        self._latency_delays.clear()
        self._drop_patterns.clear()
        self._corruption_rates.clear()
