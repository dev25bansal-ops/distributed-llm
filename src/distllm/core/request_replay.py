"""Request replay buffer and deterministic debug mode.

Stores a configurable number of recent requests with full context
(prompt, parameters, response, timing) for debugging purposes.
Supports replaying any stored request and deterministic mode
with fixed random seeds for reproducible debugging.
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from loguru import logger


@dataclass
class StoredRequest:
    """Full context of a single request for debugging/replay."""
    request_id: str
    prompt: str
    params: dict[str, Any]
    response: str | None = None
    logprobs: list[dict] | None = None
    generated_tokens: list[int] | None = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    error: str | None = None
    replay_count: int = 0
    model: str = ""


class RequestReplayBuffer:
    """Ring buffer that stores recent requests with full context.

    Supports:
      - Configurable max size (ring buffer eviction)
      - Lookup by request_id or index
      - Replay: re-runs a stored request through a provided handler
      - Export/import for sharing debugging sessions

    Usage:
        buffer = RequestReplayBuffer(max_requests=100)
        buffer.store(request_id, prompt, params)
        request = buffer.get(request_id)
        response = buffer.replay(request, handler_fn)
    """

    def __init__(self, max_requests: int = 100):
        self._max = max_requests
        self._requests: list[StoredRequest] = []
        self._index: dict[str, StoredRequest] = {}
        self._lock = threading.Lock()

    def store(
        self,
        request_id: str,
        prompt: str,
        params: dict[str, Any],
        response: str | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
        logprobs: list[dict] | None = None,
        generated_tokens: list[int] | None = None,
        model: str = "",
    ) -> None:
        """Store a completed request with its full context."""
        entry = StoredRequest(
            request_id=request_id,
            prompt=prompt,
            params=dict(params),
            response=response,
            error=error,
            duration_ms=duration_ms,
            logprobs=logprobs,
            generated_tokens=generated_tokens,
            model=model,
        )
        with self._lock:
            self._requests.append(entry)
            self._index[request_id] = entry
            if len(self._requests) > self._max:
                removed = self._requests.pop(0)
                self._index.pop(removed.request_id, None)

    def get(self, request_id: str) -> StoredRequest | None:
        with self._lock:
            return self._index.get(request_id)

    def get_by_index(self, index: int) -> StoredRequest | None:
        with self._lock:
            if 0 <= index < len(self._requests):
                return self._requests[index]
            return None

    def list_recent(self, n: int = 10) -> list[StoredRequest]:
        with self._lock:
            return list(self._requests[-n:])

    def replay(
        self,
        request_id: str,
        handler: Callable[[str, dict[str, Any]], str],
    ) -> str | None:
        """Replay a stored request through a handler function.

        Args:
            request_id: ID of the request to replay.
            handler: Callable that takes (prompt, params) and returns response.

        Returns:
            Response text, or None if request not found.
        """
        entry = self.get(request_id)
        if entry is None:
            logger.warning(f"Request {request_id} not found in replay buffer")
            return None

        entry.replay_count += 1
        start = time.time()
        try:
            response = handler(entry.prompt, dict(entry.params))
            elapsed = time.time() - start
            logger.info(f"Replayed {request_id} in {elapsed*1000:.0f}ms")
            return response
        except Exception as e:
            logger.error(f"Replay failed for {request_id}: {e}")
            return None

    def export(self, request_ids: list[str] | None = None) -> list[dict]:
        """Export requests to JSON-serializable dicts for debugging sessions."""
        entries = []
        with self._lock:
            targets = (
                [self._index[rid] for rid in request_ids if rid in self._index]
                if request_ids
                else self._requests
            )
            for e in targets:
                entries.append({
                    "request_id": e.request_id,
                    "prompt": e.prompt,
                    "params": e.params,
                    "response": e.response,
                    "error": e.error,
                    "duration_ms": e.duration_ms,
                    "model": e.model,
                    "timestamp": e.timestamp,
                })
        return entries

    def import_requests(self, entries: list[dict]) -> int:
        """Import requests from exported dicts."""
        count = 0
        for data in entries:
            rid = data.get("request_id", str(uuid.uuid4()))
            self.store(
                request_id=rid,
                prompt=data.get("prompt", ""),
                params=data.get("params", {}),
                response=data.get("response"),
                error=data.get("error"),
                duration_ms=data.get("duration_ms", 0.0),
                model=data.get("model", ""),
            )
            count += 1
        return count

    def size(self) -> int:
        with self._lock:
            return len(self._requests)

    def clear(self) -> None:
        with self._lock:
            self._requests.clear()
            self._index.clear()


class DeterministicMode:
    """Manages deterministic debug mode with fixed seeds.

    When enabled, all random operations use fixed seeds for
    reproducible outputs. Useful for debugging regressions
    and verifying fixes.

    Usage:
        det = DeterministicMode(seed=42, enabled=False)
        with det:
            outputs = model.generate(...)  # deterministic
    """

    def __init__(self, seed: int = 42, enabled: bool = False):
        self._seed = seed
        self._enabled = enabled
        self._original_state: dict[str, Any] = {}

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self, seed: int | None = None) -> None:
        """Enable deterministic mode with an optional new seed."""
        self._seed = seed or self._seed
        self._enabled = True
        self._apply_seed()
        logger.info(f"Deterministic mode enabled (seed={self._seed})")

    def disable(self) -> None:
        self._enabled = False
        logger.info("Deterministic mode disabled")

    def _apply_seed(self) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __enter__(self) -> DeterministicMode:
        if self._enabled:
            self._apply_seed()
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# Singleton for global access
_replay_buffer: RequestReplayBuffer | None = None
_deterministic_mode: DeterministicMode | None = None


def get_replay_buffer(max_requests: int = 100) -> RequestReplayBuffer:
    global _replay_buffer
    if _replay_buffer is None:
        _replay_buffer = RequestReplayBuffer(max_requests)
    return _replay_buffer


def get_deterministic_mode(seed: int = 42, enabled: bool = False) -> DeterministicMode:
    global _deterministic_mode
    if _deterministic_mode is None:
        _deterministic_mode = DeterministicMode(seed, enabled)
    return _deterministic_mode
