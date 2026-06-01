"""Channel for mid-stream parameter updates during generation.

Allows the API layer to push parameter changes (temperature, top_p, top_k)
to an in-flight request without interrupting generation.

Usage::

    from distllm.core.param_update_channel import ParamUpdateChannel, GenerationParams

    channel = ParamUpdateChannel()
    channel.register("req-1", GenerationParams(temperature=0.5))

    # In generation loop:
    params = channel.get("req-1")
    if params:
        temperature = params.temperature

    # API layer pushes update:
    channel.update("req-1", temperature=0.3)

    # Cleanup:
    channel.unregister("req-1")
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class GenerationParams:
    """Generation parameters that can be updated mid-stream."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_tokens: int = 128

    def update(self, **kwargs: float | int) -> None:
        """Update parameters from keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ParamUpdateChannel:
    """Thread-safe channel for mid-stream parameter updates.

    Each request registers at the start of generation and unregisters
    at the end.  The API layer can push updates at any time via
    :meth:`update`.
    """

    def __init__(self) -> None:
        self._channels: dict[str, GenerationParams] = {}
        self._lock = threading.Lock()

    def register(
        self, request_id: str, params: GenerationParams | None = None
    ) -> None:
        """Register a request for parameter updates.

        Args:
            request_id: The request to register.
            params: Initial parameters. If None, uses defaults.
        """
        with self._lock:
            self._channels[request_id] = params or GenerationParams()

    def update(self, request_id: str, **kwargs: float | int) -> None:
        """Push parameter updates for an in-flight request.

        Args:
            request_id: The request to update.
            **kwargs: Parameter names and values (temperature, top_p, top_k, max_tokens).
        """
        with self._lock:
            params = self._channels.get(request_id)
            if params is not None:
                params.update(**kwargs)

    def get(self, request_id: str) -> GenerationParams | None:
        """Get current parameters for a request, or None if not registered."""
        with self._lock:
            return self._channels.get(request_id)

    def unregister(self, request_id: str) -> None:
        """Unregister a request from the channel."""
        with self._lock:
            self._channels.pop(request_id, None)

    def list_requests(self) -> list[str]:
        """Return list of registered request IDs."""
        with self._lock:
            return list(self._channels.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._channels)

    def __contains__(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._channels
