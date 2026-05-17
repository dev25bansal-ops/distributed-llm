"""Mutable generation parameters for mid-generation updates.

Provides thread-safe parameter updates during streaming generation.
"""

import threading
from dataclasses import dataclass, field


@dataclass
class GenerationParams:
    """Mutable generation parameters that can be updated mid-generation.

    Attributes:
        temperature: Sampling temperature. Higher values increase randomness.
        top_p: Nucleus sampling threshold. Only tokens with cumulative probability <= top_p are considered.
        top_k: Top-k sampling. Only the top_k most likely tokens are considered. 0 means disabled.
    """

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0


class ParamUpdateChannel:
    """Thread-safe channel for updating generation parameters mid-stream.

    Keyed by request_id, allowing per-request parameter updates during
    streaming generation.

    Attributes:
        channels: Dict mapping request_id to GenerationParams.
        _lock: Threading lock for concurrent access safety.
    """

    def __init__(self):
        self.channels: dict[str, GenerationParams] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str, params: GenerationParams | None = None) -> None:
        """Register a new request with optional initial params.

        Args:
            request_id: Unique request identifier.
            params: Initial generation params. Creates defaults if None.
        """
        with self._lock:
            self.channels[request_id] = params or GenerationParams()

    def update(self, request_id: str, **kwargs) -> GenerationParams | None:
        """Update params for a request. Only provided fields are changed.

        Args:
            request_id: Unique request identifier.
            **kwargs: Fields to update (temperature, top_p, top_k).

        Returns:
            Updated GenerationParams, or None if request not found.
        """
        with self._lock:
            if request_id not in self.channels:
                return None
            params = self.channels[request_id]
            if "temperature" in kwargs:
                params.temperature = kwargs["temperature"]
            if "top_p" in kwargs:
                params.top_p = kwargs["top_p"]
            if "top_k" in kwargs:
                params.top_k = kwargs["top_k"]
            return params

    def get(self, request_id: str) -> GenerationParams | None:
        """Get current params for a request.

        Args:
            request_id: Unique request identifier.

        Returns:
            Current GenerationParams, or None if not registered.
        """
        with self._lock:
            return self.channels.get(request_id)

    def unregister(self, request_id: str) -> None:
        """Remove a request from the channel.

        Args:
            request_id: Unique request identifier.
        """
        with self._lock:
            self.channels.pop(request_id, None)

    def list_requests(self) -> list[str]:
        """List all active request IDs.

        Returns:
            List of active request_id strings.
        """
        with self._lock:
            return list(self.channels.keys())
