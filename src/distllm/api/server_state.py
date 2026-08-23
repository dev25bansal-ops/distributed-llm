"""Server-side application state — single proxy to the shared AppState.

Routes use ``g`` from ``distllm.api.api_state``.
Server code uses ``state`` from this module.
Both reference the same ``AppState`` singleton so they stay in sync.
"""

from __future__ import annotations

from typing import Any

from distllm.api.api_state import _state as _shared_state


class _ServerState:
    """Server-side state proxy that delegates to the shared AppState.

    This ensures ``state.coordinator`` and ``g.coordinator`` always
    return the same object — no dual bookkeeping.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_shared_state, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_shared_state, name, value)


state = _ServerState()
