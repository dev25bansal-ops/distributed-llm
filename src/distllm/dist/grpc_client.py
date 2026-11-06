"""gRPC client pool for MoE expert routing.

The MoE router (``core/moe_orchestrator.py``) resolves the client for a
node through :class:`GrpcClientPool`. Registered clients must expose::

    call_expert_forward_batch(expert_ids, hidden_states, layer_idx)

returning expert outputs shaped like the stacked ``hidden_states`` input,
and may additionally expose the single-token ``call_expert_forward`` form.

Note: the node.proto service does not yet define an ExpertForward RPC —
until the transport layer registers real clients here, routing falls back
to the identity path with a logged warning (per-token semantics).
"""

from __future__ import annotations

import threading
from typing import Any

from loguru import logger

# Optional dependency detection for torch (used lazily by torch_stack)
_TORCH_AVAILABLE: bool
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a helpful ImportError if torch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for expert forward batching; "
            "install with pip install distributed-llm[self-hosted]"
        )


class GrpcClientPool:
    """Process-wide registry of per-node gRPC clients."""

    _clients: dict[str, Any] = {}
    _lock = threading.Lock()

    @classmethod
    def register_client(cls, node_id: str, client: Any) -> None:
        with cls._lock:
            cls._clients[node_id] = client

    @classmethod
    def unregister_client(cls, node_id: str) -> None:
        with cls._lock:
            cls._clients.pop(node_id, None)

    @classmethod
    def get_client(cls, node_id: str) -> Any | None:
        with cls._lock:
            return cls._clients.get(node_id)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._clients.clear()


def call_expert_forward_batch(
    client: Any,
    expert_ids: list[int],
    hidden_states: Any,
    layer_idx: int,
) -> Any:
    """Invoke the batched expert forward on *client*.

    Prefers the batched method; falls back to a loop over the single-token
    method for older clients that never grew the batch surface.
    """
    batch_fn = getattr(client, "call_expert_forward_batch", None)
    if callable(batch_fn):
        return batch_fn(expert_ids, hidden_states, layer_idx)

    single_fn = getattr(client, "call_expert_forward", None)
    if callable(single_fn):
        return torch_stack([single_fn(eid, h, layer_idx) for eid, h in zip(expert_ids, hidden_states)])
    raise AttributeError("client exposes neither call_expert_forward_batch nor call_expert_forward")


def torch_stack(tensors):
    _require_torch()
    return torch.stack(list(tensors), dim=0)
