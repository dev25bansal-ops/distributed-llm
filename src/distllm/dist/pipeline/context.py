"""Pipeline context dataclasses for distributed inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class NodeForwardContext:
    """Context for a single node gRPC forward pass in the pipeline.

    Groups all parameters needed by _execute_node_grpc into one object
    to avoid a 12-parameter method signature.
    """

    node_id: str
    node_kv_caches: dict[str, list | None]
    current_hidden: torch.Tensor | None
    request_id: str
    draft_tokens: list[int] | None
    input_ids: torch.Tensor
    seq_len: int
    batch_size: int
    is_first_node: bool
    is_last_node: bool

    @classmethod
    def build(
        cls,
        node_id: str,
        node_kv_caches: dict[str, list | None],
        current_hidden: torch.Tensor | None,
        request_id: str,
        draft_tokens: list[int] | None,
        input_ids: torch.Tensor,
        is_first_node: bool,
        is_last_node: bool,
    ) -> NodeForwardContext:
        """Build a context from the available tensors."""
        if input_ids is not None:
            seq_len = input_ids.shape[1]
            batch_size = input_ids.shape[0]
        elif current_hidden is not None:
            seq_len = current_hidden.shape[1]
            batch_size = current_hidden.shape[0]
        else:
            seq_len = 1
            batch_size = 1
        return cls(
            node_id=node_id,
            node_kv_caches=node_kv_caches,
            current_hidden=current_hidden,
            request_id=request_id,
            draft_tokens=draft_tokens,
            input_ids=input_ids,
            seq_len=seq_len,
            batch_size=batch_size,
            is_first_node=is_first_node,
            is_last_node=is_last_node,
        )


@dataclass
class NodeCheckpoint:
    """Checkpoint saved after a node completes its forward pass.

    Stores the output hidden state and KV cache so the pipeline can
    resume from this point if the next node fails.
    """

    request_id: str
    node_id: str
    node_index: int
    hidden_state: torch.Tensor | None = None
    kv_cache: dict[str, list] | list | None = None
    input_ids: torch.Tensor | None = None
    draft_tokens: list[int] | None = None
