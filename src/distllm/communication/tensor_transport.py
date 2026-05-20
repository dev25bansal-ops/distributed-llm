"""Tensor transport utilities for gRPC communication.

Handles tensor parsing, response building, and debug logging
for forward pass requests between distributed nodes.
"""

import torch
from loguru import logger

from distllm.communication.node_pb2 import (
    ForwardPassRequest, ForwardPassResponse,
)
from distllm.communication.serializers import (
    proto_to_kv_cache,
    proto_to_tensor,
    tensor_to_proto,
    kv_cache_to_proto,
)
from distllm.errors import InputValidationError


# Debug mode configuration — set via CLI --debug
class DebugConfig:
    """Module-level debug configuration for tensor shape logging."""
    enabled = False


def set_debug_mode(enabled: bool) -> None:
    """Enable or disable debug mode for tensor shape logging."""
    DebugConfig.enabled = enabled


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return DebugConfig.enabled


def _parse_forward_request(request, device: str) -> dict:
    """Parse input tensors from a ForwardPassRequest proto.

    Returns a dict with keys: input_ids, hidden_states, attention_mask,
    position_ids, past_key_values.
    """
    past_key_values = None
    if request.HasField('kv_cache') and request.use_cache:
        past_key_values = proto_to_kv_cache(request.kv_cache, device).cache

    input_ids = None
    if request.input_ids:
        input_ids = torch.tensor(request.input_ids, dtype=torch.long, device=device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    hidden_states = None
    if request.HasField('hidden_states'):
        hidden_states = proto_to_tensor(request.hidden_states, device)
    elif input_ids is None:
        raise InputValidationError("Either hidden_states or input_ids must be provided", "input")

    attention_mask = None
    if request.HasField('attention_mask'):
        attention_mask = proto_to_tensor(request.attention_mask, device)

    position_ids = None
    if request.HasField('position_ids'):
        position_ids = proto_to_tensor(request.position_ids, device)

    return {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


def _build_forward_response(
    request_id: str,
    output: torch.Tensor,
    new_past_kv,
    draft_tokens: list | None = None,
) -> ForwardPassResponse:
    """Build a ForwardPassResponse from model output.

    Handles draft token verification and KV cache serialization.
    """
    from distllm.core.kv_cache import KVCache  # lazy import to avoid circular dep
    response = ForwardPassResponse(
        request_id=request_id,
        output=tensor_to_proto(output),
        success=True,
    )

    # Handle speculative decoding: verify draft tokens
    if draft_tokens and output.dim() == 3:
        num_positions = min(len(draft_tokens), output.shape[1])
        for i in range(num_positions):
            token_at_pos = torch.argmax(output[:, i, :], dim=-1).item()
            response.verified_tokens.append(token_at_pos)

    if new_past_kv:
        new_cache = KVCache()
        new_cache.set_all(new_past_kv)
        response.kv_cache.CopyFrom(kv_cache_to_proto(new_cache))

    return response


def _log_forward_debug(node_id: str, request, tensors: dict, output: torch.Tensor | None = None) -> None:
    """Log tensor shapes for debugging."""
    if not is_debug_mode():
        return

    input_ids = tensors.get("input_ids")
    hidden_states = tensors.get("hidden_states")
    attention_mask = tensors.get("attention_mask")
    past_key_values = tensors.get("past_key_values")

    if input_ids is not None:
        logger.debug(f"[{node_id}] ForwardPass input_ids shape: {input_ids.shape}")
    if hidden_states is not None:
        logger.debug(f"[{node_id}] ForwardPass hidden_states shape: {hidden_states.shape}")
    if attention_mask is not None:
        logger.debug(f"[{node_id}] ForwardPass attention_mask shape: {attention_mask.shape}")
    if past_key_values:
        cache_len = past_key_values[0][0].shape[-2]
        logger.debug(f"[{node_id}] ForwardPass KV cache seq_len: {cache_len}")
    if hasattr(request, 'draft_tokens') and request.draft_tokens:
        logger.debug(f"[{node_id}] ForwardPass draft_tokens: {list(request.draft_tokens)}")
    if output is not None:
        logger.debug(f"[{node_id}] ForwardPass output shape: {output.shape}")
