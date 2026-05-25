"""Debug mode for tensor shape logging and diagnostics."""

import torch
from loguru import logger

from distllm.errors import InputValidationError


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
    """Parse input tensors from a forward request."""
    past_key_values = None

    input_ids = None
    if hasattr(request, 'input_ids') and request.input_ids:
        input_ids = torch.tensor(request.input_ids, dtype=torch.long, device=device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    hidden_states = None
    if hasattr(request, 'hidden_states') and request.hidden_states is not None:
        hidden_states = request.hidden_states
    elif input_ids is None:
        raise InputValidationError("Either hidden_states or input_ids must be provided", "input")

    attention_mask = None
    if hasattr(request, 'attention_mask') and request.attention_mask is not None:
        attention_mask = request.attention_mask

    position_ids = None
    if hasattr(request, 'position_ids') and request.position_ids is not None:
        position_ids = request.position_ids

    return {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


def _build_forward_response(request_id: str, output: torch.Tensor, new_past_kv, draft_tokens: list | None = None):
    """Build a forward response (stub — legacy protobuf compat removed)."""
    return {"request_id": request_id, "output": output}


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
