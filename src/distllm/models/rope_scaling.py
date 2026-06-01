"""RoPE scaling configuration and application for long-context inference."""

from __future__ import annotations

from typing import Any

from loguru import logger


__all__ = ["build_rope_scaling_config", "apply_rope_scaling"]


def build_rope_scaling_config(
    model_type: str = "llama",
    original_max_pos: int = 4096,
    target_max_pos: int = 131072,
    scaling_type: str = "yarn",
    rope_theta: float = 10000.0,
    attention_head_dim: int = 128,
) -> dict:
    """Build RoPE scaling configuration for extending context to 128K+.

    Supports NTK-aware, YaRN, and linear scaling methods.

    Args:
        model_type: Model architecture (llama, mistral, gemma, qwen2).
        original_max_pos: Original max position embeddings (e.g., 4096).
        target_max_pos: Desired max position embeddings (e.g., 131072).
        scaling_type: Scaling method: "linear", "ntk", "ntk_aware", "yarn".
        rope_theta: Base theta for RoPE (default 10000.0).
        attention_head_dim: Dimension per attention head (default 128).

    Returns:
        Dict suitable for setting model config's rope_scaling field.

    Raises:
        ValueError: If scaling_type is not recognized.
    """
    scale = target_max_pos / original_max_pos

    if scaling_type == "linear":
        return {
            "type": "linear",
            "factor": scale,
        }

    if scaling_type in ("ntk", "ntk_aware"):
        rope_theta_scaled = rope_theta * (scale ** (attention_head_dim / (attention_head_dim - 2)))
        return {
            "type": "ntk",
            "factor": scale,
            "rope_theta": rope_theta_scaled,
            "original_max_position_embeddings": original_max_pos,
        }

    if scaling_type == "yarn":
        return {
            "type": "yarn",
            "factor": scale,
            "original_max_position_embeddings": original_max_pos,
            "attention_factor": 1.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }

    raise ValueError(f"Unknown RoPE scaling type: {scaling_type}. Supported: linear, ntk, ntk_aware, yarn")


def apply_rope_scaling(
    model: Any,
    target_context_len: int = 131072,
    scaling_type: str = "yarn",
) -> bool:
    """Apply RoPE scaling to a loaded model for extended context.

    Modifies the model's config in-place and re-initializes RoPE
    embeddings if possible.

    Args:
        model: Loaded HuggingFace model.
        target_context_len: Desired context window length.
        scaling_type: Scaling method ("linear", "ntk", "ntk_aware", "yarn").

    Returns:
        True if scaling was applied, False if model doesn't support it.
    """
    config = getattr(model, "config", None)
    if config is None:
        logger.warning("Model has no config, cannot apply RoPE scaling")
        return False

    original_max_pos = getattr(config, "max_position_embeddings", 4096)
    head_dim = getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    model_type = getattr(config, "model_type", "llama")

    rope_config = build_rope_scaling_config(
        model_type=model_type,
        original_max_pos=original_max_pos,
        target_max_pos=target_context_len,
        scaling_type=scaling_type,
        rope_theta=rope_theta,
        attention_head_dim=head_dim,
    )

    config.max_position_embeddings = target_context_len
    config.rope_scaling = rope_config

    if "theta" in rope_config:
        config.rope_theta = rope_config["theta"]

    logger.info(
        f"Applied {scaling_type} RoPE scaling: "
        f"{original_max_pos} -> {target_context_len} "
        f"(factor={target_context_len / original_max_pos:.1f}x)"
    )
    return True
