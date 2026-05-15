"""VRAM-aware quantization method selection for distributed LLM inference.

Selects the optimal quantization method based on available node VRAM
and model size, balancing memory savings against quality degradation.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class NodeVRAMInfo:
    """VRAM information for a worker node.

    Attributes:
        total_memory: Total VRAM in bytes.
        available_memory: Available VRAM in bytes.
        device_type: "cuda" or "cpu".
    """
    total_memory: int = 0
    available_memory: int = 0
    device_type: str = "cpu"


def select_for_node(
    node_info: NodeVRAMInfo,
    model_size_bytes: int,
    target_latency_ms: Optional[float] = None,
) -> str:
    """Select quantization method based on node VRAM and model size.

    Rules:
    - If VRAM < model_size * 1.2 → 4-bit (most aggressive)
    - If VRAM < model_size * 1.8 → 8-bit (moderate)
    - Otherwise → none (full precision)

    Args:
        node_info: Node VRAM information.
        model_size_bytes: Estimated model size in bytes (all parameters).
        target_latency_ms: Optional latency target (unused in V1, reserved).

    Returns:
        Quantization method string: "bnb_4bit", "bnb_8bit", or "none".
    """
    if node_info.device_type != "cuda" or node_info.available_memory == 0:
        logger.debug("Non-GPU node or unknown VRAM, using no quantization")
        return "none"

    available = node_info.available_memory

    if available < model_size_bytes * 1.2:
        logger.info(
            f"VRAM {available / 1e9:.1f}GB < model {model_size_bytes / 1e9:.1f}GB * 1.2, "
            f"selecting 4-bit quantization"
        )
        return "bnb_4bit"

    if available < model_size_bytes * 1.8:
        logger.info(
            f"VRAM {available / 1e9:.1f}GB < model {model_size_bytes / 1e9:.1f}GB * 1.8, "
            f"selecting 8-bit quantization"
        )
        return "bnb_8bit"

    logger.debug(
        f"VRAM {available / 1e9:.1f}GB sufficient for model {model_size_bytes / 1e9:.1f}GB, "
        f"no quantization needed"
    )
    return "none"


def estimate_model_size_bytes(hidden_size: int, num_layers: int, vocab_size: int, dtype_bytes: int = 2) -> int:
    """Estimate model parameter size in bytes.

    Rough estimate based on transformer architecture:
    - Embedding: vocab_size * hidden_size
    - Per layer: 4 * (hidden_size^2) (attention + MLP weights)
    - LM head: vocab_size * hidden_size

    Args:
        hidden_size: Model hidden dimension.
        num_layers: Number of transformer layers.
        vocab_size: Vocabulary size.
        dtype_bytes: Bytes per parameter (2 for fp16, 4 for fp32).

    Returns:
        Estimated total model size in bytes.
    """
    embedding_params = vocab_size * hidden_size
    layer_params = 4 * (hidden_size ** 2)
    total_params = embedding_params + num_layers * layer_params + embedding_params
    return total_params * dtype_bytes


def build_quantization_config(method: str, **kwargs) -> Optional["BitsAndBytesConfig"]:
    """Build a BitsAndBytesConfig from method string and optional overrides.

    Args:
        method: Quantization method ("none", "bnb_4bit", "bnb_8bit").
        **kwargs: Optional overrides for quantization parameters.

    Returns:
        BitsAndBytesConfig instance, or None for "none".
    """
    if method == "none":
        return None

    from transformers import BitsAndBytesConfig

    if method == "bnb_8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=kwargs.get("llm_int8_threshold", 6.0),
        )

    if method == "bnb_4bit":
        import torch
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_map.get(kwargs.get("bnb_4bit_compute_dtype", "float16"), torch.float16),
            bnb_4bit_quant_type=kwargs.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=kwargs.get("bnb_4bit_use_double_quant", True),
        )

    logger.warning(f"Unknown quantization method: {method}, falling back to none")
    return None
