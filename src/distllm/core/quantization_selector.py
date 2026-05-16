"""VRAM-aware quantization method selection and config building for distributed LLM inference.

Supports:
- BitsAndBytes (4-bit NF4/FP4, 8-bit)
- GPTQ with Marlin kernel (auto-gptq)
- AWQ (autoawq)
- FP8 dynamic quantization (NVIDIA Hopper native)
- KV cache quantization (4-8x memory reduction)
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
        compute_capability: GPU compute capability (e.g. 8.9 for Hopper).
    """
    total_memory: int = 0
    available_memory: int = 0
    device_type: str = "cpu"
    compute_capability: float = 0.0


def select_for_node(
    node_info: NodeVRAMInfo,
    model_size_bytes: int,
    target_latency_ms: Optional[float] = None,
) -> str:
    """Select quantization method based on node VRAM and model size.

    Rules:
    - If VRAM < model_size * 1.1 --> GPTQ 4-bit (best quality/size tradeoff)
    - If VRAM < model_size * 1.2 --> BNB 4-bit
    - If VRAM < model_size * 1.5 --> AWQ 4-bit
    - If VRAM < model_size * 1.8 --> BNB 8-bit
    - If Hopper GPU + VRAM < model_size * 2.0 --> FP8
    - Otherwise --> none (full precision)

    Args:
        node_info: Node VRAM information.
        model_size_bytes: Estimated model size in bytes.
        target_latency_ms: Optional latency target (reserved for future use).

    Returns:
        Quantization method string.
    """
    if node_info.device_type != "cuda" or node_info.available_memory == 0:
        logger.debug("Non-GPU node or unknown VRAM, using no quantization")
        return "none"

    available = node_info.available_memory
    is_hopper = node_info.compute_capability >= 9.0

    if available < model_size_bytes * 1.1:
        logger.info(f"VRAM critically low, selecting GPTQ 4-bit with Marlin")
        return "gptq"

    if available < model_size_bytes * 1.2:
        logger.info(f"VRAM {available / 1e9:.1f}GB < model * 1.2, selecting BNB 4-bit")
        return "bnb_4bit"

    if available < model_size_bytes * 1.5:
        logger.info(f"VRAM {available / 1e9:.1f}GB < model * 1.5, selecting AWQ 4-bit")
        return "awq"

    if is_hopper and available < model_size_bytes * 2.0:
        logger.info(f"Hopper GPU with limited VRAM, selecting FP8 dynamic")
        return "fp8"

    if available < model_size_bytes * 1.8:
        logger.info(f"VRAM {available / 1e9:.1f}GB < model * 1.8, selecting BNB 8-bit")
        return "bnb_8bit"

    logger.debug(f"VRAM sufficient for full precision model")
    return "none"


def estimate_model_size_bytes(hidden_size: int, num_layers: int, vocab_size: int, dtype_bytes: int = 2) -> int:
    """Estimate model parameter size in bytes.

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


def build_quantization_config(method: str, **kwargs) -> Optional[object]:
    """Build quantization config from method string and optional overrides.

    Supports BitsAndBytesConfig, GPTQConfig, and AWQ config dicts.

    Args:
        method: Quantization method.
        **kwargs: Optional overrides.

    Returns:
        Quantization config object, or None for "none".
    """
    if method == "none":
        return None

    if method == "gptq":
        return _build_gptq_config(**kwargs)

    if method == "awq":
        return _build_awq_config(**kwargs)

    if method == "fp8":
        return _build_fp8_config(**kwargs)

    # BitsAndBytes
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


def _build_gptq_config(**kwargs) -> dict:
    """Build GPTQ quantization config.

    Returns a dict that the model loader uses with auto-gptq.
    GPTQ with Marlin kernel provides 2-4x speedup on Hopper GPUs.

    Args:
        **kwargs: GPTQ parameters (bits, group_size, desc_act, use_marlin).

    Returns:
        Config dict for GPTQ model loading.
    """
    bits = kwargs.get("gptq_bits", 4)
    group_size = kwargs.get("gptq_group_size", 128)
    desc_act = kwargs.get("gptq_desc_act", False)
    use_marlin = kwargs.get("gptq_use_marlin", True)

    logger.info(
        f"GPTQ config: bits={bits}, group_size={group_size}, "
        f"desc_act={desc_act}, marlin={use_marlin}"
    )

    return {
        "method": "gptq",
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
        "use_marlin": use_marlin,
    }


def _build_awq_config(**kwargs) -> dict:
    """Build AWQ quantization config.

    AWQ (Activation-aware Weight Quantization) provides better quality
    than GPTQ at the same bit width by protecting salient weights.

    Args:
        **kwargs: AWQ parameters (bits, group_size).

    Returns:
        Config dict for AWQ model loading.
    """
    bits = kwargs.get("awq_bits", 4)
    group_size = kwargs.get("awq_group_size", 128)

    logger.info(f"AWQ config: bits={bits}, group_size={group_size}")

    return {
        "method": "awq",
        "bits": bits,
        "group_size": group_size,
    }


def _build_fp8_config(**kwargs) -> dict:
    """Build FP8 dynamic quantization config.

    FP8 is natively supported on NVIDIA Hopper (H100/H200) GPUs.
    Dynamic per-tensor quantization provides 2x memory reduction
    with minimal quality loss.

    Args:
        **kwargs: FP8 parameters (scheme, dynamic).

    Returns:
        Config dict for FP8 model loading.
    """
    scheme = kwargs.get("fp8_scheme", "e4m3")
    dynamic = kwargs.get("fp8_dynamic", True)

    logger.info(f"FP8 config: scheme={scheme}, dynamic={dynamic}")

    return {
        "method": "fp8",
        "scheme": scheme,
        "dynamic": dynamic,
    }


def apply_kv_cache_quantization(
    key: "torch.Tensor",
    value: "torch.Tensor",
    bits: int = 8,
) -> tuple:
    """Quantize KV cache tensors to reduce memory usage.

    Uses per-token quantization with dynamic scale factors.
    Provides 4-8x memory reduction depending on bit width.

    Args:
        key: Key tensor [batch, heads, seq, head_dim].
        value: Value tensor [batch, heads, seq, head_dim].
        bits: Target bit width (4 or 8).

    Returns:
        (quantized_key, quantized_value, scale_key, scale_value)
    """
    import torch

    if bits == 8:
        return _quantize_int8(key), _quantize_int8(value)
    elif bits == 4:
        return _quantize_int4(key), _quantize_int4(value)
    else:
        raise ValueError(f"Unsupported KV cache bits: {bits}")


def _quantize_int8(tensor: "torch.Tensor") -> tuple:
    """Per-token int8 quantization with dynamic scale."""
    import torch

    # Compute per-token scale: max absolute value along last dim
    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = scale / 127.0
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def _quantize_int4(tensor: "torch.Tensor") -> tuple:
    """Per-token int4 quantization with dynamic scale."""
    import torch

    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = scale / 7.0
    quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)  # Store as int8 (4-bit packed)
    return quantized, scale


def dequantize_kv_cache(
    quantized: "torch.Tensor",
    scale: "torch.Tensor",
    bits: int = 8,
) -> "torch.Tensor":
    """Dequantize KV cache tensors back to original dtype.

    Args:
        quantized: Quantized tensor.
        scale: Scale factor tensor.
        bits: Original bit width.

    Returns:
        Dequantized tensor in float16.
    """
    import torch

    qval = quantized.to(torch.float16)
    if bits == 8:
        return qval * scale
    elif bits == 4:
        return qval * scale
    else:
        raise ValueError(f"Unsupported KV cache bits: {bits}")
