"""Shared model size estimation — single source of truth for parameter counts,
layer counts, hidden dimensions, and VRAM requirements.

Consolidates duplicated implementations from ``cli/deploy.py``,
``cli/tune.py``, and ``cli/cost_avoid.py`` into one authoritative module.

Usage::

    from distllm.core.model_sizing import estimate_model_size

    params_b = estimate_model_size("meta-llama/Llama-3.1-8B")
    # -> 8.0

    vram_gb = estimate_vram_gb(params_b, dtype="int8")
    # -> 8
"""

from __future__ import annotations

# ── Known model parameter counts ──────────────────────────────────────────
# Maps model name patterns to (params_b, num_layers, hidden_dim, num_kv_heads).
# Extended for popular open-source models.
_MODEL_SPECS: dict[str, tuple[float, int, int, int]] = {
    # Llama 3.x
    "llama3.2-1b":    (1.0,   16, 2048,   8),
    "llama3.2-3b":    (3.0,   28, 3072,   8),
    "llama3.1-8b":    (8.0,   32, 4096,   8),
    "llama3-8b":      (8.0,   32, 4096,   8),
    "llama3-70b":     (70.0,  80, 8192,   8),
    "llama3.1-70b":   (70.0,  80, 8192,   8),
    "llama3-70b":     (70.0,  80, 8192,   8),
    "llama2-7b":      (7.0,   32, 4096,   32),
    "llama2-13b":     (13.0,  40, 5120,   40),
    "llama2-70b":     (70.0,  80, 8192,   8),
    # Mistral
    "mistral-7b":     (7.0,   32, 4096,   8),
    "mixtral-8x7b":   (47.0,  32, 4096,   8),
    "mixtral-8x22b":  (141.0, 56, 6144,   8),
    # Gemma
    "gemma-2b":       (2.0,   18, 2048,   1),
    "gemma-7b":       (7.0,   28, 3072,   16),
    "gemma2-9b":      (9.0,   42, 3584,   16),
    "gemma2-27b":     (27.0,  46, 4608,   32),
    # Qwen
    "qwen-0.5b":      (0.5,   24, 1024,   16),
    "qwen-1.5b":      (1.5,   24, 2048,   16),
    "qwen-1.8b":      (1.8,   24, 2048,   16),
    "qwen-4b":        (4.0,   24, 2048,   2),
    "qwen-7b":        (7.0,   32, 4096,   32),
    "qwen-14b":       (14.0,  40, 5120,   40),
    "qwen-32b":       (32.0,  64, 6144,   8),
    "qwen2.5-7b":     (7.0,   28, 3584,   4),
    "qwen2.5-14b":    (14.0,  48, 5120,   8),
    "qwen2.5-32b":    (32.0,  64, 6144,   8),
    "qwen2.5-72b":    (72.0,  80, 8192,   8),
    # DeepSeek
    "deepseek-7b":    (7.0,   30, 4096,   32),
    "deepseek-67b":   (67.0,  95, 8192,   95),
    # Phi
    "phi-1.5b":       (1.5,   24, 2048,   24),
    "phi-2b":         (2.0,   32, 2560,   32),
    "phi-3-mini":     (3.8,   32, 3072,   32),
    "phi-3-small":    (7.0,   32, 4096,   32),
    "phi-3-medium":   (14.0,  40, 5120,   40),
    "phi-3.5-moe":    (41.9,  32, 4096,   32),
    # Falcon
    "falcon-7b":      (7.0,   32, 4544,   64),
    "falcon-40b":     (40.0,  60, 8192,   128),
    "falcon-180b":    (180.0, 80, 14848,  64),
    # Code models
    "codellama-7b":   (7.0,   32, 4096,   32),
    "codellama-13b":  (13.0,  40, 5120,   40),
    "codellama-34b":  (34.0,  48, 8192,   8),
    "starcoder-1b":   (1.0,   24, 2048,   1),
    "starcoder-3b":   (3.0,   36, 3072,   4),
    "starcoder-7b":   (7.0,   32, 4096,   8),
    "starcoder2-3b":  (3.0,   30, 3072,   2),
    "starcoder2-7b":  (7.0,   32, 4096,   4),
    "starcoder2-15b": (15.0,  40, 6144,   8),
    # StableLM
    "stablelm-3b":    (3.0,   16, 2560,   4),
    "stablelm-12b":   (12.0,  40, 5120,   8),
    # Dbrx
    "dbrx-132b":      (132.0, 40, 6144,   16),
    # Common general patterns
    "1.5b":           (1.5,   24, 2048,   16),
    "2b":             (2.0,   24, 2560,   8),
    "3b":             (3.0,   28, 3072,   8),
    "7b":             (7.0,   32, 4096,   32),
    "8b":             (8.0,   32, 4096,   8),
    "13b":            (13.0,  40, 5120,   40),
    "14b":            (14.0,  40, 5120,   8),
    "20b":            (20.0,  48, 6144,   8),
    "32b":            (32.0,  64, 6144,   8),
    "34b":            (34.0,  56, 8192,   8),
    "40b":            (40.0,  60, 8192,   8),
    "65b":            (65.0,  80, 8192,   8),
    "70b":            (70.0,  80, 8192,   8),
    "120b":           (120.0, 96, 10240,  8),
    "180b":           (180.0, 80, 14848,  8),
}

_BYTES_PER_PARAM: dict[str, float] = {
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
    "int8": 1,
    "8bit": 1,
    "int4": 0.5,
    "4bit": 0.5,
    "fp8": 1,
    "float8": 1,
}


def _normalize_name(model_name: str) -> str:
    """Normalize a model name for lookup."""
    name = model_name.lower().replace("/", "-").replace("_", "-").replace(".", "-")

    # Strip common prefixes
    for prefix in ["hf-", "huggingface-", "models-"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _find_spec(model_name: str) -> tuple[float, int, int, int] | None:
    """Look up model specs, matching by name pattern."""
    name = _normalize_name(model_name)

    # Exact match first
    if name in _MODEL_SPECS:
        return _MODEL_SPECS[name]

    # Check if any key is a substring of the name
    for key, spec in _MODEL_SPECS.items():
        if key in name:
            return spec

    return None


def estimate_model_size(model_name: str) -> float:
    """Estimate model parameter count in billions from the model name.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        Estimated parameter count in billions (e.g. 7.0 for Llama-2-7B).
    """
    spec = _find_spec(model_name)
    if spec:
        return spec[0]

    # Fallback: parse size from name pattern like "7b", "70b", "350m"
    name = model_name.lower()
    import re
    m = re.search(r'(\d+(?:\.\d+)?)\s*[bB]', name)
    if m:
        return float(m.group(1))

    m = re.search(r'(\d+)\s*[mM]', name)
    if m:
        return float(m.group(1)) / 1000.0

    return 7.0  # Sensible default


def estimate_num_layers(model_name: str) -> int:
    """Estimate the number of transformer layers from the model name.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        Estimated layer count.
    """
    spec = _find_spec(model_name)
    if spec:
        return spec[1]

    # Fallback based on size
    params_b = estimate_model_size(model_name)
    if params_b >= 100:
        return 80
    if params_b >= 50:
        return 60
    if params_b >= 30:
        return 48
    if params_b >= 10:
        return 40
    if params_b >= 5:
        return 32
    return 24


def estimate_hidden_dim(model_name: str) -> int:
    """Estimate the hidden dimension from the model name.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        Estimated hidden dimension.
    """
    spec = _find_spec(model_name)
    if spec:
        return spec[2]

    params_b = estimate_model_size(model_name)
    if params_b >= 100:
        return 10240
    if params_b >= 50:
        return 8192
    if params_b >= 30:
        return 6144
    if params_b >= 10:
        return 5120
    if params_b >= 5:
        return 4096
    if params_b >= 2:
        return 2560
    return 2048


def estimate_num_kv_heads(model_name: str) -> int:
    """Estimate the number of KV attention heads from the model name.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        Estimated KV head count.
    """
    spec = _find_spec(model_name)
    if spec:
        return spec[3]

    params_b = estimate_model_size(model_name)
    if params_b >= 50:
        return 8
    if params_b >= 10:
        return 8
    if params_b >= 5:
        return 8
    return 32


def estimate_head_dim(model_name: str) -> int:
    """Estimate the head dimension from hidden dim and KV heads.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        Estimated head dimension.
    """
    hidden = estimate_hidden_dim(model_name)
    kv_heads = estimate_num_kv_heads(model_name)
    return hidden // kv_heads if kv_heads else 128


def estimate_vram_gb(
    params_b: float,
    dtype: str = "fp16",
    num_gpus: int = 1,
    overhead_factor: float = 1.2,
) -> float:
    """Estimate VRAM required in GB for a model.

    Accounts for dtype, quantization, and multi-GPU distribution.

    Args:
        params_b: Model size in billions of parameters.
        dtype: Data type ("fp16", "bf16", "fp32", "int8", "int4").
        num_gpus: Number of GPUs to distribute across.
        overhead_factor: Multiplier for additional memory (activations, KV cache).

    Returns:
        Estimated VRAM in GB per GPU.
    """
    bytes_per = _BYTES_PER_PARAM.get(dtype.lower(), 2.0)
    total_bytes = params_b * 1e9 * bytes_per * overhead_factor
    per_gpu = total_bytes / max(num_gpus, 1)
    return per_gpu / (1024 ** 3)


def estimate_vram_per_layer(
    params_b: float,
    dtype: str = "fp16",
    quantization: str = "",
) -> float:
    """Estimate VRAM per layer in GB.

    Args:
        params_b: Model size in billions.
        dtype: Base data type.
        quantization: Optional quantization ("4bit", "8bit").

    Returns:
        Estimated VRAM per layer in GB.
    """
    bytes_per = _BYTES_PER_PARAM.get(dtype.lower(), 2.0)
    if "4bit" in quantization:
        bytes_per = 0.5
    elif "8bit" in quantization:
        bytes_per = 1.0
    total_bytes = params_b * 1e9 * bytes_per
    layers = estimate_num_layers(f"{params_b}b")
    return (total_bytes / max(layers, 1)) / 1e9


def model_info(model_name: str) -> dict:
    """Return a complete info dict for a model name.

    Returns:
        Dict with keys: params_b, num_layers, hidden_dim, num_kv_heads,
        head_dim, vram_fp16_gb, vram_int8_gb, vram_int4_gb.
    """
    params_b = estimate_model_size(model_name)
    return {
        "params_b": params_b,
        "num_layers": estimate_num_layers(model_name),
        "hidden_dim": estimate_hidden_dim(model_name),
        "num_kv_heads": estimate_num_kv_heads(model_name),
        "head_dim": estimate_head_dim(model_name),
        "vram_fp16_gb": round(estimate_vram_gb(params_b, "fp16"), 1),
        "vram_int8_gb": round(estimate_vram_gb(params_b, "int8"), 1),
        "vram_int4_gb": round(estimate_vram_gb(params_b, "int4"), 1),
    }
