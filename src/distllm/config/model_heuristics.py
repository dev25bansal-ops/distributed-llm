"""Shared model estimation heuristics — single source of truth for model size,
layer count, hidden dimension, VRAM estimates, and related parameters.

All CLI modules should import from here instead of defining their own
``_estimate_*`` functions. This eliminates 4-way duplication across
``deploy.py``, ``tune.py``, ``cost_avoid.py``, and ``doctor.py``.
"""

from __future__ import annotations

import re

# ── Model profile database ──────────────────────────────────────────────────
# Maps (approximate param count) -> (layers, hidden_dim, kv_heads, head_dim)
# Values are rough approximations for common model architectures.
_MODEL_PROFILES: dict[str, dict[str, int]] = {
    "70b": {"layers": 80, "hidden": 8192, "kv_heads": 8, "head_dim": 128},
    "65b": {"layers": 80, "hidden": 8192, "kv_heads": 8, "head_dim": 128},
    "40b": {"layers": 60, "hidden": 6656, "kv_heads": 8, "head_dim": 128},
    "34b": {"layers": 48, "hidden": 6656, "kv_heads": 8, "head_dim": 128},
    "33b": {"layers": 48, "hidden": 6656, "kv_heads": 8, "head_dim": 128},
    "14b": {"layers": 40, "hidden": 5120, "kv_heads": 40, "head_dim": 128},
    "13b": {"layers": 40, "hidden": 5120, "kv_heads": 40, "head_dim": 128},
    "8b":  {"layers": 32, "hidden": 4096, "kv_heads": 8, "head_dim": 128},
    "7b":  {"layers": 32, "hidden": 4096, "kv_heads": 32, "head_dim": 128},
    "3b":  {"layers": 28, "hidden": 3072, "kv_heads": 8, "head_dim": 128},
    "1b":  {"layers": 24, "hidden": 2048, "kv_heads": 8, "head_dim": 128},
    "0.5b": {"layers": 16, "hidden": 1024, "kv_heads": 4, "head_dim": 128},
    "350m": {"layers": 16, "hidden": 1024, "kv_heads": 4, "head_dim": 128},
}

# Default profile for unknown models (conservative 7B-class estimate)
_DEFAULT_PROFILE: dict[str, int] = {"layers": 32, "hidden": 4096, "kv_heads": 32, "head_dim": 128}


def _extract_size_key(name: str) -> str | None:
    """Extract model size key from a model name like '70b', '8b', '350m'.

    Uses regex to find 'NNb', 'NN.Bb', or 'NNNm' patterns.
    Returns the matched key lowercased, or ``None`` if nothing matches.
    """
    name_lower = name.lower().replace("-", "")

    # Try exact matches first (most common)
    for size in sorted(_MODEL_PROFILES.keys(), key=len, reverse=True):
        if size in name_lower:
            return size

    # Try numeric regex extraction
    m = re.search(r"(\d+)(?:\.(\d+))?[bB]", name_lower)
    if m:
        whole = m.group(1)
        frac = m.group(2) or ""
        return f"{whole}b" if not frac else f"{whole}.{frac}b"

    m = re.search(r"(\d+)[mM]", name_lower)
    if m:
        return f"{m.group(1)}m"

    return None


def estimate_params(model_name: str) -> float:
    """Estimate parameter count in billions from model name.

    Returns a float (e.g. 70.0, 13.0, 0.5). Defaults to 7.0 if unknown.
    """
    key = _extract_size_key(model_name)
    if key is None:
        return 7.0

    # Direct lookups for standard sizes
    size_map: dict[str, float] = {
        "70b": 70.0, "65b": 65.0, "40b": 40.0, "34b": 34.0,
        "33b": 33.0, "14b": 14.0, "13b": 13.0, "8b": 8.0,
        "7b": 7.0, "3b": 3.0, "1b": 1.0, "0.5b": 0.5, "350m": 0.35,
    }
    if key in size_map:
        return size_map[key]

    # Try parsing 'NNb' pattern
    m = re.match(r"(\d+(?:\.\d+)?)b", key)
    if m:
        return float(m.group(1))
    # Try parsing 'NNNm' pattern
    m = re.match(r"(\d+)m", key)
    if m:
        return int(m.group(1)) / 1000

    return 7.0


def estimate_layers(model_name: str) -> int:
    """Estimate number of transformer layers from model name."""
    key = _extract_size_key(model_name)
    if key and key in _MODEL_PROFILES:
        return _MODEL_PROFILES[key]["layers"]
    return _DEFAULT_PROFILE["layers"]


def estimate_hidden_dim(model_name: str) -> int:
    """Estimate hidden dimension from model name."""
    key = _extract_size_key(model_name)
    if key and key in _MODEL_PROFILES:
        return _MODEL_PROFILES[key]["hidden"]
    return _DEFAULT_PROFILE["hidden"]


def estimate_kv_heads(model_name: str) -> int:
    """Estimate number of KV heads (for GQA) from model name."""
    key = _extract_size_key(model_name)
    if key and key in _MODEL_PROFILES:
        return _MODEL_PROFILES[key]["kv_heads"]
    return _DEFAULT_PROFILE["kv_heads"]


def estimate_head_dim(model_name: str) -> int:
    """Estimate head dimension from model name."""
    key = _extract_size_key(model_name)
    if key and key in _MODEL_PROFILES:
        return _MODEL_PROFILES[key]["head_dim"]
    return _DEFAULT_PROFILE["head_dim"]


def estimate_model_size_bytes(model_name: str) -> int:
    """Estimate fp16 model size in bytes from model name."""
    params_b = estimate_params(model_name)
    return int(params_b * 1e9 * 2)  # fp16 = 2 bytes per param


def estimate_vram(
    model_name_or_params: str | float,
    dtype: str = "float16",
    quantization: str = "none",
) -> float:
    """Estimate VRAM in GB for a model.

    Args:
        model_name_or_params: Model name (e.g. ``"meta-llama/Llama-2-70b"``)
            or parameter count in billions (e.g. ``70.0``).
        dtype: Data type (``"float16"``, ``"float32"``, ``"bfloat16"``).
        quantization: Quantization (``"none"``, ``"int8"``, ``"int4"``, etc.).

    Returns:
        Estimated VRAM in GB.
    """
    if isinstance(model_name_or_params, (int, float)):
        params_b = float(model_name_or_params)
    else:
        params_b = estimate_params(model_name_or_params)

    bytes_per_param = 2 if dtype in ("float16", "bfloat16") else 4
    if "4bit" in quantization or quantization in ("int4", "nf4"):
        bytes_per_param = 0.5
    elif "8bit" in quantization or quantization == "int8" or quantization == "fp8":
        bytes_per_param = 1

    return params_b * 1e9 * bytes_per_param / 1e9  # already in GB


def estimate_vram_per_layer(
    model_name: str,
    dtype: str = "float16",
    quantization: str = "none",
) -> float:
    """Estimate VRAM per transformer layer in GB."""
    total_gb = estimate_vram(model_name, dtype, quantization)
    n_layers = estimate_layers(model_name)
    return total_gb / max(n_layers, 1)
