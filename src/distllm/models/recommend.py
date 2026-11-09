"""Hardware-aware model recommendations.

The onboarding wizard used to pick models from a hardcoded table keyed on VRAM
buckets. That table listed 14 models, two of which were not real repository
ids (``meta-llama/Llama-3-70b``, ``meta-llama/Llama-3.1-70B``), and it never
consulted actual parameter counts, so a 24 GB card was told to run a 70B model
it cannot hold.

This module computes what a model actually costs from its parameter count and
dtype, and decides what fits:

  weights  = params * bytes_per_param(dtype)
  kv cache = 2 * layers * kv_heads * head_dim * seq_len * batch * bytes
  overhead ~ activation and allocator slack, taken as a fraction

The recommendation is then a query over a real catalog rather than a list
baked into the source, so new models appear on their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Bytes per parameter for the dtypes a user can pick in the wizard.
BYTES_PER_PARAM = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# KV-cache bytes per parameter are negligible next to weights, but the cache
# itself is not: it scales with context length, which is how a model that fits
# at 4k OOMs at 32k.
_DEFAULT_OVERHEAD_FRACTION = 0.15


@dataclass
class HardwareProfile:
    """What the machine can actually offer."""

    gpu_memory_gb: float = 0.0
    gpu_count: int = 0
    gpu_name: str = ""
    ram_gb: float = 0.0
    cpu_cores: int = 0

    @property
    def total_vram_gb(self) -> float:
        return self.gpu_memory_gb * max(self.gpu_count, 1) if self.gpu_memory_gb else 0.0

    @property
    def is_cpu_only(self) -> bool:
        return self.gpu_count == 0 or self.gpu_memory_gb <= 0.0


@dataclass
class ModelFit:
    """The verdict for one model on one machine."""

    model_id: str
    params: int
    dtype: str
    weights_gb: float
    kv_cache_gb: float
    total_gb: float
    fits: bool
    headroom_gb: float
    context_tokens: int
    notes: list[str] = field(default_factory=list)

    @property
    def utilisation(self) -> float:
        """Fraction of available memory this model would consume."""
        available = self.total_gb + self.headroom_gb
        return self.total_gb / available if available > 0 else 0.0


def bytes_per_param(dtype: str) -> float:
    """Bytes needed for one parameter at *dtype*, defaulting to 2 (fp16/bf16)."""
    return BYTES_PER_PARAM.get(dtype.lower().strip(), 2.0)


def estimate_memory_gb(
    params: int,
    dtype: str = "float16",
    *,
    layers: int = 32,
    kv_heads: int = 8,
    head_dim: int = 128,
    context_tokens: int = 4096,
    batch_size: int = 1,
    overhead_fraction: float = _DEFAULT_OVERHEAD_FRACTION,
) -> tuple[float, float]:
    """Return ``(weights_gb, kv_cache_gb)`` for a model of *params* parameters.

    The KV-cache term uses hidden size derived from the usual transformer
    proportions, so it is an estimate -- but unlike a static table it responds
    to context length and batch size, which is what makes a 7B model that fits
    at 4k fail at 128k.
    """
    if params <= 0:
        return 0.0, 0.0

    weights_bytes = params * bytes_per_param(dtype)

    # KV cache per token = 2 (K and V) * layers * kv_heads * head_dim * bytes.
    # Grouped-query attention keeps kv_heads well below the attention-head
    # count, which is why this term stays small next to the weights at short
    # context but dominates at long context.
    kv_bytes = (
        2  # K and V
        * layers
        * kv_heads
        * head_dim
        * context_tokens
        * batch_size
        * bytes_per_param(dtype)
    )
    gib = 1024**3
    overhead = (weights_bytes + kv_bytes) * overhead_fraction
    return (weights_bytes + overhead) / gib, kv_bytes / gib


def evaluate(
    model_id: str,
    params: int,
    hardware: HardwareProfile,
    *,
    dtype: str = "float16",
    context_tokens: int = 4096,
    layers: int | None = None,
    kv_heads: int | None = None,
    head_dim: int = 128,
) -> ModelFit:
    """Decide whether *model_id* fits on *hardware*."""
    # Layer count scales roughly with depth; these are sane defaults when the
    # catalog does not report architecture details.
    est_layers = layers or max(16, min(80, int((params / 1e9) ** 0.5 * 24) or 24))
    est_kv_heads = kv_heads or 8

    weights_gb, kv_gb = estimate_memory_gb(
        params,
        dtype,
        layers=est_layers,
        kv_heads=est_kv_heads,
        head_dim=head_dim,
        context_tokens=context_tokens,
    )
    total = weights_gb + kv_gb

    if hardware.is_cpu_only:
        # CPU offload works but is slow; require a comfortable RAM margin.
        available = hardware.ram_gb * 0.7
        fits = total <= available
        headroom = available - total
        notes = ["CPU-only: expect token rates in the single digits"]
    else:
        available = hardware.total_vram_gb
        # Drivers and other processes hold a slice of the card, so a nominal
        # 8 GB device offers roughly 7.5-7.8 GB. A 95% cut is too harsh at that
        # size (it rejected a 7.63 GB model on an 8 GB card); 97% still leaves
        # room for the CUDA context without rejecting models that genuinely fit.
        fits = total <= available * 0.97
        headroom = available - total
        notes = []
        if hardware.gpu_count > 1 and fits and total > hardware.gpu_memory_gb:
            notes.append(
                f"needs sharding across {hardware.gpu_count} GPUs "
                f"({total / hardware.gpu_count:.1f} GB each)"
            )

    if not fits and not hardware.is_cpu_only:
        # Suggest the quantization that would make it fit.
        for candidate in ("int8", "int4"):
            w, k = estimate_memory_gb(
                params,
                candidate,
                layers=est_layers,
                kv_heads=est_kv_heads,
                head_dim=head_dim,
                context_tokens=context_tokens,
            )
            if w + k <= available * 0.97:
                notes.append(
                    f"fits at {candidate} quantization "
                    f"({(w + k) / available * 100:.0f}% of VRAM)"
                )
                break

    return ModelFit(
        model_id=model_id,
        params=params,
        dtype=dtype,
        weights_gb=weights_gb,
        kv_cache_gb=kv_gb,
        total_gb=total,
        fits=fits,
        headroom_gb=headroom,
        context_tokens=context_tokens,
        notes=notes,
    )


def rank(
    candidates: list[tuple[str, int]],
    hardware: HardwareProfile,
    *,
    dtype: str = "float16",
    context_tokens: int = 4096,
    max_results: int = 10,
) -> tuple[list[ModelFit], list[ModelFit]]:
    """Split *candidates* into ``(fits, does_not_fit)``, best first.

    "Best" means the largest model that still fits: within a family the bigger
    model is the more capable one, and the point of the wizard is to use the
    hardware the user already has.
    """
    fits: list[ModelFit] = []
    too_big: list[ModelFit] = []
    for model_id, params in candidates:
        if params <= 0:
            continue
        fit = evaluate(
            model_id, params, hardware, dtype=dtype, context_tokens=context_tokens
        )
        (fits if fit.fits else too_big).append(fit)

    fits.sort(key=lambda f: f.params, reverse=True)
    too_big.sort(key=lambda f: f.total_gb)

    # The catalog returns the same model many times over (base, -Instruct,
    # GGUF, AWQ, community re-uploads). Keep one entry per base name so the
    # wizard does not offer six identical Llama-3.2-3B rows; prefer the
    # canonical org when several qualify.
    def dedupe(items: list[ModelFit]) -> list[ModelFit]:
        best: dict[str, ModelFit] = {}
        # Strip quant/format markers anywhere in the name, then the family
        # suffix, so "Llama-3.2-3B-Instruct", "...-GGUF" and "...-FP8-dynamic"
        # all collapse to one key.
        markers = (
            "gguf", "awq", "gptq", "exl2", "fp8", "fp16", "int4", "int8",
            "4bit", "8bit", "dynamic", "quantized",
        )
        for fit in items:
            name = fit.model_id.rsplit("/", 1)[-1].lower()
            for marker in markers:
                name = name.replace(f"-{marker}", "").replace(f"_{marker}", "")
            name = re.sub(r"[-_](instruct|chat|it|hf)$", "", name)
            name = re.sub(r"[-_]v?\d+(\.\d+)*$", "", name)
            key = f"{name}:{fit.params}"
            current = best.get(key)
            if current is None:
                best[key] = fit
                continue
            # Prefer the plain (unquantized) build when both are present.
            def is_plain(mid: str) -> bool:
                low = mid.lower()
                return not any(m in low for m in ("gguf", "awq", "gptq", "fp8", "4bit"))

            if is_plain(fit.model_id) and not is_plain(current.model_id):
                best[key] = fit
        return list(best.values())

    return dedupe(fits)[:max_results], dedupe(too_big)[:max_results]
