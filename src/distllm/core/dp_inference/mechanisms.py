"""Core DP operations: gradient clipping, noise injection, Gumbel mechanism.

Extracted from :mod:`distllm.core.dp_inference`.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Core DP operations
# ---------------------------------------------------------------------------


_DEFAULT_MAX_GRAD_NORM = 1.0


def clip_gradients(
    grads: list[torch.Tensor],
    max_norm: float = _DEFAULT_MAX_GRAD_NORM,
    *,
    clip_per_layer: bool = True,
) -> list[torch.Tensor]:
    """Clip gradients to bounded L2 norm.

    In DP-SGD, gradient clipping ensures bounded sensitivity, which
    determines how much noise is needed for privacy.

    Args:
        grads: List of gradient tensors.
        max_norm: Maximum allowed L2 norm per layer (or globally).
        clip_per_layer: If True, clip each tensor individually.
            If False, compute the global norm across all tensors and
            rescale if exceeded.

    Returns:
        Clipped gradient tensors.
    """
    if clip_per_layer:
        clipped = []
        for g in grads:
            norm = g.norm().item()
            if norm > max_norm:
                clipped.append(g * (max_norm / norm))
            else:
                clipped.append(g.clone())
        return clipped
    else:
        # Global norm clipping
        global_norm = math.sqrt(sum(g.norm().item() ** 2 for g in grads))
        if global_norm <= max_norm:
            return [g.clone() for g in grads]
        scale = max_norm / global_norm
        return [g.clone() * scale for g in grads]


def dp_noise_injection(
    grads: list[torch.Tensor],
    sigma: float,
    *,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Add calibrated Gaussian noise to gradients for differential privacy.

    Args:
        grads: List of gradient tensors.
        sigma: Noise scale (standard deviation).
        seed: Optional seed for reproducibility in testing.

    Returns:
        Gradients with Gaussian noise added.  The original tensors are
        not modified.
    """
    noisy_grads: list[torch.Tensor] = []
    for g in grads:
        if sigma > 0:
            generator: torch.Generator | None = None
            if seed is not None:
                generator = torch.Generator(device=g.device)
                generator.manual_seed(seed)
            noise = torch.randn_like(g, generator=generator) * sigma
            noisy_grads.append(g + noise)
        else:
            noisy_grads.append(g.clone())
    return noisy_grads


def gumbel_noise_mechanism(
    logits: torch.Tensor,
    noise_scale: float = 1.0,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Perturb output logits with Gumbel noise for DP sampling.

    This implements the Gumbel noise mechanism: add Gumbel(0, scale)
    noise to each logit, then sample from the resulting distribution.
    This is equivalent to the exponential mechanism for categorical
    outcomes.

    Args:
        logits: Raw logits tensor of shape ``(batch, vocab)`` or ``(vocab,)``.
        noise_scale: Scale of the Gumbel distribution.
        seed: Optional seed for reproducibility.

    Returns:
        Logits with added Gumbel noise.
    """
    if noise_scale <= 0:
        return logits.clone()

    batched = logits.dim() == 2
    if not batched:
        logits = logits.unsqueeze(0)

    noisy_logits = logits.clone()
    for i in range(logits.shape[0]):
        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(seed + i)
        # Gumbel(0, scale): u = Uniform(0,1) -> noise = -scale * log(-log(u))
        u = torch.rand(logits.shape[1], generator=generator).clamp(min=1e-10, max=1 - 1e-10)
        gumbel = -noise_scale * torch.log(-torch.log(u))
        noisy_logits[i] = noisy_logits[i] + gumbel

    return noisy_logits.squeeze(0) if not batched else noisy_logits


__all__ = [
    "clip_gradients",
    "dp_noise_injection",
    "gumbel_noise_mechanism",
]
