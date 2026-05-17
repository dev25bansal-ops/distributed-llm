"""NTK-aware, YaRN, and other context extension methods for RoPE.

Extends the context length of pretrained transformers beyond their
original training length using RoPE interpolation techniques:

- Linear scaling: uniform frequency scaling (original PI)
- NTK-aware: neural tangent kernel scaling (preserves high frequencies)
- YaRN: yet another RoPE extensioN method (best quality)
- Dynamic NTK: adjusts scale factor based on sequence length
- Log-NTK: logarithmic frequency spacing

All methods work by modifying the frequencies used in Rotary Position
Embedding (RoPE) without retraining the model.
"""

from __future__ import annotations

import math
from enum import Enum
import torch
from loguru import logger


class ScalingMethod(Enum):
    LINEAR = "linear"
    NTK_AWARE = "ntk_aware"
    YARN = "yarn"
    DYNAMIC_NTK = "dynamic_ntk"
    LOG_NTK = "log_ntk"
    NONE = "none"


def _compute_rope_frequencies(
    head_dim: int,
    base: float = 10000.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute standard RoPE frequencies (no scaling)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    return inv_freq.to(dtype=dtype)


def _apply_linear_scaling(
    inv_freq: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """Linear scaling: divide frequencies by scale factor."""
    return inv_freq / scale_factor


def _apply_ntk_scaling(
    inv_freq: torch.Tensor,
    head_dim: int,
    base: float,
    scale_factor: float,
) -> torch.Tensor:
    """NTK-aware scaling: modify base instead of frequencies.

    New base = base * scale_factor ^ (head_dim / (head_dim - 2))
    This preserves high-frequency components better than linear scaling.
    """
    new_base = base * (scale_factor ** (head_dim / (head_dim - 2)))
    inv_freq = 1.0 / (new_base ** (torch.arange(0, head_dim, 2, device=inv_freq.device).float() / head_dim))
    return inv_freq


def _compute_yarn_frequencies(
    head_dim: int,
    base: float,
    scale_factor: float,
    original_max_len: int,
    extended_max_len: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """YaRN frequency computation with length scaling.

    Uses a linear ramp to blend between NTK and PI scaling based on
    wavelength relative to the original context length.
    """
    inv_freq = _compute_rope_frequencies(head_dim, base, device, dtype)
    wavelengths = 2 * math.pi / inv_freq

    # Ramp function: blend NTK and linear based on wavelength
    ramp = torch.where(
        wavelengths < original_max_len,
        torch.zeros_like(wavelengths),
        torch.where(
            wavelengths > extended_max_len,
            torch.ones_like(wavelengths),
            (wavelengths - original_max_len) / (extended_max_len - original_max_len),
        ),
    )

    # Apply NTK scaling to long wavelengths, linear to short
    ntk_inv_freq = _apply_ntk_scaling(inv_freq.clone(), head_dim, base, scale_factor)
    linear_inv_freq = _apply_linear_scaling(inv_freq.clone(), scale_factor)

    blended = (1 - ramp) * ntk_inv_freq + ramp * linear_inv_freq
    return blended


class RoPEScaling:
    """RoPE frequency scaling for context length extension.

    Applies various scaling methods to extend the effective context
    length of pretrained transformers without retraining.

    Usage:
        scaling = RoPEScaling(
            method=ScalingMethod.YARN,
            scale_factor=2.0,
            original_max_len=4096,
        )
        cos, sin = scaling.get_rotations(seq_len=8192, head_dim=128)
    """

    def __init__(
        self,
        method: ScalingMethod = ScalingMethod.NTK_AWARE,
        scale_factor: float = 2.0,
        original_max_len: int = 4096,
        base: float = 10000.0,
        head_dim: int = 128,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self._method = method
        self._scale_factor = scale_factor
        self._original_max_len = original_max_len
        self._base = base
        self._head_dim = head_dim
        self._device = device
        self._dtype = dtype
        self._compute_frequencies()

    def _compute_frequencies(self) -> None:
        """Compute the scaled RoPE frequencies."""
        if self._method == ScalingMethod.NONE:
            self._inv_freq = _compute_rope_frequencies(self._head_dim, self._base, self._device, self._dtype)
        elif self._method == ScalingMethod.LINEAR:
            inv_freq = _compute_rope_frequencies(self._head_dim, self._base, self._device, self._dtype)
            self._inv_freq = _apply_linear_scaling(inv_freq, self._scale_factor)
        elif self._method == ScalingMethod.NTK_AWARE:
            self._inv_freq = _apply_ntk_scaling(
                _compute_rope_frequencies(self._head_dim, self._base, self._device, self._dtype),
                self._head_dim, self._base, self._scale_factor,
            )
        elif self._method == ScalingMethod.YARN:
            extended = int(self._original_max_len * self._scale_factor)
            self._inv_freq = _compute_yarn_frequencies(
                self._head_dim, self._base, self._scale_factor,
                self._original_max_len, extended, self._device, self._dtype,
            )
        elif self._method == ScalingMethod.DYNAMIC_NTK:
            self._inv_freq = _compute_rope_frequencies(self._head_dim, self._base, self._device, self._dtype)
        elif self._method == ScalingMethod.LOG_NTK:
            scale = 1.0 + math.log(self._scale_factor)
            inv_freq = _compute_rope_frequencies(self._head_dim, self._base, self._device, self._dtype)
            new_base = self._base * (scale ** (self._head_dim / (self._head_dim - 2)))
            self._inv_freq = 1.0 / (new_base ** (torch.arange(0, self._head_dim, 2, device=self._device).float() / self._head_dim))

    def update_scale_factor(self, scale_factor: float) -> None:
        self._scale_factor = scale_factor
        self._compute_frequencies()

    def get_rotations(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for the RoPE rotations.

        For dynamic NTK, the scale factor is adjusted per sequence length.

        Args:
            seq_len: Target sequence length.
            offset: Position offset (for incremental decoding).

        Returns:
            (cos, sin) tensors of shape (seq_len, head_dim).
        """
        if self._method == ScalingMethod.DYNAMIC_NTK:
            dynamic_scale = max(1.0, seq_len / self._original_max_len)
            inv_freq = _apply_ntk_scaling(
                _compute_rope_frequencies(self._head_dim, self._base, self._device, self._dtype),
                self._head_dim, self._base, dynamic_scale,
            )
        else:
            inv_freq = self._inv_freq

        positions = torch.arange(offset, offset + seq_len, device=self._device, dtype=self._dtype)
        angles = torch.outer(positions, inv_freq)
        cos = angles.cos()
        sin = angles.sin()
        return cos, sin

    def apply_rope(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply rotary position embeddings to a tensor.

        Args:
            x: Input tensor (..., seq_len, head_dim).
            positions: Optional position indices (seq_len,).

        Returns:
            Tensor with RoPE applied.
        """
        seq_len = x.shape[-2]
        cos, sin = self.get_rotations(seq_len)

        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    @property
    def effective_max_len(self) -> int:
        return int(self._original_max_len * self._scale_factor)

    def summary(self) -> str:
        return (
            f"RoPEScaling: method={self._method.value}, "
            f"scale={self._scale_factor}x, "
            f"{self._original_max_len} -> {self.effective_max_len} tokens, "
            f"base={self._base}, head_dim={self._head_dim}"
        )
