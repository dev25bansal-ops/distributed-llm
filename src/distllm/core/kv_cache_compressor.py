"""Block-level KV cache compression engine.

Compresses individual KV cache blocks in-place without touching
neighboring blocks.  Supports FP8, INT8, INT4, and NF4 quantization
with per-head or per-tensor scaling.

Usage::

    from distllm.core.kv_cache_compressor import BlockCompressor

    compressor = BlockCompressor(method="fp8")
    compressed_k, scale_k = compressor.compress(key_tensor)
    restored_k = compressor.decompress(compressed_k, scale_k)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from loguru import logger

CompressMethod = Literal["fp8", "int8", "int4", "nf4"]


@dataclass
class CompressedBlock:
    """A compressed KV cache block."""
    block_id: int
    layer_idx: int
    key_compressed: torch.Tensor
    value_compressed: torch.Tensor
    key_scale: torch.Tensor
    value_scale: torch.Tensor
    original_dtype: torch.dtype
    method: CompressMethod


class BlockCompressor:
    """Compresses and decompresses individual KV cache blocks.

    Args:
        method: Quantization method — "fp8", "int8", "int4", or "nf4".
        scale_granularity: "per_head" (default) or "per_tensor".
    """

    def __init__(
        self,
        method: CompressMethod = "fp8",
        scale_granularity: str = "per_head",
    ):
        if method not in ("fp8", "int8", "int4", "nf4"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.scale_granularity = scale_granularity
        self._stats = {
            "compress_calls": 0,
            "decompress_calls": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
        }

    @property
    def compression_ratio(self) -> float:
        orig = self._stats["total_original_bytes"]
        comp = self._stats["total_compressed_bytes"]
        return comp / max(orig, 1)

    def _compute_scale(
        self, tensor: torch.Tensor, max_val: float,
    ) -> torch.Tensor:
        """Compute per-head or per-tensor scale factor."""
        if self.scale_granularity == "per_head":
            # Shape: (num_heads, seq_len, head_dim) → scale per head
            amax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        else:
            amax = tensor.abs().amax().clamp(min=1e-12).reshape(1, 1, 1)
        return amax / max_val

    def compress(
        self, tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a tensor. Returns (quantized, scale).

        Args:
            tensor: FP16/FP32 tensor of shape (num_heads, seq_len, head_dim).

        Returns:
            (quantized_tensor, scale_tensor).
        """
        self._stats["compress_calls"] += 1
        self._stats["total_original_bytes"] += tensor.element_size() * tensor.numel()

        if self.method == "fp8":
            if not hasattr(torch, "float8_e4m3fn"):
                # Fallback to int8 if FP8 not available
                return self._compress_int8(tensor)
            return self._compress_fp8(tensor)
        elif self.method == "int8":
            return self._compress_int8(tensor)
        elif self.method == "int4":
            return self._compress_int4(tensor)
        elif self.method == "nf4":
            return self._compress_nf4(tensor)
        return tensor, torch.empty(0)

    def decompress(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Decompress a quantized tensor back to the original dtype.

        Args:
            quantized: Compressed tensor.
            scale: Scale factor from compress().
            target_dtype: Output dtype.

        Returns:
            Decompressed tensor.
        """
        self._stats["decompress_calls"] += 1

        if scale.numel() == 0:
            return quantized.to(target_dtype)

        if self.method == "fp8" and quantized.dtype == torch.float8_e4m3fn:
            return (quantized.float() * scale).to(target_dtype)
        elif self.method in ("int8", "int4"):
            return (quantized.float() * scale).to(target_dtype)
        elif self.method == "nf4":
            return self._decompress_nf4(quantized, scale, target_dtype)
        return quantized.to(target_dtype)

    def compress_block(
        self,
        block_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> CompressedBlock:
        """Compress both K and V tensors of a block."""
        k_comp, k_scale = self.compress(key)
        v_comp, v_scale = self.compress(value)
        self._stats["total_compressed_bytes"] += (
            k_comp.element_size() * k_comp.numel() +
            v_comp.element_size() * v_comp.numel() +
            k_scale.element_size() * k_scale.numel() * 2
        )
        return CompressedBlock(
            block_id=block_id,
            layer_idx=layer_idx,
            key_compressed=k_comp,
            value_compressed=v_comp,
            key_scale=k_scale,
            value_scale=v_scale,
            original_dtype=key.dtype,
            method=self.method,
        )

    def decompress_block(
        self, block: CompressedBlock,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress both K and V tensors of a compressed block."""
        k = self.decompress(block.key_compressed, block.key_scale, block.original_dtype)
        v = self.decompress(block.value_compressed, block.value_scale, block.original_dtype)
        return k, v

    # ── Internal compression methods ──────────────────────────────────

    def _compress_fp8(
        self, tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._compute_scale(tensor, 448.0)
        quantized = (tensor / scale).to(torch.float8_e4m3fn)
        return quantized, scale

    def _compress_int8(
        self, tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._compute_scale(tensor, 127.0)
        quantized = (tensor / scale).clamp(-128, 127).to(torch.int8)
        return quantized, scale

    def _compress_int4(
        self, tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._compute_scale(tensor, 7.0)
        quantized = (tensor / scale).clamp(-7, 7).to(torch.int8)
        return quantized, scale

    def _compress_nf4(
        self, tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """NormalFloat4 quantization (QLoRA-style).

        Maps values to the 16 NF4 quantiles of a standard normal distribution.
        """
        # NF4 quantile levels (from QLoRA paper)
        nf4_levels = torch.tensor([
            -1.0, -0.6962, -0.5251, -0.3949,
            -0.2844, -0.1848, -0.0911, 0.0,
            0.0796, 0.1609, 0.2461, 0.3379,
            0.4407, 0.5626, 0.7230, 1.0,
        ], device=tensor.device, dtype=torch.float32)

        # Normalize tensor to [-1, 1]
        absmax = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        normalized = tensor.float() / absmax

        # Find nearest NF4 level
        distances = (normalized.unsqueeze(-1) - nf4_levels.unsqueeze(0).unsqueeze(0)).abs()
        indices = distances.argmin(dim=-1)

        # Pack two 4-bit values per byte
        flat_indices = indices.reshape(-1)
        if flat_indices.numel() % 2 == 1:
            flat_indices = torch.cat([flat_indices, torch.zeros(1, dtype=torch.long, device=flat_indices.device)])
        packed = (flat_indices[::2] | (flat_indices[1::2] << 4)).to(torch.uint8)

        # Reshape to (batch, num_heads, seq_len, ceil(head_dim/2)).
        # Using ``*tensor.shape[:-1]`` matches all dims except the last
        # (which is packed to ceil(head_dim/2)).  For odd head_dim values
        # the packed count may undershoot the naive ``*tensor.shape[:-1], -1``
        # reshape by up to ``(H * S) // 2`` elements, so we compute the
        # exact trailing dimension.
        packed_last_dim = packed.numel() // (tensor.shape[0] * tensor.shape[1] * tensor.shape[2])
        return packed.reshape(*tensor.shape[:-1], packed_last_dim), absmax

    @staticmethod
    def _decompress_nf4(
        packed: torch.Tensor,
        scale: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decompress NF4-packed tensor.

        The original tensor had shape ``(batch, num_heads, seq_len, head_dim)``.
        The packed tensor has shape ``(batch, num_heads, seq_len, ceil(head_dim / 2))``
        with two 4-bit values packed per byte.

        The scale tensor from ``amax(dim=-1, keepdim=True)`` has shape
        ``(batch, num_heads, seq_len, 1)``.
        """
        nf4_levels = torch.tensor([
            -1.0, -0.6962, -0.5251, -0.3949,
            -0.2844, -0.1848, -0.0911, 0.0,
            0.0796, 0.1609, 0.2461, 0.3379,
            0.4407, 0.5626, 0.7230, 1.0,
        ], device=packed.device, dtype=torch.float32)

        # Unpack: each byte → two 4-bit indices
        flat = packed.reshape(-1)
        lo = (flat & 0x0F).long()
        hi = ((flat >> 4) & 0x0F).long()
        indices = torch.stack([lo, hi], dim=-1).reshape(-1)

        # During compression, at most 1 zero-padding element may have been
        # appended (when total element count was odd). Trim it:
        indices = indices[:indices.numel() - (indices.numel() % 2)]

        # The original number of elements per group (batch * num_heads * seq_len)
        # can be derived from the unpacked index count. This avoids the old
        # bug: truncating to scale.shape[-2] * scale.shape[-1] = seq_len * 1,
        # which drops all but the first ``seq_len`` indices.
        total_groups = scale.numel()
        head_dim = indices.numel() // total_groups
        original_shape = (*packed.shape[:-1], head_dim)
        values = nf4_levels[indices].reshape(original_shape).to(target_dtype)
        return (values * scale.to(target_dtype))

    def stats(self) -> dict:
        return {
            **self._stats,
            "compression_ratio": round(self.compression_ratio, 4),
            "method": self.method,
            "scale_granularity": self.scale_granularity,
        }

    def __repr__(self) -> str:
        return (
            f"BlockCompressor(method={self.method}, "
            f"ratio={self.compression_ratio:.2f}, "
            f"calls={self._stats['compress_calls']})"
        )
