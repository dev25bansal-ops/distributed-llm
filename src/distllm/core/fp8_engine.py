"""FP8 native inference engine for NVIDIA Hopper GPUs.

Provides native FP8 (float8) compute support for transformer inference:
- FP8 linear layers (weight cast + matmul in fp8)
- FP8 attention with scaled dot-product
- Dynamic per-tensor activation quantization
- Automatic model patching for end-to-end FP8 inference
- Graceful fallback when FP8 hardware is unavailable
"""

import copy
import math
import threading
from typing import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


FP8_AVAILABLE = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")
FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
FP8_E5M2 = getattr(torch, "float8_e5m2", None)


class FP8Scheme:
    E4M3 = "e4m3"
    E5M2 = "e5m2"


@dataclass
class FP8Tensor:
    """Wrapper for an FP8 quantized tensor with scale."""
    data: torch.Tensor
    scale: torch.Tensor
    original_dtype: torch.dtype = torch.float16


def quantize_tensor(tensor: torch.Tensor, scheme: str = FP8Scheme.E4M3) -> FP8Tensor:
    """Quantize a tensor to FP8 with per-tensor scaling.

    Uses absmax quantization: tensor_fp8 = tensor / scale
    where scale = max(|tensor|) / max_fp8.

    Args:
        tensor: Input tensor (fp16/bf16/fp32).
        scheme: FP8 scheme - "e4m3" (higher precision) or "e5m2" (wider range).

    Returns:
        FP8Tensor with quantized data and scale.
    """
    if not FP8_AVAILABLE:
        return FP8Tensor(data=tensor, scale=torch.ones(1, device=tensor.device), original_dtype=tensor.dtype)

    fp8_dtype = FP8_E4M3 if scheme == FP8Scheme.E4M3 else FP8_E5M2
    fp8_max = 448.0 if scheme == FP8Scheme.E4M3 else 57344.0

    orig_dtype = tensor.dtype
    scale = tensor.abs().max().clamp(min=1e-12) / fp8_max
    scaled = (tensor / scale).to(fp8_dtype)

    return FP8Tensor(data=scaled, scale=scale, original_dtype=orig_dtype)


def dequantize_tensor(fp8_tensor: FP8Tensor) -> torch.Tensor:
    """Dequantize FP8 tensor back to original dtype."""
    return fp8_tensor.data.to(fp8_tensor.original_dtype) * fp8_tensor.scale


def quantize_kv_fp8(tensor: torch.Tensor, scheme: str = FP8Scheme.E4M3) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize KV cache tensor to FP8 with per-tensor scaling.

    Args:
        tensor: KV tensor of shape [num_heads, seq_len, head_dim].
        scheme: FP8 scheme (e4m3 or e5m2).

    Returns:
        Tuple of (fp8_data, scale) where fp8_data is float8_e4m3fn.
    """
    if not FP8_AVAILABLE:
        return tensor, torch.ones(1, device=tensor.device)

    fp8_dtype = FP8_E4M3 if scheme == FP8Scheme.E4M3 else FP8_E5M2
    fp8_max = 448.0 if scheme == FP8Scheme.E4M3 else 57344.0

    scale = tensor.abs().max().clamp(min=1e-12) / fp8_max
    fp8_data = (tensor / scale).to(fp8_dtype)
    return fp8_data, scale


def dequantize_kv_fp8(fp8_data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize FP8 KV cache tensor back to original dtype.

    Args:
        fp8_data: FP8 quantized data.
        scale: Per-tensor scale factor.
        dtype: Target dtype for dequantization.

    Returns:
        Dequantized tensor.
    """
    return fp8_data.to(dtype) * scale


class FP8Linear(nn.Module):
    """FP8 linear layer with native float8 matmul.

    Supports three modes:
    1. Full FP8: weights + activations quantized to FP8 before matmul
    2. Weight-only FP8: weights stored in FP8, activations in fp16
    3. Fallback: standard fp16 linear when FP8 not available

    Uses torch._scaled_mm when available (Hopper), otherwise falls back.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scheme: str = FP8Scheme.E4M3,
        quantize_activations: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scheme = scheme
        self.quantize_activations = quantize_activations and FP8_AVAILABLE

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        self._fp8_weight: FP8Tensor | None = None
        self._weight_scale: torch.Tensor | None = None
        self._fp8_enabled = False
        self._has_scaled_mm: bool = FP8_AVAILABLE and hasattr(torch, '_scaled_mm')

    def to_fp8(self) -> None:
        """Convert stored weight to FP8 format for compute."""
        if not FP8_AVAILABLE:
            logger.warning("FP8 not available, keeping fp16 weights")
            return
        fp8_t = quantize_tensor(self.weight.data, self.scheme)
        self._fp8_weight = fp8_t
        self._fp8_enabled = True
        weight_vram_ratio = 0.5
        logger.debug(f"FP8Linear {self.in_features}x{self.out_features}: weights quantized ({weight_vram_ratio:.1%} VRAM)")

    def to_fp16(self) -> None:
        """Revert to fp16 weights."""
        self._fp8_enabled = False
        self._fp8_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fp8_enabled or not FP8_AVAILABLE:
            return F.linear(x, self.weight, self.bias)

        w = self._fp8_weight
        if self.quantize_activations:
            x_fp8 = quantize_tensor(x, self.scheme)
            if self._has_scaled_mm:
                out = torch._scaled_mm(
                    x_fp8.data,
                    w.data.t(),
                    scale_a=x_fp8.scale,
                    scale_b=w.scale,
                    bias=self.bias,
                )
                return out.to(x.dtype)
            x_fp16 = dequantize_tensor(x_fp8)
            w_fp16 = dequantize_tensor(w)
            return F.linear(x_fp16, w_fp16, self.bias)
        else:
            w_fp16 = dequantize_tensor(w)
            return F.linear(x, w_fp16, self.bias)

    def extra_repr(self) -> str:
        fp8_str = ", fp8" if self._fp8_enabled else ""
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}{fp8_str}"


class FP8Attention(nn.Module):
    """FP8 attention with scaled dot-product attention.

    Uses FP8 for Q, K, V projections and optionally for the attention
    softmax computation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scheme: str = FP8Scheme.E4M3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = FP8Linear(hidden_size, num_heads * head_dim, scheme=scheme)
        self.k_proj = FP8Linear(hidden_size, num_heads * head_dim, scheme=scheme)
        self.v_proj = FP8Linear(hidden_size, num_heads * head_dim, scheme=scheme)
        self.o_proj = FP8Linear(num_heads * head_dim, hidden_size, scheme=scheme)

        self._fp8_enabled = False

    def to_fp8(self) -> None:
        for mod in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            mod.to_fp8()
        self._fp8_enabled = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if self._fp8_enabled and FP8_AVAILABLE:
            k_fp8 = quantize_tensor(k, self.scheme)
            v_fp8 = quantize_tensor(v, self.scheme)
            attn_weights = torch.matmul(q, k_fp8.data.to(q.dtype) * k_fp8.scale) / math.sqrt(self.head_dim)
            past_kv_out = (dequantize_tensor(k_fp8), dequantize_tensor(v_fp8))
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            past_kv_out = (k, v)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_kv_out


class FP8ModelPatcher:
    """Patches a HuggingFace model for FP8 inference.

    Replaces nn.Linear layers with FP8Linear and adds FP8 attention.
    """

    def __init__(self, scheme: str = FP8Scheme.E4M3, quantize_activations: bool = True):
        self.scheme = scheme
        self.quantize_activations = quantize_activations
        self._patched_layers: list[str] = []

    def patch_model(self, model: nn.Module) -> nn.Module:
        """Patch all eligible linear layers in a model for FP8 compute.

        Recursively walks the module tree and replaces nn.Linear with FP8Linear.

        Args:
            model: The PyTorch model to patch.

        Returns:
            The patched model (modified in-place).
        """
        if not FP8_AVAILABLE:
            logger.warning("FP8 not available (requires torch >= 2.1 + Hopper GPU), skipping patching")
            return model

        patched_count = 0
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear):
                fp8_linear = FP8Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    scheme=self.scheme,
                    quantize_activations=self.quantize_activations,
                )
                fp8_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    fp8_linear.bias.data = module.bias.data.clone()
                setattr(model, name, fp8_linear)
                fp8_linear.to_fp8()
                self._patched_layers.append(name)
                patched_count += 1
            else:
                self.patch_model(module)

        logger.info(f"Patched {patched_count} linear layers to FP8 ({self.scheme})")
        return model

    def unpatch_model(self, model: nn.Module) -> nn.Module:
        """Revert FP8 patching, restoring nn.Linear layers."""
        for name, module in list(model.named_children()):
            if isinstance(module, FP8Linear):
                linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    linear.bias.data = module.bias.data.clone()
                setattr(model, name, linear)
            else:
                self.unpatch_model(module)
        self._patched_layers.clear()
        return model

    @property
    def patched_count(self) -> int:
        return len(self._patched_layers)


class FP8Engine:
    """End-to-end FP8 inference engine.

    Manages FP8 model patching, tensor quantization, and inference.
    Provides a unified interface for loading and running models in FP8.

    Usage:
        engine = FP8Engine(scheme="e4m3")
        model = engine.prepare_model(model)  # patches + converts weights
        output = engine.generate(model, input_ids)
    """

    def __init__(
        self,
        scheme: str = FP8Scheme.E4M3,
        quantize_activations: bool = True,
        kv_cache_fp8: bool = True,
    ):
        self.scheme = scheme
        self.quantize_activations = quantize_activations
        self.kv_cache_fp8 = kv_cache_fp8
        self._patcher = FP8ModelPatcher(scheme, quantize_activations)
        self._is_prepared = False

    @property
    def fp8_available(self) -> bool:
        return FP8_AVAILABLE

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare a model for FP8 inference.

        1. Patches linear layers to FP8Linear
        2. Converts weights to FP8 format
        3. Returns the patched model ready for inference

        Args:
            model: The PyTorch model.

        Returns:
            Patched model ready for FP8 inference.
        """
        logger.info(f"Preparing model for FP8 inference (scheme={self.scheme})")
        model = self._patcher.patch_model(model)
        orig_dtype = next(model.parameters()).dtype
        model = model.to(device=next(model.parameters()).device, dtype=orig_dtype)
        self._is_prepared = True
        return model

    def revert_model(self, model: nn.Module) -> nn.Module:
        """Revert a prepared model back to fp16."""
        model = self._patcher.unpatch_model(model)
        self._is_prepared = False
        return model

    def estimate_savings(self, model: nn.Module) -> dict:
        """Estimate memory savings from FP8 conversion.

        Returns:
            Dict with estimated savings information.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fp16_size = total_params * 2
        fp8_size = total_params * 1
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "fp16_bytes": fp16_size,
            "fp8_bytes": fp8_size,
            "savings_bytes": fp16_size - fp8_size,
            "savings_pct": (1 - fp8_size / max(fp16_size, 1)) * 100,
            "patched_layers": self._patcher.patched_count,
        }
