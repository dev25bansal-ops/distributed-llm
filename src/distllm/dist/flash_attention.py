"""FlashAttention integration for fused O(n) memory attention.

Provides a drop-in replacement for standard attention that uses the
flash-attn package for 2-3x faster prefill on long contexts and
O(n) memory instead of O(n²).

Gracefully falls back to standard attention when flash-attn is unavailable.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger

__all__ = [
    "FlashAttentionWrapper",
]


class FlashAttentionWrapper:
    # Class-level mask cache: seq_len -> cached causal mask tensor
    _mask_cache: dict[int, torch.Tensor] = {}

    def __init__(self, causal: bool = True, num_heads: int | None = None, head_dim: int | None = None):
        self.causal = causal
        self._flash_attn_fn = None
        self._available = False
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._attempt_load()

    def _attempt_load(self) -> None:
        try:
            from flash_attn import flash_attn_func
            self._flash_attn_fn = flash_attn_func
            self._available = True
            logger.info("FlashAttention: using flash-attn fused kernel")
        except ImportError:
            logger.debug("FlashAttention: flash-attn not installed, using PyTorch SDPA fallback")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        q, k, v, was_transposed = self._ensure_format(q, k, v)

        if self._available and q.device.type == "cuda":
            return self._flash_forward(q, k, v, dropout_p, softmax_scale)
        else:
            return self._sdpa_forward(q, k, v, attention_mask, dropout_p, softmax_scale, was_transposed)

    def _flash_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float | None,
    ) -> torch.Tensor:
        out = self._flash_attn_fn(
            q, k, v,
            dropout_p=dropout_p,
            causal=self.causal,
            softmax_scale=softmax_scale,
        )
        batch, seq_len, num_heads, head_dim = out.shape
        return out.reshape(batch, seq_len, num_heads * head_dim)

    def _sdpa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout_p: float,
        softmax_scale: float | None,
        was_transposed: bool,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if self.causal:
            # Use is_causal=True in scaled_dot_product_attention instead of
            # manually creating the mask. This avoids 1.28GB temporary
            # allocations per forward pass for long sequences.
            pass  # Handled by is_causal parameter below

        if attention_mask is not None:
            attn_mask = attention_mask

        # Use is_causal for causal attention (avoids mask materialization)
        use_causal = self.causal and attn_mask is None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=self.causal and attn_mask is None,
        )

        out = out.transpose(1, 2)
        batch, seq_len, num_heads, head_dim = out.shape
        return out.reshape(batch, seq_len, num_heads * head_dim)

    def _ensure_format(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        was_transposed = False
        if q.dim() == 4:
            if self._head_dim is not None and q.shape[3] == self._head_dim:
                if self._num_heads is not None and q.shape[1] == self._num_heads:
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
                    v = v.transpose(1, 2)
                    was_transposed = True
            elif q.shape[1] < q.shape[2]:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                was_transposed = True
        return q, k, v, was_transposed

    def stats(self) -> dict:
        return {
            "available": self._available,
            "causal": self.causal,
            "backend": "flash_attn" if self._available else "sdpa_fallback",
        }


def apply_flash_attention_to_model(model: torch.nn.Module) -> int:
    patched = 0
    flash_attn = FlashAttentionWrapper()

    if not flash_attn.is_available:
        logger.debug("FlashAttention not available, skipping model patching")
        return 0

    for _name, module in model.named_modules():
        module_type = type(module).__name__
        if "Attention" in module_type and hasattr(module, "q_proj") and hasattr(module, "k_proj"):
            if _patch_attention_module(module, flash_attn):
                patched += 1

    if patched > 0:
        logger.info(f"FlashAttention: patched {patched} attention modules")
    return patched


def _get_qkv_projections(module: torch.nn.Module) -> tuple[Any, Any, Any, Any] | None:
    if all(hasattr(module, attr) for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return module.q_proj, module.k_proj, module.v_proj, module.o_proj

    if hasattr(module, "query_key_value") and hasattr(module, "dense"):
        qkv_proj = module.query_key_value
        o_proj = module.dense
        return qkv_proj, qkv_proj, qkv_proj, o_proj

    for q_name, k_name, v_name, o_name in [
        ("q_proj", "k_proj", "v_proj", "dense"),
        ("Wq", "Wk", "Wv", "Wo"),
        ("q", "k", "v", "o"),
    ]:
        if all(hasattr(module, attr) for attr in [q_name, k_name, v_name, o_name]):
            return (
                getattr(module, q_name),
                getattr(module, k_name),
                getattr(module, v_name),
                getattr(module, o_name),
            )

    return None


def _patch_attention_module(module: torch.nn.Module, flash_attn: FlashAttentionWrapper) -> bool:
    if hasattr(module, "_flash_attn_patched"):
        return False

    qkv = _get_qkv_projections(module)
    if qkv is None:
        return False

    q_proj, k_proj, v_proj, o_proj = qkv
    original_forward = module.forward
    is_falcon = hasattr(module, "query_key_value")

    def flash_forward(*args, **kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            return original_forward(*args, **kwargs)

        bsz, q_len, _ = hidden_states.shape
        num_heads = getattr(module, "num_heads", None) or getattr(module, "num_attention_heads", 32)

        if is_falcon:
            num_kv_heads = getattr(module, "num_kv_heads", 8)
            head_dim = hidden_states.shape[-1] // num_heads
            qkv_out = q_proj(hidden_states)
            qkv_out = qkv_out.view(bsz, q_len, -1)
            kv_head_dim = head_dim
            q = qkv_out[:, :, :num_heads * head_dim].view(bsz, q_len, num_heads, head_dim)
            k = qkv_out[:, :, num_heads * head_dim:num_heads * head_dim + num_kv_heads * kv_head_dim] \
                .view(bsz, q_len, num_kv_heads, kv_head_dim)
            v = qkv_out[:, :, num_heads * head_dim + num_kv_heads * kv_head_dim:] \
                .view(bsz, q_len, num_kv_heads, kv_head_dim)
        else:
            head_dim = getattr(module, "head_dim", None) or hidden_states.shape[-1] // num_heads
            q = q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim)
            k = k_proj(hidden_states).view(bsz, q_len, num_heads, head_dim)
            v = v_proj(hidden_states).view(bsz, q_len, num_heads, head_dim)

        attn_output = flash_attn.forward(q, k, v)
        attn_output = o_proj(attn_output)

        return (attn_output, None)

    module.forward = flash_forward
    module._flash_attn_patched = True
    return True
