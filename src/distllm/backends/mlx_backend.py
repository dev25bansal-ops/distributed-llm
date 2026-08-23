"""MLX backend adapter for Apple Silicon (M1/M2/M3/M4) Macs.

Uses Apple's MLX framework for efficient inference on Apple Silicon's
unified memory architecture.  Macs with 48-192GB unified memory can
run models up to ~70B parameters (INT4) that would otherwise require
multiple NVIDIA GPUs.

Requires: ``pip install mlx`` (and optionally ``mlx-lm`` for easy model loading).

Priority: returns 10 for "mps" devices (best-in-class for Apple Silicon),
effectively replacing the PyTorch MPS backend for inference workloads.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import torch
from loguru import logger

from distllm.backends.protocol import BackendAdapter
from distllm.errors import ModelLoadError


class MLXNodeAdapter(BackendAdapter):
    """Apple MLX backend for inference on Apple Silicon unified memory.

    Uses mlx-lm for model loading and generation, providing native
    performance on M1/M2/M3/M4 Macs without Rosetta or CUDA emulation.

    Args:
        model_name: HuggingFace model name or path.
        max_kv_size: Max KV cache size in tokens per layer (default 4096).
        quant_bits: Quantization bits (0=no quant, 4=QLoRA-style, 8=INT8).
        **kwargs: Additional kwargs forwarded to mlx_lm.load().
    """

    def __init__(
        self,
        model_name: str,
        max_kv_size: int = 4096,
        quant_bits: int = 0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self._max_kv_size = max_kv_size
        self._quant_bits = quant_bits
        self._extra_kwargs = kwargs

        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """Load model via mlx-lm into Apple Silicon unified memory."""
        try:
            from mlx_lm import load as mlx_load

            logger.info(f"[MLX] Loading {self.model_name} on Apple Silicon...")
            quant = "q4_0" if self._quant_bits == 4 else None
            self._model, self._tokenizer = mlx_load(
                self.model_name,
                tokenizer_config={"trust_remote_code": self._extra_kwargs.get("trust_remote_code", False)},
            )
            logger.info(f"[MLX] Model loaded: {self.model_name}")
        except ImportError as e:
            raise ModelLoadError(
                self.model_name,
                "mlx-lm not installed. Run: pip install mlx mlx-lm",
            ) from e
        except Exception as e:
            self._model = None
            self._tokenizer = None
            logger.error(f"[MLX] Failed to load model {self.model_name}: {e}")
            raise ModelLoadError(self.model_name, str(e)) from e

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run a forward pass.

        MLX models use their own array type — we convert PyTorch tensors,
        run inference, and convert back.
        """
        if self._model is None:
            raise ModelLoadError("mlx", "MLX not loaded. Call load_model() first.")

        import mlx.core as mx

        if input_ids is not None:
            return self._forward_input_ids(input_ids, mx)
        if hidden_states is not None:
            return self._forward_full_model(hidden_states, mx)
        raise ValueError("Either input_ids or hidden_states must be provided")

    def _forward_input_ids(
        self, input_ids: torch.Tensor, mx: Any,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass from token IDs through the full model.

        The whole sequence is evaluated in a SINGLE model call so causal
        attention covers prior context.  Scoring each token as an
        isolated length-1 sequence (the previous behavior) drops all
        attention to earlier tokens and yields wrong logits whenever
        seq_len > 1.
        """
        import numpy as np

        ids_np = input_ids.detach().cpu().numpy()
        if ids_np.ndim == 1:
            ids_np = ids_np[np.newaxis, :]

        per_row_logits = []
        for row in range(ids_np.shape[0]):
            mlx_input = mx.array(ids_np[row].tolist()).reshape(1, -1)
            out = self._model(mlx_input)
            # Newer mlx-lm versions return (logits, cache); accept both.
            logits = out[0] if isinstance(out, tuple) else out
            per_row_logits.append(torch.from_numpy(np.asarray(logits, dtype=np.float32)))

        if len(per_row_logits) == 1:
            logits_tensor = per_row_logits[0]
        else:
            logits_tensor = torch.cat(per_row_logits, dim=0)
        return logits_tensor, []

    def _forward_full_model(
        self, hidden_states: torch.Tensor, mx: Any,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Full-model forward from hidden states (for compatibility)."""
        return self._forward_input_ids(hidden_states.argmax(dim=-1), mx)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Single-shot generation via mlx-lm."""
        if self._model is None:
            raise ModelLoadError("mlx", "MLX not loaded. Call load_model() first.")

        try:
            from mlx_lm import generate as mlx_generate
            response = mlx_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response
        except ImportError:
            raise ModelLoadError("mlx", "mlx-lm not available for generate()")

    def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        self._tokenizer = None
        import gc
        gc.collect()

    def health_check(self) -> bool:
        """Check if MLX and model are available."""
        if self._model is None:
            return False
        try:
            import mlx.core
            return mlx.core.metal.is_available()
        except Exception:
            return False

    # ── Metadata classmethods ─────────────────────────────────────────────

    @classmethod
    def display_name(cls) -> str:
        return "MLX (Apple Silicon)"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mlx
            return True
        except ImportError:
            return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        return { "mps": 10, "cpu": 2 }.get(device_type, 0)


__all__ = ["MLXNodeAdapter"]
