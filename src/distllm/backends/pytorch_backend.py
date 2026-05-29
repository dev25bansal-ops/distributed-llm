"""PyTorch/HuggingFace backend adapter for distributed inference."""

from __future__ import annotations

from typing import Any
from loguru import logger
import torch

from distllm.backends.protocol import BackendAdapter
from distllm.errors import ModelLoadError
from distllm.models.partitioner import ModelPartitioner


class PyTorchNodeAdapter(BackendAdapter):
    """Wraps ModelPartitioner to serve as a per-node inference engine.

    This is the original PyTorch-based backend used by the legacy WorkerNode.
    It loads a subset of model layers per node and runs them with
    HuggingFace transformers.

    Args:
        model_name: HuggingFace model name or path.
        device: Target device ("auto", "cuda", "cpu").
        dtype: Model dtype ("float16", "float32", "bfloat16").
        quantization_config: Optional quantization configuration.
        compression_config: Optional compression configuration.
        layer_start: First layer index (inclusive) for pipeline parallelism.
        layer_end: Last layer index (exclusive) for pipeline parallelism.
        total_layers: Total layers in the full model.
        trust_remote_code: Whether to trust HuggingFace remote code.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "float16",
        quantization_config: Any | None = None,
        compression_config: Any | None = None,
        layer_start: int = 0,
        layer_end: int = 0,
        total_layers: int = 0,
        trust_remote_code: bool | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.compression_config = compression_config
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.total_layers = total_layers
        self._trust_remote_code = trust_remote_code

        self.partitioner: ModelPartitioner | None = None
        self._is_first = layer_start == 0
        self._is_last = layer_end >= total_layers - 1

    def _get_device(self) -> str:
        if self.device == "auto":
            from distllm.core.device_registry import detect_platform
            return detect_platform()
        return self.device

    def load_model(self):
        """Load assigned model layers via ModelPartitioner."""
        target_device = self._get_device()

        try:
            self.partitioner = ModelPartitioner(
                model_name=self.model_name,
                device=target_device,
                dtype=self.dtype,
                quantization_config=self.quantization_config,
                compression_config=self.compression_config,
            )

            self.partitioner.load_layer_subset(
                self.layer_start, self.layer_end, self.total_layers, device=target_device
            )
        except Exception as e:
            self.partitioner = None
            logger.error(f"[PyTorch] Failed to load model {self.model_name}: {e}")
            raise ModelLoadError(self.model_name, str(e)) from e

        logger.info(
            f"[PyTorch] Model loaded: {self.model_name} "
            f"layers {self.layer_start}-{self.layer_end} of {self.total_layers}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through assigned layers with KV cache support.

        For first node: if input_ids provided, embed them first.
        For middle nodes: process hidden states directly.
        For last node: compute logits after layers.
        """
        if self.partitioner is None:
            raise RuntimeError("Model not loaded. Call load_model() before forward().")

        if input_ids is not None and self.partitioner.embed_input is not None:
            position_offset = 0
            if past_key_values and len(past_key_values) > 0:
                position_offset = past_key_values[0][0].shape[-2]
            hidden_states = self.partitioner.embed_input(input_ids, position_offset=position_offset)

        output, new_kv = self.partitioner.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        if self._is_last:
            output = self.partitioner.get_logits(output)

        return output, new_kv

    def shutdown(self):
        """Release PyTorch model resources (cross-platform)."""
        if self.partitioner is not None:
            self.partitioner = None
            try:
                from distllm.core.device_registry import detect_platform
                plat = detect_platform()
                if plat in ("cuda", "rocm"):
                    torch.cuda.empty_cache()
                elif plat == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                elif plat == "xpu" and hasattr(torch.xpu, "empty_cache"):
                    torch.xpu.empty_cache()
            except Exception:
                pass
            logger.info("[PyTorch] Engine shut down")

    @classmethod
    def display_name(cls) -> str:
        return "PyTorch / HF Transformers"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        return { "cuda": 5, "cpu": 5, "mps": 7, "rocm": 5, "xpu": 1 }.get(device_type, 3)
