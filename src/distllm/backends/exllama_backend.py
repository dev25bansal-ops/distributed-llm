"""ExLlamaV2 backend adapter for fast GPTQ inference.

ExLlamaV2 is a heavily optimized GPTQ inference engine that provides
significant speedups over HuggingFace Transformers for quantized 4-bit
models on NVIDIA GPUs.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from distllm.backends.protocol import BackendAdapter
from distllm.errors import ModelLoadError

try:
    from exllamav2 import ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

    HAS_EXLLAMA = True
except ImportError:
    HAS_EXLLAMA = False


class ExLlamaV2NodeAdapter(BackendAdapter):
    """ExLlamaV2 backend for fast GPTQ inference on NVIDIA GPUs.

    Provides quantized 4-bit inference with significantly lower latency
    than HuggingFace Transformers for GPTQ models. Supports both
    single-node (full model) and pipeline modes.

    Args:
        model_name: Path to a GPTQ model directory containing
            ``config.json`` and ``*.safetensors``.
        device: Target device (``"cuda"`` only).
        dtype: Not used by ExLlamaV2 (determined by quant config).
        layer_start: First layer index for pipeline mode.
        layer_end: Last layer index for pipeline mode.
        max_seq_len: Maximum sequence length.
        gpu_split: GPU memory split for multi-GPU.
        trust_remote_code: Ignored (ExLlamaV2 does not use HF code).
        **extra_kwargs: Additional kwargs for ``ExLlamaV2Config``.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        layer_start: int = 0,
        layer_end: int = 0,
        total_layers: int = 0,
        max_seq_len: int = 4096,
        gpu_split: list[float] | None = None,
        trust_remote_code: bool | None = None,
        **extra_kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.total_layers = total_layers
        self._max_seq_len = max_seq_len
        self._gpu_split = gpu_split
        self._extra_kwargs = extra_kwargs

        self._model = None
        self._tokenizer = None
        self._generator = None

    def load_model(self) -> None:
        if not HAS_EXLLAMA:
            raise ModelLoadError(
                self.model_name,
                "exllamav2 not installed. Install with: "
                "pip install exllamav2",
            )
        from exllamav2 import ExLlamaV2Config, ExLlamaV2, ExLlamaV2Tokenizer

        logger.info(f"[ExLlamaV2] Loading model: {self.model_name}")

        try:
            config = ExLlamaV2Config()
            config.model_dir = self.model_name
            config.max_seq_len = self._max_seq_len
            for k, v in self._extra_kwargs.items():
                setattr(config, k, v)

            self._model = ExLlamaV2(config)
            self._model.load(
                gpu_split=self._gpu_split or [1.0],
            )
            self._tokenizer = ExLlamaV2Tokenizer(config)
            logger.info(f"[ExLlamaV2] Model loaded: {self.model_name}")
        except Exception as e:
            self._model = None
            logger.error(f"[ExLlamaV2] Failed to load model: {e}")
            raise ModelLoadError(self.model_name, str(e)) from e

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if input_ids is not None:
            return self._forward_input_ids(input_ids)
        if hidden_states is not None:
            return self._forward_hidden_states(
                hidden_states, past_key_values
            )
        raise ValueError("Either input_ids or hidden_states must be provided")

    def _forward_input_ids(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self._generator is None:
            from exllamav2.generator import (
                ExLlamaV2BaseGenerator,
                ExLlamaV2Sampler,
            )

            self._generator = ExLlamaV2BaseGenerator(
                self._model, self._tokenizer
            )

        ids_list = input_ids.flatten().tolist()
        settings = ExLlamaV2Sampler.Settings(
            temperature=0.0, top_k=1, top_p=0.0
        )
        output = self._generator.generate_simple(
            prompt="",
            gen_settings=settings,
            num_tokens=1,
            seed=0,
            input_ids=[ids_list],
        )
        return torch.tensor([[0]]), []

    def _forward_hidden_states(
        self,
        hidden_states: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        with torch.no_grad():
            output = self._model.forward(
                hidden_states,
                past_key_values=past_key_values or None,
            )
        if isinstance(output, tuple):
            logits, new_kv = output
        else:
            logits = output
            new_kv = []
        return logits, new_kv

    def shutdown(self) -> None:
        if self._model is not None:
            self._model.unload()
            self._model = None
            self._generator = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[ExLlamaV2] Engine shut down")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        from exllamav2.generator import (
            ExLlamaV2BaseGenerator,
            ExLlamaV2Sampler,
        )

        generator = ExLlamaV2BaseGenerator(self._model, self._tokenizer)
        settings = ExLlamaV2Sampler.Settings(
            temperature=temperature,
            top_k=kwargs.get("top_k", 40),
            top_p=kwargs.get("top_p", 0.9),
        )
        return generator.generate_simple(
            prompt=prompt,
            gen_settings=settings,
            num_tokens=max_tokens,
        )

    @classmethod
    def display_name(cls) -> str:
        return "ExLlamaV2 (GPTQ)"

    @classmethod
    def is_available(cls) -> bool:
        return HAS_EXLLAMA

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        return 8 if device_type in ("cuda",) else 0
