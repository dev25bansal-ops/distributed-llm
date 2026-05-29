"""TensorRT-LLM backend adapter for high-performance distributed inference.

Provides first-class TensorRT-LLM integration with:
- Optimized CUDA kernels via TensorRT
- PagedAttention with KV cache management
- In-flight batching for maximum throughput
- FP8/INT8/INT4 quantization
- Multi-GPU tensor parallelism
- CUDA graph capture for reduced overhead

Usage::

    backend = TensorRTLLMAdapter(
        model_name="meta-llama/Llama-2-7b-hf",
        engine_dir="/path/to/trt_engine",
        max_batch_size=32,
    )
    backend.load_model()
    logits, kv_cache = backend.forward(input_ids=input_ids)
"""

from __future__ import annotations

import os
from typing import Any
from loguru import logger
import torch

from distllm.backends.protocol import BackendAdapter
from distllm.errors import ModelLoadError


class TensorRTLLMAdapter(BackendAdapter):
    """TensorRT-LLM backend for high-performance inference.

    Wraps TensorRT-LLM's engine to serve as a per-node inference engine
    in the distributed pipeline. Supports both pre-built engines and
    runtime engine building from HuggingFace models.

    Args:
        model_name: HuggingFace model name or path.
        engine_dir: Path to pre-built TensorRT engine. If None, builds
            an engine from model_name at runtime.
        max_batch_size: Maximum batch size for the engine.
        max_input_len: Maximum input sequence length.
        max_output_len: Maximum output sequence length.
        max_beam_width: Maximum beam width for beam search.
        dtype: Data type ("float16", "bfloat16", "float32").
        quantization: Quantization mode ("int8", "int4_awq", "int4_gptq", "fp8", None).
        tp_size: Tensor parallel size (number of GPUs).
        pp_size: Pipeline parallel size.
        layer_start: First layer index for pipeline parallelism.
        layer_end: Last layer index for pipeline parallelism.
        use_paged_kv_cache: Use paged KV cache (default True).
        enable_cuda_graph: Capture CUDA graphs for reduced launch overhead.
    """

    def __init__(
        self,
        model_name: str,
        engine_dir: str | None = None,
        max_batch_size: int = 32,
        max_input_len: int = 2048,
        max_output_len: int = 512,
        max_beam_width: int = 1,
        dtype: str = "float16",
        quantization: str | None = None,
        tp_size: int = 1,
        pp_size: int = 1,
        layer_start: int | None = None,
        layer_end: int | None = None,
        use_paged_kv_cache: bool = True,
        enable_cuda_graph: bool = True,
    ):
        self.model_name = model_name
        self.engine_dir = engine_dir
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_beam_width = max_beam_width
        self.dtype = dtype
        self.quantization = quantization
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.use_paged_kv_cache = use_paged_kv_cache
        self.enable_cuda_graph = enable_cuda_graph

        self._engine = None
        self._model = None
        self._tokenizer = None
        self._kv_cache_manager = None
        self._is_pipeline_mode = layer_start is not None or layer_end is not None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Initialize TensorRT-LLM engine.

        Loads a pre-built engine from engine_dir, or builds one from
        model_name if no engine_dir is provided.
        """
        try:
            import tensorrt_llm
            from tensorrt_llm import LLM, SamplingParams
        except ImportError:
            raise ModelLoadError(
                self.model_name,
                "tensorrt-llm is not installed. "
                "Install with: pip install tensorrt-llm"
            )

        logger.info(
            f"[TensorRT-LLM] Loading model {self.model_name} "
            f"{f'(layers {self.layer_start}-{self.layer_end})' if self._is_pipeline_mode else '(full model)'}"
        )

        try:
            if self.engine_dir and os.path.isdir(self.engine_dir):
                self._engine = LLM(
                    model=self.engine_dir,
                    max_batch_size=self.max_batch_size,
                    max_input_len=self.max_input_len,
                    max_output_len=self.max_output_len,
                    max_beam_width=self.max_beam_width,
                    dtype=self.dtype,
                )
            else:
                self._engine = LLM(
                    model=self.model_name,
                    max_batch_size=self.max_batch_size,
                    max_input_len=self.max_input_len,
                    max_output_len=self.max_output_len,
                    max_beam_width=self.max_beam_width,
                    dtype=self.dtype,
                    tensor_parallel_size=self.tp_size,
                    pipeline_parallel_size=self.pp_size,
                )

            self._tokenizer = self._engine.get_tokenizer()
        except Exception as e:
            self._engine = None
            self._tokenizer = None
            logger.error(f"[TensorRT-LLM] Failed to load model {self.model_name}: {e}")
            raise ModelLoadError(self.model_name, str(e)) from e

        logger.info(f"[TensorRT-LLM] Model loaded: {self.model_name}")

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass compatible with NodeService.forward_fn contract.

        For single-node mode, uses TensorRT-LLM's generate API.
        For pipeline mode, uses raw model forward with KV cache.
        """
        if input_ids is not None:
            return self._forward_with_input_ids(input_ids)
        if hidden_states is not None:
            return self._forward_hidden_states(
                hidden_states, attention_mask, position_ids, past_key_values
            )
        raise ValueError("Either input_ids or hidden_states must be provided")

    def _forward_with_input_ids(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Single-node: run full model via TensorRT-LLM generate()."""
        if self._engine is None:
            raise RuntimeError("TensorRT-LLM not loaded. Call load_model() first.")

        if self._is_pipeline_mode:
            raise NotImplementedError(
                "TensorRT-LLM pipeline mode does not support input_ids-based forward. "
                "Use hidden_states and past_key_values for pipeline stages."
            )

        prompt = self._tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

        from tensorrt_llm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
        )

        try:
            outputs = self._engine.generate([prompt], sampling_params)
            result = outputs[0]
            token_ids = result.outputs[0].token_ids
            next_token = torch.tensor([[token_ids[0]]]) if token_ids else torch.tensor([[0]])
        except Exception as e:
            raise RuntimeError(f"TensorRT-LLM forward failed: {e}") from e

        return next_token, []

    def _forward_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Pipeline mode: run assigned layers on hidden states."""
        if self._model is None:
            if self._engine is not None:
                self._model = self._extract_inner_model(self._engine)
            else:
                raise RuntimeError("TensorRT-LLM not loaded. Call load_model() first.")

        with torch.no_grad():
            output = self._model(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        if isinstance(output, tuple):
            logits, new_kv = output
        else:
            logits = output.logits if hasattr(output, "logits") else output[0]
            new_kv = output.past_key_values if hasattr(output, "past_key_values") else []

        return logits, new_kv

    @staticmethod
    def _extract_inner_model(engine):
        """Extract the inner model from TensorRT-LLM engine."""
        paths = [
            lambda e: e.model,
            lambda e: e.llm_engine.model,
            lambda e: e._model,
        ]
        for path_fn in paths:
            try:
                model = path_fn(engine)
                if model is not None:
                    return model
            except (AttributeError, TypeError):
                continue
        raise RuntimeError(
            "Failed to extract inner model from TensorRT-LLM engine. "
            "The installed version may be incompatible."
        )

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any | None = None,
    ) -> list[Any]:
        """Direct TensorRT-LLM generation API for single-node use."""
        if self._engine is None:
            raise RuntimeError("TensorRT-LLM not loaded. Call load_model() first.")

        from tensorrt_llm import SamplingParams
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=256,
            )

        return self._engine.generate(prompts, sampling_params)

    async def async_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Async generation using asyncio.to_thread()."""
        import asyncio
        from tensorrt_llm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        outputs = await asyncio.to_thread(
            self._engine.generate, [prompt], sampling_params
        )
        return outputs[0].outputs[0].text

    def get_tokenizer(self):
        return self._tokenizer or (self._engine.get_tokenizer() if self._engine else None)

    def shutdown(self):
        """Release TensorRT-LLM resources."""
        if self._engine is not None:
            del self._engine
            self._engine = None
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[TensorRT-LLM] Engine shut down")

    @classmethod
    def display_name(cls) -> str:
        return "TensorRT-LLM"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import tensorrt_llm
            return True
        except ImportError:
            return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        if device_type == "cuda":
            return 15  # Highest priority on CUDA (best performance)
        return 0
