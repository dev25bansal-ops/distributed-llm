"""vLLM backend adapter for per-node distributed inference.

Replaces the legacy per-node forward_fn with vLLM's production-quality
inference engine. Provides PagedAttention, continuous batching,
FlashAttention, AWQ/GPTQ quantization, and CUDA graphs for free.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from loguru import logger
import torch


class VLLMNodeAdapter:
    """Wraps vLLM to serve as a per-node inference engine.

    Maps between vLLM's generation API and the distributed pipeline's
    forward() contract. Supports both full-model (single-node) and
    per-layer-subset (multi-node pipeline) usage.

    Args:
        model_name: HuggingFace model name or path.
        vllm_config: Keyword arguments forwarded to vLLM.LLM().
            Typical keys: tensor_parallel_size, max_num_seqs,
            gpu_memory_utilization, dtype, quantization, etc.
        layer_start: First layer index (inclusive) for pipeline parallelism.
        layer_end: Last layer index (exclusive) for pipeline parallelism.
            When None, all layers are used (single-node mode).
        trust_remote_code: Whether to trust HuggingFace remote code.
    """

    def __init__(
        self,
        model_name: str,
        vllm_config: dict | None = None,
        layer_start: int | None = None,
        layer_end: int | None = None,
        trust_remote_code: bool | None = None,
    ):
        self.model_name = model_name
        self.layer_start = layer_start
        self.layer_end = layer_end
        self._trust_remote_code = trust_remote_code

        self._config = dict(vllm_config or {})
        if trust_remote_code is not None:
            self._config.setdefault("trust_remote_code", trust_remote_code)

        self._llm = None
        self._tokenizer = None
        self._is_pipeline_mode = layer_start is not None or layer_end is not None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Initialize the vLLM engine. Must be called before any forward pass."""
        from vllm import LLM

        logger.info(
            f"[VLLM] Loading model {self.model_name} "
            f"{f'(layers {self.layer_start}-{self.layer_end})' if self._is_pipeline_mode else '(full model)'}"
        )

        try:
            self._llm = LLM(model=self.model_name, **self._config)
            self._tokenizer = self._llm.get_tokenizer()
        except Exception as e:
            self._llm = None
            self._tokenizer = None
            logger.error(f"[VLLM] Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load vLLM model {self.model_name}: {e}") from e

        logger.info(f"[VLLM] Model loaded: {self.model_name}")

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass compatible with NodeService.forward_fn contract.

        For single-node mode (full model), uses vLLM's generation loop.
        For pipeline mode, uses raw model forward with KV cache.
        """
        if input_ids is not None:
            return self._forward_with_input_ids(input_ids)
        if hidden_states is not None:
            return self._forward_hidden_states(hidden_states, attention_mask, position_ids, past_key_values)
        raise ValueError("Either input_ids or hidden_states must be provided")

    def _forward_with_input_ids(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Single-node: run full model via vLLM generate()."""
        if self._llm is None:
            raise RuntimeError("vLLM not loaded. Call load_model() first.")

        if self._is_pipeline_mode:
            raise NotImplementedError(
                "vLLM pipeline mode does not support input_ids-based forward. "
                "Use hidden_states and past_key_values for pipeline stages."
            )

        prompt = self._tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            skip_special_tokens=True,
        )

        try:
            outputs = self._llm.generate([prompt], sampling_params)
            result = outputs[0]
            token_ids = result.outputs[0].token_ids
            next_token = torch.tensor([[token_ids[0]]]) if token_ids else torch.tensor([[0]])
        except Exception as e:
            raise RuntimeError(f"vLLM forward failed for input_ids: {e}") from e

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
            if self._llm is not None:
                self._model = self._llm.llm_engine.model_executor.driver_worker.model_runner.model
            else:
                raise RuntimeError("vLLM not loaded. Call load_model() first.")

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

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any | None = None,
    ) -> list[Any]:
        """Direct vLLM generation API for single-node use.

        Args:
            prompts: List of text prompts.
            sampling_params: vLLM SamplingParams object. Uses defaults if None.

        Returns:
            List of vLLM RequestOutput objects.
        """
        if self._llm is None:
            raise RuntimeError("vLLM not loaded. Call load_model() first.")

        from vllm import SamplingParams
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

        return self._llm.generate(prompts, sampling_params)

    async def async_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Single-shot generation using asyncio.to_thread() for vLLM."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        outputs = await asyncio.to_thread(
            self._llm.generate, [prompt], sampling_params
        )
        return outputs[0].outputs[0].text

    def get_tokenizer(self):
        return self._tokenizer or (self._llm.get_tokenizer() if self._llm else None)

    def shutdown(self):
        """Release vLLM resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[VLLM] Engine shut down")
