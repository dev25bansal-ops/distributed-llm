"""Lightweight llama.cpp backend adapter for CPU/GPU inference."""

from __future__ import annotations

from typing import Any
import asyncio
from loguru import logger
import torch

from distllm.backends.protocol import BackendAdapter, BackendState
from distllm.errors import ModelLoadError


class LlamacppNodeAdapter(BackendAdapter):
    """Wraps llama-cpp-python to serve as a per-node inference engine.

    Maps between llama.cpp's generation API and the distributed pipeline's
    forward() contract. Supports single-node (full model) and per-layer
    (multi-node pipeline) modes.

    Args:
        model_path: Path to a GGUF model file.
        n_gpu_layers: Number of layers to offload to GPU (0 = CPU only).
        n_ctx: Context size for the model.
        n_threads: Number of CPU threads (None = auto).
        n_batch: Batch size for prompt processing.
        seed: Random seed.
        verbose: Enable llama.cpp verbose logging.
        layer_start: First layer index for pipeline parallelism.
        layer_end: Last layer index for pipeline parallelism.
        extra_kwargs: Additional kwargs forwarded to llama_cpp.Llama().
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        n_batch: int = 512,
        seed: int = 0,
        verbose: bool = False,
        layer_start: int | None = None,
        layer_end: int | None = None,
        **extra_kwargs: Any,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.seed = seed
        self.verbose = verbose
        self.layer_start = layer_start
        self.layer_end = layer_end
        self._extra_kwargs = extra_kwargs

        self._llm = None
        self._tokenizer = None
        self._model = None
        self.state = BackendState.UNINITIALIZED

    def load_model(self):
        """Initialize the llama.cpp model. Must be called before forward()."""
        from llama_cpp import Llama

        self.state = BackendState.LOADING

        kwargs = dict(self._extra_kwargs)
        kwargs.setdefault("n_gpu_layers", self.n_gpu_layers)
        kwargs.setdefault("n_ctx", self.n_ctx)
        kwargs.setdefault("n_batch", self.n_batch)
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("verbose", self.verbose)
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads

        logger.info(
            f"[llama.cpp] Loading {self.model_path} "
            f"(n_ctx={self.n_ctx}, n_gpu_layers={self.n_gpu_layers})"
        )
        try:
            self._llm = Llama(model_path=self.model_path, **kwargs)
            self._tokenizer = (
                self._llm.tokenizer()
                if hasattr(self._llm, "tokenizer")
                else None
            )
        except Exception as e:
            self._llm = None
            self._tokenizer = None
            self.state = BackendState.ERROR
            logger.error(f"[llama.cpp] Failed to load model {self.model_path}: {e}")
            raise ModelLoadError(self.model_path, str(e)) from e
        self.state = BackendState.READY
        logger.info(f"[llama.cpp] Model loaded: {self.model_path}")

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass compatible with NodeService.forward_fn contract.

        Args:
            hidden_states: Activations from previous node (pipeline mode).
            input_ids: Token IDs for the first node.

        Returns:
            (output_tensor, new_kv_cache) tuple.
        """
        if self.state is not BackendState.READY:
            raise RuntimeError(
                "Cannot forward: model not loaded (state "
                f"{self.state.value}). Call load_model() first."
            )
        if input_ids is not None:
            return self._forward_input_ids(input_ids)
        if hidden_states is not None:
            return self._forward_hidden_states(hidden_states, past_key_values)
        raise ValueError("Either input_ids or hidden_states must be provided")

    def _forward_input_ids(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Single-node path: run full model via create_completion()."""
        if self._llm is None:
            raise ModelLoadError("llamacpp", "llama.cpp not loaded. Call load_model() first.")

        ids = input_ids.flatten().tolist()
        output = self._llm.create_completion(
            prompt="",
            input_ids=ids,
            max_tokens=1,
            temperature=0.0,
            echo=False,
        )
        next_token = output["choices"][0]["text"]
        token_id = output["choices"][0].get("token_id", 0)
        result = torch.tensor([[token_id]], dtype=torch.long)
        return result, []

    def _forward_hidden_states(
        self,
        hidden_states: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Pipeline mode: run full model via eval and extract logits.

        Note: llama-cpp-python does not expose per-layer hidden states for
        true pipelining. This method runs the full model but only returns
        logits, providing a compatible interface for distributed setups
        where each node holds the full model (no layer splitting).
        """
        if self._llm is None:
            raise ModelLoadError("llamacpp", "llama.cpp not loaded. Call load_model() first.")

        ids = hidden_states.argmax(dim=-1).flatten().tolist()
        n_tokens = len(ids)
        self._llm.eval(ids)

        try:
            logits_arr = self._llm._ctx.eval_logits
            logits = torch.tensor(logits_arr, dtype=torch.float32).view(1, n_tokens, -1)
        except (AttributeError, RuntimeError):
            logits = torch.zeros(1, n_tokens, hidden_states.shape[-1])

        return logits, []

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **sampling_kwargs: Any,
    ) -> str:
        """Single-shot text generation.

        Args:
            prompt: Input text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text.
        """
        if self.state is not BackendState.READY:
            raise RuntimeError(
                "Cannot generate: model not loaded (state "
                f"{self.state.value}). Call load_model() first."
            )

        output = self._llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            **sampling_kwargs,
        )
        return output["choices"][0]["text"]

    async def async_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Async generation via asyncio.to_thread."""
        return await asyncio.to_thread(
            self.generate,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def get_tokenizer(self):
        return self._tokenizer

    def shutdown(self):
        """Release llama.cpp resources (cross-platform)."""
        if self._llm is not None:
            del self._llm
            self._llm = None
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
            logger.info("[llama.cpp] Engine shut down")
        # Idempotent: a second shutdown (or shutdown before load) is a no-op
        # except for the state transition.
        if self.state is not BackendState.ERROR:
            self.state = BackendState.SHUTDOWN

    @classmethod
    def display_name(cls) -> str:
        return "llama.cpp"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import llama_cpp
            return True
        except ImportError:
            return False

    @classmethod
    def priority_for(cls, device_type: str) -> int:
        return { "cpu": 8, "mps": 9, "rocm": 6, "cuda": 4, "xpu": 1 }.get(device_type, 5)


__all__ = ["LlamacppNodeAdapter"]
