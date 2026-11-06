"""Lightweight llama.cpp backend adapter for CPU/GPU inference."""

from __future__ import annotations

from typing import Any
import asyncio
from loguru import logger
import torch

from distllm.backends.model_validation import validate_model_path
from distllm.backends.protocol import BackendAdapter, BackendState
from distllm.errors import ModelLoadError, OOMError


def _is_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower() or "oom" in type(exc).__name__.lower()


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
        validate_model_path(model_path, allow_local_paths=True)
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
            if _is_oom(e):
                raise OOMError(self.model_path, detail=str(e)) from e
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
        """Single-node forward: text-API generate of exactly one token.

        PERF-BE-2 (llama.cpp leg): ``create_completion(max_tokens=1)`` is a
        text API — llama-cpp-python exposes no supported token->logit
        forward, and its deterministic logits surface is ``_ctx.eval_logits``
        (the one the pipeline path already uses). Calling create_completion
        here is the honest single-node contract (return the sampled id), not
        a native forward; PERF_BE2_NATIVE_FORWARD says so. This path keeps
        passing input_ids positionally — llama.cpp evaluates the ids
        directly, with no decode->re-encode round trip (that loss only
        exists where a text prompt is the only input).
        """
        if self._llm is None:
            raise ModelLoadError("llamacpp", "llama.cpp not loaded. Call load_model() first.")

        ids = input_ids.flatten().tolist()
        try:
            output = self._llm.create_completion(
                prompt="",
                input_ids=ids,
                max_tokens=1,
                temperature=0.0,
                echo=False,
            )
        except Exception as e:
            if _is_oom(e):
                raise OOMError(self.model_path, detail=str(e)) from e
            raise
        next_token = output["choices"][0]["text"]
        token_id = output["choices"][0].get("token_id", 0)
        result = torch.tensor([[token_id]], dtype=torch.long)
        return result, []

    # PERF-BE-2: no native token->logit forward on the text-API surface
    # (see _forward_input_ids). The pipeline _forward_hidden_states path
    # uses the native eval/eval_logits surface when the installed
    # llama-cpp-python exposes it, and raises NotImplementedError otherwise.
    PERF_BE2_NATIVE_FORWARD = False

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
        except AttributeError as e:
            # SECURITY/CORRECTNESS: the tensor logits are unavailable without
            # eval_logits. Returning torch.zeros fabricates logits that decode
            # as uniform argmax, silently corrupting inference. Fail loudly
            # instead of serving garbage — same contract as WebGPUNodeAdapter.
            raise NotImplementedError(
                "llama.cpp eval_logits is unavailable — llama-cpp-python does "
                "not expose per-token logits for the loaded model. Refusing to "
                "fabricate torch.zeros logits for pipeline forward(). "
                f"(n_tokens={n_tokens}, hidden_dim={hidden_states.shape[-1]})"
            ) from e

        try:
            logits = torch.tensor(logits_arr, dtype=torch.float32).view(1, n_tokens, -1)
        except (RuntimeError, ValueError) as e:
            raise NotImplementedError(
                "llama.cpp eval_logits shape is incompatible with pipeline "
                f"forward() (n_tokens={n_tokens}, hidden_dim={hidden_states.shape[-1]})"
            ) from e

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
