"""Abstract base class and protocol for all inference backend adapters.

Every inference backend (PyTorch, vLLM, llama.cpp, ExLlamaV2, ONNX, etc.)
must implement ``BackendAdapter`` to be usable by the distributed pipeline.

Usage:
    class MyBackend(BackendAdapter):
        name = "mybackend"
        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BackendAdapter(ABC):
    """Interface that all inference backends must implement.

    Each backend provides:
      - ``load_model()`` — initialize the engine and load weights
      - ``forward()`` — run a forward pass through assigned layers
      - ``shutdown()`` — release GPU memory and resources

    Plus metadata ``@classmethod`` methods for auto-detection and selection.
    """

    # ── Required instance methods ──────────────────────────────────────

    @abstractmethod
    def load_model(self) -> None:
        """Initialize the inference engine and load model weights.

        Must be called once before any call to ``forward()``.
        """

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run a forward pass through the assigned layers.

        Args:
            hidden_states: Activations from the previous pipeline node.
            attention_mask: Causal + padding mask.
            position_ids: Position indices for RoPE.
            past_key_values: KV cache tuples from previous iterations.
            input_ids: Token IDs (first node only).

        Returns:
            Tuple of (output_tensor, new_kv_cache_list).
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release all resources (GPU memory, threads, files)."""

    # ── Optional instance methods ──────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Single-shot text generation (useful for single-node mode).

        The default implementation raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate()"
        )

    def get_tokenizer(self) -> Any:
        """Return the backend's tokenizer, if available."""
        return None

    # ── Required metadata classmethods ─────────────────────────────────

    @classmethod
    @abstractmethod
    def display_name(cls) -> str:
        """Human-readable backend name (e.g. \"PyTorch / HF Transformers\")."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check whether the required dependencies are installed."""

    @classmethod
    @abstractmethod
    def priority_for(cls, device_type: str) -> int:
        """Return a priority score for the given device type.

        Args:
            device_type: ``"cuda"``, ``"cpu"``, ``"mps"``, ``"rocm"``, etc.

        Returns:
            Priority score where:
              - ``0`` = not supported on this device
              - ``1`` = fallback (works but not optimal)
              - ``5`` = good support
              - ``10`` = best-in-class for this device
        """

    # ── Optional health / load methods ──────────────────────────────────

    def health_check(self) -> bool:
        """Return ``True`` if the backend is healthy and ready to serve.

        Override in subclasses to perform actual health probes (e.g. ping
        the inference server, verify GPU memory).  The default
        implementation always returns ``True``.
        """
        return True

    def current_load(self) -> float:
        """Return the current load factor in ``[0.0, 1.0]``.

        ``0.0`` means idle, ``1.0`` means fully saturated.  Used by
        health-aware selection to prefer less-loaded backends.  The
        default implementation always returns ``0.0``.
        """
        return 0.0

    # ── Optional metadata classmethods ─────────────────────────────────

    @classmethod
    def description(cls) -> str:
        return cls.__doc__.strip().split("\n")[0] if cls.__doc__ else ""

    @classmethod
    def version(cls) -> str:
        return "1.0.0"
