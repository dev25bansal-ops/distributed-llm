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
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class BackendState(Enum):
    """Lifecycle state of a backend adapter.

    Adapters start ``UNINITIALIZED``, move to ``LOADING`` while
    ``load_model()`` runs, then ``READY`` on success or ``ERROR`` on
    failure, and ``SHUTDOWN`` once ``shutdown()`` has run.
    """

    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class ForwardInput:
    """Bundle of all parameters for a single forward pass.

    Using this dataclass reduces the parameter count of ``forward()`` and
    makes it easier to store, log, or replay requests.  It is fully
    backward-compatible — the existing ``forward()`` signature still works.
    """

    hidden_states: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    input_ids: torch.Tensor | None = None


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

    # NOTE: ``forward()`` currently takes 5 optional parameters, making it
    # a fat interface (ISP violation).  It is kept as-is for backward
    # compatibility, but new code should consider splitting it into narrower
    # sub-interfaces:
    #
    #   ``EncoderForward`` — pure encoder (hidden_states + mask)
    #   ``DecoderForward`` — decoder (hidden_states + mask + kv-cache)
    #   ``PreFillForward`` — first-token prefill (input_ids + position_ids)
    #
    # Alternatively, callers can bundle parameters via the ``ForwardInput``
    # dataclass defined at module level.
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

    @classmethod
    def probe_health(cls) -> bool:
        """Class-level health probe: report whether the adapter itself is
        usable without constructing an instance.

        Distinct from ``is_available()`` (which checks the third-party
        runtime): a probe answers "is this adapter class coherent and
        importable" — which it is, by definition, when this method runs.
        Subclasses may override with a real health check.  Fail-closed:
        any error returns ``False``.
        """
        try:
            return True
        except Exception:
            return False

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
