"""Backend configuration models for the plugin system.

Provides a unified ``BackendConfig`` that wraps backend-specific settings
and standardises the constructor kwargs passed to ``BackendAdapter``
subclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendConfig:
    """Unified configuration for any ``BackendAdapter``.

    Fields shared across all backends are at the top level; backend-specific
    settings go into ``backend_kwargs`` or in the dedicated nested configs
    (``vllm``, ``llamacpp``, ``exllama``, ``onnx``).

    Usage:
        config = BackendConfig.from_preferred("vllm", model_name="...")
        config = BackendConfig(model_name="...", vllm=dict(tensor_parallel_size=2))
    """

    # ── Shared fields ─────────────────────────────────────────────────
    model_name: str = ""
    device: str = "auto"
    dtype: str = "float16"
    trust_remote_code: bool = False
    max_model_len: int | None = None

    # ── Pipeline fields ────────────────────────────────────────────────
    layer_start: int = 0
    layer_end: int = 0
    total_layers: int = 0

    # ── Backend-specific settings ──────────────────────────────────────
    vllm: dict[str, Any] = field(default_factory=dict)
    llamacpp: dict[str, Any] = field(default_factory=dict)
    exllama: dict[str, Any] = field(default_factory=dict)
    onnx: dict[str, Any] = field(default_factory=dict)
    pytorch: dict[str, Any] = field(default_factory=dict)

    # ── Extra kwargs forwarded to the backend constructor ──────────────
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_backend_kwargs(self, backend_name: str) -> dict[str, Any]:
        """Build the kwargs dict for a specific backend's constructor.

        The output always includes ``model_name``, ``device``, ``dtype``,
        ``layer_start``, ``layer_end``, ``trust_remote_code``, plus the
        backend-specific dict merged with ``backend_kwargs``.
        """
        specific = getattr(self, backend_name, {})
        kwargs = dict(
            model_name=self.model_name,
            device=self.device,
            dtype=self.dtype,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            total_layers=self.total_layers,
            trust_remote_code=self.trust_remote_code,
        )
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        kwargs.update(specific)
        kwargs.update(self.backend_kwargs)
        return kwargs

    @classmethod
    def from_preferred(
        cls,
        backend_name: str,
        **overrides: Any,
    ) -> BackendConfig:
        """Create a config that targets a particular backend.

        Any keyword arguments override the default fields.

        Example:
            BackendConfig.from_preferred("vllm", model_name="meta-llama/Llama-2-7b")
        """
        return cls(**overrides)
