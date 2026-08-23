"""Multi-model manager: track and switch between registered models.

Wraps a :class:`~distllm.core.multi_model_serving.ModelRegistry` with the
notion of a CURRENT model, so the coordinator can hot-swap which weights
serve inference without losing sight of the full catalog.
"""

from __future__ import annotations

from typing import Any


class MultiModelManager:
    """Current-model pointer over a shared model registry."""

    def __init__(
        self,
        model_name: str = "",
        pipeline: Any = None,
        model_registry: Any = None,
    ) -> None:
        self.model_name = model_name
        self.pipeline = pipeline
        self.registry = model_registry

    def get_model_name(self, requested: str | None = None) -> str:
        """Resolve the model to use: explicit request wins, else current."""
        return requested or self.model_name or ""

    def set_model(self, name: str) -> bool:
        """Switch the current model (must be registered)."""
        if self.registry is not None and name not in self.registry:
            return False
        self.model_name = name
        return True

    def list_models(self) -> list[str]:
        """All registered model names."""
        if self.registry is None:
            return [self.model_name] if self.model_name else []
        return self.registry.list_models()
