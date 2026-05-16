"""Thread-safe model registry for multi-model serving."""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from distllm.errors import ModelNotFoundError


@dataclass
class ModelEntry:
    """Metadata for a registered model."""
    name: str
    path: str
    total_layers: int
    registered_at: float = field(default_factory=time.time)


class ModelRegistry:
    """Thread-safe registry for multi-model serving.

    Tracks model name -> path -> layer metadata.
    Enforces max_models limit and manages a default model.
    """

    def __init__(self, max_models: int = 4):
        self._max_models = max_models
        self._models: Dict[str, ModelEntry] = {}
        self._default_model: Optional[str] = None
        self._lock = threading.Lock()

    def register(self, name: str, path: str, total_layers: int) -> ModelEntry:
        """Register a model. Raises ValueError if max_models exceeded."""
        with self._lock:
            if name not in self._models and len(self._models) >= self._max_models:
                raise ModelNotFoundError(name,
                    f"Maximum models ({self._max_models}) already registered. "
                    f"Remove a model before registering '{name}'."
                )
            entry = ModelEntry(name=name, path=path, total_layers=total_layers)
            self._models[name] = entry
            if self._default_model is None:
                self._default_model = name
            return entry

    def get(self, name: str) -> Optional[ModelEntry]:
        """Get a model by name."""
        with self._lock:
            return self._models.get(name)

    def list_models(self) -> List[ModelEntry]:
        """Return all registered models."""
        with self._lock:
            return list(self._models.values())

    @property
    def default_model(self) -> Optional[str]:
        """Get the default model name."""
        with self._lock:
            return self._default_model

    @default_model.setter
    def default_model(self, name: str) -> None:
        """Set the default model name. Raises if model not registered."""
        with self._lock:
            if name not in self._models:
                raise ModelNotFoundError(name)
            self._default_model = name

    def is_registered(self, name: str) -> bool:
        """Check if a model is registered."""
        with self._lock:
            return name in self._models

    def remove(self, name: str) -> bool:
        """Remove a model. Returns True if it was removed."""
        with self._lock:
            if name not in self._models:
                return False
            del self._models[name]
            if self._default_model == name:
                # Set new default to first remaining model
                self._default_model = next(iter(self._models), None)
            return True
