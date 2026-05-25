"""Multi-model concurrent serving with GPU memory budgets, hot-swap, and LRU eviction.

Extends the existing MultiModelManager with:
- Per-model GPU memory budgets
- Hot-swap: load/unload models via API without restart
- LRU model eviction based on usage
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from loguru import logger

class ModelEntry:
    def __init__(self, name='', path='', total_layers=0):
        self.name = name
        self.path = path
        self.total_layers = total_layers

class ModelRegistry:
    def __init__(self, max_models=4):
        self._models = {}
        self.default_model = None
        self.max_models = max_models

        def register(self, name, path, total_layers):
            self._models[name] = ModelEntry(name, path, total_layers)
            return self._models[name]

        def get(self, name):
            return self._models.get(name)

        def remove(self, name):
            return self._models.pop(name, None)

        def list_models(self):
            return list(self._models.keys())


@dataclass
class ModelInstance:
    """A loaded model instance with its resource tracking."""
    name: str
    path: str
    model: Optional[object] = None  # The actual model (ModelPartitioner or similar)
    tokenizer: Optional[object] = None
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    request_count: int = 0
    memory_budget_gb: float = 0.0  # Max GPU memory for this model
    actual_memory_gb: float = 0.0  # Current GPU memory usage
    is_loading: bool = False  # True while model is being loaded


class ModelMemoryBudget:
    """Tracks per-model GPU memory budgets and enforces limits."""

    def __init__(self, total_gpu_memory_gb: float = 0.0):
        self.total_gpu_memory_gb = total_gpu_memory_gb
        self._budgets: dict[str, float] = {}  # model_name -> budget_gb
        self._usage: dict[str, float] = {}  # model_name -> current_usage_gb
        self._lock = threading.Lock()

    def set_budget(self, model_name: str, budget_gb: float) -> None:
        """Set the GPU memory budget for a model."""
        with self._lock:
            self._budgets[model_name] = budget_gb

    def get_budget(self, model_name: str) -> float | None:
        return self._budgets.get(model_name)

    def update_usage(self, model_name: str, usage_gb: float) -> None:
        """Update the current GPU memory usage for a model."""
        with self._lock:
            self._usage[model_name] = usage_gb

    def get_usage(self, model_name: str) -> float:
        return self._usage.get(model_name, 0.0)

    def total_allocated_gb(self) -> float:
        return sum(self._usage.values())

    def available_gb(self) -> float:
        if self.total_gpu_memory_gb <= 0:
            return float("inf")
        return max(0.0, self.total_gpu_memory_gb - self.total_allocated_gb())

    def can_fit(self, model_name: str, required_gb: float) -> bool:
        """Check if a model's required memory fits within its budget and available GPU."""
        budget = self._budgets.get(model_name, 0.0)
        current_usage = self._usage.get(model_name, 0.0)
        # Only enforce budget if explicitly set
        if budget > 0 and current_usage + required_gb > budget:
            return False
        return required_gb <= self.available_gb()

    def remove_model(self, model_name: str) -> None:
        with self._lock:
            self._usage.pop(model_name, None)
            self._budgets.pop(model_name, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_gpu_memory_gb": self.total_gpu_memory_gb,
                "total_allocated_gb": round(self.total_allocated_gb(), 2),
                "available_gb": round(self.available_gb(), 2),
                "budgets": {k: round(v, 2) for k, v in self._budgets.items()},
                "usage": {k: round(v, 2) for k, v in self._usage.items()},
            }


class ModelHotSwapManager:
    """Manages concurrent model serving with hot-swap and LRU eviction.

    Multiple models can be loaded simultaneously, each with its own GPU
    memory budget. When memory is full, the least recently used model
    is evicted to make room for new ones.
    """

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        total_gpu_memory_gb: float = 0.0,
        max_models: int = 4,
        on_load_model: Callable | None = None,
        on_unload_model: Callable | None = None,
    ):
        self.registry = model_registry or ModelRegistry(max_models=max_models)
        self.memory_budget = ModelMemoryBudget(total_gpu_memory_gb=total_gpu_memory_gb)
        self._max_models = max_models

        # Loaded models: name -> ModelInstance
        self._loaded: dict[str, ModelInstance] = {}
        self._lock = threading.RLock()

        # Callbacks for actual model loading/unloading
        self._on_load_model = on_load_model
        self._on_unload_model = on_unload_model

        # Stats
        self._total_loads = 0
        self._total_unloads = 0
        self._total_evictions = 0

    def set_callbacks(
        self,
        on_load_model: Callable,
        on_unload_model: Callable,
    ) -> None:
        """Set callbacks for actual model loading/unloading.

        on_load_model(name, path) -> (model, tokenizer, memory_gb)
        on_unload_model(name, model, tokenizer) -> None
        """
        self._on_load_model = on_load_model
        self._on_unload_model = on_unload_model

    def register_model(
        self,
        name: str,
        path: str,
        total_layers: int,
        memory_budget_gb: float = 0.0,
    ) -> ModelEntry:
        """Register a model in the registry with an optional memory budget.

        Note: max_models limits loaded models, not registered ones.
        """
        if memory_budget_gb > 0:
            self.memory_budget.set_budget(name, memory_budget_gb)

        entry = ModelEntry(name=name, path=path, total_layers=total_layers)
        self.registry._models[name] = entry
        if self.registry.default_model is None:
            self.registry.default_model = name
        return entry

    def load_model(self, name: str) -> bool:
        """Load a registered model into GPU memory.

        If memory is insufficient, evicts LRU models automatically.

        Returns:
            True if model was loaded successfully.
        """
        with self._lock:
            entry = self.registry.get(name)
            if entry is None:
                logger.warning(f"Model '{name}' not registered")
                return False

            # Already loaded? Just update access time
            if name in self._loaded:
                self._loaded[name].last_used_at = time.time()
                self._loaded[name].request_count += 1
                return True

            # Check if at max models
            if len(self._loaded) >= self._max_models:
                if not self._evict_lru():
                    logger.error(f"Cannot load '{name}': no evictable models and at capacity")
                    return False

            if self._on_load_model is None:
                logger.error("No load callback set")
                return False

            # Load the model
            instance = ModelInstance(name=name, path=entry.path, is_loading=True)
            self._loaded[name] = instance

            try:
                model, tokenizer, memory_gb = self._on_load_model(name, entry.path)
                instance.model = model
                instance.tokenizer = tokenizer
                instance.actual_memory_gb = memory_gb
                instance.is_loading = False
                instance.last_used_at = time.time()
                instance.request_count = 1

                self.memory_budget.update_usage(name, memory_gb)
                self._total_loads += 1

                logger.info(
                    f"Hot-swap: loaded model '{name}' ({memory_gb:.2f} GB, "
                    f"total: {self.memory_budget.total_allocated_gb():.2f} GB)"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load model '{name}': {e}")
                del self._loaded[name]
                return False

    def unload_model(self, name: str) -> bool:
        """Unload a model from GPU memory (without removing from registry).

        Returns:
            True if model was unloaded.
        """
        with self._lock:
            instance = self._loaded.pop(name, None)
            if instance is None:
                return False

            if self._on_unload_model and instance.model is not None:
                try:
                    self._on_unload_model(name, instance.model, instance.tokenizer)
                except Exception as e:
                    logger.error(f"Error unloading model '{name}': {e}")

            self.memory_budget.remove_model(name)
            self._total_unloads += 1

            logger.info(
                f"Hot-swap: unloaded model '{name}' "
                f"(total: {self.memory_budget.total_allocated_gb():.2f} GB)"
            )
            return True

    def remove_model(self, name: str) -> bool:
        """Fully remove a model (unload + unregister)."""
        self.unload_model(name)
        return self.registry.remove(name)

    def get_model(self, name: str) -> ModelInstance | None:
        """Get a loaded model instance."""
        instance = self._loaded.get(name)
        if instance:
            instance.last_used_at = time.time()
            instance.request_count += 1
        return instance

    def list_loaded_models(self) -> list[dict]:
        """List all currently loaded models with their stats."""
        with self._lock:
            return [
                {
                    "name": inst.name,
                    "path": inst.path,
                    "loaded_at": inst.loaded_at,
                    "last_used_at": inst.last_used_at,
                    "request_count": inst.request_count,
                    "memory_gb": round(inst.actual_memory_gb, 2),
                    "is_loading": inst.is_loading,
                }
                for inst in self._loaded.values()
            ]

    def _evict_lru(self) -> bool:
        """Evict the least recently used loaded model.

        Returns:
            True if a model was evicted.
        """
        lru_name = None
        lru_time = float("inf")
        for name, inst in self._loaded.items():
            if inst.is_loading:
                continue  # Don't evict a model being loaded
            if inst.last_used_at < lru_time:
                lru_time = inst.last_used_at
                lru_name = name

        if lru_name is None:
            return False

        logger.info(f"LRU eviction: unloading '{lru_name}' (last used {time.time() - lru_time:.0f}s ago)")
        self.unload_model(lru_name)
        self._total_evictions += 1
        return True

    def get_total_memory_usage(self) -> float:
        return self.memory_budget.total_allocated_gb()

    def stats(self) -> dict:
        with self._lock:
            return {
                "loaded_models": len(self._loaded),
                "max_models": self._max_models,
                "total_loads": self._total_loads,
                "total_unloads": self._total_unloads,
                "total_evictions": self._total_evictions,
                "memory": self.memory_budget.stats(),
                "loaded": self.list_loaded_models(),
            }
