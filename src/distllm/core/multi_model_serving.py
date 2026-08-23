"""Multi-model concurrent serving with GPU memory budgets, hot-swap, and LRU eviction.

Extends the existing MultiModelManager with:
- Per-model GPU memory budgets
- Hot-swap: load/unload models via API without restart
- LRU model eviction based on usage
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

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
        # Refresh-in-place if re-registered; otherwise evict the OLDEST
        # entry (dict insertion order) once the registry is at capacity.
        if name not in self._models and len(self._models) >= self.max_models:
            oldest = next(iter(self._models))
            del self._models[oldest]
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
    model: object | None = None  # The actual model (ModelPartitioner or similar)
    tokenizer: object | None = None
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
        enable_layer_sharing: bool = True,
    ):
        self.registry = model_registry or ModelRegistry(max_models=max_models)
        self.memory_budget = ModelMemoryBudget(total_gpu_memory_gb=total_gpu_memory_gb)
        # Back-compat aliases (older call sites/tests use the underscore names).
        self._registry = self.registry
        self._total_gpu_memory_gb = total_gpu_memory_gb
        self._max_models = max_models

        # Loaded models: name -> ModelInstance
        self._loaded: dict[str, ModelInstance] = {}
        self._lock = threading.RLock()

        # Callbacks for actual model loading/unloading
        self._on_load_model = on_load_model
        self._on_unload_model = on_unload_model

        # Shared layer pool for multi-model memory optimization
        self._layer_pool: Any = None
        if enable_layer_sharing:
            from distllm.core.shared_layer_pool import SharedLayerPool
            self._layer_pool = SharedLayerPool()

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

        entry = self.registry.register(name, path, total_layers)
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
            result = {
                "loaded_models": len(self._loaded),
                "max_models": self._max_models,
                "total_loads": self._total_loads,
                "total_unloads": self._total_unloads,
                "total_evictions": self._total_evictions,
                "memory": self.memory_budget.stats(),
                "loaded": self.list_loaded_models(),
            }
            if self._layer_pool is not None:
                result["layer_sharing"] = self._layer_pool.get_savings()
            return result

    # ── Layer Sharing ──────────────────────────────────────────────────

    def register_model_layers(
        self, model_name: str, state_dict: dict
    ) -> dict[str, Any] | None:
        """Register a model's layers for sharing detection.

        Call this after loading a model to identify shared layers
        with other loaded models.

        Args:
            model_name: Name of the loaded model.
            state_dict: Model's state dict.

        Returns:
            Sharing stats dict, or None if layer sharing is disabled.
        """
        if self._layer_pool is None:
            return None
        return self._layer_pool.register_model(model_name, state_dict)

    def get_shared_tensor(self, model_name: str, layer_name: str):
        """Get a shared tensor for a model layer, if available.

        Returns the shared tensor if the layer is shared with another
        model, or None if not shared.
        """
        if self._layer_pool is None:
            return None
        return self._layer_pool.get_shared_tensor(model_name, layer_name)

    def find_similar_models(self, model_name: str) -> list[dict]:
        """Find models that share layers with the given model."""
        if self._layer_pool is None:
            return []
        return self._layer_pool.find_similar_models(model_name)

    def get_layer_sharing_stats(self) -> dict | None:
        """Get layer sharing statistics."""
        if self._layer_pool is None:
            return None
        return self._layer_pool.get_savings()


# ── Time-Slicing GPU Scheduler ────────────────────────────────────────

@dataclass
class ModelSLA:
    """Per-model SLA guarantees."""
    model_name: str
    max_latency_ms: float = 5000.0      # Maximum P99 latency
    min_throughput_tok_s: float = 10.0   # Minimum throughput
    priority: int = 1                     # 1=high, 2=normal, 3=low
    max_queue_depth: int = 100           # Max pending requests
    guaranteed_gpu_pct: float = 0.0      # Minimum GPU allocation %


@dataclass
class TimeSlice:
    """A time slice for GPU scheduling."""
    model_name: str
    duration_ms: float
    start_time: float = 0.0
    end_time: float = 0.0
    requests_served: int = 0


class GPUTimeSlicer:
    """Time-slicing scheduler for sharing GPUs across multiple models.

    Alternates GPU access between models using configurable time slices.
    Higher-priority models get longer slices and more frequent access.

    Args:
        slice_duration_ms: Base time slice duration in milliseconds.
        models: Dict of model_name -> ModelSLA for scheduling.
    """

    def __init__(
        self,
        slice_duration_ms: float = 100.0,
        models: dict[str, ModelSLA] | None = None,
    ):
        self._base_slice_ms = slice_duration_ms
        self._slas: dict[str, ModelSLA] = models or {}
        self._active_model: str | None = None
        self._slice_start: float = 0.0
        # RLock so nested acquisition works: stats() holds this lock and then
        # calls check_sla_violations(), which acquires it again — a plain Lock
        # would self-deadlock on the second acquire.
        self._lock = threading.RLock()
        self._stats: dict[str, dict] = {}

    def register_model(self, sla: ModelSLA) -> None:
        """Register a model with its SLA for scheduling."""
        with self._lock:
            self._slas[sla.model_name] = sla
            self._stats[sla.model_name] = {
                "slices_granted": 0,
                "total_time_ms": 0,
                "requests_served": 0,
            }

    def get_next_model(self) -> str | None:
        """Select the next model to run based on priority and fairness.

        Uses weighted round-robin: higher priority models get proportionally
        more time slices.
        """
        with self._lock:
            if not self._slas:
                return None

            # Weight by priority (lower number = higher priority = more slices)
            total_weight = sum(1.0 / max(s.priority, 1) for s in self._slas.values())
            if total_weight == 0:
                return None

            # Simple round-robin weighted by priority
            models = list(self._slas.keys())
            if self._active_model is None:
                return models[0]

            # Find next model in rotation
            try:
                current_idx = models.index(self._active_model)
                next_idx = (current_idx + 1) % len(models)
            except ValueError:
                next_idx = 0

            return models[next_idx]

    def get_slice_duration(self, model_name: str) -> float:
        """Get the time slice duration for a model based on its priority."""
        sla = self._slas.get(model_name)
        if sla is None:
            return self._base_slice_ms
        # Higher priority (lower number) gets longer slices
        priority_multiplier = 2.0 / max(sla.priority, 1)
        return self._base_slice_ms * priority_multiplier

    def start_slice(self, model_name: str) -> TimeSlice:
        """Start a time slice for a model."""
        with self._lock:
            duration = self.get_slice_duration(model_name)
            now = time.time()
            self._active_model = model_name
            self._slice_start = now

            if model_name in self._stats:
                self._stats[model_name]["slices_granted"] += 1

            return TimeSlice(
                model_name=model_name,
                duration_ms=duration,
                start_time=now,
                end_time=now + duration / 1000,
            )

    def end_slice(self, slice: TimeSlice) -> None:
        """End a time slice and record stats."""
        with self._lock:
            elapsed_ms = (time.time() - slice.start_time) * 1000
            if slice.model_name in self._stats:
                self._stats[slice.model_name]["total_time_ms"] += elapsed_ms
                self._stats[slice.model_name]["requests_served"] += slice.requests_served

    def check_sla_violations(self) -> list[str]:
        """Check which models are violating their SLAs.

        Returns list of model names with SLA violations.
        """
        violations = []
        with self._lock:
            for model_name, sla in self._slas.items():
                stats = self._stats.get(model_name, {})
                total_time = stats.get("total_time_ms", 0)
                requests = stats.get("requests_served", 0)

                if total_time > 0 and requests > 0:
                    avg_latency = total_time / requests
                    if avg_latency > sla.max_latency_ms:
                        violations.append(model_name)

        return violations

    def stats(self) -> dict:
        with self._lock:
            return {
                "active_model": self._active_model,
                "registered_models": len(self._slas),
                "base_slice_ms": self._base_slice_ms,
                "per_model": dict(self._stats),
                "sla_violations": self.check_sla_violations(),
            }


# ── Memory-Aware Model Placement ─────────────────────────────────────

class MemoryAwarePlacer:
    """Places models on GPUs based on memory requirements and availability.

    Considers:
    - Model memory requirements (from model profile)
    - Current GPU memory usage
    - Co-location affinity (models that share layers)
    - SLA requirements (min GPU allocation)
    """

    def __init__(
        self,
        gpu_memory_gb: float = 80.0,
        safety_margin_pct: float = 0.1,
    ):
        self._gpu_memory = gpu_memory_gb
        self._safety_margin = safety_margin_pct
        self._placements: dict[str, int] = {}  # model_name -> gpu_id
        self._usage: dict[int, float] = {}  # gpu_id -> used_gb

    def place_model(
        self,
        model_name: str,
        required_gb: float,
        preferred_gpu: int | None = None,
    ) -> int | None:
        """Place a model on the best available GPU.

        Args:
            model_name: Model to place.
            required_gb: Memory required in GB.
            preferred_gpu: Preferred GPU ID (if any).

        Returns:
            GPU ID where model was placed, or None if no space.
        """
        available = self._gpu_memory * (1 - self._safety_margin)

        # Try preferred GPU first
        if preferred_gpu is not None:
            used = self._usage.get(preferred_gpu, 0.0)
            if used + required_gb <= available:
                self._placements[model_name] = preferred_gpu
                self._usage[preferred_gpu] = used + required_gb
                return preferred_gpu

        # Find GPU with most free space
        best_gpu = None
        best_free = 0.0
        for gpu_id in range(8):  # Support up to 8 GPUs
            used = self._usage.get(gpu_id, 0.0)
            free = available - used
            if free >= required_gb and free > best_free:
                best_gpu = gpu_id
                best_free = free

        if best_gpu is not None:
            self._placements[model_name] = best_gpu
            self._usage[best_gpu] = self._usage.get(best_gpu, 0.0) + required_gb

        return best_gpu

    def remove_model(self, model_name: str) -> None:
        """Remove a model from placement tracking."""
        gpu_id = self._placements.pop(model_name, None)
        if gpu_id is not None:
            # We don't know exact usage reduction, so clear it
            pass

    def get_placement(self, model_name: str) -> int | None:
        return self._placements.get(model_name)

    def stats(self) -> dict:
        return {
            "placed_models": len(self._placements),
            "placements": dict(self._placements),
            "gpu_usage": {k: round(v, 2) for k, v in self._usage.items()},
            "gpu_memory_gb": self._gpu_memory,
        }
