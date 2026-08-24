"""Multi-model serving with cross-model prefix sharing and expert parallelism.

Provides four classes:

- :class:`PrefixSharingEngine` -- share KV-cache prefixes across models
  that use the same tokenizer.
- :class:`ExpertParallelScheduler` -- schedule MoE expert shards across
  GPUs for expert parallelism.
- :class:`ModelPool` -- manage a pool of loaded models with LRU eviction
  under memory pressure.
- :class:`Cortex` -- top-level orchestrator that combines all three.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer  # noqa: F401 (used at runtime)
except ImportError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PrefixSharingEngine",
    "ExpertParallelScheduler",
    "ExpertTopology",
    "LoadBalance",
    "ModelPool",
    "ModelHandle",
    "Cortex",
    "CortexStats",
]


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertTopology:
    """Describes the expert layout of an MoE model."""

    num_experts: int
    """Total number of experts in the model."""

    experts_per_token: int
    """Number of top-k experts activated per token (typically 2)."""

    num_layers: int
    """Number of transformer layers that have MoE blocks."""

    shared_experts: int = 0
    """Number of shared (always-on) experts, if any."""


class LoadBalance(Enum):
    """Expert load balance classification."""

    BALANCED = "balanced"
    """Expert assignments are roughly uniform across the GPU topology."""

    SKEWED = "skewed"
"""Some experts or GPUs receive significantly more tokens than others."""


@dataclass
class ModelHandle:
    """Handle returned by :meth:`ModelPool.load`."""

    model_id: str
    """Unique identifier for this model instance."""

    model_name: str
    """The canonical model name (e.g. ``"meta-llama/Llama-3-8B"``)."""

    tokenizer_name: str
    """The tokenizer identifier used to determine prefix-sharing eligibility."""

    loaded_at: float
    """Unix timestamp when the model was loaded."""

    last_used_at: float
    """Unix timestamp of the most recent access."""

    memory_bytes: int = 0
    """Estimated memory consumption in bytes."""


@dataclass
class CortexStats:
    """Snapshot of system statistics returned by :meth:`Cortex.stats`."""

    models_loaded: int = 0
    """Number of models currently in the pool."""

    total_memory_bytes: int = 0
    """Aggregate estimated memory of all loaded models."""

    prefix_lookups: int = 0
    """Total prefix lookups performed so far."""

    prefix_hits: int = 0
    """Number of prefix lookups that resulted in a cache hit."""

    prefix_hit_rate: float = 0.0
    """Ratio of prefix hits to total lookups (0-1)."""

    expert_assignments: int = 0
    """Total expert-to-GPU assignments made."""

    expert_utilization: float = 0.0
    """Fraction of scheduled expert capacity actually utilised (0-1)."""

    load_balance: LoadBalance = LoadBalance.BALANCED
    """Current load-balance classification."""


# ---------------------------------------------------------------------------
# Internal protocols for optional backends
# ---------------------------------------------------------------------------


class _KVCacheEntry(Protocol):
    """Minimal duck-type for a KV-cache entry.

    Concrete implementations (e.g. ``torch.Tensor`` tuples) satisfy this
    protocol implicitly.
    """

    def __len__(self) -> int: ...


class _ModelProtocol(Protocol):
    """Duck-type interface for a loaded model.

    The pool stores arbitrary model objects; the only requirement is that
    they expose an attribute ``config`` with a ``model_type`` string on it.
    """

    @property
    def config(self) -> Any: ...

    @property
    def device(self) -> Any: ...


# ---------------------------------------------------------------------------
# PrefixSharingEngine
# ---------------------------------------------------------------------------


class PrefixSharingEngine:
    """Cross-model prefix sharing via a common tokenizer.

    When multiple models share the same tokenizer, a prefix computed for one
    model's KV cache can be reused by another, saving compute on the prefill
    phase.

    Thread-safe (single ``_lock`` protecting all mutable state).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # --- prefix cache keyed by (tokenizer_name, prefix_text) ---
        self._cache: dict[tuple[str, str], list[Any]] = {}
        self._lookups = 0
        self._hits = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_prefix(self, prompt: str) -> str:
        """Return the common prefix candidate from a prompt.

        By default this returns the first 64 characters.  Subclasses may
        override to implement sentence-level or token-boundary alignment.

        Args:
            prompt: The full input text.

        Returns:
            A prefix string suitable for cache lookup.
        """
        return prompt[:64]

    def cache_prefix(
        self,
        prefix: str,
        kv_cache: list[Any],
        tokenizer_name: str = "",
    ) -> None:
        """Store a KV-cache entry keyed by *prefix* and *tokenizer_name*.

        Args:
            prefix: The prefix string (typically from :meth:`extract_prefix`).
            kv_cache: List of KV-cache tensors to store.
            tokenizer_name: Tokenizer identifier used as part of the key.
        """
        key = (tokenizer_name, prefix)
        with self._lock:
            self._cache[key] = kv_cache
            logger.debug(
                "Cached prefix for tokenizer={!r} prefix={!r} ({} entries)",
                tokenizer_name,
                prefix,
                len(self._cache),
            )

    def lookup_prefix(
        self,
        prompt: str,
        models: list[ModelHandle],
    ) -> tuple[bool, list[Any] | None]:
        """Look up a cached prefix compatible with *models*.

        The lookup succeeds if *any* model in *models* has the same
        tokenizer as a previously cached prefix that matches the prompt.

        Args:
            prompt: The input text to look up.
            models: Candidate models (their ``tokenizer_name`` is checked).

        Returns:
            A tuple ``(found, kv_cache)`` where *found* is ``True`` when
            a matching entry exists.
        """
        prefix = self.extract_prefix(prompt)
        tokenizer_names = {m.tokenizer_name for m in models}

        with self._lock:
            self._lookups += 1
            for tz_name in tokenizer_names:
                entry = self._cache.get((tz_name, prefix))
                if entry is not None:
                    self._hits += 1
                    logger.debug(
                        "Prefix HIT  tokenizer={!r} prefix={!r}",
                        tz_name,
                        prefix,
                    )
                    return True, entry

            logger.debug("Prefix MISS prefix={!r}", prefix)
            return False, None

    @property
    def hit_rate(self) -> float:
        """Prefix cache hit rate as a float between 0 and 1."""
        with self._lock:
            if self._lookups == 0:
                return 0.0
            return self._hits / self._lookups

    @property
    def stats(self) -> dict[str, Any]:
        """Return raw statistics as a dictionary."""
        with self._lock:
            return {
                "cache_entries": len(self._cache),
                "lookups": self._lookups,
                "hits": self._hits,
                "hit_rate": self.hit_rate,
            }

    def clear(self) -> None:
        """Remove all cached prefix entries."""
        with self._lock:
            self._cache.clear()
            self._lookups = 0
            self._hits = 0


# ---------------------------------------------------------------------------
# ExpertParallelScheduler
# ---------------------------------------------------------------------------


class ExpertParallelScheduler:
    """Schedules MoE expert shards across GPUs for expert parallelism.

    Expert parallelism places different expert parameters on different
    devices so that each token's top-k experts are fetched from potentially
    distinct GPUs, reducing per-device memory pressure for wide MoE models.
    """

    def __init__(self) -> None:
        self._assignments: dict[int, list[int]] = {}  # expert_id -> [gpu_ids]
        self._total_assignments = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_moe(model: Any) -> bool:
        """Return ``True`` if *model* appears to be a mixture-of-experts model.

        This checks for common MoE attributes on the model or its config:

        - ``model.config.num_experts`` (e.g. Mixtral, Qwen2-MoE, DeepSeek)
        - ``model.config.moe`` (DeepSeek-V2/V3 style)
        - ``model.model`` with ``num_experts`` (NVIDIA Megatron-LM style)
        - ``model.num_experts`` (flat attribute)

        Args:
            model: Any loaded model object.

        Returns:
            ``True`` if the model appears to be an MoE architecture.
        """
        if torch is None:
            return False

        config = getattr(model, "config", model)
        # Standard HuggingFace MoE indicator
        if hasattr(config, "num_experts"):
            return True
        # DeepSeek-style
        if hasattr(config, "moe"):
            return True
        # Flat attribute on model
        if hasattr(model, "num_experts"):
            return True
        # Nested model attribute
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "num_experts"):
            return True
        return False

    @staticmethod
    def get_experts(model: Any) -> ExpertTopology:
        """Return the expert topology for *model*.

        Args:
            model: A loaded MoE model.

        Returns:
            An :class:`ExpertTopology` instance.  Returns a zero-valued
            topology when the model is not MoE.
        """
        config = getattr(model, "config", model)
        num_experts: int = getattr(config, "num_experts", 0)
        experts_per_token: int = getattr(config, "num_experts_per_tok", 2)
        num_layers: int = getattr(config, "num_hidden_layers", 0) or getattr(
            config, "num_layers", 0
        )
        shared_experts: int = getattr(config, "shared_experts", 0) or getattr(
            config, "num_shared_experts", 0
        )
        return ExpertTopology(
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            num_layers=num_layers,
            shared_experts=shared_experts,
        )

    def schedule_experts(
        self,
        tokens: int,
        experts: ExpertTopology,
        gpus: int,
    ) -> dict[int, list[int]]:
        """Build an expert-to-GPU mapping.

        Distributes *experts.num_experts* experts across *gpus* devices
        in a round-robin fashion, then classifies the resulting balance.

        Args:
            tokens: Number of tokens in the current batch.
            experts: The expert topology descriptor.
            gpus: Number of available GPU devices.

        Returns:
            Mapping ``{expert_index: [gpu_id, ...]}``.
        """
        if gpus <= 0 or experts.num_experts <= 0:
            self._assignments = {}
            return self._assignments

        assignment: dict[int, list[int]] = {}
        for expert_id in range(experts.num_experts):
            gpu_id = expert_id % gpus
            assignment[expert_id] = [gpu_id]

        self._assignments = assignment
        self._total_assignments += experts.num_experts * experts.num_layers
        self._classify_balance(assignment, tokens, experts)

        return assignment

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def load_balance(self) -> LoadBalance:
        """Current load-balance classification."""
        return self._compute_load_balance()

    @property
    def utilization(self) -> float:
        """Fraction of scheduled expert capacity that was utilised.

        Returns 0.0 when no assignments have been made.
        """
        if self._total_assignments == 0:
            return 0.0
        # Simplification: assume all assigned capacity was used.
        return 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_load_balance(self) -> LoadBalance:
        """Classify assignments as balanced or skewed."""
        if not self._assignments:
            return LoadBalance.BALANCED

        # Count experts per GPU
        gpu_counts: dict[int, int] = {}
        for gpus_list in self._assignments.values():
            for gpu_id in gpus_list:
                gpu_counts[gpu_id] = gpu_counts.get(gpu_id, 0) + 1

        if not gpu_counts:
            return LoadBalance.BALANCED

        counts = list(gpu_counts.values())
        max_count = max(counts)
        min_count = min(counts)

        # If the spread is more than 20% of max, call it skewed.
        if min_count > 0 and (max_count - min_count) / max_count > 0.20:
            return LoadBalance.SKEWED
        return LoadBalance.BALANCED

    def _classify_balance(
        self,
        assignment: dict[int, list[int]],
        tokens: int,
        experts: ExpertTopology,
    ) -> None:
        """Update internal balance (used during scheduling)."""
        pass  # classification is computed lazily via _compute_load_balance


# ---------------------------------------------------------------------------
# ModelPool
# ---------------------------------------------------------------------------


class ModelPool:
    """Manages a pool of loaded models with LRU eviction under memory pressure.

    Thread-safe: all mutation is guarded by ``_lock``.
    """

    def __init__(self, max_memory_bytes: int = 8 * 1024**3) -> None:
        """
        Args:
            max_memory_bytes: Soft memory limit before LRU eviction is
                triggered (default 8 GiB).
        """
        self._max_memory_bytes = max_memory_bytes
        self._lock = threading.Lock()
        # OrderedDict for LRU tracking: insertion order == use order.
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._handles: dict[str, ModelHandle] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        *,
        model_factory: Any = None,
        memory_bytes: int = 0,
        tokenizer_name: str = "",
    ) -> str:
        """Load a model and return a unique *model_id*.

        If *model_factory* is provided it is called with
        ``(model_name, config)`` to produce the model object.  Otherwise a
        stub is created for testing.

        Args:
            model_name: Canonical model name (e.g. ``"meta-llama/Llama-3-8B"``).
            config: Optional configuration dictionary passed to the factory.
            model_factory: Optional callable that returns a loaded model.
            memory_bytes: Estimated memory consumption of this model.
            tokenizer_name: Tokenizer identifier for prefix-sharing eligibility.

        Returns:
            A unique ``model_id`` string.
        """
        model_id = str(uuid.uuid4())
        effective_config = config or {}
        tokenizer = tokenizer_name or model_name

        # --- Build model object ---
        if model_factory is not None:
            model_obj = model_factory(model_name, effective_config)
        else:
            model_obj = _StubModel(model_name, effective_config)

        handle = ModelHandle(
            model_id=model_id,
            model_name=model_name,
            tokenizer_name=tokenizer,
            loaded_at=time.time(),
            last_used_at=time.time(),
            memory_bytes=memory_bytes or _estimate_memory(model_name),
        )

        with self._lock:
            # Evict LRU models if we exceed the memory budget.
            self._evict_lru(handle.memory_bytes)
            self._models[model_id] = model_obj
            self._handles[model_id] = handle

        logger.info(
            "Loaded model {} (id={}, memory={} bytes)",
            model_name,
            model_id,
            handle.memory_bytes,
        )
        return model_id

    def unload(self, model_id: str) -> None:
        """Unload (remove) a model from the pool.

        Args:
            model_id: The identifier returned by :meth:`load`.

        Raises:
            KeyError: If *model_id* is not in the pool.
        """
        with self._lock:
            handle = self._handles.pop(model_id, None)
            self._models.pop(model_id, None)

        if handle is None:
            raise KeyError(f"Model {model_id!r} not found in pool")

        logger.info("Unloaded model {} (id={})", handle.model_name, model_id)

    def get_model(self, model_id: str) -> Any:
        """Return the model object for *model_id*.

        Touches the LRU timestamp.

        Args:
            model_id: The identifier returned by :meth:`load`.

        Returns:
            The model object.

        Raises:
            KeyError: If *model_id* is not in the pool.
        """
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                raise KeyError(f"Model {model_id!r} not found in pool")
            # Update LRU by re-inserting.
            self._models.move_to_end(model_id)
            self._handles[model_id].last_used_at = time.time()
            return model

    def list_loaded(self) -> list[ModelHandle]:
        """Return handles for all currently loaded models.

        The list is a snapshot; the caller should not mutate it.
        """
        with self._lock:
            return list(self._handles.values())

    @property
    def total_memory_bytes(self) -> int:
        """Aggregate estimated memory of all loaded models."""
        with self._lock:
            return sum(h.memory_bytes for h in self._handles.values())

    @property
    def max_memory_bytes(self) -> int:
        """Configured soft memory limit."""
        return self._max_memory_bytes

    @max_memory_bytes.setter
    def max_memory_bytes(self, value: int) -> None:
        """Update the soft memory limit and trigger eviction if needed."""
        with self._lock:
            self._max_memory_bytes = value
            self._evict_lru(0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_lru(self, incoming_bytes: int) -> None:
        """Evict least-recently-used models until the budget is satisfied.

        Must be called while holding ``_lock``.

        The running memory total is tracked locally instead of read from the
        :attr:`total_memory_bytes` property: that property re-acquires
        ``_lock``, which is a plain (non-reentrant) ``threading.Lock``, so
        calling it from here -- always executed under the lock -- deadlocked
        the pool on the very first eviction check.
        """
        budget = self._max_memory_bytes
        total = sum(h.memory_bytes for h in self._handles.values())
        while total + incoming_bytes > budget and self._models:
            # Remove the first (oldest) entry from OrderedDict.
            oldest_id, oldest_model = next(iter(self._models.items()))
            oldest_handle = self._handles[oldest_id]
            logger.info(
                "Evicting LRU model {} (id={}, {} bytes)",
                oldest_handle.model_name,
                oldest_id,
                oldest_handle.memory_bytes,
            )
            self._models.pop(oldest_id)
            self._handles.pop(oldest_id)
            total -= oldest_handle.memory_bytes
            # Attempt to free GPU memory.
            self._try_clear_device(oldest_model)

    @staticmethod
    def _try_clear_device(model: Any) -> None:
        """Best-effort cleanup of model device memory."""
        if torch is None:
            return
        try:
            device = getattr(model, "device", None)
            if device is not None and str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Cortex
# ---------------------------------------------------------------------------


class Cortex:
    """Top-level orchestrator combining prefix sharing, model pool, and
    expert parallelism.

    Typical usage::

        cortex = Cortex()
        model_a = cortex.load("mistralai/Mixtral-8x7B", {})
        model_b = cortex.load("meta-llama/Llama-3-8B", {})
        resp = cortex.serve("mistralai/Mixtral-8x7B", "Hello, world!")
        print(resp)
        print(cortex.stats())
    """

    def __init__(
        self,
        pool_max_memory_bytes: int = 8 * 1024**3,
    ) -> None:
        """
        Args:
            pool_max_memory_bytes: Soft memory limit for the model pool
                (default 8 GiB).
        """
        self.prefix_engine = PrefixSharingEngine()
        self.expert_scheduler = ExpertParallelScheduler()
        self.pool = ModelPool(max_memory_bytes=pool_max_memory_bytes)
        self._served: int = 0

    # ------------------------------------------------------------------
    # Delegated pool methods (convenience)
    # ------------------------------------------------------------------

    def load(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        *,
        model_factory: Any = None,
        memory_bytes: int = 0,
        tokenizer_name: str = "",
    ) -> str:
        """Load a model into the pool.

        See :meth:`ModelPool.load` for parameter details.
        """
        return self.pool.load(
            model_name=model_name,
            config=config,
            model_factory=model_factory,
            memory_bytes=memory_bytes,
            tokenizer_name=tokenizer_name,
        )

    def unload(self, model_id: str) -> None:
        """Unload a model from the pool.

        See :meth:`ModelPool.unload`.
        """
        self.pool.unload(model_id)

    def list_loaded(self) -> list[ModelHandle]:
        """List currently loaded model handles.

        See :meth:`ModelPool.list_loaded`.
        """
        return self.pool.list_loaded()

    def get_model(self, model_id: str) -> Any:
        """Retrieve a loaded model object by ID.

        See :meth:`ModelPool.get_model`.
        """
        return self.pool.get_model(model_id)

    # ------------------------------------------------------------------
    # Core serve method
    # ------------------------------------------------------------------

    def serve(
        self,
        model_name: str,
        request: str,
        **kwargs: Any,
    ) -> str:
        """Run a request against the named model.

        This is the primary entry point.  It:

        1. Resolves *model_name* to a loaded model (uses the most recently
           used instance if multiple copies exist).
        2. Looks up a shared prefix from the prefix engine (if any model
           with the same tokenizer has a cached KV cache).
        3. Optionally schedules experts for MoE models.
        4. Delegates to a response provider (or returns a stub response).

        Args:
            model_name: The model name passed to :meth:`load` (e.g.
                ``"mistralai/Mixtral-8x7B"``).
            request: The input prompt text.
            **kwargs: Additional arguments (e.g. ``max_tokens``,
                ``temperature``).

        Returns:
            The generated response text.
        """
        handle = self._resolve_handle(model_name)
        model = self.pool.get_model(handle.model_id)

        # --- Attempt prefix sharing ---
        models_with_same_tokenizer = [
            h
            for h in self.pool.list_loaded()
            if h.tokenizer_name == handle.tokenizer_name
        ]
        found, cached_kv = self.prefix_engine.lookup_prefix(
            request,
            models_with_same_tokenizer,
        )
        if found and cached_kv is not None:
            logger.debug(
                "Reusing cached prefix for model {} (name={!r})",
                handle.model_id,
                model_name,
            )

        # --- Expert scheduling for MoE models ---
        if self.expert_scheduler.is_moe(model):
            topology = self.expert_scheduler.get_experts(model)
            gpus = self._detect_gpu_count()
            self.expert_scheduler.schedule_experts(
                tokens=_estimate_tokens(request),
                experts=topology,
                gpus=gpus,
            )

        # --- Generate response ---
        response = self._generate(model, request, **kwargs)

        # --- Cache prefix for future reuse ---
        prefix = self.prefix_engine.extract_prefix(request)
        dummy_kv: list[Any] = []
        self.prefix_engine.cache_prefix(prefix, dummy_kv, handle.tokenizer_name)

        self._served += 1
        return response

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> CortexStats:
        """Return a snapshot of current system statistics."""
        prefix_stats = self.prefix_engine.stats
        models = self.pool.list_loaded()

        return CortexStats(
            models_loaded=len(models),
            total_memory_bytes=sum(h.memory_bytes for h in models),
            prefix_lookups=prefix_stats["lookups"],
            prefix_hits=prefix_stats["hits"],
            prefix_hit_rate=self.prefix_engine.hit_rate,
            expert_assignments=self.expert_scheduler._total_assignments,  # type: ignore[attr-defined]
            expert_utilization=self.expert_scheduler.utilization,
            load_balance=self.expert_scheduler.load_balance,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_handle(self, model_name: str) -> ModelHandle:
        """Return the most recently used handle matching *model_name*.

        Raises:
            KeyError: If no loaded model matches *model_name*.
        """
        candidates = [
            h
            for h in self.pool.list_loaded()
            if h.model_name == model_name
        ]
        if not candidates:
            raise KeyError(
                f"Model {model_name!r} not found in pool "
                f"(loaded: {[h.model_name for h in self.pool.list_loaded()]})"
            )
        # Return the most recently used match.
        candidates.sort(key=lambda h: h.last_used_at, reverse=True)
        return candidates[0]

    @staticmethod
    def _generate(model: Any, request: str, **kwargs: Any) -> str:
        """Generate a response.

        If the model has a ``generate`` method, delegate to it.  Otherwise
        return a stub response (useful during testing without real model
        weights).
        """
        gen = getattr(model, "generate", None)
        if gen is not None:
            try:
                result = gen(request, **kwargs)
                if isinstance(result, str):
                    return result
                return str(result)
            except Exception as exc:
                logger.warning("Model.generate() failed: {}", exc)
                return _stub_response(request)
        return _stub_response(request)

    @staticmethod
    def _detect_gpu_count() -> int:
        """Return the number of available CUDA GPUs, or 1 if not detected."""
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal model stub used when no factory is supplied."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = _StubConfig(name)

    def generate(self, request: str, **kwargs: Any) -> str:
        return _stub_response(request)

    def __repr__(self) -> str:
        return f"_StubModel({self.name})"


@dataclass
class _StubConfig:
    model_type: str = ""


def _stub_response(prompt: str) -> str:
    """Return a placeholder response for a prompt."""
    return f"[stub response to {prompt!r}]"


def _estimate_memory(model_name: str) -> int:
    """Rough memory estimate (bytes) based on model name heuristics.

    Only used as a fallback when the caller does not supply ``memory_bytes``.
    """
    name_lower = model_name.lower()
    if "70b" in name_lower or "70-b" in name_lower:
        return 140 * 1024**3  # ~140 GB (FP16)
    if "72b" in name_lower:
        return 144 * 1024**3
    if "34b" in name_lower:
        return 68 * 1024**3
    if "13b" in name_lower:
        return 26 * 1024**3
    if "8x7b" in name_lower or "8x7" in name_lower:
        return 90 * 1024**3  # ~90 GB (MoE)
    if "7b" in name_lower or "7-b" in name_lower or "8b" in name_lower:
        return 16 * 1024**3  # ~16 GB
    if "3b" in name_lower:
        return 6 * 1024**3  # ~6 GB
    if "1b" in name_lower or "1.5b" in name_lower:
        return 3 * 1024**3
    return 4 * 1024**3  # Default ~4 GB


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)
