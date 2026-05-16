"""LoRA adapter management for multi-tenant serving.

S-LoRA style: shared base model + per-request adapter weights in one batch.
- VRAM-resident adapter pool with LRU eviction
- Adapter warmup API (pre-load before traffic)
- Adapter ranking for multi-tenant priority
- Per-sequence adapter routing in batched inference
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from loguru import logger


@dataclass
class AdapterInfo:
    """Metadata about a loaded adapter."""
    adapter_id: str
    path: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    vram_bytes: int = 0
    rank: int = 0  # Multi-tenant priority (higher = more important)
    tenant_id: str = ""


class AdapterPool:
    """VRAM-resident LoRA adapter pool with LRU eviction.

    Keeps frequently used adapters in VRAM for fast switching.
    When VRAM is full, evicts least-recently-used adapters.
    """

    def __init__(self, max_vram_bytes: int = 0):
        self.max_vram_bytes = max_vram_bytes or self._detect_vram()
        # OrderedDict for LRU: adapter_id -> AdapterInfo
        self._pool: OrderedDict[str, AdapterInfo] = OrderedDict()
        self._active_adapter: Optional[str] = None
        self._lock = threading.Lock()
        self._total_vram = 0

    def _detect_vram(self) -> int:
        """Detect available GPU VRAM."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return int(props.total_memory * 0.3)  # Use 30% of VRAM for adapters
        except Exception:
            pass
        return 4 * 1024 ** 3  # Default: 4GB

    def add_adapter(self, adapter_id: str, path: str, vram_bytes: int = 0, rank: int = 0, tenant_id: str = "") -> None:
        """Register an adapter in the pool.

        Args:
            adapter_id: Unique adapter identifier.
            path: Path to adapter weights.
            vram_bytes: Estimated VRAM usage.
            rank: Multi-tenant priority rank.
            tenant_id: Tenant identifier for isolation.
        """
        with self._lock:
            if adapter_id in self._pool:
                # Update existing
                info = self._pool[adapter_id]
                info.path = path
                info.last_used = time.time()
                self._pool.move_to_end(adapter_id)
                return

            info = AdapterInfo(
                adapter_id=adapter_id,
                path=path,
                vram_bytes=vram_bytes,
                rank=rank,
                tenant_id=tenant_id,
            )
            self._pool[adapter_id] = info
            self._total_vram += vram_bytes

            # Evict LRU if over capacity
            self._evict_lru()

    def get_adapter(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get adapter info and mark as recently used."""
        with self._lock:
            if adapter_id in self._pool:
                info = self._pool[adapter_id]
                info.last_used = time.time()
                info.use_count += 1
                self._pool.move_to_end(adapter_id)
                return info
            return None

    def remove_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter from the pool."""
        with self._lock:
            if adapter_id in self._pool:
                info = self._pool.pop(adapter_id)
                self._total_vram -= info.vram_bytes
                return True
            return False

    def _evict_lru(self) -> None:
        """Evict least-recently-used adapters until under VRAM limit."""
        while self._total_vram > self.max_vram_bytes and self._pool:
            # Don't evict the active adapter
            lru_id, lru_info = self._pool.popitem(last=False)
            if lru_id == self._active_adapter:
                self._pool[lru_id] = lru_info
                break
            self._total_vram -= lru_info.vram_bytes
            logger.info(f"Evicted LRU adapter '{lru_id}' ({lru_info.vram_bytes / 1e6:.1f}MB)")

    def set_active(self, adapter_id: Optional[str]) -> None:
        """Set the active adapter."""
        with self._lock:
            if adapter_id is not None and adapter_id not in self._pool:
                raise KeyError(f"Adapter '{adapter_id}' not in pool")
            self._active_adapter = adapter_id

    @property
    def active_adapter(self) -> Optional[str]:
        return self._active_adapter

    def list_adapters(self) -> List[str]:
        return list(self._pool.keys())

    def get_stats(self) -> dict:
        return {
            "pool_size": len(self._pool),
            "total_vram_bytes": self._total_vram,
            "max_vram_bytes": self.max_vram_bytes,
            "vram_usage_pct": round(self._total_vram / max(self.max_vram_bytes, 1) * 100, 1),
            "active_adapter": self._active_adapter,
        }


class AdapterManager:
    """S-LoRA style LoRA adapter manager with per-request adapter routing.

    Maintains a shared base model and routes each request to its specified
    adapter within a batch. Adapters are loaded once and kept in a VRAM pool
    with LRU eviction.

    Usage:
        mgr = AdapterManager()
        mgr.set_base_model(model, tokenizer)
        mgr.warmup_adapters({"adapter_1": "/path/to/adapter1"})
        # During batch inference:
        batch_tags = {"adapter_ids": ["adapter_1", "adapter_2", None]}
    """

    def __init__(
        self,
        base_model: Optional[object] = None,
        tokenizer: Optional[object] = None,
        max_vram_bytes: int = 0,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self._pool = AdapterPool(max_vram_bytes=max_vram_bytes)
        # Loaded PEFT models: adapter_id -> PeftModel
        self._loaded_models: Dict[str, object] = {}
        self._lock = threading.Lock()

    def set_base_model(self, base_model: object, tokenizer: Optional[object] = None) -> None:
        """Set the base model after it has been loaded."""
        self.base_model = base_model
        if tokenizer:
            self.tokenizer = tokenizer

    def load_adapter(
        self,
        adapter_id: str,
        adapter_path: str,
        rank: int = 0,
        tenant_id: str = "",
    ) -> None:
        """Load a LoRA adapter from path or HuggingFace hub.

        Args:
            adapter_id: Unique adapter identifier.
            adapter_path: Path to adapter weights or HF repo.
            rank: Multi-tenant priority (higher = more important to keep).
            tenant_id: Tenant identifier for isolation.
        """
        from peft import PeftModel

        if self.base_model is None:
            raise RuntimeError("No base model loaded. Call set_base_model() first.")

        logger.info(f"Loading LoRA adapter '{adapter_id}' from {adapter_path}")

        # Load the adapter using PEFT
        model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            adapter_name=adapter_id,
        )
        self._loaded_models[adapter_id] = model

        # Estimate VRAM usage (rough: count trainable parameters * 2 bytes)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        vram_bytes = trainable_params * 2  # fp16

        # Register in the pool
        self._pool.add_adapter(
            adapter_id,
            adapter_path,
            vram_bytes=vram_bytes,
            rank=rank,
            tenant_id=tenant_id,
        )
        logger.info(f"Adapter '{adapter_id}' loaded ({trainable_params / 1e6:.1f}M params)")

    def set_active(self, adapter_id: Optional[str]) -> None:
        """Switch active adapter. None = base model only."""
        self._pool.set_active(adapter_id)
        if adapter_id is not None and adapter_id in self._loaded_models:
            self._loaded_models[adapter_id].set_adapter(adapter_id)
            info = self._pool.get_adapter(adapter_id)
            if info:
                info.use_count += 1
        logger.info(f"Switched to adapter '{adapter_id}'" if adapter_id else "Switched to base model")

    def warmup_adapters(self, adapters: Dict[str, str], rank_map: Optional[Dict[str, int]] = None, tenant_map: Optional[Dict[str, str]] = None) -> List[str]:
        """Pre-load adapters before traffic arrives.

        Args:
            adapters: Dict of adapter_id -> adapter_path.
            rank_map: Optional dict of adapter_id -> rank.
            tenant_map: Optional dict of adapter_id -> tenant_id.

        Returns:
            List of successfully loaded adapter IDs.
        """
        loaded = []
        for adapter_id, path in adapters.items():
            try:
                self.load_adapter(
                    adapter_id,
                    path,
                    rank=rank_map.get(adapter_id, 0) if rank_map else 0,
                    tenant_id=tenant_map.get(adapter_id, "") if tenant_map else "",
                )
                loaded.append(adapter_id)
            except Exception as e:
                logger.error(f"Failed to warmup adapter '{adapter_id}': {e}")
        return loaded

    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload an adapter to free VRAM."""
        if adapter_id in self._loaded_models:
            del self._loaded_models[adapter_id]
            return self._pool.remove_adapter(adapter_id)
        return False

    def get_adapter_for_request(self, request_adapter_id: Optional[str]) -> Optional[object]:
        """Get the PEFT model for a specific request.

        For S-LoRA style batching, each request in a batch may use
        a different adapter. This returns the appropriate model.

        Args:
            request_adapter_id: Adapter ID for the request.

        Returns:
            PeftModel instance, or None for base model.
        """
        if request_adapter_id is None:
            return None
        return self._loaded_models.get(request_adapter_id)

    def get_batch_adapter_ids(self, sequence_adapter_ids: List[Optional[str]]) -> Tuple[List[Optional[str]], int]:
        """Get unique adapter IDs used in a batch.

        Returns the set of adapters needed for the batch and counts
        how many sequences use each adapter.

        Args:
            sequence_adapter_ids: Adapter ID per sequence in the batch.

        Returns:
            (unique_adapter_ids, num_unique_adapters)
        """
        unique = list(dict.fromkeys(sequence_adapter_ids))  # Preserve order, remove dups
        return unique, len(unique)

    def rank_adapters(self) -> List[AdapterInfo]:
        """Return adapters sorted by rank (highest first) for multi-tenant priority."""
        adapters = []
        for adapter_id in self._pool.list_adapters():
            info = self._pool.get_adapter(adapter_id)
            if info:
                adapters.append(info)
        adapters.sort(key=lambda a: (a.rank, a.use_count), reverse=True)
        return adapters

    def list_adapters(self) -> List[str]:
        """Return list of loaded adapter IDs."""
        return self._pool.list_adapters()

    def get_adapter_info(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get detailed info about an adapter."""
        return self._pool.get_adapter(adapter_id)

    @property
    def active_adapter(self) -> Optional[str]:
        return self._pool.active_adapter

    def get_stats(self) -> dict:
        pool_stats = self._pool.get_stats()
        return {
            **pool_stats,
            "loaded_models": len(self._loaded_models),
        }
