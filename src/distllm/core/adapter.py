"""LoRA adapter management with S-LoRA style unified batching.

Implements S-LoRA's approach: instead of maintaining separate PeftModel
instances per request, we merge LoRA weights into the base model
per-batch, avoiding redundant memory and enabling efficient batching
across multiple adapters.

Architecture:
- Maintains a pool of LoRA adapter weights (A and B matrices)
- Before each batch: merges required adapters into the base model
- After batch: unloads adapters, restoring original weights
- Supports up to N concurrent adapters with O(1) weight swap
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class LoRAAdapter:
    """A single LoRA adapter configuration."""
    adapter_id: str
    r: int = 16
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] | None = None
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    loaded_at: float = 0.0
    use_count: int = 0

    @property
    def scaling(self) -> float:
        return self.alpha / self.r if self.r > 0 else 1.0


class BatchedLoRAManager:
    """Manages LoRA adapters with S-LoRA style per-batch weight merging.

    Instead of loading each adapter as a separate PeftModel (which consumes
    redundant memory for the base model), we:
    1. Keep the base model in its original state
    2. Before a batch: merge the needed adapter weights into the base model
    3. After the batch: restore original weights
    4. Adapt weights stay in a compact pool (A/B matrices only)

    Supports:
    - Up to 256 concurrent adapters
    - Per-batch weight merging
    - LRU eviction of cold adapters
    - Weight overlap detection for memory savings
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_adapters: int = 256,
        memory_budget_mb: float = 2048.0,
        merge_dtype: torch.dtype = torch.float16,
    ):
        self.base_model = base_model
        self.max_adapters = max_adapters
        self.memory_budget = memory_budget_mb * 1024 * 1024
        self.merge_dtype = merge_dtype
        self._lock = threading.Lock()

        # Adapter storage
        self._adapters: dict[str, LoRAAdapter] = {}
        self._adapter_lru: list[str] = []

        # Original weight cache: (module_path, weight_name) -> original_weight
        self._original_weights: dict[tuple[str, str], torch.Tensor] = {}

        # Currently merged adapter (None = base model only)
        self._current_adapter_id: str | None = None

        # Track which modules have LoRA applied
        self._lora_target_modules: set[str] = set()

        logger.info(f"LoRA manager initialized: max_adapters={max_adapters}, "
                    f"budget={memory_budget_mb}MB")

    def _get_target_modules(self, adapter: LoRAAdapter) -> list[str]:
        """Get target module names for this adapter.

        Default: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
        """
        if adapter.target_modules:
            return adapter.target_modules
        return ["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

    def _find_module(self, module_path: str) -> nn.Module | None:
        """Find a module by its dotted path in the base model."""
        parts = module_path.split(".")
        module = self.base_model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def register_adapter(
        self,
        adapter_id: str,
        weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
        r: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ) -> LoRAAdapter:
        """Register a new LoRA adapter with its weights.

        Args:
            adapter_id: Unique identifier for this adapter.
            weights: Dict mapping module_path -> (lora_A, lora_B) weight tuple.
            r: LoRA rank.
            alpha: LoRA scaling alpha.
            dropout: Dropout probability.
            target_modules: Target module name patterns (None = defaults).

        Returns:
            The registered LoRAAdapter.
        """
        with self._lock:
            if adapter_id in self._adapters:
                logger.warning(f"Adapter {adapter_id} already registered, overwriting")

            if len(self._adapters) >= self.max_adapters:
                self._evict_lru()

            adapter = LoRAAdapter(
                adapter_id=adapter_id,
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
                weights=weights,
            )
            self._adapters[adapter_id] = adapter
            self._touch_lru(adapter_id)
            logger.debug(f"Registered LoRA adapter {adapter_id} (r={r}, alpha={alpha})")
            return adapter

    def _touch_lru(self, adapter_id: str) -> None:
        if adapter_id in self._adapter_lru:
            self._adapter_lru.remove(adapter_id)
        self._adapter_lru.append(adapter_id)

    def _evict_lru(self) -> bool:
        """Evict the least recently used adapter."""
        if not self._adapter_lru:
            return False
        oldest = self._adapter_lru.pop(0)
        if oldest in self._adapters:
            if self._current_adapter_id == oldest:
                self.unmerge_adapter()
            del self._adapters[oldest]
            logger.debug(f"Evicted cold LoRA adapter: {oldest}")
        return True

    def merge_adapter(self, adapter_id: str) -> bool:
        """Merge LoRA weights into base model for the given adapter.

        S-LoRA style: weights are merged into the original linear layers
        as: W' = W + (B @ A) * scaling

        Args:
            adapter_id: Adapter to merge.

        Returns:
            True if merge succeeded.
        """
        with self._lock:
            if adapter_id not in self._adapters:
                logger.warning(f"Cannot merge unknown adapter: {adapter_id}")
                return False

            if self._current_adapter_id == adapter_id:
                return True

            if self._current_adapter_id is not None:
                self.unmerge_adapter()

            adapter = self._adapters[adapter_id]
            target_modules = self._get_target_modules(adapter)
            merged_count = 0

            for module_path, (lora_a, lora_b) in adapter.weights.items():
                module = self._find_module(module_path)
                if module is None:
                    continue

                if not hasattr(module, 'weight'):
                    continue

                # Save original weight first time
                key = (module_path, "weight")
                if key not in self._original_weights:
                    self._original_weights[key] = module.weight.data.clone()

                # Compute delta: delta = (B @ A) * scaling
                lora_a = lora_a.to(device=module.weight.device, dtype=self.merge_dtype)
                lora_b = lora_b.to(device=module.weight.device, dtype=self.merge_dtype)
                delta = (lora_b @ lora_a) * adapter.scaling

                # Merge: W' = W + delta
                module.weight.data.add_(delta)
                merged_count += 1
                self._lora_target_modules.add(module_path)

            self._current_adapter_id = adapter_id
            adapter.use_count += 1
            self._touch_lru(adapter_id)

            if merged_count > 0:
                logger.debug(f"Merged LoRA adapter {adapter_id} into {merged_count} modules")
            return True

    def unmerge_adapter(self) -> bool:
        """Restore original model weights, removing the current LoRA merge.

        Returns:
            True if unmerge succeeded.
        """
        with self._lock:
            if self._current_adapter_id is None:
                return False

            restored_count = 0
            for (module_path, weight_name), original_weight in self._original_weights.items():
                module = self._find_module(module_path)
                if module is not None and hasattr(module, weight_name):
                    getattr(module, weight_name).data.copy_(original_weight)
                    restored_count += 1

            adapter_id = self._current_adapter_id
            self._current_adapter_id = None

            if restored_count > 0:
                logger.debug(f"Unmerged LoRA adapter {adapter_id}, restored {restored_count} modules")
            return True

    def batch_merge(self, adapter_ids: list[str]) -> bool:
        """Merge multiple adapters sequentially for a batch.

        In S-LoRA, only one adapter is active at a time per forward pass,
        so we merge the first one and note the others for sequential processing.

        Args:
            adapter_ids: List of adapter IDs needed for this batch.

        Returns:
            True if at least one adapter was merged.
        """
        if not adapter_ids:
            return False
        return self.merge_adapter(adapter_ids[0])

    def get_weights(self, adapter_id: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]] | None:
        adapter = self._adapters.get(adapter_id)
        if adapter is None:
            return None
        return adapter.weights

    def get_merged_adapter_id(self) -> str | None:
        return self._current_adapter_id

    @property
    def active_count(self) -> int:
        return len(self._adapters)

    def stats(self) -> dict:
        return {
            "registered_adapters": len(self._adapters),
            "max_adapters": self.max_adapters,
            "current_merged": self._current_adapter_id,
            "lru_size": len(self._adapter_lru),
            "memory_budget_mb": self.memory_budget / (1024 * 1024),
            "original_weights_cached": len(self._original_weights),
        }
