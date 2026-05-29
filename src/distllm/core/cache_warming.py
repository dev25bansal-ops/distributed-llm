"""Cache warming: pre-populate caches with frequently used prompts.

Provides CacheWarmer with tiered warming (hot/warm/cold) and optional
CUDA graph capture for hot-tier prompts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from loguru import logger


@dataclass
class WarmUpTier:
    """A single warming tier (hot, warm, cold)."""
    name: str
    prompts: list[str] = field(default_factory=list)
    capture_cuda_graphs: bool = False
    batch_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])


@dataclass
class WarmUpStats:
    """Statistics from a warming run."""
    warmed: int = 0
    failed: int = 0
    total_prompts: int = 0
    duration_seconds: float = 0.0
    cuda_graphs_captured: int = 0


class CacheWarmer:
    """Pre-populates caches with frequently used prompts.

    Supports tiered warming (hot/warm/cold) with optional CUDA graph
    capture for hot-tier prompts to eliminate launch overhead.
    """

    def __init__(self):
        self._tiers: list[WarmUpTier] = []
        self._last_stats = WarmUpStats()

    def add_tier(
        self,
        name: str,
        prompts: list[str],
        capture_cuda_graphs: bool = False,
        batch_sizes: list[int] | None = None,
    ) -> None:
        """Add a warming tier."""
        tier = WarmUpTier(
            name=name,
            prompts=prompts,
            capture_cuda_graphs=capture_cuda_graphs,
            batch_sizes=batch_sizes or [1, 2, 4, 8, 16, 32],
        )
        self._tiers.append(tier)

    def warm(self, prompts: list[str], coordinator: Any) -> int:
        """Warm a list of prompts using the coordinator.

        Returns:
            Number of prompts successfully warmed.
        """
        count = 0
        for prompt in prompts:
            try:
                coordinator.generate(prompt)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to warm prompt: {e}")
        return count

    def warm_from_file(self, path: str, coordinator: Any) -> int:
        """Warm prompts loaded from a JSON file.

        The file can be either a JSON list of strings or a dict with a 'prompts' key.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Warming file not found: {path}")

        with open(filepath) as f:
            data = json.load(f)

        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict) and "prompts" in data:
            prompts = data["prompts"]
        else:
            prompts = []

        return self.warm(prompts, coordinator)

    def run(self, coordinator: Any) -> WarmUpStats:
        """Run warming across all tiers.

        Returns:
            WarmUpStats with counts and duration.
        """
        import torch

        start = time.time()
        stats = WarmUpStats()

        for tier in self._tiers:
            stats.total_prompts += len(tier.prompts)
            for prompt in tier.prompts:
                try:
                    coordinator.generate(prompt)
                    stats.warmed += 1
                except Exception as e:
                    logger.warning(f"Failed to warm prompt in tier '{tier.name}': {e}")
                    stats.failed += 1

            # Capture CUDA graphs for hot tier
            if tier.capture_cuda_graphs and torch.cuda.is_available():
                captured = self._capture_cuda_graphs_for_tier(tier, coordinator)
                stats.cuda_graphs_captured += captured

        stats.duration_seconds = time.time() - start
        self._last_stats = stats
        return stats

    def _capture_cuda_graphs_for_tier(self, tier: WarmUpTier, coordinator: Any) -> int:
        """Capture CUDA graphs for the given tier's batch sizes."""
        import torch

        captured = 0
        for batch_size in tier.batch_sizes:
            try:
                if hasattr(coordinator, "scheduler") and coordinator.scheduler is not None:
                    if hasattr(coordinator.scheduler, "capture_cuda_graph"):
                        coordinator.scheduler.capture_cuda_graph(batch_size)
                        captured += 1
            except Exception as e:
                logger.debug(f"CUDA graph capture failed for batch_size={batch_size}: {e}")
        return captured

    def get_stats(self) -> WarmUpStats:
        """Return the last warming run stats."""
        return self._last_stats

    @classmethod
    def from_config(cls, path: str) -> CacheWarmer:
        """Create a CacheWarmer from a JSON config file.

        Config format:
            {
                "tiers": [
                    {"name": "hot", "prompts": ["p1", "p2"], "capture_cuda_graphs": true},
                    {"name": "warm", "prompts": ["p3"]},
                    {"name": "cold", "prompts_file": "/path/to/prompts.json"}
                ]
            }
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(filepath) as f:
            config = json.load(f)

        warmer = cls()
        for tier_cfg in config.get("tiers", []):
            prompts = tier_cfg.get("prompts", [])

            # Load prompts from file if specified
            if "prompts_file" in tier_cfg:
                pf = Path(tier_cfg["prompts_file"])
                if pf.exists():
                    with open(pf) as f:
                        data = json.load(f)
                        prompts = data if isinstance(data, list) else data.get("prompts", [])

            warmer.add_tier(
                name=tier_cfg.get("name", "default"),
                prompts=prompts,
                capture_cuda_graphs=tier_cfg.get("capture_cuda_graphs", False),
            )

        return warmer
