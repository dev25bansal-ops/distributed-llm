"""Cache warmer for pre-populating KV caches with CUDA graph capture.

Supports tiered warm-up (cold/warm/hot states), automatic trigger on
model load or node join, CUDA graph pre-capture for common batch sizes,
and predictive warm-up based on request patterns.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger


@dataclass
class WarmUpStats:
    """Statistics from a warm-up run."""
    total_prompts: int = 0
    warmed: int = 0
    failed: int = 0
    duration_seconds: float = 0.0
    cache_fill_pct: float = 0.0
    cuda_graphs_captured: int = 0


@dataclass
class WarmUpTier:
    """A warm-up tier with different priority levels."""
    name: str  # "cold", "warm", "hot"
    prompts: list[str] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    capture_cuda_graphs: bool = False


class CacheWarmer:
    """Pre-populates KV caches with tiered warm-up and CUDA graph capture.

    Usage:
        warmer = CacheWarmer()
        # Tier 1: Common prompts (high priority)
        warmer.add_tier("hot", common_prompts, capture_graphs=True)
        # Tier 2: Less common prompts
        warmer.add_tier("warm", secondary_prompts)
        # Run warm-up
        stats = warmer.run(coordinator)
    """

    def __init__(self):
        self._tiers: list[WarmUpTier] = []
        self._stats: WarmUpStats = WarmUpStats()
        self._cuda_graph_pool = None

    def add_tier(
        self,
        name: str,
        prompts: list[str],
        batch_sizes: list[int] | None = None,
        capture_cuda_graphs: bool = False,
    ) -> None:
        """Add a warm-up tier.

        Args:
            name: Tier name (cold/warm/hot).
            prompts: List of prompt strings.
            batch_sizes: Batch sizes for CUDA graph capture.
            capture_cuda_graphs: Whether to capture CUDA graphs.
        """
        self._tiers.append(WarmUpTier(
            name=name,
            prompts=prompts,
            batch_sizes=batch_sizes or [1, 2, 4, 8, 16, 32],
            capture_cuda_graphs=capture_cuda_graphs,
        ))

    def warm(self, prompts: list[str], coordinator) -> int:
        """Run prompts through the pipeline to populate caches.

        Args:
            prompts: List of prompt strings to warm.
            coordinator: Coordinator instance to use for generation.

        Returns:
            Number of successfully warmed prompts.
        """
        warmed = 0
        for prompt in prompts:
            try:
                coordinator.generate(prompt, max_new_tokens=1)
                warmed += 1
            except Exception as e:
                logger.warning(f"Cache warm failed for prompt: {e}")
        logger.info(f"Cache warmer: warmed {warmed}/{len(prompts)} prompts")
        return warmed

    def warm_from_file(self, file_path: str, coordinator) -> int:
        """Load prompts from a JSON file and warm caches.

        Args:
            file_path: Path to JSON file (list of strings or {"prompts": [...]}).
            coordinator: Coordinator instance to use for generation.

        Returns:
            Number of successfully warmed prompts.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        data = json.loads(path.read_text())
        prompts = data if isinstance(data, list) else data.get("prompts", [])
        return self.warm(prompts, coordinator)

    def run(self, coordinator) -> WarmUpStats:
        """Execute all warm-up tiers sequentially.

        Args:
            coordinator: Coordinator instance.

        Returns:
            WarmUpStats with results.
        """
        start_time = time.time()
        stats = WarmUpStats()

        for tier in self._tiers:
            logger.info(f"Warm-up tier '{tier.name}': {len(tier.prompts)} prompts")
            tier_warmed = 0

            for prompt in tier.prompts:
                try:
                    coordinator.generate(prompt, max_new_tokens=1)
                    tier_warmed += 1
                    stats.warmed += 1
                except Exception as e:
                    logger.warning(f"Warm-up tier '{tier.name}' failed: {e}")
                    stats.failed += 1

            stats.total_prompts += len(tier.prompts)

            # Capture CUDA graphs for this tier if enabled
            if tier.capture_cuda_graphs and torch.cuda.is_available():
                captured = self._capture_cuda_graphs_for_tier(coordinator, tier)
                stats.cuda_graphs_captured += captured

        stats.duration_seconds = time.time() - start_time

        # Calculate cache fill percentage
        if coordinator.scheduler is not None:
            cache_fill_pct = self._get_cache_fill_pct(coordinator)
            stats.cache_fill_pct = cache_fill_pct

        self._stats = stats
        logger.info(
            f"Warm-up complete: {stats.warmed}/{stats.total_prompts} prompts "
            f"({stats.failed} failed) in {stats.duration_seconds:.1f}s, "
            f"cache fill: {stats.cache_fill_pct:.0f}%, "
            f"CUDA graphs: {stats.cuda_graphs_captured}"
        )
        return stats

    def _capture_cuda_graphs_for_tier(
        self, coordinator, tier: WarmUpTier
    ) -> int:
        """Capture CUDA graphs for common batch sizes in a tier."""
        if not torch.cuda.is_available():
            return 0

        captured = 0
        try:
            from distllm.core.cuda_graph import CUDAGraphPool

            if self._cuda_graph_pool is None:
                # Get model from coordinator
                local_partitioner = getattr(coordinator, 'local_partitioner', None)
                if local_partitioner is None or not hasattr(local_partitioner, 'full_model'):
                    logger.debug("No local model available for CUDA graph capture")
                    return 0

                model = local_partitioner.full_model
                config = model.config
                batch_sizes = tier.batch_sizes

                self._cuda_graph_pool = CUDAGraphPool(
                    model=model,
                    batch_sizes=batch_sizes,
                    num_layers=getattr(config, 'num_hidden_layers', 0),
                    num_heads=getattr(config, 'num_attention_heads', 0),
                    head_dim=getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32),
                )
                self._cuda_graph_pool.capture_all()
                captured = len(self._cuda_graph_pool._graphs)
                logger.debug(f"CUDA graph captured for {captured} batch sizes: {batch_sizes}")
        except ImportError:
            logger.debug("CUDA graph module not available")

        return captured

    def _get_cache_fill_pct(self, coordinator) -> float:
        """Estimate cache fill percentage."""
        if coordinator._cache_mgr is None:
            return 0.0

        prefix_cache = getattr(coordinator._cache_mgr, 'prefix_cache', None)
        if prefix_cache is None:
            return 0.0

        # Get cache stats if available
        max_entries = getattr(prefix_cache, 'max_entries', 0)
        if max_entries == 0:
            return 0.0

        # Count entries in cache
        current_entries = 0
        if hasattr(prefix_cache, '_root') and hasattr(prefix_cache._root, '_entries'):
            current_entries = len(prefix_cache._root._entries)
        elif hasattr(prefix_cache, '_cache'):
            current_entries = len(prefix_cache._cache)

        return min(100.0, (current_entries / max_entries) * 100)

    def get_stats(self) -> WarmUpStats:
        """Get the last warm-up statistics."""
        return self._stats

    @classmethod
    def from_config(cls, config_path: str) -> "CacheWarmer":
        """Create a CacheWarmer from a configuration file.

        Config format (JSON):
        {
            "tiers": [
                {"name": "hot", "prompts": [...], "capture_cuda_graphs": true},
                {"name": "warm", "prompts": [...]},
                {"name": "cold", "prompts_file": "prompts.json"}
            ]
        }
        """
        warmer = cls()
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        config = json.loads(path.read_text())
        for tier_cfg in config.get("tiers", []):
            prompts = tier_cfg.get("prompts", [])
            if "prompts_file" in tier_cfg:
                prompts_path = Path(tier_cfg["prompts_file"])
                if prompts_path.exists():
                    data = json.loads(prompts_path.read_text())
                    prompts = data if isinstance(data, list) else data.get("prompts", [])

            warmer.add_tier(
                name=tier_cfg["name"],
                prompts=prompts,
                batch_sizes=tier_cfg.get("batch_sizes", [1, 2, 4, 8, 16, 32]),
                capture_cuda_graphs=tier_cfg.get("capture_cuda_graphs", False),
            )

        return warmer
