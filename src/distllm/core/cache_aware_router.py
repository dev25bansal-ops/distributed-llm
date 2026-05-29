"""F4: Cache-aware request routing.

Routes requests to the node with the best cache affinity,
eliminating redundant prefill computation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger


class CacheAwareRouter:
    """Routes requests to nodes based on cache affinity.

    Combines cache hit potential with node load to select the
    optimal node for each request.
    """

    def __init__(
        self,
        cache_weight: float = 0.7,
        load_weight: float = 0.3,
    ):
        self._cache_weight = cache_weight
        self._load_weight = load_weight
        self._route_stats: dict[str, dict[str, int]] = {}

    async def route(
        self,
        tokens: list[int],
        nodes: dict[str, Any],
        cache_manager: Any = None,
    ) -> str | None:
        """Select the best node for a request based on cache affinity.

        Args:
            tokens: Token IDs of the request.
            nodes: Dict of {node_id: node_info} with load metrics.
            cache_manager: CacheManager instance for checking cache state.

        Returns:
            Best node ID, or None if no nodes available.
        """
        if not nodes:
            return None

        scores: dict[str, float] = {}

        for node_id, node_info in nodes.items():
            # Cache affinity score (0-1)
            cache_score = await self._check_cache_affinity(
                node_id, tokens, cache_manager
            )

            # Load score (0-1, lower is better)
            load_score = self._get_load_score(node_info)

            # Combined score
            combined = (
                self._cache_weight * cache_score
                + self._load_weight * (1.0 - load_score)
            )
            scores[node_id] = combined

        # Select node with highest score
        best_node = max(scores, key=scores.get)

        # Track routing stats
        if best_node not in self._route_stats:
            self._route_stats[best_node] = {"routed": 0, "cache_hits": 0}
        self._route_stats[best_node]["routed"] += 1

        if scores[best_node] > 0.5:  # Threshold for "likely cache hit"
            self._route_stats[best_node]["cache_hits"] += 1

        return best_node

    async def _check_cache_affinity(
        self,
        node_id: str,
        tokens: list[int],
        cache_manager: Any,
    ) -> float:
        """Check how much of the prefix is cached on a node.

        Returns:
            Cache affinity score (0-1).
        """
        if cache_manager is None:
            return 0.0

        try:
            # Check if we have cache info for this node
            if hasattr(cache_manager, 'prefix_cache') and cache_manager.prefix_cache is not None:
                match_len, _ = cache_manager.prefix_cache.lookup(tokens)
                if match_len > 0:
                    return min(1.0, match_len / max(len(tokens), 1))
        except Exception:
            pass

        return 0.0

    def _get_load_score(self, node_info: Any) -> float:
        """Get load score for a node (0 = idle, 1 = overloaded)."""
        if isinstance(node_info, dict):
            return node_info.get("load", 0.5)
        if hasattr(node_info, 'load'):
            return node_info.load
        return 0.5

    def get_route_stats(self) -> dict[str, dict[str, int]]:
        """Return routing statistics per node."""
        return dict(self._route_stats)
