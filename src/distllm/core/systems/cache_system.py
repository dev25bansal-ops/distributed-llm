"""CacheSystem: prefix cache, KV cache, gossip, persistence.

Groups: CacheManager, CachePersistenceManager, GossipProtocol, PredictiveCacheManager
"""

from typing import Any



class CacheSystem:
    """Manages all caching: prefix, KV, gossip, persistence.

    Composes CacheManager, CachePersistenceManager, GossipProtocol,
    and PredictiveCacheManager into a single interface.
    """

    def __init__(
        self,
        max_cache_size: int = 1000,
        persistence_path: str = "",
        enable_gossip: bool = False,
        enable_predictive: bool = False,
    ):
        from distllm.core.cache_manager import CacheManager
        from distllm.core.cache_persistence import CachePersistenceManager
        from distllm.core.gossip_protocol import GossipProtocol, GossipClient
        from distllm.core.cache_index import CacheIndex
        from distllm.core.predictive_cache import PredictiveCacheManager

        self.cache_mgr = CacheManager(max_cache_size=max_cache_size)

        self.persistence = CachePersistenceManager(
            persistence_path=persistence_path,
        ) if persistence_path else None

        self.gossip_protocol = None
        self.gossip_client = None
        self.cache_index = None
        if enable_gossip:
            self.cache_index = CacheIndex()
            self.gossip_protocol = GossipProtocol(cache_index=self.cache_index)
            self.gossip_client = GossipClient()

        self.predictive = PredictiveCacheManager() if enable_predictive else None

    @property
    def prefix_cache(self):
        return self.cache_mgr.prefix_cache

    def lookup_prefix(self, tokens: list) -> tuple[int, Any]:
        return self.cache_mgr.lookup(tokens)

    def store_prefix(self, tokens: list, data: Any) -> None:
        self.cache_mgr.store(tokens, data)

    def get_match_len(self, tokens: list) -> int:
        return self.cache_mgr.get_match_len(tokens)

    def get_stats(self) -> dict:
        stats = self.cache_mgr.get_stats()
        if self.persistence:
            stats["persistence"] = self.persistence.get_stats()
        if self.gossip_protocol:
            stats["gossip"] = self.gossip_protocol.get_stats()
        if self.predictive:
            stats["predictive"] = self.predictive.stats
        return stats

    def warm_cache(self, prompts: list[str], tokenizer: Any) -> int:
        """Pre-populate cache with common prefixes."""
        return self.cache_mgr.warm_cache(prompts, tokenizer)

    def observe_request(self, token_ids: list) -> None:
        """Record request for predictive caching."""
        if self.predictive:
            self.predictive.observe_request(token_ids)
