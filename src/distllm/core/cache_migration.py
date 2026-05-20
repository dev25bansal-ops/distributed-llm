"""Cache migration for cross-cluster KV cache warmup.

Wraps existing GossipTransport.request_kv_cache() for explicit
cross-cluster cache warmup and migration between nodes.
"""

from __future__ import annotations

from loguru import logger


class CacheMigrator:
    """Migrates KV caches between clusters for cache warming.

    Uses the existing gossip transport infrastructure to transfer
    cached KV data from source nodes to destination nodes across clusters.
    """

    def __init__(self, gossip_transport=None) -> None:
        self._transport = gossip_transport

    def set_transport(self, transport) -> None:
        """Set the gossip transport to use for cache transfer."""
        self._transport = transport

    def migrate_cache(
        self,
        src_node_url: str,
        dst_node_url: str,
        prefix_hashes: list[str],
    ) -> bool:
        """Migrate KV cache entries from source to destination node.

        Args:
            src_node_url: URL of the source node holding the cache.
            dst_node_url: URL of the destination node to warm.
            prefix_hashes: List of prefix hashes to migrate.

        Returns:
            True if all migrations succeeded.
        """
        if self._transport is None:
            logger.warning("Cache migration skipped: no gossip transport configured")
            return False

        success_count = 0
        for prefix_hash in prefix_hashes:
            try:
                # Fetch KV cache from source
                kv_data = self._transport.request_kv_cache(src_node_url, prefix_hash)
                if kv_data is None:
                    logger.debug(f"Cache entry not found on source: {prefix_hash}")
                    continue

                # Send to destination for warming
                import urllib.request
                import json

                payload = json.dumps({
                    "prefix_hash": prefix_hash,
                    "kv_data": kv_data,
                }).encode()

                url = f"{dst_node_url.rstrip('/')}/api/v1/cache/warm"
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        success_count += 1
                        logger.debug(f"Cache migrated: {prefix_hash}")
            except Exception as e:
                logger.warning(f"Cache migration failed for {prefix_hash}: {e}")

        logger.info(
            f"Cache migration: {success_count}/{len(prefix_hashes)} entries migrated "
            f"from {src_node_url} to {dst_node_url}"
        )
        return success_count == len(prefix_hashes)

    def warm_cache_on_cluster(
        self,
        cluster_edge_url: str,
        prefix_hashes: list[str],
    ) -> bool:
        """Warm KV caches on a remote cluster via its edge node.

        Args:
            cluster_edge_url: URL of the target cluster's edge node.
            prefix_hashes: List of prefix hashes to warm.

        Returns:
            True if all caches were warmed successfully.
        """
        success_count = 0
        for prefix_hash in prefix_hashes:
            try:
                import urllib.request
                import json

                # Request cache warm on the remote cluster
                payload = json.dumps({
                    "prefix_hash": prefix_hash,
                    "action": "warm",
                }).encode()

                url = f"{cluster_edge_url.rstrip('/')}/api/v1/cache/warm"
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status == 200:
                        success_count += 1
            except Exception as e:
                logger.warning(f"Cache warm failed on {cluster_edge_url} for {prefix_hash}: {e}")

        logger.info(
            f"Cache warm on cluster: {success_count}/{len(prefix_hashes)} entries "
            f"warmed on {cluster_edge_url}"
        )
        return success_count == len(prefix_hashes)
