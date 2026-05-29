"""Cross-cluster KV cache migration.

Migrates cached KV entries between clusters by fetching from a source
cluster's cache API and pushing to a destination cluster.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from loguru import logger


class CacheMigrator:
    """Migrates KV cache entries between clusters.

    Uses an optional transport object for fetching KV data from the source,
    and HTTP POST to push to the destination's /api/v1/cache/warm endpoint.
    """

    def __init__(self):
        self._transport: Any = None

    def set_transport(self, transport: Any) -> None:
        """Set the transport for fetching KV cache from source nodes."""
        self._transport = transport

    def migrate_cache(
        self,
        source_url: str,
        dest_url: str,
        prefix_hashes: list[str],
    ) -> bool:
        """Migrate cache entries from source to destination.

        Args:
            source_url: URL of the source cluster.
            dest_url: URL of the destination cluster.
            prefix_hashes: List of prefix hashes to migrate.

        Returns:
            True if all entries migrated successfully, False otherwise.
        """
        if self._transport is None:
            logger.warning("No transport set for cache migration")
            return False

        if not prefix_hashes:
            return True

        all_success = True
        for prefix_hash in prefix_hashes:
            try:
                kv_data = self._transport.request_kv_cache(prefix_hash)
                if kv_data is None:
                    logger.debug(f"Source missing cache for {prefix_hash}")
                    all_success = False
                    continue

                success = self._push_to_dest(dest_url, prefix_hash, kv_data)
                if not success:
                    all_success = False

            except Exception as e:
                logger.warning(f"Migration failed for {prefix_hash}: {e}")
                all_success = False

        return all_success

    def warm_cache_on_cluster(
        self,
        cluster_url: str,
        prefix_hashes: list[str],
    ) -> bool:
        """Send warm requests to a cluster for the given prefix hashes.

        Args:
            cluster_url: URL of the cluster to warm.
            prefix_hashes: List of prefix hashes to warm.

        Returns:
            True if all warm requests succeeded, False otherwise.
        """
        if not prefix_hashes:
            return True

        all_success = True
        for prefix_hash in prefix_hashes:
            try:
                payload = json.dumps({
                    "prefix_hash": prefix_hash,
                    "action": "warm",
                }).encode()

                req = urllib.request.Request(
                    f"{cluster_url.rstrip('/')}/api/v1/cache/warm",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                with urllib.request.urlopen(req) as resp:
                    if resp.status != 200:
                        all_success = False

            except Exception as e:
                logger.warning(f"Warm request failed for {prefix_hash}: {e}")
                all_success = False

        return all_success

    def _push_to_dest(self, dest_url: str, prefix_hash: str, kv_data: Any) -> bool:
        """Push KV cache data to the destination cluster."""
        try:
            payload = json.dumps({
                "prefix_hash": prefix_hash,
                "kv_data": str(kv_data),
            }).encode()

            req = urllib.request.Request(
                f"{dest_url.rstrip('/')}/api/v1/cache/warm",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req) as resp:
                return resp.status == 200

        except Exception as e:
            logger.warning(f"Push to {dest_url} failed for {prefix_hash}: {e}")
            return False
