"""Cross-cluster KV cache migration — move cached prefixes between clusters.

When a request spills over to a federated peer, its prompt's KV cache
can be streamed to the peer so that the peer does not re-execute the
prefill phase.  This eliminates redundant computation at the cost of
one WAN round-trip for the cache transfer.

Architecture::

    Cluster A (source)                  Cluster B (destination)
         │                                      │
         │  1. Compute KV cache for prompt      │
         │  2. Serialise via protobuf           │
         │  3. Stream pages via gRPC ──────────► │  4. Deserialise & load
         │                                      │  5. Start decode from
         │                                      │     cached prefix
         │◄── 6. Acknowledge (or redirect) ─────│

The migration is incremental — pages are streamed as they are computed,
so the destination can begin decoding as soon as the first page arrives
(pipeline parallelism across clusters).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from loguru import logger

from distllm.dist.cache_digest import KVCacheDigest, ContentRouter


@dataclass
class KVMigrationResult:
    """Outcome of a single KV cache migration."""
    success: bool
    cluster_id: str
    prefix_hash: str
    pages_transferred: int
    bytes_transferred: int
    transfer_time_ms: float
    error: str = ""


class KVCacheMigrator:
    """Streams KV cache pages to a peer cluster for zero-copy spillover.

    Usage::

        migrator = KVCacheMigrator()

        # During spillover:
        result = await migrator.migrate_prefix(
            source_cluster="cluster-a",
            target_cluster="cluster-b",
            target_url="https://cluster-b:8000",
            kv_pages=[...],
            prefix_hash="h4f8a...",
        )
    """

    def __init__(
        self,
        chunk_size_pages: int = 4,
        max_concurrent_migrations: int = 4,
        timeout_s: float = 30.0,
    ):
        self._chunk_size = chunk_size_pages
        self._semaphore = asyncio.Semaphore(max_concurrent_migrations)
        self._timeout_s = timeout_s
        self._content_router = ContentRouter()

        # Metrics
        self._total_migrations = 0
        self._successful_migrations = 0
        self._total_bytes = 0
        self._total_time_ms = 0.0

    async def migrate_prefix(
        self,
        target_url: str,
        prefix_hash: str,
        kv_pages: list[dict[str, Any]],
        cluster_id: str = "",
    ) -> KVMigrationResult:
        """Migrate KV cache pages to a peer cluster.

        Pages are streamed in chunks of *chunk_size_pages* so the peer
        can begin decoding before all pages arrive (pipeline parallelism).

        Args:
            target_url: Base URL of the destination cluster coordinator.
            prefix_hash: Hash of the cached token prefix (for lookup on
                the destination side).
            kv_pages: List of KV cache page dicts, each containing
                ``key`` and ``value`` tensors (as lists or numpy arrays).
            cluster_id: Destination cluster ID (for metrics).

        Returns:
            :class:`KVMigrationResult` with transfer statistics.
        """
        async with self._semaphore:
            return await self._migrate(
                target_url, prefix_hash, kv_pages, cluster_id,
            )

    async def _migrate(
        self,
        target_url: str,
        prefix_hash: str,
        kv_pages: list[dict[str, Any]],
        cluster_id: str,
    ) -> KVMigrationResult:
        t0 = time.monotonic()
        pages_transferred = 0
        bytes_transferred = 0

        url = f"{target_url.rstrip('/')}/api/v1/cache/migrate"
        total_pages = len(kv_pages)

        try:
            # Stream pages in chunks.
            for start in range(0, total_pages, self._chunk_size):
                chunk = kv_pages[start:start + self._chunk_size]
                payload = {
                    "prefix_hash": prefix_hash,
                    "page_offset": start,
                    "total_pages": total_pages,
                    "pages": [
                        {
                            "key": page.get("key", []),
                            "value": page.get("value", []),
                        }
                        for page in chunk
                    ],
                }

                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self._timeout_s),
                ) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()

                pages_transferred += len(chunk)
                bytes_transferred += _estimate_page_bytes(chunk)

            elapsed_ms = (time.monotonic() - t0) * 1000

            self._total_migrations += 1
            self._successful_migrations += 1
            self._total_bytes += bytes_transferred
            self._total_time_ms += elapsed_ms

            logger.info(
                f"KV migration to {cluster_id}: {pages_transferred}/{total_pages} "
                f"pages, {bytes_transferred / 1024:.0f} KB in {elapsed_ms:.0f}ms"
            )

            return KVMigrationResult(
                success=True,
                cluster_id=cluster_id,
                prefix_hash=prefix_hash,
                pages_transferred=pages_transferred,
                bytes_transferred=bytes_transferred,
                transfer_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._total_migrations += 1

            logger.warning(
                f"KV migration to {cluster_id} failed after "
                f"{pages_transferred}/{total_pages} pages: {e}"
            )

            return KVMigrationResult(
                success=False,
                cluster_id=cluster_id,
                prefix_hash=prefix_hash,
                pages_transferred=pages_transferred,
                bytes_transferred=bytes_transferred,
                transfer_time_ms=elapsed_ms,
                error=str(e),
            )

    async def warm_remote_cache(
        self,
        target_url: str,
        prefix_hash: str,
        kv_pages: list[dict[str, Any]],
    ) -> bool:
        """One-shot: migrate and block until acknowledged.

        Convenience wrapper for callers that want a simple
        ``True/False`` result.
        """
        result = await self.migrate_prefix(
            target_url=target_url,
            prefix_hash=prefix_hash,
            kv_pages=kv_pages,
        )
        return result.success

    # ── Observability ─────────────────────────────────────────────────

    def get_metrics(self) -> dict[str, Any]:
        return {
            "total_migrations": self._total_migrations,
            "successful_migrations": self._successful_migrations,
            "total_bytes_transferred": self._total_bytes,
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_transfer_speed_kbps": round(
                (self._total_bytes / 1024)
                / max(self._total_time_ms / 1000, 0.001)
            ) if self._total_time_ms > 0 else 0,
        }


def _estimate_page_bytes(pages: list[dict[str, Any]]) -> int:
    """Rough byte-count estimate for a list of KV pages.

    Used for metrics only — not a serialization path.
    """
    total = 0
    for page in pages:
        for key in ("key", "value"):
            data = page.get(key, [])
            if isinstance(data, (list, tuple)):
                total += len(data) * 2  # assume float16
            elif hasattr(data, "nbytes"):
                total += data.nbytes
    return total
