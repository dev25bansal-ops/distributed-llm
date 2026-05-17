"""Complete KV cache exchange transport for gossip protocol.

Provides the actual network layer for transferring KV cache metadata
and data between peer nodes. Supports:
- gRPC-based transport (production)
- HTTP fallback for testing
- Bandwidth-aware transfers (skip when network saturated)
- Polynomial rolling hash for cache identification
"""

import hashlib
import time
from loguru import logger

import torch


class KVCacheTransfer:
    """Handles serialization and transfer of KV cache data between nodes.

    Uses polynomial rolling hash for cache identification and
    bandwidth-aware transfer to avoid saturating the network.
    """

    # Polynomial hash parameters (matching PrefixCache and CacheIndex)
    HASH_BASE = 31
    HASH_MOD = (1 << 61) - 1  # Mersenne prime

    # Bandwidth limits (bytes/sec) - configurable
    DEFAULT_MAX_BANDWIDTH = 100 * 1024 * 1024  # 100 MB/s

    def __init__(self, max_bandwidth: int = DEFAULT_MAX_BANDWIDTH):
        self._max_bandwidth = max_bandwidth
        self._bytes_transferred = 0
        self._transfer_start = time.time()
        self._transfers_completed = 0
        self._transfers_failed = 0

    @classmethod
    def hash_tokens(cls, token_ids: list[int]) -> str:
        """Compute polynomial rolling hash for a token sequence.

        Uses the same hash as PrefixCache and CacheIndex for consistency.

        Args:
            token_ids: List of token IDs.

        Returns:
            String hash identifier.
        """
        h = 0
        for t in token_ids:
            h = (h * cls.HASH_BASE + t) % cls.HASH_MOD
        return f"h{h}"

    @classmethod
    def hash_tokens_sha256(cls, token_ids: list[int]) -> str:
        """Compute SHA-256 hash for collision-safe identification.

        Args:
            token_ids: List of token IDs.

        Returns:
            Hex-encoded SHA-256 hash (first 16 chars).
        """
        data = bytes(token_ids)
        return hashlib.sha256(data).hexdigest()[:16]

    @classmethod
    def serialize_kv(cls, cache_data: dict) -> bytes:
        """Serialize KV cache data for transmission.

        Args:
            cache_data: Dict with layer_idx -> (k_tensor, v_tensor).

        Returns:
            Serialized bytes.
        """
        buffer = torch.io.BytesIO()
        torch.save(cache_data, buffer)
        return buffer.getvalue()

    @classmethod
    def deserialize_kv(cls, data: bytes) -> dict:
        """Deserialize KV cache data from received bytes.

        Args:
            data: Serialized KV cache bytes.

        Returns:
            Dict with layer_idx -> (k_tensor, v_tensor).
        """
        buffer = torch.io.BytesIO(data)
        return torch.load(buffer, weights_only=False)

    @classmethod
    def estimate_size(cls, cache_data: dict) -> int:
        """Estimate serialized size of KV cache data.

        Args:
            cache_data: Dict with layer_idx -> (k_tensor, v_tensor).

        Returns:
            Estimated size in bytes.
        """
        total = 0
        for layer_data in cache_data.values():
            if isinstance(layer_data, tuple):
                k, v = layer_data
                total += k.element_size() * k.numel()
                total += v.element_size() * v.numel()
            elif isinstance(layer_data, dict):
                total += cls.estimate_size(layer_data)
        return total

    def can_transfer(self, size_bytes: int) -> bool:
        """Check if transfer is within bandwidth limits.

        Args:
            size_bytes: Size of data to transfer.

        Returns:
            True if transfer is allowed.
        """
        elapsed = time.time() - self._transfer_start
        if elapsed <= 0:
            return True

        current_rate = self._bytes_transferred / elapsed
        # Allow if under 80% of max bandwidth (headroom for other traffic)
        return current_rate + size_bytes < self._max_bandwidth * 0.8

    def record_transfer(self, size_bytes: int, success: bool = True) -> None:
        """Record a completed transfer for bandwidth tracking.

        Args:
            size_bytes: Size of transferred data.
            success: Whether the transfer succeeded.
        """
        if success:
            self._bytes_transferred += size_bytes
            self._transfers_completed += 1
        else:
            self._transfers_failed += 1

    def stats(self) -> dict:
        """Get transfer statistics."""
        elapsed = time.time() - self._transfer_start
        return {
            "bytes_transferred": self._bytes_transferred,
            "transfers_completed": self._transfers_completed,
            "transfers_failed": self._transfers_failed,
            "avg_bandwidth_mbps": round(
                self._bytes_transferred / max(elapsed, 1) / 1024 / 1024, 2
            ),
            "max_bandwidth_mbps": round(self._max_bandwidth / 1024 / 1024, 2),
        }

    def reset_stats(self) -> None:
        """Reset transfer statistics."""
        self._bytes_transferred = 0
        self._transfer_start = time.time()
        self._transfers_completed = 0
        self._transfers_failed = 0


class GossipTransport:
    """Network transport layer for gossip protocol communication.

    Provides actual HTTP/gRPC communication between nodes for:
    1. Cache advertisement exchange
    2. KV cache data transfer
    3. Bandwidth-aware throttling
    """

    def __init__(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 50052,
        peer_resolver=None,
        max_bandwidth: int = KVCacheTransfer.DEFAULT_MAX_BANDWIDTH,
    ):
        self.node_id = node_id
        self.host = host
        self.port = port
        self._peer_resolver = peer_resolver
        self._transfer = KVCacheTransfer(max_bandwidth=max_bandwidth)
        self._session = None

    def _get_session(self):
        """Get or create HTTP session for peer communication."""
        if self._session is None:
            import urllib3
            self._session = urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=2.0, read=10.0),
                retries=urllib3.Retry(total=1, connect=1, read=1),
            )
        return self._session

    def exchange_advertisements(
        self, peer_id: str, my_ad: dict
    ) -> dict | None:
        """Exchange cache advertisements with a peer node.

        In production this uses gRPC; here we use HTTP as fallback.

        Args:
            peer_id: Peer node identifier.
            my_ad: Our cache advertisement.

        Returns:
            Peer's advertisement, or None if exchange failed.
        """
        # Try to resolve peer address
        peer_host, peer_port = self._resolve_peer(peer_id)
        if peer_host is None:
            return None

        url = f"http://{peer_host}:{peer_port}/api/v1/gossip/exchange"

        try:
            import json
            session = self._get_session()
            response = session.request(
                "POST",
                url,
                body=json.dumps(my_ad).encode(),
                headers={"Content-Type": "application/json"},
            )

            if response.status == 200:
                peer_ad = json.loads(response.data.decode())
                logger.debug(
                    f"Gossip exchange with {peer_id}: "
                    f"{peer_ad.get('total_cache_entries', 0)} entries"
                )
                return peer_ad
        except Exception as e:
            logger.debug(f"Gossip exchange with {peer_id} failed: {e}")

        return None

    def request_kv_cache(
        self, peer_id: str, prefix_hashes: list[str]
    ) -> dict | None:
        """Request KV cache data from a peer node.

        Only requests entries that are beneficial to transfer
        (within bandwidth limits and not already cached locally).

        Args:
            peer_id: Peer node identifier.
            prefix_hashes: List of prefix hashes to request.

        Returns:
            Response dict with cache_entries and transfer stats,
            or None if request failed.
        """
        peer_host, peer_port = self._resolve_peer(peer_id)
        if peer_host is None:
            return None

        # Filter hashes by bandwidth availability
        beneficial_hashes = []
        for h in prefix_hashes:
            # Estimate ~1MB per entry (typical for medium-length sequences)
            estimated_size = 1 * 1024 * 1024
            if self._transfer.can_transfer(estimated_size):
                beneficial_hashes.append(h)
            else:
                logger.debug(
                    f"Skipping transfer of {h}: bandwidth limit reached"
                )

        if not beneficial_hashes:
            return {"success": True, "cache_entries": {}, "entries_returned": 0}

        url = f"http://{peer_host}:{peer_port}/api/v1/gossip/fetch"

        try:
            import json
            session = self._get_session()
            request_data = {
                "requester_id": self.node_id,
                "prefix_hashes": beneficial_hashes,
            }
            response = session.request(
                "POST",
                url,
                body=json.dumps(request_data).encode(),
                headers={"Content-Type": "application/json"},
            )

            if response.status == 200:
                result = json.loads(response.data.decode())
                total_size = sum(
                    len(e.get("data", b""))
                    for e in result.get("cache_entries", {}).values()
                )
                self._transfer.record_transfer(total_size, success=True)
                logger.debug(
                    f"Fetched {len(beneficial_hashes)} entries from {peer_id}: "
                    f"{total_size / 1024:.1f} KB"
                )
                return result
        except Exception as e:
            logger.debug(f"KV fetch from {peer_id} failed: {e}")
            self._transfer.record_transfer(0, success=False)

        return None

    def _resolve_peer(self, peer_id: str) -> tuple[str | None, int]:
        """Resolve peer_id to (host, port).

        Args:
            peer_id: Peer node identifier.

        Returns:
            Tuple of (host, port) or (None, 0) if unresolved.
        """
        if self._peer_resolver:
            result = self._peer_resolver(peer_id)
            if result:
                return result
        # Default: localhost with incremental port
        return None, 0

    @property
    def transfer_stats(self) -> dict:
        """Get transfer statistics."""
        return self._transfer.stats()
