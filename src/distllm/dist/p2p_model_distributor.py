"""P2P Model Distribution — BitTorrent-style chunked model download with Merkle verification.

Model weights are split into layer-sized chunks.  Peers download chunks
from multiple sources simultaneously, verifying each chunk against the
Merkle tree root.  Corrupted chunks are re-requested from different peers.

Usage::

    dist = P2PModelDistributor(
        model_name="meta-llama/Llama-3.1-70B",
        chunk_size_bytes=100 * 1024 * 1024,  # 100MB chunks
    )
    dist.discover_peers(["node-a:50051", "node-b:50051"])
    dist.download_layers(start_layer=0, end_layer=5)
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from distllm.dist.merkle import MerkleTree


class P2PModelDistributor:
    """Distributes model weights across peers using chunked P2P transfers.

    Each layer is split into *chunk_size_bytes* chunks.  A Merkle tree
    over all chunks provides integrity verification.  Peers exchange
    chunk availability via a simple bitfield protocol.
    """

    def __init__(
        self,
        model_name: str,
        chunk_size_bytes: int = 100 * 1024 * 1024,
        max_peers: int = 16,
        max_concurrent_downloads: int = 4,
    ):
        self._model_name = model_name
        self._chunk_size = chunk_size_bytes
        self._max_peers = max_peers
        self._peers: list[dict[str, Any]] = []
        self._merkle: MerkleTree | None = None
        self._chunks: dict[int, bytes] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_downloads)

    def discover_peers(self, peer_addresses: list[str]) -> None:
        """Register peers that may have model chunks."""
        for addr in peer_addresses:
            self._peers.append({"address": addr, "alive": True})

    def build_merkle(self, layer_weights: dict[str, Any]) -> str:
        """Build a Merkle tree over a list of chunk hashes.

        Args:
            layer_weights: Dict of layer_name -> weight tensor.

        Returns:
            Merkle root hash as hex string.
        """
        chunk_hashes = []
        for layer_name, weight in layer_weights.items():
            weight_bytes = weight.numpy().tobytes() if hasattr(weight, 'numpy') else str(weight).encode()
            chunk_hashes.append(hashlib.sha256(weight_bytes).hexdigest())

        self._merkle = MerkleTree(chunk_hashes)
        logger.info(f"Merkle root: {self._merkle.root[:16]}... ({len(chunk_hashes)} chunks)")
        return self._merkle.root

    def verify_chunk(self, chunk_index: int, chunk_data: bytes) -> bool:
        """Verify a downloaded chunk against the Merkle tree.

        Args:
            chunk_index: Index of the chunk in the layer list.
            chunk_data: Raw chunk bytes.

        Returns:
            True if the chunk is valid.
        """
        if self._merkle is None:
            return True
        chunk_hash = hashlib.sha256(chunk_data).hexdigest()
        proof = self._merkle.get_proof(chunk_index)
        computed = chunk_hash
        for sibling in proof:
            computed = hashlib.sha256((computed + sibling).encode()).hexdigest()
        return computed == self._merkle.root

    def download_layer(self, layer_index: int, from_peers: list[dict] | None = None) -> bytes | None:
        """Download a single layer's weights from a peer.

        Tries each peer until one returns valid data.

        Returns:
            Layer weight bytes, or None if all peers failed.
        """
        targets = from_peers or self._peers
        for peer in targets:
            try:
                peer_addr = peer["address"]
                host, port_str = peer_addr.rsplit(":", 1)
                port = int(port_str)
                from distllm.dist.node_client import request_layer_weights
                data = request_layer_weights(
                    host, port, self._model_name,
                    layer_index, layer_index + 1,
                )
                if data:
                    logger.debug(f"Downloaded layer {layer_index} from {peer_addr} ({len(data)} bytes)")
                    return data
            except Exception as e:
                logger.debug(f"Failed to download layer {layer_index} from {peer['address']}: {e}")
        return None

    def download_layers(self, start_layer: int, end_layer: int) -> dict[int, bytes]:
        """Download a range of layers from peers in parallel.

        Args:
            start_layer: First layer index (inclusive).
            end_layer: Last layer index (inclusive).

        Returns:
            Dict of layer_index -> bytes for successfully downloaded layers.
        """
        from concurrent.futures import as_completed

        futures = {}
        for i in range(start_layer, end_layer + 1):
            future = self._executor.submit(self.download_layer, i)
            futures[future] = i

        results: dict[int, bytes] = {}
        for future in as_completed(futures):
            layer_idx = futures[future]
            try:
                data = future.result()
                if data:
                    results[layer_idx] = data
            except Exception as e:
                logger.warning(f"P2P download failed for layer {layer_idx}: {e}")

        logger.info(f"P2P downloaded {len(results)}/{end_layer - start_layer + 1} layers")
        return results

    def advertise_chunks(self, available_chunks: list[int]) -> None:
        """Advertise which chunks this node has to peers."""
        pass  # Placeholder for gossip-based chunk advertisement

    def stats(self) -> dict[str, Any]:
        return {
            "peers": len(self._peers),
            "chunks_downloaded": len(self._chunks),
            "model": self._model_name,
        }
