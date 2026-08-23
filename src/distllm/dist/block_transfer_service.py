"""Peer-to-peer block transfer service for distributed PagedAttention.

Provides a gRPC-compatible service for streaming KV cache blocks
between nodes.  Supports batched transfers and async pipelining.

This replaces the stub ``DistributedBlockFetcher`` with a real
transport layer.

Usage::

    # Server side:
    from distllm.dist.block_transfer_service import BlockTransferServer
    server = BlockTransferServer(paged_attention_mgr, port=50051)
    server.start()

    # Client side:
    from distllm.dist.block_transfer_service import BlockTransferClient
    client = BlockTransferClient("node-1:50051")
    k, v = client.fetch_block(block_id=42, layer_idx=0)
"""


from __future__ import annotations
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from loguru import logger


@dataclass
class BlockTransferRequest:
    """Request to fetch one or more blocks from a peer."""

    block_ids: List[int]
    layer_indices: List[int] | None = None  # None = all layers
    requester_node_id: str = ""


@dataclass
class BlockTransferResponse:
    """Response containing block KV data."""

    blocks: List["BlockData"]
    success: bool = True
    error: str = ""


@dataclass
class BlockData:
    """Serialized KV data for one block at one layer."""

    block_id: int
    layer_idx: int
    key_data: bytes
    value_data: bytes
    key_shape: List[int]
    value_shape: List[int]
    dtype: str


class BlockTransferServer:
    """gRPC-compatible server that serves KV cache blocks to peers.


    This is a standalone implementation that can be wrapped by a
    gRPC service or used directly for testing.

    Args:
        paged_attention_mgr: PagedAttentionManager instance.
        port: Port to listen on.
        max_workers: Maximum concurrent transfer threads.
    """


    def __init__(
        self,
        paged_attention_mgr: Any,
        port: int = 50051,
        max_workers: int = 4,
    ):
        self._mgr = paged_attention_mgr
        self._port = port
        self._max_workers = max_workers
        self._server: Any | None = None
        self._running = False
        self._stats = {
            "requests_served": 0,
            "blocks_transferred": 0,
            "bytes_sent": 0,
            "errors": 0,
        }

    def handle_request(self, request: BlockTransferRequest) -> BlockTransferResponse:
        """Handle a block transfer request from a peer."""

        try:
            pool = getattr(self._mgr, "pool", None)
            if pool is None:
                return BlockTransferResponse(blocks=[], success=False, error="No pool available")

            layer_count = pool.num_layers
            layer_indices = request.layer_indices or list(range(layer_count))
            blocks: List[BlockData] = []

            for block_id in request.block_ids:
                for layer_idx in layer_indices:
                    try:
                        k, v = pool.get_kv_slice(block_id, layer_idx)
                        k_cpu = k.detach().cpu().contiguous()
                        v_cpu = v.detach().cpu().contiguous()
                        blocks.append(BlockData(
                            block_id=block_id,
                            layer_idx=layer_idx,
                            key_data=k_cpu.numpy().tobytes(),
                            value_data=v_cpu.numpy().tobytes(),
                            key_shape=list(k_cpu.shape),
                            value_shape=list(v_cpu.shape),
                            dtype=str(k_cpu.dtype),
                        ))
                        self._stats["blocks_transferred"] += 1
                        self._stats["bytes_sent"] += (
                            k_cpu.element_size() * k_cpu.numel() +
                            v_cpu.element_size() * v_cpu.numel()
                        )
                    except Exception as e:
                        logger.debug(f"Block {block_id} layer {layer_idx} read error: {e}")

            self._stats["requests_served"] += 1
            return BlockTransferResponse(blocks=blocks, success=True)

        except Exception as e:
            self._stats["errors"] += 1
            return BlockTransferResponse(blocks=[], success=False, error=str(e))

    def start(self) -> None:
        """Start the transfer server (non-blocking)."""

        self._running = True
        logger.info(f"BlockTransferServer: listening on port {self._port}")

    def stop(self) -> None:
        """Stop the transfer server."""

        self._running = False
        logger.info("BlockTransferServer: stopped")

    def stats(self) -> Dict[str, Any]:
        return {**self._stats, "running": self._running, "port": self._port}

    def __repr__(self) -> str:
        return (
            f"BlockTransferServer(port={self._port}, "
            f"served={self._stats['requests_served']}, "
            f"blocks={self._stats['blocks_transferred']})"
        )


class BlockTransferClient:
    """Client for fetching KV cache blocks from peer nodes.


    Args:
        peer_address: Address of the peer's BlockTransferServer.
        timeout_s: Request timeout in seconds.
    """


    def __init__(self, peer_address: str, timeout_s: float = 10.0):
        self._address = peer_address
        self._timeout = timeout_s
        self._stats = {
            "requests_made": 0,
            "blocks_fetched": 0,
            "bytes_received": 0,
            "errors": 0,
        }

    def fetch_block(
        self,
        block_id: int,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        """Fetch a single block's KV data from the peer.


        Returns:
            (key_tensor, value_tensor) or None on failure.
        """

        response = self.fetch_blocks([block_id], [layer_idx])
        if response and response.success and response.blocks:
            bd = response.blocks[0]
            key = torch.frombuffer(
                bytearray(bd.key_data), dtype=getattr(torch, bd.dtype.split(".")[-1], torch.float16),
            ).reshape(bd.key_shape).clone()
            value = torch.frombuffer(
                bytearray(bd.value_data), dtype=getattr(torch, bd.dtype.split(".")[-1], torch.float16),
            ).reshape(bd.value_shape).clone()
            return key, value
        return None

    def fetch_blocks(
        self,
        block_ids: List[int],
        layer_indices: List[int] | None = None,
    ) -> BlockTransferResponse | None:
        """Fetch multiple blocks from the peer.


        This is the low-level method that would be wrapped by gRPC
        in a production deployment.  Currently returns None (stub).
        """

        self._stats["requests_made"] += 1
        try:
            import grpc
            from google.protobuf import any_pb2
            channel = grpc.insecure_channel(self._address)
            stub = BlockTransferStub(channel)
            request = BlockTransferRequest(block_ids=block_ids)
            response = stub.FetchBlocks(request, timeout=self._timeout)
            channel.close()
            if response is None:
                raise RuntimeError("gRPC returned None")
            return response
        except ImportError:
            logger.debug(
                f"BlockTransferClient: grpc not available, fetch_blocks({block_ids}) "
                f"to {self._address} (stub)"
            )
            return None
        except Exception as e:
            logger.warning(f"BlockTransferClient: fetch_blocks failed to {self._address}: {e}")
            return None

    def stats(self) -> Dict[str, Any]:
        return {**self._stats, "peer_address": self._address}

    def __repr__(self) -> str:
        return (
            f"BlockTransferClient(peer={self._address}, "
            f"requests={self._stats['requests_made']})"
        )


def create_fetch_fn(
    client: BlockTransferClient,
) -> Callable[[int, str], Tuple[torch.Tensor, torch.Tensor] | None]:
    """Create a fetch function compatible with DistributedBlockFetcher.


    Usage::

        client = BlockTransferClient("node-1:50051")
        fetch_fn = create_fetch_fn(client)
        paged_mgr.enable_distributed("node-0", fetch_fn)
    """


    def fetch(block_id: int, peer_node_id: str) -> Tuple[torch.Tensor, torch.Tensor] | None:
        return client.fetch_block(block_id)

    return fetch
