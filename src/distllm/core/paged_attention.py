"""PagedAttention: Block-table memory management for KV cache.

Solves memory fragmentation by allocating KV cache in fixed-size blocks
instead of contiguous tensors. Each sequence maintains a block-table
mapping logical block indices to physical block indices.

Supports **distributed prefix sharing**: nodes advertise their page-table
entries via a Merkle tree over the gossip protocol. On a cache miss, a
node fetches the raw block data from a peer via gRPC block streaming.

Inspired by vLLM's PagedAttention architecture.
"""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from loguru import logger

from distllm.core.merkle_tree import MerkleTree, EMPTY_HASH


@dataclass
class Block:
    """A single fixed-size KV cache block."""
    block_id: int  # Physical block index in the pool
    num_tokens: int = 0  # How many tokens are stored (<= block_size)
    ref_count: int = 1  # Reference count for prefix sharing
    last_access: float = field(default_factory=time.time)

    def is_full(self, block_size: int) -> bool:
        return self.num_tokens >= block_size

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


@dataclass
class BlockTable:
    """Maps logical block indices to physical blocks for one sequence."""
    request_id: str
    physical_blocks: List[int] = field(default_factory=list)  # Physical block IDs
    logical_to_physical: Dict[int, int] = field(default_factory=dict)  # Logical -> Physical mapping
    num_logical_blocks: int = 0

    def append_block(self, physical_block_id: int) -> int:
        """Append a new physical block and return its logical index."""
        logical_idx = self.num_logical_blocks
        self.logical_to_physical[logical_idx] = physical_block_id
        self.physical_blocks.append(physical_block_id)
        self.num_logical_blocks += 1
        return logical_idx

    def get_physical(self, logical_idx: int) -> Optional[int]:
        """Get physical block ID for a logical block index."""
        return self.logical_to_physical.get(logical_idx)

    def total_capacity(self, block_size: int) -> int:
        return self.num_logical_blocks * block_size


@dataclass
class SwapEntry:
    """Host memory swap entry for a block that was evicted from GPU."""
    block_id: int
    key_tensor: Optional[torch.Tensor] = None
    value_tensor: Optional[torch.Tensor] = None
    device: str = "cpu"
    swap_time: float = field(default_factory=time.time)


class BlockPool:
    """Fixed-size block pool for KV cache storage.

    Manages a pool of pre-allocated KV cache blocks on GPU (and optionally
    host memory for swap). Blocks are allocated on demand and freed when
    sequences complete, enabling automatic defragmentation.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        swap_to_cpu: bool = False,
        max_swap_blocks: int = 0,
        auto_expand: bool = True,
    ):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.swap_to_cpu = swap_to_cpu
        self.max_swap_blocks = max_swap_blocks
        self.auto_expand = auto_expand
        self.use_fp8 = False  # FP8 KV cache mode

        # Multi-pool: list of block pool tensors for dynamic resizing
        self._pools: List[torch.Tensor] = []
        self._pool_boundaries: List[int] = []  # Cumulative block counts per pool
        self._initial_num_blocks = num_blocks
        # FP8 scales: parallel storage for per-block scale factors [num_blocks, num_layers, 2]
        self._fp8_scales: Optional[torch.Tensor] = None
        # FP8 block pools: parallel fp8 storage (half the size of main pools)
        self._fp8_pools: List[torch.Tensor] = []

        # GPU block pool: lazy allocation starting small (scales to 70B+)
        self.num_blocks = 0
        self._free_blocks: List[int] = []
        self._block_usage: Dict[int, Block] = {}

        # Start with minimal allocation; rely on _expand_pool for growth
        initial = max(1, min(num_blocks, 64))
        self._allocate_initial_pool(initial)

        # CPU swap space
        self._swap_space: Dict[int, SwapEntry] = {}
        self._lock = threading.Lock()

        # Stats
        self._total_allocations = 0
        self._total_swaps = 0
        self._total_restores = 0
        self._total_expansions = 0

    def _allocate_initial_pool(self, num_blocks: int) -> None:
        """Pre-allocate the initial block pool on GPU."""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                pool = torch.zeros(
                    (num_blocks, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                self._pools.append(pool)
                self._pool_boundaries.append(num_blocks)
                self.num_blocks = num_blocks
                logger.info(
                    f"PagedAttention: allocated {num_blocks} blocks "
                    f"({self._pool_memory_gb:.2f} GB) on {self.device}"
                )
            else:
                logger.warning(f"PagedAttention: device {self.device} not CUDA, using CPU pool")
                pool = torch.zeros(
                    (num_blocks, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                )
                self._pools.append(pool)
                self._pool_boundaries.append(num_blocks)
                self.num_blocks = num_blocks
        except RuntimeError as e:
            logger.error(f"Failed to allocate block pool: {e}")
            raise

    def _expand_pool(self, grow_by: int | None = None) -> bool:
        """Allocate an additional block pool segment.

        Args:
            grow_by: Number of blocks to add. Defaults to max(num_blocks, 64).

        Returns:
            True if pool was expanded.
        """
        if not self.auto_expand:
            return False

        grow = grow_by or max(self.num_blocks, 64)
        new_start = self.num_blocks

        try:
            if self.device == "cuda" and torch.cuda.is_available():
                new_pool = torch.zeros(
                    (grow, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                new_pool = torch.zeros(
                    (grow, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                )

            self._pools.append(new_pool)
            self._pool_boundaries.append(new_start + grow)
            self._free_blocks.extend(range(new_start, new_start + grow))
            self.num_blocks += grow
            self._total_expansions += 1

            logger.info(
                f"PagedAttention: expanded pool by {grow} blocks "
                f"(total: {self.num_blocks}, {self._pool_memory_gb:.2f} GB)"
            )
            return True
        except RuntimeError as e:
            logger.error(f"Failed to expand block pool: {e}")
            return False

    def _get_pool_and_offset(self, block_id: int) -> Tuple[torch.Tensor, int]:
        """Find which pool contains this block and its local offset.

        Returns:
            (pool_tensor, local_offset)
        """
        for i, boundary in enumerate(self._pool_boundaries):
            prev_boundary = self._pool_boundaries[i - 1] if i > 0 else 0
            if prev_boundary <= block_id < boundary:
                return self._pools[i], block_id - prev_boundary
        raise ValueError(f"Block {block_id} not found in any pool (total: {self.num_blocks})")

    @property
    def _pool_memory_gb(self) -> float:
        """Total memory of the block pool in GB."""
        bytes_per_block = (
            self.num_layers * 2 * self.num_heads * self.block_size * self.head_dim * self.dtype.itemsize
        )
        return (self.num_blocks * bytes_per_block) / (1024 ** 3)

    @property
    def free_count(self) -> int:
        return len(self._free_blocks)

    @property
    def used_count(self) -> int:
        return self.num_blocks - len(self._free_blocks)

    @property
    def utilization(self) -> float:
        return self.used_count / self.num_blocks if self.num_blocks > 0 else 0.0

    def allocate_block(self) -> Optional[int]:
        """Allocate a free block from the pool.

        Auto-expands the pool if exhausted and auto_expand is enabled.

        Returns:
            Physical block ID, or None if pool is exhausted.
        """
        with self._lock:
            if not self._free_blocks:
                # Try auto-expand first
                if self.auto_expand and self._expand_pool():
                    pass  # New blocks added to _free_blocks
                # Try to swap out LRU block
                elif self.swap_to_cpu:
                    swapped = self._swap_lru_block()
                    if not swapped:
                        return None
                else:
                    return None

            block_id = self._free_blocks.pop()
            self._block_usage[block_id] = Block(block_id=block_id)
            self._total_allocations += 1
            return block_id

    def free_block(self, block_id: int) -> None:
        """Free a block back to the pool."""
        with self._lock:
            if block_id in self._block_usage:
                self._block_usage[block_id].ref_count -= 1
                if self._block_usage[block_id].is_free:
                    del self._block_usage[block_id]
                    self._free_blocks.append(block_id)

    def free_blocks(self, block_ids: List[int]) -> None:
        """Free multiple blocks."""
        for bid in block_ids:
            self.free_block(bid)

    def get_kv_slice(
        self,
        block_id: int,
        layer_idx: int,
        num_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K/V tensors for a specific block and layer.

        Args:
            block_id: Physical block ID.
            layer_idx: Transformer layer index.
            num_tokens: Number of valid tokens (truncates to actual used portion).

        Returns:
            (key, value) tensors of shape [num_heads, tokens, head_dim].
        """
        # If FP8 mode, read from fp8 pool and dequantize
        if self.use_fp8 and self._fp8_pools:
            fp8_pool, fp8_offset = self._get_fp8_pool_and_offset(block_id)
            fp8_block = fp8_pool[fp8_offset, layer_idx]
            key = fp8_block[0]  # [num_heads, block_size, head_dim]
            value = fp8_block[1]
            if num_tokens is not None and num_tokens < self.block_size:
                key = key[:, :num_tokens, :]
                value = value[:, :num_tokens, :]
            from distllm.core.fp8_engine import dequantize_kv_fp8
            scale_k = self._fp8_scales[block_id, layer_idx, 0]
            scale_v = self._fp8_scales[block_id, layer_idx, 1]
            key = dequantize_kv_fp8(key, scale_k)
            value = dequantize_kv_fp8(value, scale_v)
        else:
            pool, offset = self._get_pool_and_offset(block_id)
            block_data = pool[offset, layer_idx]  # [2, num_heads, block_size, head_dim]
            key = block_data[0]  # [num_heads, block_size, head_dim]
            value = block_data[1]
            if num_tokens is not None and num_tokens < self.block_size:
                key = key[:, :num_tokens, :]
                value = value[:, :num_tokens, :]

        return key, value

    def set_kv_slice(
        self,
        block_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        offset: int = 0,
    ) -> None:
        """Write K/V tensors into a specific block and layer.

        Args:
            block_id: Physical block ID.
            layer_idx: Transformer layer index.
            key: Key tensor [num_heads, num_tokens, head_dim].
            value: Value tensor [num_heads, num_tokens, head_dim].
            offset: Starting position within the block.
        """
        pool, local_offset = self._get_pool_and_offset(block_id)

        num_tokens = key.shape[-2]

        # Quantize to FP8 if enabled — store in fp8 pool, not in main pool
        if self.use_fp8:
            from distllm.core.fp8_engine import quantize_kv_fp8
            if self._fp8_scales is None:
                self._init_fp8_scales()
            fp8_k, scale_k = quantize_kv_fp8(key)
            fp8_v, scale_v = quantize_kv_fp8(value)
            self._fp8_scales[block_id, layer_idx, 0] = scale_k
            self._fp8_scales[block_id, layer_idx, 1] = scale_v
            if not self._fp8_pools:
                self._allocate_fp8_pools()
            fp8_pool, fp8_local_offset = self._get_fp8_pool_and_offset(block_id)
            fp8_block = fp8_pool[fp8_local_offset, layer_idx]
            fp8_block[0, :, offset : offset + num_tokens, :] = fp8_k
            fp8_block[1, :, offset : offset + num_tokens, :] = fp8_v
        else:
            block_data = pool[local_offset, layer_idx]
            block_data[0, :, offset : offset + num_tokens, :] = key
            block_data[1, :, offset : offset + num_tokens, :] = value

    def _init_fp8_scales(self) -> None:
        """Initialize FP8 scale storage tensor."""
        self._fp8_scales = torch.zeros(
            (self.num_blocks, self.num_layers, 2),
            dtype=torch.float32,
            device=self.device,
        )

    def enable_fp8_storage(self) -> None:
        """Enable FP8 KV cache storage for this block pool."""
        self.use_fp8 = True
        self._init_fp8_scales()

    def _allocate_fp8_pools(self) -> None:
        """Allocate parallel fp8 block pools (half memory vs self.dtype)."""
        for pool in self._pools:
            fp8_pool = torch.empty_like(pool, dtype=torch.float8_e4m3fn)
            self._fp8_pools.append(fp8_pool)

    def _get_fp8_pool_and_offset(self, block_id: int) -> tuple[torch.Tensor, int]:
        """Find the fp8 pool and local offset for a given block ID."""
        for pool_idx, boundary in enumerate(self._pool_boundaries):
            if block_id < boundary:
                prev = self._pool_boundaries[pool_idx - 1] if pool_idx > 0 else 0
                return self._fp8_pools[pool_idx], block_id - prev
        raise IndexError(f"Block {block_id} out of range")

    def _swap_lru_block(self) -> bool:
        """Swap out the least recently used block to CPU.

        Returns:
            True if a block was successfully swapped.
        """
        # Find LRU block among used blocks
        lru_id = None
        lru_time = float("inf")
        for bid, block in self._block_usage.items():
            if block.last_access < lru_time:
                lru_time = block.last_access
                lru_id = bid

        if lru_id is None:
            return False

        # Check swap space limit
        if self.max_swap_blocks > 0 and len(self._swap_space) >= self.max_swap_blocks:
            return False

        # Copy block data to CPU
        pool, offset = self._get_pool_and_offset(lru_id)
        key_data = pool[offset, :, 0].cpu().clone()
        value_data = pool[offset, :, 1].cpu().clone()

        self._swap_space[lru_id] = SwapEntry(
            block_id=lru_id,
            key_tensor=key_data,
            value_tensor=value_data,
            device="cpu",
        )
        self._total_swaps += 1
        logger.debug(f"Swapped block {lru_id} to CPU")

        # Free the GPU block
        del self._block_usage[lru_id]
        self._free_blocks.append(lru_id)
        return True

    def restore_block(self, block_id: int) -> bool:
        """Restore a swapped block from CPU to GPU.

        Returns:
            True if the block was successfully restored.
        """
        with self._lock:
            if block_id not in self._swap_space:
                return False

            entry = self._swap_space.pop(block_id)
            pool, offset = self._get_pool_and_offset(block_id)
            pool[offset, :, 0, :, :, :] = entry.key_tensor.to(self.device)
            pool[offset, :, 1, :, :, :] = entry.value_tensor.to(self.device)
            self._total_restores += 1
            logger.debug(f"Restored block {block_id} from CPU")

            self._block_usage[block_id] = Block(block_id=block_id)
            if block_id in self._free_blocks:
                self._free_blocks.remove(block_id)
            return True

    def get_swap_stats(self) -> Dict:
        return {
            "swapped_blocks": len(self._swap_space),
            "total_swaps": self._total_swaps,
            "total_restores": self._total_restores,
            "total_expansions": self._total_expansions,
            "swap_memory_gb": self._swap_memory_gb,
        }

    @property
    def _swap_memory_gb(self) -> float:
        total = 0
        for entry in self._swap_space.values():
            if entry.key_tensor is not None:
                total += entry.key_tensor.numel() * entry.key_tensor.element_size()
            if entry.value_tensor is not None:
                total += entry.value_tensor.numel() * entry.value_tensor.element_size()
        return total / (1024 ** 3)

    def stats(self) -> Dict:
        return {
            "total_blocks": self.num_blocks,
            "free_blocks": self.free_count,
            "used_blocks": self.used_count,
            "utilization": round(self.utilization, 4),
            "block_size": self.block_size,
            "pool_memory_gb": round(self._pool_memory_gb, 2),
            **self.get_swap_stats(),
        }


def _byte_hash(data: bytes) -> str:
    """SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


class DistributedBlockFetcher:
    """Callback interface for fetching remote blocks from a peer node.

    Set by :meth:`PagedAttentionManager.enable_distributed` to wire up
    the actual gRPC/HTTP transport without creating a circular import.
    """

    def __init__(self) -> None:
        self._fetch_fn: Callable[[int, str], Tuple[torch.Tensor, torch.Tensor] | None] | None = None
        self._node_id: str = ""

    def set_transport(
        self,
        node_id: str,
        fetch_fn: Callable[[int, str], Tuple[torch.Tensor, torch.Tensor] | None],
    ) -> None:
        self._node_id = node_id
        self._fetch_fn = fetch_fn

    def fetch(self, block_id: int, peer_node_id: str) -> Tuple[torch.Tensor, torch.Tensor] | None:
        if self._fetch_fn is None:
            return None
        return self._fetch_fn(block_id, peer_node_id)


class PagedAttentionManager:
    """Manages PagedAttention block tables for all active sequences.

    Each sequence gets a BlockTable mapping logical block indices to
    physical blocks in the BlockPool. When sequences complete, their
    blocks are freed back to the pool (automatic defragmentation).

    Supports distributed prefix sharing via Merkle tree sync over
    the gossip protocol (see :meth:`enable_distributed`).
    """

    def __init__(
        self,
        num_blocks: int = 256,
        block_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        swap_to_cpu: bool = False,
        max_swap_blocks: int = 0,
    ):
        self.block_size = block_size
        self.pool = BlockPool(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            swap_to_cpu=swap_to_cpu,
            max_swap_blocks=max_swap_blocks,
        )
        self._tables: Dict[str, BlockTable] = {}
        self._lock = threading.Lock()

        # Distributed prefix sharing state
        self._distributed_enabled: bool = False
        self._node_id: str = ""
        self._block_fetcher = DistributedBlockFetcher()
        # block_hash → physical block ID (local blocks)
        self._local_block_hashes: dict[str, int] = {}
        # physical block ID → block hash (reverse map for Merkle update)
        self._block_id_to_hash: dict[int, str] = {}
        # physical block ID → peer node ID that has this block (remote)
        self._remote_blocks: OrderedDict[int, str] = OrderedDict()
        self._max_remote_blocks: int = 10000
        self._merkle_tree = MerkleTree()

    # ------------------------------------------------------------------
    # Distributed prefix sharing
    # ------------------------------------------------------------------

    def enable_distributed(
        self,
        node_id: str,
        fetch_fn: Callable[[int, str], Tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> None:
        """Enable cross-node block sharing.

        Once enabled, each appended block's hash is tracked in a Merkle
        tree whose root is shared via gossip.  Blocks discovered on peers
        are fetched on demand and cached locally.

        Args:
            node_id: This node's identifier (used in gossip advertisements).
            fetch_fn: Optional callback to fetch a block from a peer
                      ``(block_id, peer_node_id) → (key, value)``.
        """
        self._distributed_enabled = True
        self._node_id = node_id
        if fetch_fn is not None:
            self._block_fetcher.set_transport(node_id, fetch_fn)

    def get_merkle_root(self) -> str:
        """Current Merkle root of local page-table block hashes."""
        if not self._distributed_enabled:
            return EMPTY_HASH
        return self._merkle_tree.root

    def get_page_table_hashes(self) -> List[str]:
        """All block hashes in the local page table, ordered by physical block ID."""
        if not self._distributed_enabled:
            return []
        return list(self._merkle_tree._leaves)

    def get_differing_blocks(self, other_root: str) -> List[int]:
        """Return physical block IDs whose hashes differ from *other_root*.

        Args:
            other_root: Merkle root from a peer node.

        Returns:
            List of physical block IDs to fetch from the peer.
        """
        if not self._distributed_enabled or other_root == EMPTY_HASH:
            return []

        other_tree = MerkleTree(list(self._merkle_tree._leaves) if self._merkle_tree.root != other_root else [])
        # We can't reconstruct the peer's leaves — but we CAN detect if
        # our local tree differs.  If roots match, nothing changed.
        if self._merkle_tree.root == other_root:
            return []

        # Get differing leaf indices, map back to physical block IDs
        diff_indices = self._merkle_tree.diff(other_tree)
        phys_ids = []
        block_list = list(self._block_id_to_hash.keys())
        for idx in diff_indices:
            if idx < len(block_list):
                phys_ids.append(block_list[idx])
        return phys_ids

    def store_remote_block_location(self, block_hash: str, peer_node_id: str) -> bool:
        """Record that a peer node has a block with the given hash.

        Returns True if a local block query should be retried with a
        remote fetch (i.e. the block isn't already cached locally).
        """
        if block_hash in self._local_block_hashes:
            return False  # already have this block locally
        # We don't have a physical ID yet — peer will provide it on fetch
        return True

    def fetch_block_from_peer(
        self, block_id: int, peer_node_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        """Fetch a remote block's KV data from *peer_node_id*.

        Returns ``(key_tensor, value_tensor)`` or None on failure.
        """
        return self._block_fetcher.fetch(block_id, peer_node_id)

    def _compute_block_hash(self, block_id: int) -> str:
        """SHA-256 hash of all KV data in this block (across all layers).

        The hash covers the concatenated key+value tensors for every
        transformer layer stored in this physical block.
        """
        pool = self.pool
        layer_count = pool.num_layers if hasattr(pool, 'num_layers') else 1
        digests: list[bytes] = []

        for layer_idx in range(layer_count):
            try:
                k, v = pool.get_kv_slice(block_id, layer_idx)
                digests.append(k.numpy().tobytes() if hasattr(k, 'numpy') else k.cpu().numpy().tobytes())
                digests.append(v.numpy().tobytes() if hasattr(v, 'numpy') else v.cpu().numpy().tobytes())
            except Exception:
                continue

        return _byte_hash(b"".join(digests)) if digests else EMPTY_HASH

    def _update_merkle_tree(self) -> None:
        """Rebuild the Merkle tree from current block hashes."""
        ordered_hashes = [
            self._block_id_to_hash[pid]
            for pid in sorted(self._block_id_to_hash.keys())
        ]
        self._merkle_tree.update(ordered_hashes)

    def register_remote_block(self, block_id: int, block_hash: str, peer_node_id: str) -> None:
        """Register a block fetched from a peer as a local cache."""
        with self._lock:
            self._remote_blocks[block_id] = peer_node_id
            self._remote_blocks.move_to_end(block_id)
            self._block_id_to_hash[block_id] = block_hash
            self._local_block_hashes[block_hash] = block_id
            if len(self._remote_blocks) > self._max_remote_blocks:
                evicted_id, _ = self._remote_blocks.popitem(last=False)
                evicted_hash = self._block_id_to_hash.pop(evicted_id, None)
                if evicted_hash:
                    self._local_block_hashes.pop(evicted_hash, None)
            self._update_merkle_tree()

    def create_sequence(self, request_id: str) -> BlockTable:
        """Create a new block table for a sequence.

        Allocates the first block eagerly.
        """
        with self._lock:
            table = BlockTable(request_id=request_id)
            # Allocate first block
            block_id = self.pool.allocate_block()
            if block_id is None:
                raise RuntimeError("No free blocks available for new sequence")
            table.append_block(block_id)
            self._tables[request_id] = table
            return table

    def append_tokens(self, request_id: str, num_tokens: int) -> List[int]:
        """Append tokens to a sequence, allocating new blocks as needed.

        When distributed mode is enabled, each newly allocated block's
        hash is computed and the Merkle tree is updated for gossip sync.

        Returns:
            List of (block_id, offset, num_tokens) tuples for the appended data.
        """
        with self._lock:
            table = self._tables.get(request_id)
            if table is None:
                raise KeyError(f"Sequence {request_id} not found")

            allocations = []
            remaining = num_tokens
            new_blocks: list[int] = []

            # Fill the last block if it has space
            if table.physical_blocks:
                last_phys = table.physical_blocks[-1]
                last_block = self.pool._block_usage.get(last_phys)
                if last_block and not last_block.is_full(self.block_size):
                    space = self.block_size - last_block.num_tokens
                    take = min(space, remaining)
                    last_block.num_tokens += take
                    allocations.append((last_phys, last_block.num_tokens - take, take))
                    remaining -= take

            # Allocate new blocks for remaining tokens
            while remaining > 0:
                block_id = self.pool.allocate_block()
                if block_id is None:
                    raise RuntimeError(
                        f"Block pool exhausted: need {remaining} more tokens but no free blocks"
                    )
                logical_idx = table.append_block(block_id)
                block = self.pool._block_usage[block_id]
                take = min(self.block_size, remaining)
                block.num_tokens = take
                allocations.append((block_id, 0, take))
                new_blocks.append(block_id)
                remaining -= take

            # Update Merkle tree for each new block (distributed mode)
            if self._distributed_enabled and new_blocks:
                for bid in new_blocks:
                    if bid not in self._block_id_to_hash:
                        bh = self._compute_block_hash(bid)
                        self._block_id_to_hash[bid] = bh
                        self._local_block_hashes[bh] = bid
                self._update_merkle_tree()

            return allocations

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        return self._tables.get(request_id)

    def get_physical_blocks(self, request_id: str) -> List[int]:
        """Get list of physical block IDs for a sequence."""
        table = self._tables.get(request_id)
        return table.physical_blocks if table else []

    def free_sequence(self, request_id: str) -> None:
        """Free all blocks for a sequence (defragmentation).

        Cleans up distributed block tracking when enabled.
        """
        with self._lock:
            table = self._tables.pop(request_id, None)
            if table:
                self.pool.free_blocks(table.physical_blocks)
                if self._distributed_enabled:
                    for pid in table.physical_blocks:
                        bh = self._block_id_to_hash.pop(pid, None)
                        if bh:
                            self._local_block_hashes.pop(bh, None)
                        self._remote_blocks.pop(pid, None)
                    self._update_merkle_tree()

    def free_layer_kv(
        self,
        request_id: str,
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> Tuple[List[Tuple[int, int, int]], bool]:
        """Store new K/V tensors into blocks for a sequence.

        Handles block allocation automatically.

        Returns:
            (allocations, needs_new_block) — allocations are (block_id, offset, num_tokens).
        """
        num_tokens = new_key.shape[-2]
        allocations = self.append_tokens(request_id, num_tokens)

        # Write the actual KV data into blocks
        offset = 0
        for block_id, block_offset, block_tokens in allocations:
            k_slice = new_key[:, offset : offset + block_tokens, :]
            v_slice = new_value[:, offset : offset + block_tokens, :]
            self.pool.set_kv_slice(block_id, layer_idx, k_slice, v_slice)
            offset += block_tokens

        return allocations, offset >= num_tokens

    def gather_kv_for_attention(
        self,
        request_id: str,
        layer_idx: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V tensors from all blocks for attention computation.

        Reconstructs the contiguous KV tensors that the attention layer expects.

        Args:
            request_id: Sequence ID.
            layer_idx: Transformer layer index.
            seq_len: Total sequence length to gather.

        Returns:
            (key, value) tensors of shape [num_heads, seq_len, head_dim].
        """
        table = self._tables.get(request_id)
        if table is None:
            raise KeyError(f"Sequence {request_id} not found")

        pool = self.pool
        num_heads = pool.num_heads
        head_dim = pool.head_dim
        dtype = pool.dtype
        device = pool.device

        key_out = torch.zeros((num_heads, seq_len, head_dim), dtype=dtype, device=device)
        value_out = torch.zeros((num_heads, seq_len, head_dim), dtype=dtype, device=device)

        pos = 0
        for phys_id in table.physical_blocks:
            block = pool._block_usage.get(phys_id)
            tokens = block.num_tokens if block else pool.block_size

            # Handle potential overflow beyond requested seq_len
            take = min(tokens, seq_len - pos)
            if take <= 0:
                break

            k, v = pool.get_kv_slice(phys_id, layer_idx, num_tokens=tokens)
            key_out[:, pos : pos + take, :] = k[:, :take, :]
            value_out[:, pos : pos + take, :] = v[:, :take, :]
            pos += take

        return key_out[:, :pos, :], value_out[:, :pos, :]

    def swap_out_sequence(self, request_id: str) -> bool:
        """Swap out all blocks of a sequence to CPU (for memory pressure).

        Returns:
            True if successfully swapped.
        """
        table = self._tables.get(request_id)
        if table is None:
            return False

        swapped = 0
        for phys_id in list(table.physical_blocks):
            if self._swap_block_to_cpu(phys_id):
                swapped += 1

        return swapped > 0

    def _swap_block_to_cpu(self, phys_id: int) -> bool:
        """Swap a single block from GPU to CPU memory."""
        pool = self.pool
        if pool is None or not pool._pools:
            return False

        # Check swap space limit
        if pool.max_swap_blocks > 0 and len(pool._swap_space) >= pool.max_swap_blocks:
            return False

        block_pool, offset = pool._get_pool_and_offset(phys_id)
        key_data = block_pool[offset, :, 0].cpu().clone()
        value_data = block_pool[offset, :, 1].cpu().clone()

        pool._swap_space[phys_id] = SwapEntry(
            block_id=phys_id,
            key_tensor=key_data,
            value_tensor=value_data,
            device="cpu",
        )
        pool._total_swaps += 1

        # Free the GPU block
        pool._block_usage.pop(phys_id, None)
        if phys_id not in pool._free_blocks:
            pool._free_blocks.append(phys_id)
        return True

    @property
    def active_sequences(self) -> int:
        return len(self._tables)

    def stats(self) -> Dict:
        return {
            "active_sequences": self.active_sequences,
            **self.pool.stats(),
        }
