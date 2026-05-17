"""PagedAttention: Block-table memory management for KV cache.

Solves memory fragmentation by allocating KV cache in fixed-size blocks
instead of contiguous tensors. Each sequence maintains a block-table
mapping logical block indices to physical block indices.

Inspired by vLLM's PagedAttention architecture.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger


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
    ):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.swap_to_cpu = swap_to_cpu
        self.max_swap_blocks = max_swap_blocks

        # GPU block pool: pre-allocate all blocks
        self.num_blocks = num_blocks
        self._free_blocks: List[int] = list(range(num_blocks))
        self._block_usage: Dict[int, Block] = {}

        # Pre-allocate block tensors on GPU
        # Shape: [num_blocks, num_layers, 2, num_heads, block_size, head_dim]
        # 2 = key + value
        self._kv_pool: Optional[torch.Tensor] = None
        self._allocate_pool()

        # CPU swap space
        self._swap_space: Dict[int, SwapEntry] = {}
        self._lock = threading.Lock()

        # Stats
        self._total_allocations = 0
        self._total_swaps = 0
        self._total_restores = 0

    def _allocate_pool(self) -> None:
        """Pre-allocate the block pool on GPU."""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                self._kv_pool = torch.zeros(
                    (self.num_blocks, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                )
                logger.info(
                    f"PagedAttention: allocated {self.num_blocks} blocks "
                    f"({self._pool_memory_gb:.2f} GB) on {self.device}"
                )
            else:
                logger.warning(f"PagedAttention: device {self.device} not CUDA, using CPU pool")
                self._kv_pool = torch.zeros(
                    (self.num_blocks, self.num_layers, 2, self.num_heads, self.block_size, self.head_dim),
                    dtype=self.dtype,
                    device="cpu",
                )
        except RuntimeError as e:
            logger.error(f"Failed to allocate block pool: {e}")
            raise

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

        Returns:
            Physical block ID, or None if pool is exhausted.
        """
        with self._lock:
            if not self._free_blocks:
                # Try to swap out LRU block
                if self.swap_to_cpu:
                    swapped = self._swap_lru_block()
                    if not swapped:
                        return None
                else:
                    return None

            block_id = self._free_blocks.pop(0)
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
                    self._free_blocks.sort()  # Keep sorted for consistency

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
        if self._kv_pool is None:
            raise RuntimeError("Block pool not allocated")

        block_data = self._kv_pool[block_id, layer_idx]  # [2, num_heads, block_size, head_dim]
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
        if self._kv_pool is None:
            raise RuntimeError("Block pool not allocated")

        num_tokens = key.shape[-2]
        block_data = self._kv_pool[block_id, layer_idx]
        block_data[0, :, offset : offset + num_tokens, :] = key
        block_data[1, :, offset : offset + num_tokens, :] = value

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
        if self._kv_pool is not None:
            key_data = self._kv_pool[lru_id, :, 0].cpu().clone()
            value_data = self._kv_pool[lru_id, :, 1].cpu().clone()

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
        self._free_blocks.sort()
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
            if self._kv_pool is not None:
                self._kv_pool[block_id, :, 0, :, :, :] = entry.key_tensor.to(self.device)
                self._kv_pool[block_id, :, 1, :, :, :] = entry.value_tensor.to(self.device)
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


class PagedAttentionManager:
    """Manages PagedAttention block tables for all active sequences.

    Each sequence gets a BlockTable mapping logical block indices to
    physical blocks in the BlockPool. When sequences complete, their
    blocks are freed back to the pool (automatic defragmentation).
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

        Returns:
            List of (block_id, offset, num_tokens) tuples for the appended data.
        """
        with self._lock:
            table = self._tables.get(request_id)
            if table is None:
                raise KeyError(f"Sequence {request_id} not found")

            allocations = []
            remaining = num_tokens

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
                remaining -= take

            return allocations

    def get_block_table(self, request_id: str) -> Optional[BlockTable]:
        return self._tables.get(request_id)

    def get_physical_blocks(self, request_id: str) -> List[int]:
        """Get list of physical block IDs for a sequence."""
        table = self._tables.get(request_id)
        return table.physical_blocks if table else []

    def free_sequence(self, request_id: str) -> None:
        """Free all blocks for a sequence (defragmentation)."""
        with self._lock:
            table = self._tables.pop(request_id, None)
            if table:
                self.pool.free_blocks(table.physical_blocks)

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
        if pool is None or pool._kv_pool is None:
            return False

        # Check swap space limit
        if pool.max_swap_blocks > 0 and len(pool._swap_space) >= pool.max_swap_blocks:
            return False

        key_data = pool._kv_pool[phys_id, :, 0].cpu().clone()
        value_data = pool._kv_pool[phys_id, :, 1].cpu().clone()

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
            pool._free_blocks.sort()
        return True

    @property
    def active_sequences(self) -> int:
        return len(self._tables)

    def stats(self) -> Dict:
        return {
            "active_sequences": self.active_sequences,
            **self.pool.stats(),
        }
