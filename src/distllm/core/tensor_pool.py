"""Pre-allocated tensor buffer pool for batch processing.

Avoids repeated torch.tensor() allocations by maintaining a pool
of reusable buffers sized for common batch shapes.
"""

import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class TensorPool:
    """Pool of pre-allocated tensor buffers for batch processing.

    Buffers are keyed by (device, dtype) and stored in shape buckets
    to minimize reallocations while avoiding excessive memory usage.

    Usage:
        pool = TensorPool()
        buffer = pool.get_buffer((batch_size, max_len), dtype=torch.long, device="cuda")
        buffer[:, :seq_len].copy_(input_data)
        pool.release(buffer)
    """

    def __init__(self, max_buffers: int = 64, growth_factor: float = 1.5):
        self.max_buffers = max_buffers
        self.growth_factor = growth_factor
        # Buckets: (device, dtype) -> list of (shape, tensor)
        self._pools: Dict[Tuple[str, torch.dtype], List[Tuple[Tuple[int, ...], torch.Tensor]]] = defaultdict(list)
        self._total_buffers = 0

    def get_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.long,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Get a buffer with at least the requested shape.

        Returns a tensor with shape >= requested shape. The caller
        should only use the portion matching the requested shape.
        """
        pool_key = (device, dtype)
        pool = self._pools[pool_key]

        # Find a buffer with sufficient size
        for i, (buf_shape, buf) in enumerate(pool):
            if len(buf_shape) == len(shape) and all(b >= s for b, s in zip(buf_shape, shape)):
                pool.pop(i)
                return buf[:list(shape)]

        # Allocate new buffer with some headroom for growth
        padded_shape = tuple(max(1, int(s * self.growth_factor)) for s in shape)
        buf = torch.empty(padded_shape, dtype=dtype, device=device)
        self._total_buffers += 1
        return buf[:list(shape)]

    def release(self, tensor: torch.Tensor) -> None:
        """Return a buffer to the pool for reuse."""
        if self._total_buffers >= self.max_buffers:
            return  # Pool full, let GC handle it

        device = str(tensor.device)
        dtype = tensor.dtype
        pool_key = (device, dtype)
        original_shape = tuple(tensor.shape)

        # Store the buffer (keep full allocated size, not the slice)
        # We can't recover the full size from a slice, so we store the slice
        # and rely on the growth factor to cover future needs
        self._pools[pool_key].append((original_shape, tensor))

    def clear(self) -> None:
        """Release all pooled buffers."""
        self._pools.clear()
        self._total_buffers = 0

    @property
    def pool_size(self) -> int:
        return sum(len(p) for p in self._pools.values())


class GPUMemoryPool:
    """Pre-allocated GPU memory pool for KV cache buffers.

    Avoids GPU memory fragmentation by pre-allocating large contiguous
    buffers and handing out slices as needed.

    Usage:
        pool = GPUMemoryPool(total_gb=8.0, block_size_mb=16)
        # Allocate a KV cache block: [batch, heads, seq, head_dim]
        cache = pool.allocate((1, 32, 512, 64), dtype=torch.float16)
        pool.free(cache)
    """

    def __init__(self, total_gb: float = 8.0, block_size_mb: float = 16.0):
        self.total_bytes = int(total_gb * 1024 * 1024 * 1024)
        self.block_bytes = int(block_size_mb * 1024 * 1024)
        self._blocks: List[torch.Tensor] = []
        self._free_blocks: List[torch.Tensor] = []
        self._allocated: Dict[int, torch.Tensor] = {}
        self._device: Optional[str] = None

    def _init_device(self, device: str) -> None:
        """Initialize the memory pool on the given device."""
        if self._device == device:
            return

        self._device = device
        num_blocks = max(1, self.total_bytes // self.block_bytes)

        # Pre-allocate blocks
        for _ in range(num_blocks):
            block = torch.empty(self.block_bytes // 4, dtype=torch.float32, device=device)
            self._blocks.append(block)
            self._free_blocks.append(block)

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> Optional[torch.Tensor]:
        """Allocate a tensor from the pool.

        Returns None if insufficient memory.
        """
        required_bytes = 1
        for s in shape:
            required_bytes *= s
        required_bytes *= torch.tensor([], dtype=dtype).element_size()

        if required_bytes > self.block_bytes:
            # Too large for pool, allocate directly
            return torch.empty(shape, dtype=dtype, device=device)

        self._init_device(device)

        if not self._free_blocks:
            return None  # Pool exhausted

        block = self._free_blocks.pop()
        # Reshape block to requested shape (view, not copy)
        tensor = block[:required_bytes // block.element_size()].view(dtype).reshape(shape)
        self._allocated[id(tensor)] = block
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool."""
        block_id = id(tensor)
        if block_id in self._allocated:
            block = self._allocated.pop(block_id)
            self._free_blocks.append(block)

    def free_all(self) -> None:
        """Free all allocated tensors."""
        self._free_blocks.extend(self._allocated.values())
        self._allocated.clear()

    @property
    def used_blocks(self) -> int:
        return len(self._allocated)

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)
