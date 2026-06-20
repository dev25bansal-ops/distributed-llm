"""KV cache block management for PagedAttention and preemption state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import torch

    from distllm.core.scheduler.sequence import Sequence

__all__ = ["KVCacheManager"]


class KVCacheManager:
    """Manages PagedAttention block allocation, CPU swap, and KV compression.

    Encapsulates all KV cache block operations that were previously spread
    across ``BatchScheduler``: block alloc/free, CPU eviction/restore,
    copy-on-write, and int4/int8 compression for preempted sequences.

    Args:
        paged_attention_mgr: Optional PagedAttention manager instance.
    """

    def __init__(self, paged_attention_mgr: object | None = None) -> None:
        self._paged_attention_mgr = paged_attention_mgr

    # ── PagedAttention manager binding ────────────────────────────────────

    def set_paged_attention(self, mgr: object) -> None:
        """Connect to PagedAttention manager for KV block-aware scheduling."""
        self._paged_attention_mgr = mgr

    # ── Block allocation ──────────────────────────────────────────────────

    def allocate_paged_blocks(self, seq: Sequence) -> list[int] | None:
        """Allocate PagedAttention blocks for a sequence.

        Called when a sequence enters the active batch. If PagedAttention
        is not configured, returns None (caller uses flat KV cache).

        Returns:
            List of allocated block IDs, or None if PagedAttention not active.
        """
        if self._paged_attention_mgr is None:
            return None
        try:
            num_tokens = len(seq.prompt_tokens) + seq.max_new_tokens
            block_ids = self._paged_attention_mgr.allocate_sequence(
                seq.request_id, num_tokens,
            )
            return block_ids
        except RuntimeError as e:
            logger.warning(
                f"PagedAttention allocation failed for {seq.request_id}: {e}"
            )
            return None

    def free_paged_blocks(self, request_id: str) -> None:
        """Free PagedAttention blocks for a completed sequence."""
        if self._paged_attention_mgr is not None:
            try:
                self._paged_attention_mgr.free_sequence(request_id)
            except Exception as e:
                logger.warning(
                    f"Failed to free paged blocks for {request_id}: {e}"
                )

    def paged_kv_block_count(self, tokens: int) -> int:
        """Estimate number of PagedAttention blocks needed for this many tokens."""
        if self._paged_attention_mgr is not None:
            block_size = getattr(self._paged_attention_mgr, 'block_size', 16)
        else:
            block_size = 16
        return (tokens + block_size - 1) // block_size

    # ── CPU swap (eviction / restore) ─────────────────────────────────────

    def swap_evict_to_cpu(
        self,
        active: dict[str, Sequence],
        lock: Any,
        min_blocks: int = 1,
    ) -> int:
        """Evict lowest-priority active sequences to CPU to free GPU blocks.

        Selects sequences with the highest numeric priority (least important:
        3=low > 2=normal > 1=high > 0=critical), breaking ties by oldest first.

        Args:
            active: The active sequences dict (request_id -> Sequence).
            lock: Threading lock protecting ``active``.
            min_blocks: Minimum number of blocks to free.

        Returns:
            Number of blocks freed.
        """
        if self._paged_attention_mgr is None:
            return 0

        # Find lowest-priority active sequences (highest numeric priority first,
        # then oldest first — evict cheap/old work before expensive/new work)
        with lock:
            candidates = sorted(
                active.values(),
                key=lambda s: (s.priority, -s.created_at),
            )

        freed = 0
        for seq in candidates:
            if freed >= min_blocks:
                break
            try:
                blocks_freed = self._paged_attention_mgr.swap_blocks_to_cpu(
                    seq.request_id,
                )
                freed += blocks_freed
                logger.debug(
                    f"Swapped {blocks_freed} blocks to CPU for {seq.request_id}"
                )
            except Exception as e:
                logger.debug(
                    f"Failed to swap blocks to CPU for {seq.request_id}: {e}"
                )
                continue
        return freed

    def restore_from_cpu(self, request_id: str) -> int:
        """Restore a sequence's blocks from CPU back to GPU."""
        if self._paged_attention_mgr is None:
            return 0
        try:
            return self._paged_attention_mgr.swap_blocks_to_gpu(request_id)
        except Exception as e:
            logger.warning(
                f"Failed to restore blocks from CPU for {request_id}: {e}"
            )
            return 0

    def copy_on_write(self, source_id: str, dest_id: str) -> None:
        """Copy-on-write for shared prefixes (beam search, speculative decoding)."""
        if self._paged_attention_mgr is not None:
            try:
                self._paged_attention_mgr.copy_on_write(source_id, dest_id)
            except Exception as e:
                logger.warning(
                    f"Copy-on-write failed from {source_id} to {dest_id}: {e}"
                )

    # ── KV compression for preemption ─────────────────────────────────────

    @staticmethod
    def _compress_tensor(tensor: "torch.Tensor", method: str) -> dict:
        """Compress a single tensor with int4 or int8 quantization.

        Returns a dict with the quantized tensor and scale factors,
        which can be restored with ``_decompress_tensor()``.
        """
        import torch

        if method == "int4":
            scale = (
                tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 7.0
            )
            quantized = (tensor / scale).clamp(-7, 7).to(torch.int8)
            return {
                "_compressed": True,
                "method": "int4",
                "data": quantized,
                "scale": scale,
            }

        if method == "int8":
            scale = (
                tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 127.0
            )
            quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
            return {
                "_compressed": True,
                "method": "int8",
                "data": quantized,
                "scale": scale,
            }

        return {"_compressed": False, "data": tensor}

    @staticmethod
    def _decompress_tensor(compressed: dict) -> "torch.Tensor":
        """Restore a tensor compressed by ``_compress_tensor()``."""
        import torch

        if not compressed.get("_compressed", False):
            return compressed["data"]

        data = compressed["data"]
        scale = compressed["scale"]
        method = compressed.get("method", "int8")

        if method == "int4":
            return data.to(torch.float16) * scale

        if method == "int8":
            return data.to(torch.float16) * scale

        return data

    def _compress_kv_for_preemption(
        self, kv_data: Any, method: str = "int4"
    ) -> Any:
        """Compress KV cache data before storing for preemption.

        Applies int4 quantization (8x reduction) to KV tensors if they
        are torch.Tensor objects.  Non-tensor data is stored as-is.

        Args:
            kv_data: KV cache data (tensor, list of tensors, dict, or any).
            method: Compression method — "int4" (8x), "int8" (4x), or "none".

        Returns:
            Compressed data with metadata for decompression.
        """
        if method == "none" or kv_data is None:
            return kv_data

        import torch

        if isinstance(kv_data, torch.Tensor):
            return self._compress_tensor(kv_data, method)

        if isinstance(kv_data, dict):
            return {
                k: self._compress_kv_for_preemption(v, method)
                for k, v in kv_data.items()
            }

        if isinstance(kv_data, (list, tuple)):
            return [
                self._compress_kv_for_preemption(item, method)
                for item in kv_data
            ]

        # Non-tensor data (strings, ints, etc.) — store as-is
        return kv_data

    def decompress_preempted_kv(self, kv_data: Any) -> Any:
        """Recursively decompress KV data that was compressed for preemption.

        Args:
            kv_data: Compressed KV data (may contain nested dicts with _compressed flag).

        Returns:
            Decompressed data ready for use.
        """
        if isinstance(kv_data, dict) and kv_data.get("_compressed"):
            return self._decompress_tensor(kv_data)

        if isinstance(kv_data, dict):
            return {k: self.decompress_preempted_kv(v) for k, v in kv_data.items()}

        if isinstance(kv_data, list):
            return [self.decompress_preempted_kv(item) for item in kv_data]

        return kv_data

    def save_kv_state(
        self,
        request_id: str,
        preempted_kv_state: dict[str, Any],
        kv_cache_state: dict | None = None,
    ) -> None:
        """Save KV cache state for a preempted sequence.

        Stores the raw KV cache data (any type) for the given request_id
        so it can be restored later via ``restore_kv_state()``.

        Args:
            request_id: The request whose KV state to save.
            preempted_kv_state: Dict storing compressed KV state (modified in-place).
            kv_cache_state: External dict mapping request_id -> KV cache data.
                If the dict contains request_id, its value is saved.
        """
        if kv_cache_state is not None and request_id in kv_cache_state:
            data = kv_cache_state[request_id]
            compressed = self._compress_kv_for_preemption(data)
            preempted_kv_state[request_id] = compressed

    def restore_kv_state(
        self,
        request_id: str,
        preempted_kv_state: dict[str, Any],
        kv_cache_state: dict | None = None,
    ) -> bool:
        """Restore KV cache state for a preempted sequence.

        Decompresses the KV data if it was compressed during preemption,
        then writes it back into the external kv_cache_state dict.

        Args:
            request_id: The request whose KV state to restore.
            preempted_kv_state: Dict storing compressed KV state (modified in-place).
            kv_cache_state: External dict to write the restored KV data into.

        Returns:
            True if KV state was found and restored, False otherwise.
        """
        saved = preempted_kv_state.pop(request_id, None)
        if saved is not None and kv_cache_state is not None:
            kv_cache_state[request_id] = self.decompress_preempted_kv(saved)
            return True
        return False
