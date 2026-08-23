"""KV cache transfer between prefill and decode nodes.

Provides ``KVCacheHandle`` (opaque token returned by prefill) and
``KVCacheTransferScheduler`` (orchestrates cross-node KV cache movement).
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger


@dataclass
class KVCacheHandle:
    """Opaque handle to a KV cache segment stored on a decode node.

    Returned by ``DisaggManager.prefill()`` and passed to
    ``DisaggManager.decode()`` so the decode pool knows which
    prefill output to continue from.
    """
    request_id: str
    decode_node_id: str
    prefill_node_id: str
    num_prefill_tokens: int
    kv_cache_key: str = ""
    created_at: float = field(default_factory=time.time)


class KVCacheTransferScheduler:
    """Orchestrates KV cache transfer from prefill nodes to decode nodes.

    Uses the existing ``block_transfer_service`` for the actual data
    movement.  Supports:
    - Direct GPU→GPU via NCCL (fastest, same cluster)
    - RDMA transfer via block_transfer_service (cross-node)
    - Pipelined transfer (start decode before transfer completes)
    """

    def __init__(self, transfer_fn: Optional[Callable] = None):
        self._transfer_fn = transfer_fn
        self._transfers_in_flight: dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()

    async def transfer(
        self,
        handle: KVCacheHandle,
        source_address: str,
        dest_address: str,
    ) -> bool:
        if self._transfer_fn is None:
            # No real transfer path configured — report failure honestly
            # instead of simulating a successful KV movement.
            logger.error(
                f"KV transfer for {handle.request_id}: no transfer_fn "
                "configured; cannot move KV cache"
            )
            return False
        try:
            result = await self._transfer_fn(
                handle.request_id,
                source_address,
                dest_address,
            )
            return bool(result)
        except Exception:
            # Propagate: a failed KV transfer must not look like a success,
            # or decode would silently continue from a nonexistent cache.
            logger.exception(f"KV transfer failed for {handle.request_id}")
            raise

    def in_flight_count(self) -> int:
        return len(self._transfers_in_flight)
