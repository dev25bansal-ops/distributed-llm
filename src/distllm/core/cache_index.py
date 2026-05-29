"""Cache index for P2P gossip-based cache lookups.

Re-exports CacheIndex from dist module and adds CacheIndexEntry for
gossip protocol integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from distllm.dist.cache import CacheIndex

__all__ = ["CacheIndex", "CacheIndexEntry"]


@dataclass
class CacheIndexEntry:
    """Metadata for a single cached KV entry in the gossip index."""
    key: str
    owner: str
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600.0
    num_tokens: int = 0
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl
