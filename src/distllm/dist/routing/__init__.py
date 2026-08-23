"""Routing subpackage for distributed inference.

Provides request routing strategies, affinity-based dispatch,
and load-aware request distribution across worker nodes.
"""

from __future__ import annotations

from .composite import CompositeRouter
from .consistent_hash import ConsistentHashRouter
from .latency_aware import LatencyAwareRouter
from .load_aware import LoadAwareRouter

__all__ = (
    "CompositeRouter",
    "ConsistentHashRouter",
    "LatencyAwareRouter",
    "LoadAwareRouter",
)
