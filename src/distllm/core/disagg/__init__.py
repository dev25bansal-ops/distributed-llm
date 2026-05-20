"""Disaggregated prefill/decode serving.

Package structure:
    types.py         — Data models, enums, dataclasses
    pool.py          — PrefillPool and DecodePool management
    kv_cache.py      — KV cache extraction, serialization, transfer
    router.py        — DisaggRouter: routes requests between pools
    orchestrator.py  — DisaggOrchestrator: full lifecycle management
    metrics.py       — Pool-level metrics collection and monitoring
    scaler.py        — Autoscalers for prefill and decode pools
    config.py        — Configuration models for disaggregated serving
"""

from distllm.core.disagg.types import (
    DisaggPhase,
    PoolStatus,
    PoolNode,
    PrefillRequest,
    PrefillResult,
    DecodeRequest,
)
from distllm.core.disagg.pool import PrefillPool, DecodePool
from distllm.core.disagg.kv_cache import extract_kv_cache
from distllm.core.disagg.router import DisaggRouter
from distllm.core.disagg.orchestrator import DisaggOrchestrator
from distllm.core.disagg.metrics import DisaggMetrics, PoolMetricsCollector
from distllm.core.disagg.scaler import (
    ScalingDecision,
    PrefillScaler,
    DecodeScaler,
)
from distllm.core.disagg.config import (
    DisaggPoolConfig,
    DisaggKVCacheConfig,
    DisaggScalingConfig,
    DisaggFullConfig,
)

__all__ = [
    "DisaggPhase",
    "PoolStatus",
    "PoolNode",
    "PrefillRequest",
    "PrefillResult",
    "DecodeRequest",
    "PrefillPool",
    "DecodePool",
    "extract_kv_cache",
    "DisaggRouter",
    "DisaggOrchestrator",
    "DisaggMetrics",
    "PoolMetricsCollector",
    "ScalingDecision",
    "PrefillScaler",
    "DecodeScaler",
    "DisaggPoolConfig",
    "DisaggKVCacheConfig",
    "DisaggScalingConfig",
    "DisaggFullConfig",
]
