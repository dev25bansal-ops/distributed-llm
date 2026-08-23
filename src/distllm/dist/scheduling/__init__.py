"""Scheduling subpackage for distributed inference.

Provides batching, workload classification, profiling,
iteration-level scheduling primitives, deadline-aware scheduling,
GPU-memory-aware batch packing, preemptive stage scheduling,
and heterogeneous batch sizing.
"""

from __future__ import annotations
from distllm.dist.scheduling.batcher import LatencyAwareBatcher, BatchGroup
from distllm.dist.scheduling.classifier import WorkloadType, classify
from distllm.dist.scheduling.deadline_scheduler import (
    DeadlineAwareBatchScheduler,
    DeadlineRequest,
    GPUMemoryBatchPacker,
    PreemptiveStageScheduler,
    StageType,
    HeterogeneousBatchSizer,
    GpuTier,
)
from distllm.dist.scheduling.iteration import IterationScheduler
from distllm.dist.scheduling.profiler import get_memory_per_sequence

__all__ = [
    "LatencyAwareBatcher",
    "BatchGroup",
    "WorkloadType",
    "classify",
    "IterationScheduler",
    "get_memory_per_sequence",
    "DeadlineAwareBatchScheduler",
    "DeadlineRequest",
    "GPUMemoryBatchPacker",
    "PreemptiveStageScheduler",
    "StageType",
    "HeterogeneousBatchSizer",
    "GpuTier",
]
