"""Distributed LLM Inference System — Path A: Inference Control Plane."""

__version__ = "0.4.0"

from distllm.core.kv_cache import KVCache, KVCacheManager
from distllm.models.partitioner import ModelPartitioner, partition_model_across_nodes, get_model_info
from distllm.core.coordinator import Coordinator
from distllm.core.resource_manager import NodeRegistration
from distllm.core.batch_scheduler import BatchScheduler, Sequence, SequenceStatus, ScheduledBatch
from distllm.core.structured_output import JSONSchemaConstraint
from distllm.core.monitor import SystemMonitor
from distllm.models.adapter import AdapterManager

try:
    from distllm.core.multi_model_serving import ModelHotSwapManager
except ImportError:
    ModelHotSwapManager = None

__all__ = [
    "__version__",
    "Coordinator",
    "KVCache", "KVCacheManager",
    "ModelPartitioner", "partition_model_across_nodes",
    "BatchScheduler", "Sequence", "SequenceStatus", "ScheduledBatch",
    "JSONSchemaConstraint",
    "SystemMonitor",
    "AdapterManager",
    "NodeRegistration",
    "ModelHotSwapManager",
]
