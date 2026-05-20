"""Coordinator configuration dataclass.

Extracted from coordinator.py __init__ to reduce the 2965-line monolith.
All original constructor parameters are preserved for backward compatibility.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class CoordinatorConfig:
    """Immutable configuration for the Coordinator."""

    # Core engine params
    model_name: str = "default"
    port: int = 50050
    dtype: str = "float16"
    trust_remote_code: bool | None = None
    max_batch_size: int = 1
    max_tokens_per_batch: int = 4096
    max_context_length: int = 8192
    discovery_mode: str = "static"

    # Cache
    prefix_cache_enabled: bool = False
    prefix_cache_max_entries: int = 1024
    prefix_cache_min_prefix_len: int = 16
    radix_tree_cache_enabled: bool = True
    chunked_prefill_enabled: bool = True
    chunked_prefill_chunk_size: int = 512

    # Feature configs (opaque, passed to subsystem constructors)
    quantization_config: Any = None
    speculative_config: Any = None
    lora_config: Any = None
    multi_model_config: Any = None
    rebalancer_config: Any = None
    cache_persistence_config: Any = None
    gossip_config: Any = None
    moe_config: Any = None
    embedding_config: Any = None
    version_config: Any = None
    hybrid_parallel_config: Any = None
    zero_copy_config: Any = None
    adaptive_precision_config: Any = None
    predictive_cache_config: Any = None
    pipeline_schedule_config: Any = None
    self_optimizing_config: Any = None
    cuda_graph_config: Any = None
    compile_config: Any = None
    slora_config: Any = None
    rag_config: Any = None
    agent_config: Any = None
    disagg_config: Any = None
    request_auditor_config: Any = None
    prompt_cache_config: Any = None
    graceful_degradation_config: Any = None
    adaptive_batching_config: Any = None
    request_fingerprinting_config: Any = None
    leaky_bucket_config: Any = None

    # Observability
    metrics_exporter: Any = None
