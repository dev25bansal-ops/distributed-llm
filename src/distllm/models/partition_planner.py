"""Partition calculation and optimization for distributed LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from transformers import AutoConfig

from distllm.security import hf_revision


__all__ = [
    "partition_model_across_nodes",
    "partition_model_gpu_aware",
    "get_model_info",
    "find_optimal_partition",
    "profile_partition_throughput",
    "PartitionProfile",
]


def _should_trust_remote_code(model_name: str, trust_remote_code: bool | None = None) -> bool:
    """Determine whether to trust remote code for a model.

    Args:
        model_name: HuggingFace model identifier
        trust_remote_code: Explicit override. If None, uses allowlist logic.

    Returns:
        True if remote code should be trusted, False otherwise.
    """
    if trust_remote_code is not None:
        return trust_remote_code

    # Import here to avoid circular dependency
    from distllm.models.partitioner import TRUSTED_MODELS_ALLOWLIST

    # Extract the model name part (last segment of HF repo path)
    model_lower = model_name.lower().split("/")[-1]

    # Extract model family (prefix before first - or . separator)
    # e.g., "qwen2-7b" -> "qwen2", "my-qwen-exploit" -> "my"
    family = model_lower.split("-")[0].split(".")[0]

    # Match model family against allowlist to prevent false positives
    # (e.g., "my-qwen-exploit" has family "my" which won't match "qwen")
    for trusted in TRUSTED_MODELS_ALLOWLIST:
        if model_lower == trusted or family == trusted:
            return True
    return False


def partition_model_across_nodes(model_name: str, num_nodes: int, trust_remote_code: bool | None = None) -> list[tuple[int, int]]:
    """Calculate layer assignments for each node using equal split."""
    trust = _should_trust_remote_code(model_name, trust_remote_code)
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust,
        revision=hf_revision(),
    )
    total_layers = config.num_hidden_layers

    layers_per_node = total_layers // num_nodes
    remainder = total_layers % num_nodes

    assignments = []
    start = 0
    for i in range(num_nodes):
        extra = 1 if i < remainder else 0
        end = start + layers_per_node + extra - 1
        assignments.append((start, end))
        start = end + 1

    return assignments


def partition_model_gpu_aware(
    node_gpus: dict[str, list],
    model_name: str,
    total_layers: int,
    trust_remote_code: bool | None = None,
    safety_margin: float = 0.1,
) -> dict[str, tuple[int, int]]:
    """Calculate VRAM-aware layer assignments for each node.

    Uses the GPUProfiler to estimate per-layer memory usage for the given
    model, then assigns layers proportionally to each node's available VRAM.

    Args:
        node_gpus: dict mapping node_id to list of objects with
            ``free_memory_bytes`` and ``total_memory_bytes`` attributes.
            If empty, falls back to equal partitioning.
        model_name: HuggingFace model identifier.
        total_layers: total number of transformer layers.
        trust_remote_code: whether to trust remote code.
        safety_margin: fraction of VRAM to leave free (default 0.1 = 10%).

    Returns:
        dict mapping node_id to (start_layer, end_layer) tuple.
    """
    if not node_gpus or total_layers <= 0:
        logger.warning("No GPU info provided, falling back to equal partitioning")
        return _fallback_equal(node_gpus, model_name, trust_remote_code)

    try:
        from distllm.dist.partition.profiles import GPUProfiler

        # Get model config for layer memory estimation
        trust = _should_trust_remote_code(model_name, trust_remote_code)
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust,
            revision=hf_revision(),
        )

        profiler = GPUProfiler()
        layer_estimates = profiler.estimate_layer_weights(
            hidden_size=getattr(config, "hidden_size", 4096),
            intermediate_size=getattr(config, "intermediate_size", 11008),
            num_layers=total_layers,
            num_heads=getattr(config, "num_attention_heads", 32),
            head_dim=getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32),
            vocab_size=getattr(config, "vocab_size", 32000),
        )

        # Estimate per-transformer-layer memory (average of all layers)
        transformer_layers = [l for l in layer_estimates if l.layer_type == "transformer"]
        if transformer_layers:
            per_layer_weight = sum(l.weight_memory_bytes for l in transformer_layers) // len(transformer_layers)
        else:
            per_layer_weight = 1024 * 1024 * 100  # 100MB fallback

    except Exception as e:
        logger.warning(f"GPU-aware profiling failed ({e}), falling back to equal partitioning")
        return _fallback_equal(node_gpus, model_name, trust_remote_code)

    # Calculate available VRAM per node (apply safety margin)
    node_vram: dict[str, int] = {}
    for node_id, gpus in node_gpus.items():
        total_free = sum(
            getattr(g, "free_memory_bytes", getattr(g, "free_memory", 0))
            for g in (gpus if isinstance(gpus, list) else [gpus])
        )
        available = int(total_free * (1 - safety_margin))
        node_vram[node_id] = available

    total_available = sum(node_vram.values())
    if total_available <= 0:
        logger.warning("No available VRAM, falling back to equal partitioning")
        return _fallback_equal(node_gpus, model_name, trust_remote_code)

    # Assign layers proportional to available VRAM
    node_ids = sorted(node_gpus.keys())
    node_layers: dict[str, int] = {}
    assigned_total = 0

    for node_id in node_ids:
        raw_layers = node_vram[node_id] // per_layer_weight if per_layer_weight > 0 else 1
        node_layers[node_id] = max(1, raw_layers)
        assigned_total += node_layers[node_id]

    # Normalize to match total_layers exactly
    if assigned_total != total_layers:
        scale = total_layers / max(assigned_total, 1)
        scaled_total = 0
        for node_id in node_ids:
            scaled = max(1, int(node_layers[node_id] * scale))
            node_layers[node_id] = scaled
            scaled_total += scaled

        # Distribute remainder to nodes with most VRAM
        remainder = total_layers - scaled_total
        if remainder > 0:
            sorted_by_vram = sorted(node_ids, key=lambda n: node_vram[n], reverse=True)
            for i in range(remainder):
                node_layers[sorted_by_vram[i % len(sorted_by_vram)]] += 1
        elif remainder < 0:
            sorted_by_vram = sorted(node_ids, key=lambda n: node_vram[n])
            for i in range(abs(remainder)):
                node_layers[sorted_by_vram[i % len(sorted_by_vram)]] = max(
                    1, node_layers[sorted_by_vram[i % len(sorted_by_vram)]] - 1,
                )

    # Convert to (start, end) tuples
    result: dict[str, tuple[int, int]] = {}
    start = 0
    for node_id in node_ids:
        count = node_layers[node_id]
        end = start + count - 1
        result[node_id] = (start, end)
        start = end + 1

    logger.info(
        f"GPU-aware partitioning for {model_name}: "
        f"{ {n: f'{s}-{e}' for n, (s, e) in result.items()} }"
    )
    return result


def _fallback_equal(
    node_gpus: dict[str, list],
    model_name: str,
    trust_remote_code: bool | None = None,
) -> dict[str, tuple[int, int]]:
    """Fallback: assign layers equally across all nodes."""
    assignments = partition_model_across_nodes(model_name, len(node_gpus), trust_remote_code)
    return {node_id: assignments[i] for i, node_id in enumerate(node_gpus)}


def get_model_info(model_name: str, trust_remote_code: bool | None = None) -> dict:
    """Get model configuration info."""
    trust = _should_trust_remote_code(model_name, trust_remote_code)
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust,
        revision=hf_revision(),
    )
    return {
        "model_type": config.model_type,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, 'num_key_value_heads', config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "rope_scaling": getattr(config, 'rope_scaling', None),
        "max_position_embeddings": getattr(config, 'max_position_embeddings', 2048),
        "head_dim": getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32),
    }


# --- Auto-Partitioning Optimizer ---

@dataclass
class PartitionProfile:
    """Profiling results for a single layer assignment."""
    node_id: str
    start_layer: int
    end_layer: int
    vram_mb: float = 0.0
    compute_ms: float = 0.0
    communication_ms: float = 0.0
    throughput: float = 0.0  # tokens/second


def profile_partition_throughput(
    model_name: str,
    num_nodes: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    trust_remote_code: bool | None = None,
    gpu_info: dict[str, list] | None = None,
) -> list[tuple[int, int, float]]:
    """Profile and find the optimal layer partition for max throughput.

    Estimates each partition's throughput by considering:
    - VRAM capacity per node (or equal split if not provided)
    - Compute cost proportional to layers assigned
    - Communication cost (proportional to activations sent between nodes)

    Args:
        model_name: HuggingFace model identifier.
        num_nodes: Number of pipeline nodes.
        batch_size: Micro-batch size for profiling.
        seq_len: Sequence length for profiling.
        trust_remote_code: Whether to trust remote HF code.
        gpu_info: Optional dict of node_id -> list of GPUInfo objects.

    Returns:
        List of (start_layer, end_layer, estimated_throughput) sorted by
        throughput descending.
    """
    trust = _should_trust_remote_code(model_name, trust_remote_code)
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust,
        revision=hf_revision(),
    )
    total_layers = config.num_hidden_layers or 32
    hidden_size = config.hidden_size or 4096
    num_heads = config.num_attention_heads or 32
    head_dim = hidden_size // num_heads

    # Estimate per-layer compute cost (relative)
    per_layer_flops = (
        4 * batch_size * seq_len * hidden_size * hidden_size  # MLP
        + 2 * batch_size * seq_len * hidden_size * (num_heads * head_dim)  # Attention
        + 4 * batch_size * seq_len * hidden_size  # LayerNorm + residual
    )

    # Activation size sent between nodes (bytes per step)
    activation_bytes = batch_size * seq_len * hidden_size * 2  # fp16

    results: list[tuple[int, int, float, float, float, float]] = []

    # Try multiple partition strategies and evaluate
    strategies = [
        ("equal", None),
    ]

    if gpu_info:
        strategies.append(("gpu_aware", gpu_info))

    for strategy_name, gpus in strategies:
        if strategy_name == "equal":
            partitions = partition_model_across_nodes(model_name, num_nodes, trust)
        else:
            result_dict = partition_model_gpu_aware(gpus, model_name, total_layers, trust)
            partitions = [result_dict[nid] for nid in sorted(result_dict.keys())]

        for start, end in partitions:
            num_assigned_layers = end - start + 1

            # Compute cost (proportional to FLOPs)
            compute_cost = num_assigned_layers * per_layer_flops

            # Communication cost (activation transfer)
            comm_cost = activation_bytes  # one send per step

            # Throughput = 1 / (compute + communication)
            total_cost = compute_cost + comm_cost
            throughput = 1.0 / max(total_cost, 1)

            # VRAM estimate
            vram_per_layer_mb = (
                hidden_size * head_dim * 2 * 2  # K/V cache per layer (fp16)
                + hidden_size * hidden_size * 4 * 2 / (1024 ** 2)  # weights (fp16)
            )
            estimated_vram_mb = num_assigned_layers * vram_per_layer_mb

            results.append((
                start, end, throughput, estimated_vram_mb, compute_cost, comm_cost
            ))

    # Sort by throughput descending
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def find_optimal_partition(
    model_name: str,
    num_nodes: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    trust_remote_code: bool | None = None,
    gpu_info: dict[str, list] | None = None,
) -> list[tuple[int, int]]:
    """Find the optimal layer partition maximizing throughput.

    Profiles multiple partition strategies and returns the best one.

    Args:
        Same as profile_partition_throughput.

    Returns:
        List of (start_layer, end_layer) tuples for the optimal partition.
    """
    profiles = profile_partition_throughput(
        model_name, num_nodes, batch_size, seq_len,
        trust_remote_code, gpu_info,
    )
    if not profiles:
        return partition_model_across_nodes(model_name, num_nodes, trust_remote_code)

    best_start, best_end = profiles[0][0], profiles[0][1]
    # Use proportional allocation: faster layer counts get more layers
    total_layers = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=_should_trust_remote_code(model_name, trust_remote_code),
        revision=hf_revision(),
    ).num_hidden_layers

    throughputs = {i * (total_layers // num_nodes): prof[2] for i, prof in enumerate(profiles[:num_nodes])}
    total_throughput = sum(throughputs.values()) or 1.0
    result = []
    current = 0
    for i in range(num_nodes):
        fraction = throughputs.get(current, 1.0) / total_throughput
        n_layers = max(1, int(total_layers * fraction)) if i < num_nodes - 1 else total_layers - current
        end = min(current + n_layers - 1, total_layers - 1)
        result.append((current, end))
        current = end + 1
    return result
