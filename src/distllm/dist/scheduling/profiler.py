"""Model-aware batch size estimation for distributed LLM inference.

Estimates optimal batch sizes based on model architecture and available
GPU memory to maximize throughput without OOM.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from distllm.utils.scheduling import group_by_length  # noqa: F401 — re-export

def get_memory_per_sequence(
    model_info: dict,
    seq_len: int,
    dtype_bytes: int = 2,
) -> int:
    hidden_size = model_info.get("hidden_size", 768)
    num_layers = model_info.get("num_layers", 12)
    num_heads = model_info.get("num_attention_heads", 12)
    num_kv_heads = model_info.get("num_key_value_heads", num_heads)

    if num_heads == 0:
        return 0

    hidden_per_head = hidden_size // num_heads
    memory = (
        2
        * num_layers
        * num_kv_heads
        * hidden_per_head
        * seq_len
        * dtype_bytes
    )
    return memory

def estimate_max_batch(
    model_info: dict,
    device_memory_bytes: int,
    target_latency_ms: Optional[float] = None,
    safety_factor: float = 0.6,
) -> Tuple[int, int]:
    avg_seq_len = 256
    mem_per_seq = get_memory_per_sequence(model_info, avg_seq_len)

    if mem_per_seq == 0:
        return 32, 4096

    usable_memory = int(device_memory_bytes * safety_factor)
    max_seqs = max(1, usable_memory // mem_per_seq)

    max_batch_size = min(max_seqs, 128)
    max_tokens_per_batch = max_batch_size * avg_seq_len

    max_batch_size = max(max_batch_size, 2)
    max_tokens_per_batch = max(max_tokens_per_batch, 512)

    return max_batch_size, max_tokens_per_batch
