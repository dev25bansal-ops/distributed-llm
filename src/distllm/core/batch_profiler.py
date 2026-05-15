"""Model-aware batch size estimation for distributed LLM inference.

Estimates optimal batch sizes based on model architecture and available
GPU memory to maximize throughput without OOM.
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger


def get_memory_per_sequence(
    model_info: dict,
    seq_len: int,
    dtype_bytes: int = 2,
) -> int:
    """Estimate KV cache memory for one sequence.

    KV cache memory per sequence:
        2 * num_layers * 2 * num_kv_heads * hidden_size_per_head * seq_len * dtype_bytes

    The factor of 2 accounts for key + value states.

    Args:
        model_info: Dict with hidden_size, num_layers, num_attention_heads,
                    num_key_value_heads (optional, defaults to num_attention_heads).
        seq_len: Sequence length in tokens.
        dtype_bytes: Bytes per parameter (2 for fp16/bf16, 4 for fp32).

    Returns:
        Estimated memory in bytes.
    """
    hidden_size = model_info.get("hidden_size", 768)
    num_layers = model_info.get("num_layers", 12)
    num_heads = model_info.get("num_attention_heads", 12)
    num_kv_heads = model_info.get("num_key_value_heads", num_heads)

    if num_heads == 0:
        return 0

    hidden_per_head = hidden_size // num_heads
    memory = (
        2  # key + value
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
    """Estimate max batch size and tokens per batch.

    Uses model architecture to estimate KV cache memory per sequence,
    then computes how many sequences fit in available device memory.

    Args:
        model_info: Dict with model architecture parameters.
        device_memory_bytes: Available GPU memory in bytes.
        target_latency_ms: Optional latency target (unused in V1).
        safety_factor: Fraction of memory to use (default 0.6 to leave headroom).

    Returns:
        Tuple of (max_batch_size, max_tokens_per_batch).
    """
    avg_seq_len = 256  # reasonable default
    mem_per_seq = get_memory_per_sequence(model_info, avg_seq_len)

    if mem_per_seq == 0:
        return 32, 4096  # fallback

    usable_memory = int(device_memory_bytes * safety_factor)
    max_seqs = max(1, usable_memory // mem_per_seq)

    # Clamp to reasonable range
    max_batch_size = min(max_seqs, 128)
    max_tokens_per_batch = max_batch_size * avg_seq_len

    # Ensure minimums
    max_batch_size = max(max_batch_size, 2)
    max_tokens_per_batch = max(max_tokens_per_batch, 512)

    return max_batch_size, max_tokens_per_batch


def group_by_length(
    sequences: List[object],
    num_buckets: int = 4,
) -> Dict[int, List[object]]:
    """Group sequences by similar total length into buckets.

    Uses log-scale bucketing to group sequences with similar context lengths,
    reducing padding waste in mixed-length batches.

    Args:
        sequences: List of Sequence objects (must have total_len property).
        num_buckets: Number of length buckets.

    Returns:
        Dict mapping bucket index to list of sequences.
    """
    import math

    buckets: Dict[int, List[object]] = {i: [] for i in range(num_buckets)}

    # Find length range
    lengths = [s.total_len for s in sequences]
    if not lengths:
        return buckets

    min_len = min(lengths)
    max_len = max(lengths)

    if min_len == max_len:
        # All same length, put everything in bucket 0
        buckets[0] = list(sequences)
        return buckets

    # Log-scale bucketing
    log_min = math.log(max(min_len, 1))
    log_max = math.log(max_len)
    log_range = log_max - log_min

    for seq in sequences:
        ln = math.log(max(seq.total_len, 1))
        bucket = min(int((ln - log_min) / log_range * num_buckets), num_buckets - 1)
        buckets[bucket].append(seq)

    return buckets
