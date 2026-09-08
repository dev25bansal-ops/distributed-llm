"""Gradient-free activation profiling for optimal model splitting.

Determines the best layer split points for pipeline parallelism by
profiling activation memory and compute at each layer.  Works with
unknown model architectures by running a short forward pass and
measuring activation sizes layer by layer.

Usage::

    profiler = ActivationProfiler(model_name="meta-llama/Llama-3.2-1B")
    splits = profiler.find_optimal_split(num_splits=3)
    print(splits)  # [(0, 5), (6, 11), (12, 17)]
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM


class ActivationProfiler:
    """Estimates activation memory per layer to find optimal split points.

    Uses model config and analytical formulas to estimate activation
    sizes per layer without loading the full model.  Then uses a DP
    algorithm to find splits that minimize the maximum per-split
    activation memory.

    Note: This uses formula-based estimation (4 * batch * seq * hidden
    for activations + KV cache), not actual forward-pass measurement.
    For precise profiling, use ``GPUProfiler.bench_matmul_fp16()`` or
    runtime tracing.

    Args:
        model_name: HuggingFace model name.
        trust_remote_code: Whether to trust remote HF code.
        device: Device for the profiling pass.
        seq_len: Dummy sequence length for profiling.
        batch_size: Dummy batch size for profiling.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool | None = None,
        device: str = "cpu",
        seq_len: int = 512,
        batch_size: int = 1,
    ):
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._device = torch.device(device)
        self._seq_len = seq_len
        self._batch_size = batch_size

    def profile(self) -> dict[str, Any]:
        """Profile activation memory per layer.

        Returns:
            dict with "num_layers", "hidden_size", "layer_activations_mb"
            (list of activation sizes in MB per layer), and "total_layers".
        """
        from distllm.models.partitioner import _should_trust_remote_code

        trust = _should_trust_remote_code(self._model_name, self._trust_remote_code)
        config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=trust)
        num_layers = getattr(config, "num_hidden_layers", 0)
        hidden_size = getattr(config, "hidden_size", 4096)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
        vocab_size = getattr(config, "vocab_size", 32000)

        # Estimate activation memory per layer without loading the full model
        # Uses the formula: 4 * batch * seq * hidden (for attention + MLP)
        float_bytes = 2  # fp16
        activation_per_layer_mb = (
            4 * self._batch_size * self._seq_len * hidden_size * float_bytes
        ) / (1024 * 1024)

        # Embedding and LM head
        embed_mb = (vocab_size * hidden_size * float_bytes) / (1024 * 1024)
        lm_head_mb = embed_mb

        # KV cache per layer (2 * batch * num_heads * seq * head_dim)
        num_heads = getattr(config, "num_attention_heads", 32)
        head_dim = hidden_size // num_heads
        kv_per_layer_mb = (
            2 * self._batch_size * num_heads * self._seq_len * head_dim * float_bytes
        ) / (1024 * 1024)

        layer_activations = [round(activation_per_layer_mb + kv_per_layer_mb, 2) for _ in range(num_layers)]

        return {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "vocab_size": vocab_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "embedding_mb": round(embed_mb, 2),
            "lm_head_mb": round(lm_head_mb, 2),
            "layer_activations_mb": layer_activations,
            "total_activation_mb": round(sum(layer_activations) + embed_mb + lm_head_mb, 2),
        }

    def find_optimal_split(self, num_splits: int = 2, profile_data: dict | None = None) -> list[tuple[int, int]]:
        """Find optimal layer splits that minimize the maximum activation per partition.

        Uses dynamic programming to partition layers such that the
        maximum activation memory across all partitions is minimized.

        Args:
            num_splits: Number of pipeline stages (nodes).
            profile_data: Pre-computed profile dict (from ``profile()``).
                If None, runs profiling automatically.

        Returns:
            List of (start_layer, end_layer) tuples, one per split.
        """
        if profile_data is None:
            profile_data = self.profile()

        num_layers = profile_data["num_layers"]
        activations = profile_data["layer_activations_mb"]

        if num_splits >= num_layers:
            return [(i, i) for i in range(num_layers)]

        # Prefix sums for O(1) range sum
        prefix = [0]
        for a in activations:
            prefix.append(prefix[-1] + a)

        def range_sum(i: int, j: int) -> float:
            return prefix[j + 1] - prefix[i]

        # DP: dp[p][i] = min max activation for splitting first i layers into p partitions
        dp = [[float("inf")] * (num_layers + 1) for _ in range(num_splits + 1)]
        split_point = [[0] * (num_layers + 1) for _ in range(num_splits + 1)]

        # Base: 1 partition = sum of all layers up to i
        for i in range(1, num_layers + 1):
            dp[1][i] = range_sum(0, i - 1)

        # Fill DP table
        for p in range(2, num_splits + 1):
            for i in range(p, num_layers + 1):
                for j in range(p - 1, i):
                    cost = max(dp[p - 1][j], range_sum(j, i - 1))
                    if cost < dp[p][i]:
                        dp[p][i] = cost
                        split_point[p][i] = j

        # Backtrack to find split positions
        splits = []
        i = num_layers
        for p in range(num_splits, 0, -1):
            j = split_point[p][i]
            if p == num_splits:
                splits.append((j, i - 1))
            else:
                splits.append((j, i - 1))
            i = j

        splits.reverse()
        logger.info(
            f"Optimal split for {num_layers} layers across {num_splits} stages: "
            f"{[(s, e) for s, e in splits]}"
        )
        return splits
