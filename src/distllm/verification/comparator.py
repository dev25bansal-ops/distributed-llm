"""Numerical comparison functions for model output verification.

Computes metrics that quantify the difference between a reference
(single-node) output and a candidate (distributed pipeline) output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass
class OutputComparison:
    """Aggregated comparison results between reference and candidate outputs.

    Fields:
        token_exact_match: Fraction of tokens that match exactly.
        token_edit_distance: Normalized Levenshtein-like distance.
        logit_cosine_sim: Cosine similarity between logit vectors.
        logit_kl_div: KL divergence (reference || candidate) averaged over positions.
        logit_max_abs_diff: Maximum absolute difference across all logit elements.
        hidden_cosine_sim: Cosine similarity of last hidden states.
        hidden_max_abs_diff: Maximum absolute difference in hidden states.
        hidden_relative_error: Mean relative error in hidden states.
        pass_threshold: Whether all metrics pass their respective thresholds.
        thresholds: Metric thresholds used for pass/fail.
    """

    token_exact_match: float = 0.0
    token_edit_distance: float = 1.0
    logit_cosine_sim: float = 0.0
    logit_kl_div: float = float("inf")
    logit_max_abs_diff: float = float("inf")
    hidden_cosine_sim: float = 0.0
    hidden_max_abs_diff: float = float("inf")
    hidden_relative_error: float = float("inf")
    pass_threshold: bool = False
    thresholds: dict[str, float] = field(default_factory=lambda: DEFAULT_THRESHOLDS)


DEFAULT_THRESHOLDS: dict[str, float] = {
    "token_exact_match": 1.0,
    "token_edit_distance": 0.0,
    "logit_cosine_sim": 0.999,
    "logit_kl_div": 0.01,
    "logit_max_abs_diff": 0.5,
    "hidden_cosine_sim": 0.999,
    "hidden_max_abs_diff": 0.1,
    "hidden_relative_error": 0.01,
}


def compare_logits(
    gold_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> dict[str, float]:
    """Compare logit tensors from reference and distributed inference.

    Args:
        gold_logits: Logits from single-node reference model.
            Shape ``(batch, seq, vocab)`` or ``(seq, vocab)``.
        candidate_logits: Logits from distributed pipeline.
            Same shape as *gold_logits*.

    Returns:
        Dictionary with keys: ``cosine_sim``, ``kl_div``, ``max_abs_diff``.
    """
    if gold_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"Logit shape mismatch: {gold_logits.shape} vs {candidate_logits.shape}"
        )

    g = gold_logits.float().flatten()
    c = candidate_logits.float().flatten()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(g.unsqueeze(0), c.unsqueeze(0)).item()

    # Max absolute difference
    max_abs = (g - c).abs().max().item()

    # KL divergence: D_KL(gold || candidate) averaged over positions
    gold_probs = torch.softmax(gold_logits.float(), dim=-1)
    candidate_probs = torch.softmax(candidate_logits.float(), dim=-1)
    eps = 1e-10
    kl = (
        gold_probs * (torch.log(gold_probs + eps) - torch.log(candidate_probs + eps))
    ).sum(dim=-1)
    kl_mean = kl.mean().item()

    return {
        "cosine_sim": round(cos_sim, 8),
        "kl_div": round(kl_mean, 8),
        "max_abs_diff": round(max_abs, 8),
    }


def compare_hidden_states(
    gold_hidden: torch.Tensor,
    candidate_hidden: torch.Tensor,
) -> dict[str, float]:
    """Compare hidden state tensors between reference and distributed.

    Args:
        gold_hidden: Hidden states from reference model.
        candidate_hidden: Hidden states from distributed pipeline.

    Returns:
        Dictionary with keys: ``cosine_sim``, ``max_abs_diff``, ``relative_error``.
    """
    if gold_hidden.shape != candidate_hidden.shape:
        raise ValueError(
            f"Hidden state shape mismatch: {gold_hidden.shape} vs {candidate_hidden.shape}"
        )

    g = gold_hidden.float().flatten()
    c = candidate_hidden.float().flatten()

    cos_sim = torch.nn.functional.cosine_similarity(g.unsqueeze(0), c.unsqueeze(0)).item()
    max_abs = (g - c).abs().max().item()

    g_norm = g.norm()
    denom = max(g_norm.item(), 1e-8)
    rel_err = ((g - c).abs() / denom).mean().item()

    return {
        "cosine_sim": round(cos_sim, 8),
        "max_abs_diff": round(max_abs, 8),
        "relative_error": round(rel_err, 8),
    }


def compare_tokens(
    gold_ids: Sequence[int],
    candidate_ids: Sequence[int],
) -> dict[str, float]:
    """Compare token sequences from reference and distributed generation.

    Args:
        gold_ids: Token IDs produced by single-node inference.
        candidate_ids: Token IDs produced by distributed inference.

    Returns:
        Dictionary with keys: ``exact_match``, ``edit_distance``.
    """
    gold = list(gold_ids)
    candidate = list(candidate_ids)

    # Exact match: fraction of candidate tokens matching gold at same position
    min_len = min(len(gold), len(candidate))
    matches = sum(1 for i in range(min_len) if gold[i] == candidate[i])
    exact_match = matches / max(len(gold), 1)

    # Normalized edit distance (Levenshtein-like, simple diff-based)
    edit_dist = _normalized_edit_distance(gold, candidate)

    return {
        "exact_match": round(exact_match, 6),
        "edit_distance": round(edit_dist, 6),
    }


def compare_text(
    gold_text: str,
    candidate_text: str,
) -> dict[str, float]:
    """Compare output text from reference and distributed generation.

    Args:
        gold_text: Output text from single-node.
        candidate_text: Output text from distributed.

    Returns:
        Dictionary with ``exact_match``, ``edit_distance`` and
        ``token_overlap`` (Jaccard similarity of word sets).
    """
    exact = 1.0 if gold_text == candidate_text else 0.0

    gold_words = set(gold_text.lower().split())
    candidate_words = set(candidate_text.lower().split())
    overlap = 0.0
    if gold_words:
        overlap = len(gold_words & candidate_words) / len(gold_words | candidate_words)

    # Token-level edit distance
    edit_dist = _normalized_edit_distance(
        gold_text.split(), candidate_text.split()
    )

    return {
        "exact_match": exact,
        "edit_distance": round(edit_dist, 6),
        "token_overlap": round(overlap, 6),
    }


def evaluate_comparison(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> OutputComparison:
    """Build an ``OutputComparison`` from a dictionary of metrics.

    Args:
        metrics: Flat dict with keys like ``token_exact_match``,
            ``logit_cosine_sim``, etc.
        thresholds: Per-metric pass/fail thresholds.
            Defaults to ``DEFAULT_THRESHOLDS``.

    Returns:
        ``OutputComparison`` with ``pass_threshold`` set.
    """
    t = dict(thresholds or DEFAULT_THRESHOLDS)
    comparison = OutputComparison(thresholds=t)

    for field_name in comparison.__dataclass_fields__:
        if field_name in metrics:
            setattr(comparison, field_name, metrics[field_name])

    comparison.pass_threshold = _all_pass(metrics, t)
    return comparison


def _all_pass(metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
    """Check all available metrics against their thresholds."""
    for key, threshold in thresholds.items():
        if key not in metrics:
            continue
        value = metrics[key]
        # Higher-is-better metrics
        if key in ("token_exact_match", "logit_cosine_sim", "hidden_cosine_sim"):
            if value < threshold:
                return False
        # Lower-is-better metrics
        else:
            if value > threshold:
                return False
    return True


def _normalized_edit_distance(a: Sequence, b: Sequence) -> float:
    """Simple normalized token-level edit distance (relative to longer seq)."""
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 0.0
    # Use Levenshtein distance for small sequences, Jaccard for large
    if m * n > 10000:
        set_a, set_b = set(a), set(b)
        jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
        return round(1.0 - jaccard, 6)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / max(m, n, 1)
