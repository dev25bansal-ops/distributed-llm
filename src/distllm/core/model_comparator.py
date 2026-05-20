"""Model Version Comparator: statistical comparison between model versions.

Supports:
  - KL divergence between probability distributions
  - Exact match rate between outputs
  - Rouge-L/ROUGE score comparison
  - Output length distribution stats
  - Per-sample diff for debugging regressions
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class ComparisonSample:
    """A single paired comparison between two model versions."""
    prompt: str
    response_a: str  # Baseline model output
    response_b: str  # Candidate model output
    logprobs_a: list[float] | None = None
    logprobs_b: list[float] | None = None
    tokens_a: list[str] | None = None
    tokens_b: list[str] | None = None


@dataclass
class ComparisonResult:
    """Statistical comparison result between two model versions."""
    exact_match_rate: float = 0.0
    avg_kl_divergence: float | None = None
    rouge_l_f1: float = 0.0
    length_diff_mean: float = 0.0
    num_samples: int = 0
    regressions: list[int] = field(default_factory=list)
    improvements: list[int] = field(default_factory=list)


def _kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute KL(P || Q) with smoothing."""
    kl = 0.0
    eps = 1e-10
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        kl += pi * math.log(pi / qi)
    return kl


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute longest common subsequence length."""
    n, m = len(x), len(y)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def _rouge_l(reference: str, candidate: str) -> dict[str, float]:
    """Compute ROUGE-L precision, recall, F1."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


class ModelVersionComparator:
    """Compares outputs between two model versions statistically.

    Usage:
        comparator = ModelVersionComparator()
        comparator.add_sample(prompt, response_a, response_b)
        result = comparator.compare()
        print(f"Exact match rate: {result.exact_match_rate:.2%}")
    """

    def __init__(self):
        self._samples: list[ComparisonSample] = []

    def add_sample(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        logprobs_a: list[float] | None = None,
        logprobs_b: list[float] | None = None,
    ) -> None:
        """Add a paired comparison sample."""
        self._samples.append(ComparisonSample(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            logprobs_a=logprobs_a,
            logprobs_b=logprobs_b,
        ))

    def compare(self) -> ComparisonResult:
        """Run all statistical comparisons across samples."""
        if not self._samples:
            return ComparisonResult()

        exact_matches = 0
        total_kl = 0.0
        kl_count = 0
        total_rouge = 0.0
        length_diffs: list[float] = []
        regressions: list[int] = []
        improvements: list[int] = []

        for idx, sample in enumerate(self._samples):
            a, b = sample.response_a, sample.response_b

            # Exact match
            if a.strip() == b.strip():
                exact_matches += 1

            # ROUGE-L
            rouge = _rouge_l(a, b)
            total_rouge += rouge["f1"]

            # Length diff
            len_diff = len(b.split()) - len(a.split())
            length_diffs.append(len_diff)

            # KL divergence (if logprobs available)
            if sample.logprobs_a and sample.logprobs_b and len(sample.logprobs_a) == len(sample.logprobs_b):
                total_kl += _kl_divergence(sample.logprobs_a, sample.logprobs_b)
                kl_count += 1

            # Track regressions/improvements (ROUGE-based)
            if rouge["f1"] < 0.7:
                regressions.append(idx)
            elif rouge["f1"] > 0.95 and a.strip() != b.strip():
                improvements.append(idx)

        n = len(self._samples)
        return ComparisonResult(
            exact_match_rate=exact_matches / n,
            avg_kl_divergence=total_kl / kl_count if kl_count > 0 else None,
            rouge_l_f1=total_rouge / n,
            length_diff_mean=sum(length_diffs) / n if length_diffs else 0.0,
            num_samples=n,
            regressions=regressions,
            improvements=improvements,
        )

    def get_regression_prompts(self, result: ComparisonResult | None = None) -> list[str]:
        """Return prompts where candidate is worse than baseline."""
        if result is None:
            result = self.compare()
        return [self._samples[i].prompt for i in result.regressions]

    def summary(self) -> str:
        """Return a human-readable summary."""
        r = self.compare()
        lines = [
            f"Samples: {r.num_samples}",
            f"Exact match rate: {r.exact_match_rate:.2%}",
            f"ROUGE-L F1: {r.rouge_l_f1:.4f}",
            f"Avg length diff: {r.length_diff_mean:.1f} tokens",
            f"Regressions: {len(r.regressions)}",
            f"Improvements: {len(r.improvements)}",
        ]
        if r.avg_kl_divergence is not None:
            lines.append(f"Avg KL divergence: {r.avg_kl_divergence:.4f}")
        return "\n".join(lines)
