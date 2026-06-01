"""Correctness test framework for distributed inference validation.

Verifies that distributed inference produces the same output as
single-node inference, within configurable tolerance thresholds.

Usage::

    verifier = CorrectnessVerifier(tolerance=1e-5)
    result = verifier.verify(
        distributed_output=logits_dist,
        reference_output=logits_ref,
        test_name="70b_pipeline_parallel",
    )
    assert result.passed
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CorrectnessResult:
    """Result of a correctness verification."""
    test_name: str
    passed: bool
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    cosine_similarity: float
    token_match_rate: float
    details: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CorrectnessVerifier:
    """Verifies distributed inference correctness against reference output.

    Supports multiple verification modes:
    - Logits comparison (numerical accuracy)
    - Token comparison (exact match)
    - Cosine similarity (directional accuracy)
    - Quantization quality (lossy but bounded)
    """

    def __init__(
        self,
        tolerance: float = 1e-5,
        cosine_threshold: float = 0.999,
        token_match_threshold: float = 0.95,
    ):
        self._tolerance = tolerance
        self._cosine_threshold = cosine_threshold
        self._token_match_threshold = token_match_threshold
        self._results: list[CorrectnessResult] = []

    def verify(
        self,
        distributed_output: torch.Tensor,
        reference_output: torch.Tensor,
        test_name: str = "",
        distributed_tokens: list[int] | None = None,
        reference_tokens: list[int] | None = None,
    ) -> CorrectnessResult:
        """Verify distributed output matches reference.

        Args:
            distributed_output: Logits from distributed inference.
            reference_output: Logits from single-node inference.
            test_name: Name of the test for reporting.
            distributed_tokens: Token IDs from distributed inference.
            reference_tokens: Token IDs from reference inference.

        Returns:
            CorrectnessResult with pass/fail and detailed metrics.
        """
        # Numerical comparison
        abs_diff = torch.abs(distributed_output - reference_output)
        max_abs_err = abs_diff.max().item()
        mean_abs_err = abs_diff.mean().item()

        # Relative error (avoid division by zero)
        ref_abs = torch.abs(reference_output).clamp(min=1e-12)
        rel_err = (abs_diff / ref_abs).max().item()

        # Cosine similarity
        dist_flat = distributed_output.flatten().float()
        ref_flat = reference_output.flatten().float()
        cos_sim = torch.nn.functional.cosine_similarity(
            dist_flat.unsqueeze(0), ref_flat.unsqueeze(0)
        ).item()

        # Token match rate
        token_match = 1.0
        if distributed_tokens and reference_tokens:
            min_len = min(len(distributed_tokens), len(reference_tokens))
            if min_len > 0:
                matches = sum(
                    1 for i in range(min_len)
                    if distributed_tokens[i] == reference_tokens[i]
                )
                token_match = matches / min_len

        # Determine pass/fail
        passed = (
            max_abs_err <= self._tolerance
            and cos_sim >= self._cosine_threshold
            and token_match >= self._token_match_threshold
        )

        result = CorrectnessResult(
            test_name=test_name,
            passed=passed,
            max_absolute_error=max_abs_err,
            mean_absolute_error=mean_abs_err,
            max_relative_error=rel_err,
            cosine_similarity=cos_sim,
            token_match_rate=token_match,
            details={
                "tolerance": self._tolerance,
                "cosine_threshold": self._cosine_threshold,
                "token_match_threshold": self._token_match_threshold,
                "output_shape": list(distributed_output.shape),
            },
        )

        self._results.append(result)
        return result

    def verify_quantization_quality(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor,
        dequantized: torch.Tensor,
        test_name: str = "",
    ) -> CorrectnessResult:
        """Verify quantization doesn't degrade output quality beyond threshold.

        Args:
            original: Original tensor.
            quantized: Quantized tensor.
            dequantized: Dequantized tensor (should approximate original).
            test_name: Test name.

        Returns:
            CorrectnessResult for the quantization quality.
        """
        return self.verify(
            distributed_output=dequantized,
            reference_output=original,
            test_name=f"quantization_{test_name}",
        )

    def verify_speculative_decoding(
        self,
        speculative_output: list[int],
        greedy_output: list[int],
        test_name: str = "",
    ) -> CorrectnessResult:
        """Verify speculative decoding matches greedy output.

        Speculative decoding should produce identical output to greedy
        decoding (temperature=0) when verification is correct.

        Args:
            speculative_output: Tokens from speculative decoding.
            greedy_output: Tokens from greedy decoding.
            test_name: Test name.

        Returns:
            CorrectnessResult for the speculative decoding quality.
        """
        min_len = min(len(speculative_output), len(greedy_output))
        if min_len == 0:
            return CorrectnessResult(
                test_name=test_name,
                passed=True,
                max_absolute_error=0,
                mean_absolute_error=0,
                max_relative_error=0,
                cosine_similarity=1.0,
                token_match_rate=1.0,
            )

        matches = sum(
            1 for i in range(min_len)
            if speculative_output[i] == greedy_output[i]
        )
        match_rate = matches / min_len

        passed = match_rate >= self._token_match_threshold

        result = CorrectnessResult(
            test_name=test_name,
            passed=passed,
            max_absolute_error=0,
            mean_absolute_error=0,
            max_relative_error=0,
            cosine_similarity=1.0,
            token_match_rate=match_rate,
            details={
                "speculative_length": len(speculative_output),
                "greedy_length": len(greedy_output),
                "matches": matches,
            },
        )

        self._results.append(result)
        return result

    def get_summary(self) -> dict:
        """Get summary of all verification results."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failed = total - passed

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / max(total, 1),
            "results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "max_abs_err": r.max_absolute_error,
                    "cosine_sim": r.cosine_similarity,
                    "token_match": r.token_match_rate,
                }
                for r in self._results
            ],
        }
