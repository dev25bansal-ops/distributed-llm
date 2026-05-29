"""Verification report generation — structured output for CI and debugging.

Produces both machine-readable JSON and human-readable console output
from accuracy verification runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from distllm.verification.comparator import (
    DEFAULT_THRESHOLDS,
    OutputComparison,
)
from distllm.verification.hash_registry import GenerationOutput, OutputHashRegistry


@dataclass
class VerificationReport:
    """Complete report from an accuracy verification run.

    Attributes:
        model_name: The model that was verified.
        num_nodes: Number of distributed nodes used.
        dtype: Model dtype used.
        temperature: Sampling temperature.
        thresholds: Metric thresholds for pass/fail.
        per_prompt: List of per-prompt results (each a dict with
            ``prompt``, ``comparison``, ``reference``, ``candidate``).
        hash_comparison: Summary of hash registry comparison.
        created_at: Unix timestamp of report creation.
        duration_ms: Total verification duration in milliseconds.
    """

    model_name: str = ""
    num_nodes: int = 0
    dtype: str = "float16"
    temperature: float = 0.0
    thresholds: dict[str, float] = field(default_factory=lambda: DEFAULT_THRESHOLDS)
    per_prompt: list[dict[str, Any]] = field(default_factory=list)
    hash_comparison: dict[str, Any] | None = None
    created_at: float = 0.0
    duration_ms: float = 0.0

    def summary(self) -> dict[str, Any]:
        """Aggregate results across all prompts.

        Returns:
            Dict with pass/fail counts, average metrics, and hash comparison.
        """
        total = len(self.per_prompt)
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_token_match": 0.0,
                "avg_logit_cosim": 0.0,
                "avg_kl_div": 0.0,
            }

        passed = sum(
            1 for p in self.per_prompt if p["comparison"].pass_threshold
        )

        avg_token_match = (
            sum(p["comparison"].token_exact_match for p in self.per_prompt) / total
        )
        avg_logit_cosim = (
            sum(p["comparison"].logit_cosine_sim for p in self.per_prompt) / total
        )
        avg_kl_div = (
            sum(p["comparison"].logit_kl_div for p in self.per_prompt) / total
        )

        result: dict[str, Any] = {
            "model_name": self.model_name,
            "num_nodes": self.num_nodes,
            "dtype": self.dtype,
            "temperature": self.temperature,
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / max(total, 1), 4),
            "avg_token_exact_match": round(avg_token_match, 6),
            "avg_logit_cosine_sim": round(avg_logit_cosim, 6),
            "avg_logit_kl_div": round(avg_kl_div, 6),
            "duration_ms": round(self.duration_ms, 2),
        }

        if self.hash_comparison:
            result["hash_comparison"] = self.hash_comparison

        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to JSON.

        Returns:
            JSON string with all verification results.
        """
        data: dict[str, Any] = {
            "model_name": self.model_name,
            "num_nodes": self.num_nodes,
            "dtype": self.dtype,
            "temperature": self.temperature,
            "thresholds": self.thresholds,
            "created_at": self.created_at,
            "duration_ms": self.duration_ms,
            "summary": self.summary(),
        }

        # Per-prompt data: serialize only comparison metrics + text hashes
        prompts_data = []
        for p in self.per_prompt:
            comp = p["comparison"]
            ref: GenerationOutput = p["reference"]
            cand: GenerationOutput = p["candidate"]
            prompts_data.append({
                "prompt": p["prompt"],
                "metrics": {
                    "token_exact_match": comp.token_exact_match,
                    "token_edit_distance": comp.token_edit_distance,
                    "logit_cosine_sim": comp.logit_cosine_sim,
                    "logit_kl_div": comp.logit_kl_div,
                    "logit_max_abs_diff": comp.logit_max_abs_diff,
                    "hidden_cosine_sim": comp.hidden_cosine_sim,
                    "hidden_max_abs_diff": comp.hidden_max_abs_diff,
                    "hidden_relative_error": comp.hidden_relative_error,
                    "pass": comp.pass_threshold,
                },
                "reference_tokens": ref.token_ids,
                "candidate_tokens": cand.token_ids,
                "reference_text": ref.text,
                "candidate_text": cand.text,
            })
        data["prompts"] = prompts_data

        return json.dumps(data, indent=indent, default=str)

    def print_human_readable(self) -> None:
        """Print a formatted human-readable report to stdout."""
        summary = self.summary()
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  Model Accuracy Verification Report")
        lines.append("=" * 60)
        lines.append(f"  Model:     {self.model_name}")
        lines.append(f"  Nodes:     {self.num_nodes}")
        lines.append(f"  Dtype:     {self.dtype}")
        lines.append(f"  Temp:      {self.temperature}")
        lines.append(f"  Duration:  {self.duration_ms:.1f} ms")
        lines.append("")

        lines.append(f"  Summary:   {summary['passed']}/{summary['total']} passed "
                      f"({summary['pass_rate']:.1%})")
        lines.append(f"  Avg token match: {summary['avg_token_exact_match']:.4f}")
        lines.append(f"  Avg logit cosim: {summary['avg_logit_cosine_sim']:.6f}")
        lines.append(f"  Avg KL div:      {summary['avg_logit_kl_div']:.6f}")
        lines.append("")

        for i, p in enumerate(self.per_prompt):
            comp = p["comparison"]
            status = "PASS" if comp.pass_threshold else "FAIL"
            lines.append(f"  [{status}] Prompt {i + 1}: {p['prompt'][:50]}")
            lines.append(f"       Token match: {comp.token_exact_match:.2%}  "
                          f"Edit dist: {comp.token_edit_distance:.4f}")
            lines.append(f"       Logit cosim: {comp.logit_cosine_sim:.6f}  "
                          f"KL div: {comp.logit_kl_div:.6f}  "
                          f"Max diff: {comp.logit_max_abs_diff:.6f}")
            if comp.hidden_cosine_sim > 0:
                lines.append(f"       Hidden cosim: {comp.hidden_cosine_sim:.6f}  "
                              f"Max diff: {comp.hidden_max_abs_diff:.6f}  "
                              f"Rel err: {comp.hidden_relative_error:.6f}")
            lines.append("")

        if self.hash_comparison:
            total = self.hash_comparison.get("total_prompts", 0)
            lines.append(f"  Hash registry: {self.hash_comparison['passed']}/"
                          f"{total} match "
                          f"({self.hash_comparison['pass_rate']:.1%})")

        lines.append("=" * 60)
        print("\n".join(lines))


def generate_report(
    comparisons: list[OutputComparison],
    per_prompt_data: list[dict[str, Any]] | None = None,
    hash_registry: OutputHashRegistry | None = None,
    thresholds: dict[str, float] | None = None,
    model_name: str = "",
    num_nodes: int = 2,
    dtype: str = "float16",
    temperature: float = 0.0,
) -> VerificationReport:
    """Build a ``VerificationReport`` from comparison results.

    Args:
        comparisons: List of ``OutputComparison`` objects, one per prompt.
        per_prompt_data: Optional detailed per-prompt data.
        hash_registry: Optional hash registry for cross-run comparison.
        thresholds: Custom metric thresholds.
        model_name: Model name for the report.
        num_nodes: Number of distributed nodes.
        dtype: Model dtype.
        temperature: Sampling temperature.

    Returns:
        A populated ``VerificationReport``.
    """
    if per_prompt_data is None:
        per_prompt_data = []
        for i, comp in enumerate(comparisons):
            per_prompt_data.append({
                "prompt": f"prompt_{i}",
                "comparison": comp,
                "reference": GenerationOutput(token_ids=[], text=""),
                "candidate": GenerationOutput(token_ids=[], text=""),
            })

    hash_comparison = None
    if hash_registry is not None:
        hash_comparison = hash_registry.summary()

    return VerificationReport(
        model_name=model_name,
        num_nodes=num_nodes,
        dtype=dtype,
        temperature=temperature,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
        per_prompt=per_prompt_data,
        hash_comparison=hash_comparison,
        created_at=time.time(),
        duration_ms=0.0,
    )
