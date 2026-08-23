"""Stub implementations for verification tests.

Provides lightweight replacements for heavyweight dependencies
(backend registry, AccuracyVerifier) so that construction and
verify-flow tests do not require real inference backends.
"""

from __future__ import annotations

from distllm.verification.comparator import DEFAULT_THRESHOLDS, OutputComparison
from distllm.verification.hash_registry import GenerationOutput
from distllm.verification.report import VerificationReport
from distllm.verification.runner import AccuracyVerifier


def select_backend(
    device_type: str | None = None,
    preferred_backend: str | None = None,
) -> None:
    """Stub: always returns None (no backends available).

    Replaces ``distllm.backends.registry.select_backend`` during tests
    so that ``AccuracyVerifier.__post_init__`` does not need real
    inference backends to be installed.
    """
    return None


class StubAccuracyVerifier(AccuracyVerifier):
    """Stub AccuracyVerifier that skips backend selection and returns
    canned verify() results.

    Overrides ``__post_init__`` to do nothing (no backend selection).
    Overrides ``verify()`` to return a ``VerificationReport`` with
    one canned ``OutputComparison`` per prompt.
    """

    def __post_init__(self) -> None:
        # Intentionally do nothing -- skip backend selection, model loading,
        # and any other real initialization.
        pass

    def verify(
        self,
        prompts: str | list[str],
        collect_hidden_states: bool = False,
        thresholds: dict[str, float] | None = None,
    ) -> VerificationReport:
        if isinstance(prompts, str):
            prompts = [prompts]

        per_prompt = [
            {
                "prompt": p,
                "comparison": OutputComparison(
                    token_exact_match=1.0,
                    token_edit_distance=0.0,
                    logit_cosine_sim=1.0,
                    logit_kl_div=0.0,
                    logit_max_abs_diff=0.0,
                    hidden_cosine_sim=1.0,
                    hidden_max_abs_diff=0.0,
                    hidden_relative_error=0.0,
                    pass_threshold=True,
                    thresholds=DEFAULT_THRESHOLDS,
                ),
                "reference": GenerationOutput(token_ids=[], text=""),
                "candidate": GenerationOutput(token_ids=[], text=""),
            }
            for p in prompts
        ]

        return VerificationReport(
            model_name=self.model_name,
            num_nodes=self.num_nodes,
            dtype=self.dtype,
            temperature=self.temperature,
            per_prompt=per_prompt,
            created_at=0.0,
            duration_ms=0.0,
        )
