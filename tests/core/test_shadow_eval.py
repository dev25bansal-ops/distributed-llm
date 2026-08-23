"""Tests for shadow evaluation and regression detection.

Includes regression coverage for B17: ShadowEvalRunner must fail closed
instead of crashing with a NameError or fabricating random evaluation scores.
``run_eval()`` / ``_run_single_suite()`` raise an explicit
``NotImplementedError`` ("not configured") rather than a ``NameError``, and no
silent random-draw "results" are returned.
"""

import inspect

import pytest

import distllm.core.shadow_eval_runner as shadow_eval_runner_mod
from distllm.core.shadow_eval_runner import (
    RegressionDetector,
    RegressionReport,
    ShadowEvalConfig,
    ShadowEvalRunner,
)


class TestRegressionDetectorCompare:
    def test_detects_regression(self):
        detector = RegressionDetector()
        baseline = {"accuracy": [0.75, 0.74, 0.76, 0.75, 0.74]}
        candidate = {"accuracy": [0.65, 0.63, 0.66, 0.64, 0.65]}
        report = detector.compare(baseline, candidate)
        assert isinstance(report, RegressionReport)
        assert report.deltas.get("accuracy", 0) < 0

    def test_no_regression_when_accuracy_improves(self):
        detector = RegressionDetector()
        baseline = {"accuracy": [0.70, 0.71, 0.69, 0.70]}
        candidate = {"accuracy": [0.80, 0.81, 0.79, 0.80]}
        report = detector.compare(baseline, candidate)
        if "accuracy" in report.deltas:
            assert report.deltas["accuracy"] > 0

    def test_report_has_required_fields(self):
        detector = RegressionDetector()
        baseline = {"mmlu": [0.72, 0.73, 0.71]}
        candidate = {"mmlu": [0.70, 0.69, 0.71]}
        report = detector.compare(baseline, candidate)
        assert hasattr(report, "metrics")
        assert hasattr(report, "deltas")
        assert hasattr(report, "is_regression")


class TestShadowEvalRunner:
    def test_config_defaults(self):
        config = ShadowEvalConfig()
        assert config.eval_suites == ["mmlu", "gsm8k", "humaneval"]

    # ── B17 regression coverage ────────────────────────────────────────────

    def test_run_eval_fails_closed_not_name_error(self):
        """run_eval() must raise NotImplementedError, never a NameError."""
        runner = ShadowEvalRunner()
        with pytest.raises(NotImplementedError, match="not configured"):
            runner.run_eval("v1.0")

    def test_run_eval_single_suite_fails_closed(self):
        """A specific requested suite must fail closed too."""
        runner = ShadowEvalRunner()
        with pytest.raises(NotImplementedError, match="not configured"):
            runner.run_eval("v1.0", eval_suite="mmlu")

    def test_no_silent_random_draws(self):
        """run_eval() must never return fabricated random scores, and the
        module must not reference the ``random`` module for eval scoring."""
        runner = ShadowEvalRunner()
        with pytest.raises(NotImplementedError):
            runner.run_eval("v1.0")

        src = inspect.getsource(shadow_eval_runner_mod)
        assert "random.gauss" not in src, "shadow eval must not fabricate gauss scores"
        assert "random.seed" not in src, "shadow eval must not seed fabricated scores"

    def test_check_regression_without_baseline_is_none(self):
        """check_regression() with no cached baseline returns None (no eval run)."""
        runner = ShadowEvalRunner()
        assert runner.check_regression("v1.0") is None
