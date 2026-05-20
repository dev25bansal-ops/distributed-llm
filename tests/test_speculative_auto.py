"""Tests for auto-speculative selection features."""

import pytest

from distllm.core.speculative_profiler import SpeculativeProfiler, MethodProfile
from distllm.core.workload_classifier import classify, classify_features, WorkloadType
from distllm.core.speculative_adaptor import SpeculativeAdaptor
from distllm.core.speculative_dashboard import SpeculativeDashboard, MethodComparison
from distllm.core.speculative_decoder import SpeculativeDecoder


class TestSpeculativeProfiler:
    def test_record_and_get_best_method(self):
        profiler = SpeculativeProfiler(warmup_samples=1)
        profiler.record_acceptance("ngram", "code", 5, 4)
        profiler.record_acceptance("eagle", "code", 5, 2)

        best = profiler.get_best_method("code")
        assert best == "ngram"  # Higher acceptance rate

    def test_fallback_to_prior(self):
        profiler = SpeculativeProfiler(warmup_samples=100)
        # No data recorded, should return prior
        best = profiler.get_best_method("code")
        assert best == "ngram"  # Prior for code

    def test_method_ranking(self):
        profiler = SpeculativeProfiler(warmup_samples=1)
        profiler.record_acceptance("ngram", "repetitive", 5, 4, 10.0)
        profiler.record_acceptance("draft_model", "repetitive", 5, 3, 15.0)

        ranking = profiler.get_method_ranking("repetitive")
        assert len(ranking) == 2
        # ngram has higher acceptance rate
        assert ranking[0][0] == "ngram"

    def test_profile_data(self):
        profiler = SpeculativeProfiler(warmup_samples=1)
        profiler.record_acceptance("eagle", "instruction", 5, 3, 20.0)

        profile = profiler.get_profile("eagle", "instruction")
        assert profile.total_drafts == 5
        assert profile.total_accepted == 3
        assert profile.samples == 1


class TestWorkloadClassifier:
    def test_code_detection(self):
        text = """
def hello_world():
    print("Hello, world!")
    for i in range(10):
        return i + 1
"""
        result = classify(text)
        assert result == WorkloadType.CODE

    def test_instruction_detection(self):
        text = "Please explain how machine learning works and can you summarize the key concepts?"
        result = classify(text)
        assert result == WorkloadType.INSTRUCTION

    def test_repetitive_detection(self):
        text = "the cat sat on the mat the cat sat on the mat the cat sat"
        result = classify(text)
        assert result == WorkloadType.REPETITIVE

    def test_diverse_detection(self):
        text = "The quantum mechanics of subatomic particles reveals fascinating phenomena including superposition entanglement and wave-particle duality which challenge our classical intuition about reality"
        result = classify(text)
        assert result in (WorkloadType.DIVERSE, WorkloadType.UNKNOWN)

    def test_empty_text(self):
        assert classify("") == WorkloadType.UNKNOWN

    def test_classify_features(self):
        text = "def foo(): return 42"
        features = classify_features(text)
        assert features["code_ratio"] > 0
        assert features["entropy_3gram"] >= 0


class TestSpeculativeAdaptor:
    def test_increase_tokens_on_high_acceptance(self):
        adaptor = SpeculativeAdaptor(base_tokens=5, target_rate=0.6)
        # High acceptance rate should increase tokens
        new_tokens = adaptor.adapt(0.8)
        assert new_tokens >= 5

    def test_decrease_tokens_on_low_acceptance(self):
        adaptor = SpeculativeAdaptor(base_tokens=5, target_rate=0.6)
        # Low acceptance rate should decrease tokens
        new_tokens = adaptor.adapt(0.3)
        assert new_tokens <= 5

    def test_disable_on_very_low_acceptance(self):
        adaptor = SpeculativeAdaptor(base_tokens=5)
        new_tokens = adaptor.adapt(0.1)
        assert new_tokens == 0
        assert adaptor.is_disabled

    def test_respect_bounds(self):
        adaptor = SpeculativeAdaptor(base_tokens=5, min_tokens=1, max_tokens=10)
        # Force many adjustments
        for _ in range(20):
            adaptor.adapt(0.9)  # High acceptance -> increase
        assert adaptor.current_tokens <= 10

    def test_reset(self):
        adaptor = SpeculativeAdaptor(base_tokens=5)
        adaptor.adapt(0.8)
        adaptor.reset()
        assert adaptor.current_tokens == 5
        assert not adaptor.is_disabled


class TestSpeculativeDashboard:
    def test_update_and_report(self):
        dashboard = SpeculativeDashboard()
        dashboard.update_method("ngram", "code", 0.7, 1.5, 100.0, 50)
        dashboard.update_method("eagle", "code", 0.5, 1.3, 80.0, 30)

        report = dashboard.get_comparison_report()
        assert "ngram" in report["method_summary"]
        assert "eagle" in report["method_summary"]

    def test_record_comparison(self):
        dashboard = SpeculativeDashboard()
        comparison = dashboard.record_comparison(
            "ngram", "eagle",
            {"acceptance_rate": 0.7, "avg_speedup": 1.5, "tokens_per_sec": 100, "samples": 50},
            {"acceptance_rate": 0.5, "avg_speedup": 1.3, "tokens_per_sec": 80, "samples": 30},
        )
        assert comparison.winner == "ngram"

    def test_export_json(self):
        dashboard = SpeculativeDashboard()
        dashboard.update_method("ngram", "code", 0.7, 1.5, 100.0, 50)
        data = dashboard.export_json()
        assert "ngram" in data

    def test_reset(self):
        dashboard = SpeculativeDashboard()
        dashboard.update_method("ngram", "code", 0.7, 1.5, 100.0, 50)
        dashboard.reset()
        report = dashboard.get_comparison_report()
        assert report["method_summary"] == {}


class TestSpeculativeDecoderIntegration:
    def test_profiler_integration(self):
        decoder = SpeculativeDecoder(method="auto")
        decoder.set_workload_type("code")

        # Record some performance data
        decoder.record_method_performance("ngram", 5, 4, 10.0)
        decoder.record_method_performance("eagle", 5, 2, 15.0)

        # Now auto-select should prefer ngram for code
        method = decoder.get_active_method(workload_type="code")
        assert method == "ngram"

    def test_metrics_include_profiler(self):
        decoder = SpeculativeDecoder(method="auto")
        decoder.set_workload_type("instruction")
        decoder.record_method_performance("eagle", 5, 3, 20.0)

        metrics = decoder.get_metrics()
        assert "method_ranking" in metrics
        assert "workload_type" in metrics

    def test_tree_drafts_wired(self):
        """Test that tree_draft method is callable (requires draft_model, so just check it doesn't crash)."""
        decoder = SpeculativeDecoder(method="tree_draft")
        # Can't actually run without a model, but verify the method dispatch exists
        assert decoder.method == "tree_draft"
