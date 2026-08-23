"""Tests for SpeculativeProfiler and MethodProfile.

Uses the import-helper pattern to avoid circular imports.
"""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_prof_mod = load_module("distllm/core/speculative_profiler.py")
SpeculativeProfiler = _prof_mod.SpeculativeProfiler
MethodProfile = _prof_mod.MethodProfile


class TestMethodProfile:
    def test_defaults(self):
        p = MethodProfile()
        assert p.method == ""
        assert p.workload_type == ""
        assert p.total_drafts == 0
        assert p.total_accepted == 0
        assert p.total_latency_s == 0.0
        assert p.samples == 0

    def test_acceptance_rate_zero_on_no_drafts(self):
        p = MethodProfile(method="ngram", workload_type="code")
        assert p.acceptance_rate == 0.0

    def test_acceptance_rate(self):
        p = MethodProfile(method="ngram", workload_type="code", total_drafts=10, total_accepted=7)
        assert p.acceptance_rate == 0.7

    def test_avg_latency_ms(self):
        p = MethodProfile(method="eagle", workload_type="code", total_latency_s=2.0, samples=4)
        assert p.avg_latency_ms == 500.0

    def test_avg_latency_ms_zero_on_no_samples(self):
        p = MethodProfile(method="ngram", workload_type="code")
        assert p.avg_latency_ms == 0.0


class TestSpeculativeProfiler:
    def test_default_construction(self):
        profiler = SpeculativeProfiler()
        assert profiler._warmup == 5

    def test_custom_warmup(self):
        profiler = SpeculativeProfiler(warmup_samples=10)
        assert profiler._warmup == 10

    def test_record_acceptance_creates_profile(self):
        profiler = SpeculativeProfiler()
        profiler.record_acceptance("ngram", "code", drafted=5, accepted=4)
        profile = profiler.get_profile("ngram", "code")
        assert profile.total_drafts == 5
        assert profile.total_accepted == 4
        assert profile.samples == 1

    def test_record_acceptance_accumulates(self):
        profiler = SpeculativeProfiler()
        profiler.record_acceptance("ngram", "code", drafted=5, accepted=4)
        profiler.record_acceptance("ngram", "code", drafted=3, accepted=2)
        profile = profiler.get_profile("ngram", "code")
        assert profile.total_drafts == 8
        assert profile.total_accepted == 6
        assert profile.samples == 2

    def test_get_profile_unknown_returns_empty(self):
        profiler = SpeculativeProfiler()
        p = profiler.get_profile("nonexistent", "code")
        assert p.total_drafts == 0

    def test_get_best_method_during_warmup_uses_prior(self):
        profiler = SpeculativeProfiler(warmup_samples=10)
        # Only 1 sample, below warmup threshold
        profiler.record_acceptance("ngram", "code", drafted=5, accepted=4)
        # During warmup, should return the prior-best for "code" workload
        best = profiler.get_best_method("code")
        assert isinstance(best, str)

    def test_get_best_method_after_warmup(self):
        profiler = SpeculativeProfiler(warmup_samples=2)
        profiler.record_acceptance("ngram", "code", drafted=5, accepted=4)
        profiler.record_acceptance("eagle", "code", drafted=5, accepted=1)
        # ngram has higher acceptance (0.8) than eagle (0.2)
        best = profiler.get_best_method("code")
        assert best == "ngram"

    def test_get_best_method_fallback(self):
        profiler = SpeculativeProfiler()
        best = profiler.get_best_method("unknown_workload")
        # Should fall back to "ngram" as default
        assert best == "ngram"

    def test_get_method_ranking_empty(self):
        profiler = SpeculativeProfiler()
        ranking = profiler.get_method_ranking("nonexistent")
        # Should use prior rates for unknown workload type
        assert len(ranking) >= 3

    def test_get_method_ranking_after_records(self):
        profiler = SpeculativeProfiler(warmup_samples=1)
        profiler.record_acceptance("ngram", "code", drafted=10, accepted=8)
        profiler.record_acceptance("eagle", "code", drafted=10, accepted=3)
        ranking = profiler.get_method_ranking("code")
        # ngram (0.8) should be first
        assert ranking[0][0] == "ngram"
        assert ranking[0][1] > ranking[1][1]

    def test_get_all_profiles_empty(self):
        profiler = SpeculativeProfiler()
        assert profiler.get_all_profiles() == {}

    def test_get_all_profiles_after_records(self):
        profiler = SpeculativeProfiler()
        profiler.record_acceptance("ngram", "code", drafted=5, accepted=4)
        profiler.record_acceptance("eagle", "instruction", drafted=5, accepted=3)
        profiles = profiler.get_all_profiles()
        assert len(profiles) == 2
        assert ("ngram", "code") in profiles
        assert ("eagle", "instruction") in profiles

    def test_record_with_latency(self):
        profiler = SpeculativeProfiler()
        profiler.record_acceptance("ngram", "code", drafted=5, accepted=4, latency_ms=100.0)
        profile = profiler.get_profile("ngram", "code")
        assert profile.total_latency_s == 0.1
        assert profile.avg_latency_ms == 100.0
