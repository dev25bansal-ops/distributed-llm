"""Tests for CacheDoctor (self-healing cache diagnostics)."""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/cache_doctor.py")
CacheDoctor = _mod.CacheDoctor
DiagnosticIssue = _mod.DiagnosticIssue


class TestDiagnosticIssue:
    def test_constructor(self):
        issue = DiagnosticIssue(
            severity="warning",
            component="prefix_cache",
            description="Low hit rate",
        )
        assert issue.severity == "warning"
        assert issue.component == "prefix_cache"
        assert issue.description == "Low hit rate"
        assert issue.auto_fixed is False
        assert issue.timestamp > 0


class TestCacheDoctor:
    def test_init(self):
        doctor = CacheDoctor()
        assert doctor._cache_manager is None
        assert doctor._issues == []
        assert doctor._tier_disabled_until == {}

    def test_diagnose_no_cache_manager(self):
        doctor = CacheDoctor()
        issues = doctor.diagnose()
        assert isinstance(issues, list)

    def test_diagnose_with_manager_no_prefix_cache(self):
        class FakeManager:
            prefix_cache = None
            def get_tier_stats(self):
                return {"local": {"hits": 5, "misses": 5}}
            def get_tier_latencies(self):
                return {"local": {"avg_ms": 1, "p50_ms": 1, "p95_ms": 2}}

        doctor = CacheDoctor(FakeManager())
        issues = doctor.diagnose()
        assert isinstance(issues, list)

    def test_diagnose_detects_low_hit_rate(self):
        class FakePrefixCache:
            def stats(self):
                return {
                    "prefix_cache_hit_rate": 0.05,
                    "prefix_cache_memory_util": 0.5,
                    "prefix_cache_hits": 10,
                    "prefix_cache_misses": 200,
                }

        class FakeManager:
            prefix_cache = FakePrefixCache()
            def get_tier_stats(self):
                return {"local": {"hits": 0, "misses": 0}}
            def get_tier_latencies(self):
                return {"local": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0}}

        doctor = CacheDoctor(FakeManager())
        issues = doctor.diagnose()
        warnings = [i for i in issues if i.severity == "warning"]
        assert any("Low hit rate" in i.description for i in warnings)

    def test_diagnose_detects_high_memory(self):
        class FakePrefixCache:
            def stats(self):
                return {
                    "prefix_cache_hit_rate": 0.5,
                    "prefix_cache_memory_util": 0.98,
                    "prefix_cache_hits": 50,
                    "prefix_cache_misses": 50,
                }

        class FakeManager:
            prefix_cache = FakePrefixCache()
            def get_tier_stats(self):
                return {"local": {"hits": 0, "misses": 0}}
            def get_tier_latencies(self):
                return {"local": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0}}

        doctor = CacheDoctor(FakeManager())
        issues = doctor.diagnose()
        warnings = [i for i in issues if i.severity == "warning"]
        assert any("Memory near limit" in i.description for i in warnings)

    def test_auto_repair_disables_empty_tier(self):
        class FakeManager:
            prefix_cache = None
            def get_tier_stats(self):
                return {"broadcast": {"hits": 0, "misses": 100}}
            def get_tier_latencies(self):
                return {"broadcast": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0}}

        doctor = CacheDoctor(FakeManager())
        doctor.diagnose()
        assert "broadcast" in doctor._tier_disabled_until
        assert doctor.is_tier_disabled("broadcast") is True

    def test_is_tier_disabled_missing_false(self):
        doctor = CacheDoctor()
        assert doctor.is_tier_disabled("nonexistent") is False

    def test_get_issues_filtered(self):
        doctor = CacheDoctor()
        doctor._issues = [
            DiagnosticIssue("critical", "system", "OOM"),
            DiagnosticIssue("warning", "disk", "Slow"),
        ]
        crits = doctor.get_issues("critical")
        assert len(crits) == 1
        assert crits[0].severity == "critical"

        warns = doctor.get_issues("warning")
        assert len(warns) == 1
        assert warns[0].severity == "warning"

    def test_clear_issues(self):
        doctor = CacheDoctor()
        doctor._issues = [DiagnosticIssue("info", "test", "test")]
        doctor.clear_issues()
        assert doctor._issues == []

    def test_health_summary(self):
        doctor = CacheDoctor()
        doctor._issues = [
            DiagnosticIssue("critical", "system", "OOM"),
            DiagnosticIssue("warning", "disk", "Slow", auto_fixed=True),
        ]
        summary = doctor.health_summary()
        assert summary["total_issues"] == 2
        assert summary["critical"] == 1
        assert summary["warnings"] == 1
        assert summary["auto_fixed"] == 1
