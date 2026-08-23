"""Regression tests for the two market-differentiator features.

1. Verified-equivalence spec-decode (spec_divergence_total metric + sampling).
2. Unified cost/carbon/latency SLA router (objective dispatch + savings/gCO2).
"""

from distllm.core.coordinator_metrics import MetricsManager
from distllm.core.spec_equivalence import SpecEquivalenceChecker
from distllm.core.unified_sla_router import Objective, UnifiedSlaRouter, SlaRouterReport
from distllm.core.cross_cloud_router import RouteDecision


# ── Feature 1: verified-equivalence ──

def test_equivalence_disabled_by_default():
    c = SpecEquivalenceChecker(sample_rate=0.0)
    assert c.enabled is False
    assert c.should_sample() is False


def test_equivalence_samples_by_rate():
    seen = []
    c = SpecEquivalenceChecker(sample_rate=1.0, rng=lambda: 0.0)  # always sample
    assert c.should_sample() is True
    c2 = SpecEquivalenceChecker(sample_rate=0.5, rng=lambda: 0.9)  # never sample
    assert c2.should_sample() is False


def test_equivalence_passes_on_identical_tokens():
    c = SpecEquivalenceChecker(sample_rate=1.0, rng=lambda: 0.0)
    assert c.check([1, 2, 3], [1, 2, 3], request_id="r1") is True
    assert c.diverged == 0
    assert c.checked == 1


def test_equivalence_flags_divergence_and_records_metric():
    mm = MetricsManager()
    c = SpecEquivalenceChecker(sample_rate=1.0, metrics=mm, rng=lambda: 0.0)
    # divergent tokens
    assert c.check([1, 2, 9], [1, 2, 3], request_id="r2") is False
    assert c.diverged == 1
    # metric recorded in the Prometheus-compatible store
    assert mm.get().get("spec_divergence_total", 0) == 1


def test_equivalence_ignores_length_mismatch():
    c = SpecEquivalenceChecker(sample_rate=1.0, rng=lambda: 0.0)
    assert c.check([1, 2], [1, 2, 3]) is False


# ── Feature 2: unified SLA router ──

class _FakeRouter:
    """Minimal stand-in for CrossCloudRouter returning canned decisions."""

    def __init__(self, cheapest, fastest, greenest):
        self._cheapest = cheapest
        self._fastest = fastest
        self._greenest = greenest
        self.calls: list[tuple] = []

    def select_provider(self, **kw):
        self.calls.append(("cheapest", kw))
        return self._cheapest

    def select_provider_fastest(self, **kw):
        self.calls.append(("fastest", kw))
        return self._fastest

    def select_provider_carbon_aware(self, **kw):
        self.calls.append(("greenest", kw))
        return self._greenest


def _decision(provider, price, latency, carbon=0.0, region="us-east-1"):
    return RouteDecision(
        provider=provider, instance_type="p4d", price_per_hour=price,
        estimated_cost=price, latency_ms=latency, region=region,
        carbon_intensity=carbon, reason="test",
    )


def _make_router():
    cheap = _decision("aws", price=2.0, latency=50.0, carbon=350.0)
    fast = _decision("gcp", price=4.0, latency=10.0, carbon=400.0)
    green = _decision("azure", price=3.0, latency=30.0, carbon=120.0)
    return _FakeRouter(cheap, fast, green)


def test_unified_cheapest_objective():
    router = UnifiedSlaRouter(_make_router())
    rep = router.route("cheapest")
    assert rep.objective is Objective.CHEAPEST
    assert rep.provider == "aws"
    assert rep.price_per_hour == 2.0


def test_unified_fastest_objective():
    router = UnifiedSlaRouter(_make_router())
    rep = router.route("fastest")
    assert rep.objective is Objective.FASTEST
    assert rep.provider == "gcp"
    assert rep.latency_ms == 10.0


def test_unified_greenest_computes_gco2_avoided():
    router = UnifiedSlaRouter(_make_router())
    rep = router.route("greenest")
    assert rep.objective is Objective.GREENEST
    assert rep.provider == "azure"
    # baseline 475 - azure 120 = 355 gCO2/kWh * 0.4 kWh ≈ 142 gCO2 avoided
    assert rep.gco2_avoided == 142.0


def test_unified_balanced_records_metrics():
    mm = MetricsManager()
    router = UnifiedSlaRouter(_make_router(), metrics=mm)
    rep = router.route("balanced")
    assert rep.objective is Objective.BALANCED
    # greenest path chosen (carbon_aware); azure carbon 120 -> gCO2 avoided 142
    assert rep.gco2_avoided == 142.0
    # metrics accumulated (gCO2 stored as int grams)
    assert mm.get().get("sla_gco2_avoided_total", 0) == 142


def test_unified_unknown_objective_defaults_balanced():
    router = UnifiedSlaRouter(_make_router())
    rep = router.route("not-a-real-objective")
    assert rep.objective is Objective.BALANCED


def test_unified_no_provider_returns_reason():
    empty = _FakeRouter(None, None, None)
    router = UnifiedSlaRouter(empty)
    rep = router.route("cheapest")
    assert rep.provider == ""
    assert "no provider" in rep.reason
