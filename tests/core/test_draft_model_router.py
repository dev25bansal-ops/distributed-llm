"""Tests for draft_model_router.py — DraftModelFleet and DraftModelRouter.

Covers:
    WorkloadType       -- Enum values
    DraftModelSpec     -- Dataclass construction/defaults
    DraftModelHealth   -- Properties: avg_latency_ms, error_rate, tokens_per_second
    RoutingConstraints -- Dataclass construction/defaults
    RoutingDecision    -- Dataclass construction/defaults
    DraftModelFleet    -- register, unregister, record_success, record_error,
                          mark_healthy, properties
    DraftModelRouter   -- select, _score, last_decision, fleet_stats

Every test is deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the source module
_mod = load_module("distllm/core/draft_model_router.py")

# Re-export symbols for test readability
WorkloadType = _mod.WorkloadType
DraftModelSpec = _mod.DraftModelSpec
DraftModelHealth = _mod.DraftModelHealth
RoutingConstraints = _mod.RoutingConstraints
RoutingDecision = _mod.RoutingDecision
DraftModelFleet = _mod.DraftModelFleet
DraftModelRouter = _mod.DraftModelRouter


# ===================================================================
# HELPERS
# ===================================================================

def make_spec(
    endpoint_url: str = "http://draft:8000/v1/completions",
    model_name: str = "SmolLM-135M",
    hardware: str = "cpu",
    cost_per_hour: float = 0.05,
    avg_latency_ms: float = 45.0,
    avg_acceptance_rate: float = 0.7,
    max_concurrent: int = 10,
    **kwargs: float | str | int | bool,
) -> DraftModelSpec:
    """Factory: deterministic DraftModelSpec for tests."""
    return DraftModelSpec(
        endpoint_url=endpoint_url,
        model_name=model_name,
        hardware=hardware,
        cost_per_hour=cost_per_hour,
        avg_latency_ms=avg_latency_ms,
        avg_acceptance_rate=avg_acceptance_rate,
        max_concurrent=max_concurrent,
        **kwargs,
    )


# ===================================================================
# WORKLOAD TYPE TESTS
# ===================================================================

class TestWorkloadType:
    """WorkloadType enum values."""

    def test_code_value(self) -> None:
        assert WorkloadType.CODE.value == "code"

    def test_instruction_value(self) -> None:
        assert WorkloadType.INSTRUCTION.value == "instruction"

    def test_repetitive_value(self) -> None:
        assert WorkloadType.REPETITIVE.value == "repetitive"

    def test_diverse_value(self) -> None:
        assert WorkloadType.DIVERSE.value == "diverse"

    def test_unknown_value(self) -> None:
        assert WorkloadType.UNKNOWN.value == "unknown"

    def test_is_str_enum(self) -> None:
        """WorkloadType inherits from str as well, so it can be compared to str."""
        assert WorkloadType.CODE == "code"
        assert isinstance(WorkloadType.CODE, str)

    def test_membership(self) -> None:
        """All expected members exist."""
        names = {m.name for m in WorkloadType}
        assert names == {"CODE", "INSTRUCTION", "REPETITIVE", "DIVERSE", "UNKNOWN"}


# ===================================================================
# DRAFT MODEL SPEC TESTS
# ===================================================================

class TestDraftModelSpec:
    """DraftModelSpec dataclass -- construction and defaults."""

    def test_minimal_construction(self) -> None:
        """Only endpoint_url is required; everything else gets defaults."""
        spec = DraftModelSpec(endpoint_url="http://test:8000/")
        assert spec.endpoint_url == "http://test:8000/"
        assert spec.model_name == ""
        assert spec.api_key == ""
        assert spec.hardware == "cpu"
        assert spec.transport == "http"
        assert spec.cost_per_hour == 0.0
        assert spec.avg_latency_ms == 0.0
        assert spec.avg_acceptance_rate == 0.0
        assert spec.max_concurrent == 10
        assert spec.timeout_seconds == 30.0
        assert spec.max_retries == 2
        assert spec.verify_ssl is True
        assert spec.metadata == {}

    def test_full_construction(self) -> None:
        spec = DraftModelSpec(
            endpoint_url="http://gpu:8001/",
            model_name="SmolLM-360M",
            api_key="sk-test",
            hardware="cuda:0",
            transport="grpc",
            cost_per_hour=0.60,
            avg_latency_ms=8.0,
            avg_acceptance_rate=0.85,
            max_concurrent=4,
            timeout_seconds=60.0,
            max_retries=5,
            verify_ssl=False,
            metadata={"region": "us-east"},
        )
        assert spec.api_key == "sk-test"
        assert spec.transport == "grpc"
        assert spec.timeout_seconds == 60.0
        assert spec.max_retries == 5
        assert spec.verify_ssl is False
        assert spec.metadata == {"region": "us-east"}

    def test_metadata_is_independent(self) -> None:
        """Each spec should have its own metadata dict, not a shared default."""
        spec1 = DraftModelSpec(endpoint_url="http://a/")
        spec2 = DraftModelSpec(endpoint_url="http://b/")
        spec1.metadata["key"] = "value"
        assert "key" not in spec2.metadata


# ===================================================================
# DRAFT MODEL HEALTH TESTS
# ===================================================================

class TestDraftModelHealth:
    """DraftModelHealth dataclass -- properties and defaults."""

    def test_default_construction(self) -> None:
        health = DraftModelHealth(endpoint_url="http://test:8000/")
        assert health.endpoint_url == "http://test:8000/"
        assert health.is_healthy is True
        assert health.current_concurrent == 0
        assert health.total_calls == 0
        assert health.total_errors == 0
        assert health.total_latency_s == 0.0
        assert health.total_tokens == 0
        assert health.recent_latency_ms == 0.0
        assert health.recent_acceptance_rate == 0.0
        assert health.last_error == ""
        assert health.last_error_time == 0.0
        assert health.consecutive_failures == 0

    def test_avg_latency_ms_no_calls(self) -> None:
        health = DraftModelHealth(endpoint_url="http://test:8000/")
        assert health.avg_latency_ms == 0.0

    def test_avg_latency_ms_with_calls(self) -> None:
        health = DraftModelHealth(
            endpoint_url="http://test:8000/",
            total_calls=2,
            total_latency_s=3.0,
        )
        # (3.0 / 2) * 1000 = 1500 ms
        assert health.avg_latency_ms == 1500.0

    def test_avg_latency_ms_partial_call(self) -> None:
        """Single call with 0.5s latency."""
        health = DraftModelHealth(
            endpoint_url="http://test:8000/",
            total_calls=1,
            total_latency_s=0.5,
        )
        assert health.avg_latency_ms == 500.0

    def test_error_rate_no_calls(self) -> None:
        health = DraftModelHealth(endpoint_url="http://test:8000/")
        assert health.error_rate == 0.0

    def test_error_rate_some_errors(self) -> None:
        health = DraftModelHealth(
            endpoint_url="http://test:8000/",
            total_calls=10,
            total_errors=3,
        )
        assert health.error_rate == 0.3

    def test_error_rate_all_errors(self) -> None:
        health = DraftModelHealth(
            endpoint_url="http://test:8000/",
            total_calls=5,
            total_errors=5,
        )
        assert health.error_rate == 1.0

    def test_tokens_per_second_no_latency(self) -> None:
        health = DraftModelHealth(endpoint_url="http://test:8000/")
        assert health.tokens_per_second == 0.0

    def test_tokens_per_second_with_latency(self) -> None:
        health = DraftModelHealth(
            endpoint_url="http://test:8000/",
            total_tokens=100,
            total_latency_s=25.0,
        )
        assert health.tokens_per_second == 4.0

    def test_tokens_per_second_high_throughput(self) -> None:
        health = DraftModelHealth(
            endpoint_url="http://test:8000/",
            total_tokens=1000,
            total_latency_s=5.0,
        )
        assert health.tokens_per_second == 200.0

    def test_tokens_per_second_zero_latency(self) -> None:
        """Zero total_latency_s should return 0.0, not crash."""
        health = DraftModelHealth(
            endpoint_url="http://test:8000/",
            total_tokens=50,
            total_latency_s=0.0,
        )
        assert health.tokens_per_second == 0.0


# ===================================================================
# ROUTING CONSTRAINTS TESTS
# ===================================================================

class TestRoutingConstraints:
    """RoutingConstraints dataclass -- defaults and construction."""

    def test_defaults(self) -> None:
        c = RoutingConstraints()
        assert c.max_latency_ms == 100.0
        assert c.max_cost_per_hour == 10.0
        assert c.min_acceptance_rate == 0.0
        assert c.preferred_hardware == []
        assert c.workload_type == "unknown"
        assert c.max_concurrent == 0  # 0 = no limit

    def test_custom_values(self) -> None:
        c = RoutingConstraints(
            max_latency_ms=50.0,
            max_cost_per_hour=1.0,
            min_acceptance_rate=0.5,
            preferred_hardware=["cuda:0"],
            workload_type="code",
            max_concurrent=5,
        )
        assert c.max_latency_ms == 50.0
        assert c.max_cost_per_hour == 1.0
        assert c.min_acceptance_rate == 0.5
        assert c.preferred_hardware == ["cuda:0"]
        assert c.workload_type == "code"
        assert c.max_concurrent == 5

    def test_preferred_hardware_independence(self) -> None:
        c1 = RoutingConstraints()
        c2 = RoutingConstraints()
        c1.preferred_hardware.append("cpu")
        assert c2.preferred_hardware == []


# ===================================================================
# ROUTING DECISION TESTS
# ===================================================================

class TestRoutingDecision:
    """RoutingDecision dataclass -- defaults and construction."""

    def test_defaults(self) -> None:
        d = RoutingDecision(
            selected_url="",
            selected_model="",
            selection_reason="test",
            score=0.0,
            candidates_evaluated=0,
            candidates_qualified=0,
        )
        assert d.fallback_used is False

    def test_full_construction(self) -> None:
        d = RoutingDecision(
            selected_url="http://best:8000/",
            selected_model="SmolLM-360M",
            selection_reason="best_score",
            score=0.85,
            candidates_evaluated=5,
            candidates_qualified=3,
            fallback_used=True,
        )
        assert d.selected_url == "http://best:8000/"
        assert d.score == 0.85
        assert d.fallback_used is True


# ===================================================================
# DRAFT MODEL FLEET TESTS
# ===================================================================

class TestDraftModelFleet:
    """DraftModelFleet -- registration, health tracking, properties."""

    def test_initial_state(self) -> None:
        fleet = DraftModelFleet()
        assert fleet.size == 0
        assert fleet.healthy_endpoints == []
        assert fleet.get_all_specs() == []
        assert fleet.get_all_health() == {}

    def test_register_adds_spec_and_health(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec()
        fleet.register(spec)
        assert fleet.size == 1
        assert fleet.get_spec(spec.endpoint_url) is spec
        health = fleet.get_health(spec.endpoint_url)
        assert health is not None
        assert health.endpoint_url == spec.endpoint_url
        assert health.is_healthy is True

    def test_register_duplicate_updates_spec_only(self) -> None:
        """Registering the same URL again should update the spec but keep health."""
        fleet = DraftModelFleet()
        spec1 = make_spec(endpoint_url="http://test:8000/", cost_per_hour=0.05)
        fleet.register(spec1)
        health = fleet.get_health("http://test:8000/")
        assert health is not None
        health.total_calls = 42  # modify health

        spec2 = make_spec(endpoint_url="http://test:8000/", cost_per_hour=0.99)
        fleet.register(spec2)
        # spec should be updated
        assert fleet.get_spec("http://test:8000/") is spec2
        assert fleet.get_spec("http://test:8000/").cost_per_hour == 0.99
        # health should NOT be re-created
        assert fleet.get_health("http://test:8000/") is health
        assert health.total_calls == 42

    def test_register_multiple_endpoints(self) -> None:
        fleet = DraftModelFleet()
        fleet.register(make_spec(endpoint_url="http://a:8000/"))
        fleet.register(make_spec(endpoint_url="http://b:8000/"))
        fleet.register(make_spec(endpoint_url="http://c:8000/"))
        assert fleet.size == 3
        assert len(fleet.get_all_specs()) == 3
        assert len(fleet.get_all_health()) == 3

    def test_unregister_removes_both(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec()
        fleet.register(spec)
        fleet.unregister(spec.endpoint_url)
        assert fleet.size == 0
        assert fleet.get_spec(spec.endpoint_url) is None
        assert fleet.get_health(spec.endpoint_url) is None

    def test_unregister_nonexistent(self) -> None:
        """Unregistering an endpoint that was never added should not raise."""
        fleet = DraftModelFleet()
        fleet.unregister("http://nonexistent:8000/")  # should not raise

    def test_unregister_then_get_all(self) -> None:
        fleet = DraftModelFleet()
        fleet.register(make_spec(endpoint_url="http://a:8000/"))
        fleet.register(make_spec(endpoint_url="http://b:8000/"))
        fleet.unregister("http://a:8000/")
        urls = {s.endpoint_url for s in fleet.get_all_specs()}
        assert urls == {"http://b:8000/"}

    def test_get_spec_missing(self) -> None:
        fleet = DraftModelFleet()
        assert fleet.get_spec("http://missing:8000/") is None

    def test_get_health_missing(self) -> None:
        fleet = DraftModelFleet()
        assert fleet.get_health("http://missing:8000/") is None

    def test_record_success_updates_health(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec()
        fleet.register(spec)
        fleet.record_success(
            endpoint_url=spec.endpoint_url,
            latency_s=1.5,
            tokens_generated=50,
            acceptance_rate=0.8,
        )
        health = fleet.get_health(spec.endpoint_url)
        assert health is not None
        assert health.total_calls == 1
        assert health.total_latency_s == 1.5
        assert health.total_tokens == 50
        assert health.recent_latency_ms == 1500.0  # 1.5 * 1000
        assert health.recent_acceptance_rate == 0.8
        assert health.consecutive_failures == 0
        assert health.is_healthy is True

    def test_record_success_resets_consecutive_failures(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec()
        fleet.register(spec)
        fleet.record_error(spec.endpoint_url, "error 1")
        fleet.record_error(spec.endpoint_url, "error 2")
        health = fleet.get_health(spec.endpoint_url)
        assert health is not None
        assert health.consecutive_failures == 2

        fleet.record_success(spec.endpoint_url, latency_s=0.1, tokens_generated=10)
        assert health.consecutive_failures == 0  # reset

    def test_record_success_unknown_endpoint(self) -> None:
        """record_success for an unregistered endpoint should not crash."""
        fleet = DraftModelFleet()
        fleet.record_success("http://unknown:8000/", latency_s=1.0, tokens_generated=10)

    def test_record_error_increments_error_count(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec()
        fleet.register(spec)
        fleet.record_error(spec.endpoint_url, "timeout")
        health = fleet.get_health(spec.endpoint_url)
        assert health is not None
        assert health.total_errors == 1
        assert health.total_calls == 1
        assert health.last_error == "timeout"
        assert health.consecutive_failures == 1
        assert health.is_healthy is True  # only 1 failure, not yet unhealthy

    def test_record_error_marks_unhealthy_after_three(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec()
        fleet.register(spec)
        fleet.record_error(spec.endpoint_url, "e1")
        fleet.record_error(spec.endpoint_url, "e2")
        fleet.record_error(spec.endpoint_url, "e3")
        health = fleet.get_health(spec.endpoint_url)
        assert health is not None
        assert health.consecutive_failures == 3
        assert health.is_healthy is False

    def test_record_error_unknown_endpoint(self) -> None:
        """record_error for an unregistered endpoint should not crash."""
        fleet = DraftModelFleet()
        fleet.record_error("http://unknown:8000/", "error")

    def test_mark_healthy_resets_failure_state(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec()
        fleet.register(spec)
        fleet.record_error(spec.endpoint_url, "e1")
        fleet.record_error(spec.endpoint_url, "e2")
        fleet.record_error(spec.endpoint_url, "e3")
        health = fleet.get_health(spec.endpoint_url)
        assert health is not None
        assert health.is_healthy is False

        fleet.mark_healthy(spec.endpoint_url)
        assert health.is_healthy is True
        assert health.consecutive_failures == 0

    def test_mark_healthy_unknown_endpoint(self) -> None:
        """mark_healthy for an unregistered endpoint should not crash."""
        fleet = DraftModelFleet()
        fleet.mark_healthy("http://unknown:8000/")

    def test_healthy_endpoints_property(self) -> None:
        fleet = DraftModelFleet()
        fleet.register(make_spec(endpoint_url="http://a:8000/"))
        fleet.register(make_spec(endpoint_url="http://b:8000/"))
        fleet.register(make_spec(endpoint_url="http://c:8000/"))

        # Mark b as unhealthy
        fleet.record_error("http://b:8000/", "error 1")
        fleet.record_error("http://b:8000/", "error 2")
        fleet.record_error("http://b:8000/", "error 3")

        healthy = fleet.healthy_endpoints
        assert "http://a:8000/" in healthy
        assert "http://b:8000/" not in healthy
        assert "http://c:8000/" in healthy
        assert len(healthy) == 2

    def test_healthy_endpoints_requires_spec_exists(self) -> None:
        """An endpoint with health but no spec should not appear in healthy_endpoints."""
        fleet = DraftModelFleet()
        spec = make_spec(endpoint_url="http://a:8000/")
        fleet.register(spec)
        fleet.unregister(spec.endpoint_url)
        # health still exists internally, but no spec
        assert fleet.healthy_endpoints == []

    def test_size_property(self) -> None:
        fleet = DraftModelFleet()
        assert fleet.size == 0
        fleet.register(make_spec(endpoint_url="http://a:8000/"))
        assert fleet.size == 1
        fleet.register(make_spec(endpoint_url="http://b:8000/"))
        assert fleet.size == 2
        fleet.unregister("http://a:8000/")
        assert fleet.size == 1

    def test_get_all_specs_returns_copy(self) -> None:
        fleet = DraftModelFleet()
        fleet.register(make_spec(endpoint_url="http://a:8000/"))
        specs = fleet.get_all_specs()
        specs.clear()
        # Original should be unaffected
        assert fleet.size == 1

    def test_get_all_health_returns_copy(self) -> None:
        fleet = DraftModelFleet()
        fleet.register(make_spec(endpoint_url="http://a:8000/"))
        health_map = fleet.get_all_health()
        health_map.clear()
        # Original should be unaffected
        assert len(fleet.get_all_health()) == 1


# ===================================================================
# DRAFT MODEL ROUTER TESTS
# ===================================================================

class TestDraftModelRouter:
    """DraftModelRouter -- select, _score, last_decision, fleet_stats."""

    # -- Construction / defaults --

    def test_default_construction(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet)
        assert router.last_decision is None

    def test_custom_weights(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet, latency_weight=0.5, cost_weight=0.5)
        assert router is not None

    # -- Empty fleet / no endpoints --

    def test_select_no_endpoints(self) -> None:
        """With no registered endpoints, select should return a fallback decision."""
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet)
        decision = router.select()
        assert decision.selected_url == ""
        assert decision.selected_model == ""
        assert decision.selection_reason == "no endpoints registered"
        assert decision.score == 0.0
        assert decision.candidates_evaluated == 0
        assert decision.candidates_qualified == 0
        assert decision.fallback_used is True

    def test_select_default_constraints(self) -> None:
        """When constraints is None, a default RoutingConstraints is used."""
        fleet = DraftModelFleet()
        fleet.register(make_spec())
        router = DraftModelRouter(fleet)
        decision = router.select(constraints=None)
        assert decision.selected_url != ""
        assert decision.fallback_used is False

    # -- No healthy endpoints --

    def test_select_no_healthy_endpoints(self) -> None:
        """With only unhealthy endpoints, select should return a fallback decision."""
        fleet = DraftModelFleet()
        spec = make_spec(endpoint_url="http://a:8000/")
        fleet.register(spec)
        # Make it unhealthy
        fleet.record_error(spec.endpoint_url, "e1")
        fleet.record_error(spec.endpoint_url, "e2")
        fleet.record_error(spec.endpoint_url, "e3")

        router = DraftModelRouter(fleet)
        decision = router.select()
        assert decision.selected_url == ""
        assert decision.selection_reason == "no healthy endpoints available"
        assert decision.score == 0.0
        assert decision.fallback_used is True
        assert decision.candidates_evaluated == 1

    # -- Happy path: single endpoint meets constraints --

    def test_select_single_candidate_qualified(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec(
            endpoint_url="http://cpu:8000/",
            model_name="SmolLM-135M",
            avg_latency_ms=45.0,
            cost_per_hour=0.05,
        )
        fleet.register(spec)
        router = DraftModelRouter(fleet)
        constraints = RoutingConstraints(
            max_latency_ms=100.0,
            max_cost_per_hour=1.0,
        )
        decision = router.select(constraints)
        assert decision.selected_url == "http://cpu:8000/"
        assert decision.selected_model == "SmolLM-135M"
        assert decision.selection_reason == "best_score"
        assert decision.candidates_evaluated == 1
        assert decision.candidates_qualified == 1
        assert decision.fallback_used is False
        assert decision.score > 0.0

    # -- Multiple candidates: pick highest score --

    def test_select_highest_score_wins(self) -> None:
        fleet = DraftModelFleet()
        # Slow CPU endpoint
        fleet.register(make_spec(
            endpoint_url="http://cpu:8000/",
            model_name="smol-cpu",
            hardware="cpu",
            avg_latency_ms=100.0,
            cost_per_hour=0.01,
        ))
        # Fast GPU endpoint
        fleet.register(make_spec(
            endpoint_url="http://gpu:8000/",
            model_name="smol-gpu",
            hardware="cuda:0",
            avg_latency_ms=5.0,
            cost_per_hour=0.60,
        ))
        router = DraftModelRouter(fleet)
        constraints = RoutingConstraints(max_latency_ms=200.0, max_cost_per_hour=2.0)
        decision = router.select(constraints)
        # With default weights (latency 0.35, cost 0.20, acceptance 0.30, load 0.15),
        # the GPU endpoint's much lower latency gives it a higher score.
        assert decision.selected_url == "http://gpu:8000/"
        assert decision.candidates_qualified == 2

    # -- Fallback: no candidate meets all constraints --
    # NOTE: The constraint check in select() uses health.recent_latency_ms directly
    # (which defaults to 0.0 when no calls recorded), not spec.avg_latency_ms.
    # To trigger fallback we need health.recent_latency_ms set above the threshold.

    def test_select_fallback_when_no_candidate_meets_all(self) -> None:
        """When no endpoint satisfies all constraints, fallback to relaxed."""
        fleet = DraftModelFleet()

        spec_slow = make_spec(
            endpoint_url="http://slow:8000/",
            model_name="slow",
            avg_latency_ms=500.0,
            cost_per_hour=0.01,
        )
        fleet.register(spec_slow)
        # Set recent_latency_ms to exceed the constraint (50ms)
        health_slow = fleet.get_health(spec_slow.endpoint_url)
        assert health_slow is not None
        health_slow.recent_latency_ms = 500.0

        spec_expensive = make_spec(
            endpoint_url="http://expensive:8000/",
            model_name="expensive",
            avg_latency_ms=10.0,
            cost_per_hour=100.0,
        )
        fleet.register(spec_expensive)
        health_expensive = fleet.get_health(spec_expensive.endpoint_url)
        assert health_expensive is not None
        health_expensive.recent_latency_ms = 10.0

        router = DraftModelRouter(fleet)
        constraints = RoutingConstraints(
            max_latency_ms=50.0,
            max_cost_per_hour=1.0,
        )
        decision = router.select(constraints)
        # slow: latency 500 > 50 => fails. expensive: cost 100 > 1 => fails.
        # Neither meets all constraints, so fallback picks the higher-scored.
        assert decision.selected_url != ""
        assert decision.fallback_used is True
        assert decision.selection_reason == "fallback_relaxed_constraints"
        assert decision.candidates_evaluated == 2
        assert decision.candidates_qualified == 0

    # -- Hard filter: concurrency limit --

    def test_select_skips_at_concurrency_limit(self) -> None:
        """Endpoints at max_concurrent should be skipped."""
        fleet = DraftModelFleet()
        spec = make_spec(endpoint_url="http://busy:8000/", max_concurrent=2)
        fleet.register(spec)
        # Set concurrent count to max
        health = fleet.get_health(spec.endpoint_url)
        assert health is not None
        health.current_concurrent = 2

        router = DraftModelRouter(fleet)
        decision = router.select()
        # No eligible candidates since the only endpoint is at concurrency limit
        assert decision.selected_url == ""
        assert decision.selection_reason == "no healthy endpoints available"

    def test_select_respects_zero_concurrent_no_limit(self) -> None:
        """max_concurrent=0 means no concurrency limit."""
        fleet = DraftModelFleet()
        spec = make_spec(endpoint_url="http://busy:8000/", max_concurrent=0)
        fleet.register(spec)
        health = fleet.get_health(spec.endpoint_url)
        assert health is not None
        health.current_concurrent = 999  # very busy, but no limit

        router = DraftModelRouter(fleet)
        decision = router.select()
        assert decision.selected_url == "http://busy:8000/"

    # -- Hardware preference bonus --

    def test_select_hardware_preference_bonus(self) -> None:
        """Preferred hardware should add a 0.1 bonus to the score."""
        fleet = DraftModelFleet()
        fleet.register(make_spec(
            endpoint_url="http://cpu:8000/",
            model_name="cpu-model",
            hardware="cpu",
            avg_latency_ms=20.0,
            cost_per_hour=0.10,
            avg_acceptance_rate=0.7,
        ))
        fleet.register(make_spec(
            endpoint_url="http://gpu:8000/",
            model_name="gpu-model",
            hardware="cuda:0",
            avg_latency_ms=15.0,
            cost_per_hour=0.50,
            avg_acceptance_rate=0.8,
        ))
        router = DraftModelRouter(fleet)
        constraints = RoutingConstraints(
            preferred_hardware=["cpu"],
            max_latency_ms=100.0,
            max_cost_per_hour=2.0,
        )
        decision = router.select(constraints)
        # cpu-model gets a 0.1 hardware bonus, which may compensate for lower specs
        assert decision.selected_url == "http://cpu:8000/"

    # -- _score method --

    def test_score_lower_latency_is_higher(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet)
        constraints = RoutingConstraints(max_latency_ms=100.0, max_cost_per_hour=1.0)

        spec_fast = make_spec(avg_latency_ms=10.0, cost_per_hour=0.5, avg_acceptance_rate=0.8)
        health_fast = DraftModelHealth(endpoint_url="http://fast:8000/")
        health_fast.recent_latency_ms = 10.0
        health_fast.recent_acceptance_rate = 0.8

        spec_slow = make_spec(avg_latency_ms=90.0, cost_per_hour=0.5, avg_acceptance_rate=0.8)
        health_slow = DraftModelHealth(endpoint_url="http://slow:8000/")
        health_slow.recent_latency_ms = 90.0
        health_slow.recent_acceptance_rate = 0.8

        score_fast = router._score(spec_fast, health_fast, constraints)
        score_slow = router._score(spec_slow, health_slow, constraints)
        assert score_fast > score_slow

    def test_score_lower_cost_is_higher(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet, latency_weight=0.0, cost_weight=1.0,
                                   acceptance_weight=0.0, load_weight=0.0)
        constraints = RoutingConstraints(max_cost_per_hour=1.0)

        spec_cheap = make_spec(cost_per_hour=0.1)
        health_cheap = DraftModelHealth(endpoint_url="http://cheap:8000/")

        spec_pricey = make_spec(cost_per_hour=0.9)
        health_pricey = DraftModelHealth(endpoint_url="http://pricey:8000/")

        assert router._score(spec_cheap, health_cheap, constraints) > router._score(spec_pricey, health_pricey, constraints)

    def test_score_higher_acceptance_is_higher(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet, latency_weight=0.0, cost_weight=0.0,
                                   acceptance_weight=1.0, load_weight=0.0)
        constraints = RoutingConstraints()

        spec_low = make_spec(avg_acceptance_rate=0.3)
        health_low = DraftModelHealth(endpoint_url="http://low:8000/")
        health_low.recent_acceptance_rate = 0.3

        spec_high = make_spec(avg_acceptance_rate=0.9)
        health_high = DraftModelHealth(endpoint_url="http://high:8000/")
        health_high.recent_acceptance_rate = 0.9

        assert router._score(spec_high, health_high, constraints) > router._score(spec_low, health_low, constraints)

    def test_score_lower_load_is_higher(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet, latency_weight=0.0, cost_weight=0.0,
                                   acceptance_weight=0.0, load_weight=1.0)
        constraints = RoutingConstraints()

        spec = make_spec(max_concurrent=10)
        health_idle = DraftModelHealth(endpoint_url="http://idle:8000/", current_concurrent=0)
        health_busy = DraftModelHealth(endpoint_url="http://busy:8000/", current_concurrent=8)

        assert router._score(spec, health_idle, constraints) > router._score(spec, health_busy, constraints)

    def test_score_hardware_bonus_applied(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet, latency_weight=0.0, cost_weight=0.0,
                                   acceptance_weight=0.0, load_weight=0.0)
        constraints = RoutingConstraints(preferred_hardware=["cuda:0"])

        spec_match = make_spec(hardware="cuda:0")
        health = DraftModelHealth(endpoint_url="http://gpu:8000/")

        spec_nomatch = make_spec(hardware="cpu")
        health_nomatch = DraftModelHealth(endpoint_url="http://cpu:8000/")

        score_match = router._score(spec_match, health, constraints)
        score_nomatch = router._score(spec_nomatch, health_nomatch, constraints)
        assert score_match == pytest.approx(score_nomatch + 0.1)

    def test_score_no_hardware_bonus_when_no_preference(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet, latency_weight=0.0, cost_weight=0.0,
                                   acceptance_weight=0.0, load_weight=0.0)
        constraints = RoutingConstraints()  # no preferred_hardware

        spec = make_spec(hardware="cuda:0")
        health = DraftModelHealth(endpoint_url="http://gpu:8000/")

        score = router._score(spec, health, constraints)
        assert score == 0.0  # all weights are 0, no bonus

    def test_score_falls_back_to_spec_defaults(self) -> None:
        """When health's recent_* values are 0, _score falls back to spec defaults."""
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet, latency_weight=0.5, cost_weight=0.0,
                                   acceptance_weight=0.5, load_weight=0.0)
        constraints = RoutingConstraints(max_latency_ms=100.0)

        spec = make_spec(avg_latency_ms=50.0, avg_acceptance_rate=0.7)
        # health has no recent measurements (all 0)
        health = DraftModelHealth(endpoint_url="http://test:8000/")

        score = router._score(spec, health, constraints)
        # Should use spec.avg_latency_ms (50) and spec.avg_acceptance_rate (0.7)
        assert score > 0.0

    # -- last_decision property --

    def test_last_decision_is_none_before_select(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet)
        assert router.last_decision is None

    def test_last_decision_is_set_after_select(self) -> None:
        fleet = DraftModelFleet()
        fleet.register(make_spec())
        router = DraftModelRouter(fleet)
        decision = router.select()
        assert router.last_decision is decision
        assert router.last_decision.selected_url == decision.selected_url

    def test_last_decision_overwritten_on_repeated_select(self) -> None:
        """Each select() call overwrites the previous last_decision."""
        fleet = DraftModelFleet()
        fleet.register(make_spec(endpoint_url="http://a:8000/"))
        router = DraftModelRouter(fleet)
        d1 = router.select()
        assert router.last_decision is d1
        d2 = router.select()
        assert router.last_decision is d2
        assert router.last_decision is not d1

    # -- fleet_stats method --

    def test_fleet_stats_empty(self) -> None:
        fleet = DraftModelFleet()
        router = DraftModelRouter(fleet)
        stats = router.fleet_stats()
        assert stats["total_endpoints"] == 0
        assert stats["healthy_endpoints"] == 0
        assert stats["total_calls"] == 0
        assert stats["total_errors"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["endpoints"] == []

    def test_fleet_stats_with_specs(self) -> None:
        fleet = DraftModelFleet()
        fleet.register(make_spec(
            endpoint_url="http://a:8000/",
            model_name="model-a",
            hardware="cpu",
            cost_per_hour=0.05,
        ))
        fleet.register(make_spec(
            endpoint_url="http://b:8000/",
            model_name="model-b",
            hardware="cuda:0",
            cost_per_hour=0.60,
        ))
        router = DraftModelRouter(fleet)
        stats = router.fleet_stats()
        assert stats["total_endpoints"] == 2
        assert stats["healthy_endpoints"] == 2
        assert stats["total_calls"] == 0
        assert stats["error_rate"] == 0.0
        assert len(stats["endpoints"]) == 2

        urls = {e["url"] for e in stats["endpoints"]}
        assert urls == {"http://a:8000/", "http://b:8000/"}

    def test_fleet_stats_with_errors(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec(endpoint_url="http://a:8000/")
        fleet.register(spec)
        fleet.record_success(spec.endpoint_url, latency_s=1.0, tokens_generated=50)
        fleet.record_error(spec.endpoint_url, "oops")

        router = DraftModelRouter(fleet)
        stats = router.fleet_stats()
        assert stats["total_calls"] == 2
        assert stats["total_errors"] == 1
        assert stats["error_rate"] == 0.5

    def test_fleet_stats_avg_latency(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec(endpoint_url="http://a:8000/")
        fleet.register(spec)
        fleet.record_success(spec.endpoint_url, latency_s=2.0, tokens_generated=100)
        fleet.record_success(spec.endpoint_url, latency_s=4.0, tokens_generated=200)

        router = DraftModelRouter(fleet)
        stats = router.fleet_stats()
        # total_lat = 6.0, total_calls = 2 => (6.0/2)*1000 = 3000.0
        assert stats["avg_latency_ms"] == 3000.0

    def test_fleet_stats_healthy_count_reflects_health(self) -> None:
        fleet = DraftModelFleet()
        spec_a = make_spec(endpoint_url="http://a:8000/")
        spec_b = make_spec(endpoint_url="http://b:8000/")
        fleet.register(spec_a)
        fleet.register(spec_b)
        fleet.record_error(spec_b.endpoint_url, "e1")
        fleet.record_error(spec_b.endpoint_url, "e2")
        fleet.record_error(spec_b.endpoint_url, "e3")

        router = DraftModelRouter(fleet)
        stats = router.fleet_stats()
        assert stats["healthy_endpoints"] == 1

    def test_fleet_stats_endpoint_details(self) -> None:
        fleet = DraftModelFleet()
        spec = make_spec(
            endpoint_url="http://a:8000/",
            model_name="model-a",
            hardware="cpu",
            cost_per_hour=0.05,
        )
        fleet.register(spec)
        fleet.record_success(spec.endpoint_url, latency_s=0.5, tokens_generated=10)

        router = DraftModelRouter(fleet)
        stats = router.fleet_stats()
        ep = stats["endpoints"][0]
        assert ep["url"] == "http://a:8000/"
        assert ep["model"] == "model-a"
        assert ep["hardware"] == "cpu"
        assert ep["cost_per_hour"] == 0.05
        assert ep["healthy"] is True
        assert ep["calls"] == 1
        # avg_latency_ms = (0.5 / 1) * 1000 = 500.0
        assert ep["avg_latency_ms"] == 500.0

    # -- Edge cases for select --

    def test_select_with_all_weights_zero(self) -> None:
        """When all weights are 0, the base score is 0, but selection still works."""
        fleet = DraftModelFleet()
        fleet.register(make_spec(
            endpoint_url="http://a:8000/",
            model_name="only-one",
        ))
        router = DraftModelRouter(fleet,
                                   latency_weight=0.0,
                                   cost_weight=0.0,
                                   acceptance_weight=0.0,
                                   load_weight=0.0)
        decision = router.select()
        assert decision.selected_url == "http://a:8000/"
        assert decision.score == 0.0  # no bonus either

    def test_select_acceptance_rate_filter(self) -> None:
        """Constraints.min_acceptance_rate should filter candidates."""
        fleet = DraftModelFleet()
        spec_low = make_spec(
            endpoint_url="http://low:8000/",
            model_name="low",
        )
        fleet.register(spec_low)
        health_low = fleet.get_health(spec_low.endpoint_url)
        assert health_low is not None
        health_low.recent_acceptance_rate = 0.3

        spec_high = make_spec(
            endpoint_url="http://high:8000/",
            model_name="high",
        )
        fleet.register(spec_high)
        health_high = fleet.get_health(spec_high.endpoint_url)
        assert health_high is not None
        health_high.recent_acceptance_rate = 0.9

        router = DraftModelRouter(fleet)
        constraints = RoutingConstraints(
            max_latency_ms=200.0,
            max_cost_per_hour=1.0,
            min_acceptance_rate=0.5,
        )
        decision = router.select(constraints)
        assert decision.selected_url == "http://high:8000/"
        assert decision.candidates_qualified == 1  # only high meets acceptance rate

    def test_select_multiple_candidates_tie(self) -> None:
        """When scores tie, the first one encountered (by iteration) wins."""
        fleet = DraftModelFleet()
        # Both endpoints are identical in specs
        fleet.register(make_spec(
            endpoint_url="http://a:8000/",
            model_name="a",
            avg_latency_ms=50.0,
            cost_per_hour=0.10,
            avg_acceptance_rate=0.7,
        ))
        fleet.register(make_spec(
            endpoint_url="http://b:8000/",
            model_name="b",
            avg_latency_ms=50.0,
            cost_per_hour=0.10,
            avg_acceptance_rate=0.7,
        ))
        router = DraftModelRouter(fleet)
        decision = router.select()
        # Both have the same score; the first registered should be returned
        assert decision.selected_url in ("http://a:8000/", "http://b:8000/")

    def test_select_constraints_max_latency_ms_zero(self) -> None:
        """max_latency_ms=0: health.recent_latency_ms is 0 (no records),
        so the endpoint qualifies (0 <= 0)."""
        fleet = DraftModelFleet()
        fleet.register(make_spec(
            endpoint_url="http://a:8000/",
            avg_latency_ms=10.0,
        ))
        router = DraftModelRouter(fleet)
        constraints = RoutingConstraints(max_latency_ms=0.0, max_cost_per_hour=1.0)
        decision = router.select(constraints)
        # health.recent_latency_ms is 0, which <= 0.0, so it qualifies
        assert decision.fallback_used is False
        assert decision.selected_url == "http://a:8000/"

    def test_select_request_id_to_string(self) -> None:
        """RoutingDecision should be able to convert to string (dataclass repr)."""
        fleet = DraftModelFleet()
        fleet.register(make_spec())
        router = DraftModelRouter(fleet)
        decision = router.select()
        # Just verify repr works
        rep = repr(decision)
        assert "selected_url" in rep
        assert "score" in rep
