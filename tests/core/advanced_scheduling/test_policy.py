"""Tests for SchedulingPolicy, DefaultPolicy, SarathiPolicy, CompositePolicy."""

from __future__ import annotations

from typing import Protocol

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_policy = load_module("distllm/core/advanced_scheduling/policy.py")
SchedulingPolicy = _policy.SchedulingPolicy
DefaultPolicy = _policy.DefaultPolicy
SarathiPolicy = _policy.SarathiPolicy
CompositePolicy = _policy.CompositePolicy


class TestSchedulingPolicyProtocol:
    """Test suite for SchedulingPolicy protocol."""

    def test_is_runtime_checkable(self) -> None:
        """SchedulingPolicy should be a runtime-checkable protocol."""
        assert isinstance(DefaultPolicy(), SchedulingPolicy)
        assert isinstance(SarathiPolicy(), SchedulingPolicy)
        assert isinstance(CompositePolicy(), SchedulingPolicy)

    def test_custom_implementation(self) -> None:
        """A class implementing the protocol methods should be recognized."""

        class MyPolicy:
            def compute_budget(self, base_budget: object) -> object:
                return base_budget

            def on_before_schedule(self, sequences: list) -> list:
                return sequences

        assert isinstance(MyPolicy(), SchedulingPolicy)

    def test_missing_method_not_instance(self) -> None:
        """A class missing required methods is NOT an instance."""

        class IncompletePolicy:
            pass

        assert not isinstance(IncompletePolicy(), SchedulingPolicy)


class TestDefaultPolicy:
    """Test suite for DefaultPolicy."""

    def test_compute_budget_passthrough(self) -> None:
        policy = DefaultPolicy()
        budget = object()
        assert policy.compute_budget(budget) is budget

    def test_on_before_schedule_passthrough(self) -> None:
        policy = DefaultPolicy()
        seqs = ["a", "b"]
        assert policy.on_before_schedule(seqs) is seqs


class TestSarathiPolicy:
    """Test suite for SarathiPolicy."""

    def test_default_construction(self) -> None:
        policy = SarathiPolicy()
        assert policy.pressure_threshold == 0.8
        assert policy.prefill_scale_under_pressure == 0.5

    def test_custom_values(self) -> None:
        policy = SarathiPolicy(pressure_threshold=0.9, prefill_scale_under_pressure=0.3)
        assert policy.pressure_threshold == 0.9
        assert policy.prefill_scale_under_pressure == 0.3

    def test_compute_budget_passthrough(self) -> None:
        policy = SarathiPolicy()
        budget = object()
        assert policy.compute_budget(budget) is budget

    def test_on_before_schedule_passthrough(self) -> None:
        policy = SarathiPolicy()
        seqs = ["a", "b"]
        assert policy.on_before_schedule(seqs) is seqs

    def test_should_disable_pressure_adaptation_default(self) -> None:
        policy = SarathiPolicy()
        assert policy.should_disable_pressure_adaptation() is False


class TestCompositePolicy:
    """Test suite for CompositePolicy."""

    def test_default_construction(self) -> None:
        policy = CompositePolicy()
        assert policy.policies == []

    def test_init_with_policies(self) -> None:
        inner = DefaultPolicy()
        policy = CompositePolicy(policies=[inner])
        assert policy.policies == [inner]

    def test_compute_budget_applies_all(self) -> None:
        budget = {"calls": 0}

        class IncrementPolicy:
            def compute_budget(self, b: object) -> object:
                b["calls"] += 1
                return b

            def on_before_schedule(self, s: list) -> list:
                return s

        policy = CompositePolicy(policies=[IncrementPolicy(), IncrementPolicy()])
        result = policy.compute_budget(budget)
        assert result["calls"] == 2

    def test_on_before_schedule_applies_all(self) -> None:
        seqs = [1, 2]

        class AppendPolicy:
            def compute_budget(self, b: object) -> object:
                return b

            def on_before_schedule(self, s: list) -> list:
                s.append(0)
                return s

        policy = CompositePolicy(policies=[AppendPolicy(), AppendPolicy()])
        result = policy.on_before_schedule(seqs)
        assert len(result) == 4

    def test_empty_policies_passthrough(self) -> None:
        policy = CompositePolicy()
        budget = object()
        assert policy.compute_budget(budget) is budget
        seqs = ["a"]
        assert policy.on_before_schedule(seqs) is seqs
