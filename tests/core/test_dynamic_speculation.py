"""Tests for the dynamic speculation length controller."""

from __future__ import annotations

from distllm.core.dynamic_speculation import DynamicSpeculationController


class TestDynamicSpeculationController:
    def test_init_defaults(self):
        ctrl = DynamicSpeculationController()
        assert ctrl.current == 5
        assert ctrl.acceptance_rate == 0.0

    def test_high_acceptance_increases_candidates(self):
        """Easy input (all tokens accepted) should max out candidates."""
        ctrl = DynamicSpeculationController(
            initial_candidates=3, min_candidates=1, max_candidates=10,
        )
        for _ in range(20):
            ctrl.update(5, 5)  # 100% acceptance
        assert ctrl.current >= 8  # Should be near max

    def test_low_acceptance_decreases_candidates(self):
        """Hard input (no tokens accepted) should min out candidates."""
        ctrl = DynamicSpeculationController(
            initial_candidates=5, min_candidates=1, max_candidates=10,
        )
        for _ in range(20):
            ctrl.update(0, 5)  # 0% acceptance
        assert ctrl.current == 1

    def test_stable_at_target_rate(self):
        """Matching the target rate should keep candidates stable."""
        ctrl = DynamicSpeculationController(
            initial_candidates=5, min_candidates=1, max_candidates=10,
            target_acceptance_rate=0.7,
        )
        for _ in range(30):
            ctrl.update(7, 10)  # 70% acceptance
        # Should stay within 1 of initial
        assert 4 <= ctrl.current <= 6, f"{ctrl.current} != ~5"

    def test_sliding_window_smooths_oscillations(self):
        """Alternating high/low acceptance should produce stable average."""
        ctrl = DynamicSpeculationController(
            initial_candidates=5, min_candidates=1, max_candidates=10,
            window_size=10,
        )
        for i in range(20):
            if i % 2 == 0:
                ctrl.update(5, 5)  # 100%
            else:
                ctrl.update(0, 5)  # 0%
        # Average ~50% — below 70% target, so candidates should decrease
        assert ctrl.current < 5  # Should have decreased from initial

    def test_adaptation_delay_respected(self):
        """No adaptation during the delay period."""
        ctrl = DynamicSpeculationController(
            initial_candidates=5, adaptation_delay=10,
        )
        for _ in range(9):
            ctrl.update(5, 5)
        assert ctrl.current == 5  # Still initial

    def test_adaptation_after_delay(self):
        """Adaptation begins after the delay period."""
        ctrl = DynamicSpeculationController(
            initial_candidates=5, adaptation_delay=5,
        )
        for _ in range(10):
            ctrl.update(5, 5)
        assert ctrl.current > 5  # Should have increased

    def test_reset_clears_window(self):
        ctrl = DynamicSpeculationController()
        ctrl.update(5, 5)
        ctrl.update(4, 5)
        assert ctrl.acceptance_rate > 0
        ctrl.reset(candidates=3)
        assert ctrl.acceptance_rate == 0.0
        assert ctrl.current == 3

    def test_partial_acceptance_moves_toward_target(self):
        ctrl = DynamicSpeculationController(
            initial_candidates=3, min_candidates=1, max_candidates=10,
            target_acceptance_rate=0.5,
        )
        for _ in range(15):
            ctrl.update(2, 4)  # 50% — exactly target
        assert 2 <= ctrl.current <= 4  # Stable near initial

    def test_edge_case_all_rejected(self):
        ctrl = DynamicSpeculationController(
            initial_candidates=10, min_candidates=1, max_candidates=10,
        )
        for _ in range(25):
            ctrl.update(0, 10)
        assert ctrl.current == 1

    def test_edge_case_all_accepted(self):
        ctrl = DynamicSpeculationController(
            initial_candidates=1, min_candidates=1, max_candidates=10,
        )
        for _ in range(20):
            ctrl.update(5, 5)
        assert ctrl.current == 10  # Should max out
