"""Tests for DraftQualityScorer -- draft model quality scoring and auto-selection.

Covers:
    DraftStats           -- dataclass fields and defaults
    DraftQualityScorer   -- construction, record, get_acceptance_rate,
                            select_best_draft, get_all_stats, get_leaderboard

All tests are deterministic (no network, no GPU, no time.sleep).
No MagicMock -- real objects or lightweight stubs only.
"""

from __future__ import annotations

import math

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

# Bootstrap fake packages for distllm namespace
bootstrap_fake_packages()

# Load the module under test
_mod = load_module("distllm/core/draft_quality_scorer.py")

# Re-export symbols for test readability
DraftStats = _mod.DraftStats
DraftQualityScorer = _mod.DraftQualityScorer


# ===================================================================
# DraftStats unit tests
# ===================================================================


class TestDraftStats:
    """DraftStats dataclass fields and defaults."""

    def test_default_values(self) -> None:
        """All numeric counters start at zero, rate at 0.0, last_updated set."""
        stats = DraftStats(draft_model="draft-a", target_model="target-x")
        assert stats.draft_model == "draft-a"
        assert stats.target_model == "target-x"
        assert stats.total_accepted == 0
        assert stats.total_proposed == 0
        assert stats.total_calls == 0
        assert stats.avg_acceptance_rate == 0.0
        assert stats.last_updated > 0  # time.time() produced a real value

    def test_explicit_values(self) -> None:
        """All fields can be set explicitly via constructor."""
        stats = DraftStats(
            draft_model="d1",
            target_model="t1",
            total_accepted=10,
            total_proposed=20,
            total_calls=2,
            avg_acceptance_rate=0.5,
            last_updated=12345.0,
        )
        assert stats.draft_model == "d1"
        assert stats.target_model == "t1"
        assert stats.total_accepted == 10
        assert stats.total_proposed == 20
        assert stats.total_calls == 2
        assert stats.avg_acceptance_rate == 0.5
        assert stats.last_updated == 12345.0


# ===================================================================
# DraftQualityScorer construction
# ===================================================================


class TestDraftQualityScorerConstruction:
    """Construction and default parameter values."""

    def test_default_decay_factor(self) -> None:
        """Default decay factor is 0.95."""
        scorer = DraftQualityScorer()
        assert scorer._decay_factor == 0.95

    def test_custom_decay_factor(self) -> None:
        """Custom decay factor is accepted and stored."""
        scorer = DraftQualityScorer(decay_factor=0.8)
        assert scorer._decay_factor == 0.8

    def test_decay_factor_zero(self) -> None:
        """Zero decay factor (no-memory EMA) is accepted."""
        scorer = DraftQualityScorer(decay_factor=0.0)
        assert scorer._decay_factor == 0.0

    def test_decay_factor_one(self) -> None:
        """Decay factor of 1.0 (no update) is accepted."""
        scorer = DraftQualityScorer(decay_factor=1.0)
        assert scorer._decay_factor == 1.0

    def test_internal_stats_empty(self) -> None:
        """Internal stats dict starts empty."""
        scorer = DraftQualityScorer()
        assert scorer._stats == {}


# ===================================================================
# DraftQualityScorer.record
# ===================================================================


class TestDraftQualityScorerRecord:
    """Recording speculative decoding results."""

    def test_first_record_creates_entry(self) -> None:
        """First call to record creates a new entry in _stats."""
        scorer = DraftQualityScorer()
        scorer.record("draft-a", "target-x", accepted=4, total=8)

        key = ("draft-a", "target-x")
        assert key in scorer._stats
        stats = scorer._stats[key]
        assert stats.draft_model == "draft-a"
        assert stats.target_model == "target-x"
        assert stats.total_accepted == 4
        assert stats.total_proposed == 8
        assert stats.total_calls == 1
        # First record: rate = accepted / total = 0.5
        assert stats.avg_acceptance_rate == 0.5

    def test_second_record_updates_entry(self) -> None:
        """Second call updates existing entry incrementally and applies EMA."""
        scorer = DraftQualityScorer(decay_factor=0.9)
        scorer.record("draft-a", "target-x", accepted=4, total=8)  # rate = 0.5
        scorer.record("draft-a", "target-x", accepted=6, total=8)  # rate = 0.75

        key = ("draft-a", "target-x")
        stats = scorer._stats[key]
        assert stats.total_accepted == 10
        assert stats.total_proposed == 16
        assert stats.total_calls == 2
        # EMA: 0.9 * 0.5 + 0.1 * 0.75 = 0.45 + 0.075 = 0.525
        expected = 0.9 * 0.5 + 0.1 * 0.75
        assert math.isclose(stats.avg_acceptance_rate, expected)

    def test_three_records_apply_ema_correctly(self) -> None:
        """EMA is correctly applied over multiple updates."""
        scorer = DraftQualityScorer(decay_factor=0.8)
        # rate_1 = 4/8 = 0.5
        scorer.record("d", "t", accepted=4, total=8)
        # rate_2 = 8/8 = 1.0
        scorer.record("d", "t", accepted=8, total=8)
        # rate_3 = 2/8 = 0.25
        scorer.record("d", "t", accepted=2, total=8)

        stats = scorer._stats[("d", "t")]
        assert stats.total_calls == 3
        assert stats.total_accepted == 14
        assert stats.total_proposed == 24
        # EMA = 0.8*(0.8*0.5 + 0.2*1.0) + 0.2*0.25
        #     = 0.8*(0.4 + 0.2) + 0.05
        #     = 0.8*0.6 + 0.05
        #     = 0.48 + 0.05
        #     = 0.53
        expected = 0.8 * (0.8 * 0.5 + 0.2 * 1.0) + 0.2 * 0.25
        assert math.isclose(stats.avg_acceptance_rate, expected)

    def test_different_draft_target_pairs_independent(self) -> None:
        """Records for different (draft, target) pairs do not interfere."""
        scorer = DraftQualityScorer()
        scorer.record("draft-a", "target-x", accepted=4, total=8)
        scorer.record("draft-b", "target-x", accepted=6, total=8)
        scorer.record("draft-a", "target-y", accepted=7, total=8)

        assert len(scorer._stats) == 3
        assert scorer._stats[("draft-a", "target-x")].avg_acceptance_rate == 0.5
        assert scorer._stats[("draft-b", "target-x")].avg_acceptance_rate == 0.75
        assert scorer._stats[("draft-a", "target-y")].avg_acceptance_rate == 0.875

    def test_all_accepted(self) -> None:
        """Perfect acceptance rate (all tokens accepted)."""
        scorer = DraftQualityScorer()
        scorer.record("d", "t", accepted=8, total=8)
        assert scorer._stats[("d", "t")].avg_acceptance_rate == 1.0

    def test_none_accepted(self) -> None:
        """Zero acceptance rate (no tokens accepted)."""
        scorer = DraftQualityScorer()
        scorer.record("d", "t", accepted=0, total=8)
        assert scorer._stats[("d", "t")].avg_acceptance_rate == 0.0

    def test_all_accepted_zero_total(self) -> None:
        """Edge case: zero total tokens (prevents division by zero)."""
        scorer = DraftQualityScorer()
        # accepted=0, total=0 -> rate = 0 / max(0, 1) = 0
        scorer.record("d", "t", accepted=0, total=0)
        assert scorer._stats[("d", "t")].avg_acceptance_rate == 0.0

    def test_partial_accepted_zero_total(self) -> None:
        """Edge case: partial accepted with zero total (clamped to 1)."""
        scorer = DraftQualityScorer()
        # accepted=3, total=0 -> rate = 3 / max(0, 1) = 3.0 (unusual but handled)
        scorer.record("d", "t", accepted=3, total=0)
        assert scorer._stats[("d", "t")].avg_acceptance_rate == 3.0


# ===================================================================
# DraftQualityScorer.get_acceptance_rate
# ===================================================================


class TestGetAcceptanceRate:
    """Querying acceptance rates."""

    def test_returns_rate_for_recorded_pair(self) -> None:
        """Returns the avg_acceptance_rate for a recorded pair."""
        scorer = DraftQualityScorer()
        scorer.record("d", "t", accepted=4, total=8)
        rate = scorer.get_acceptance_rate("d", "t")
        assert rate == 0.5

    def test_returns_none_for_unrecorded_pair(self) -> None:
        """Returns None when no data exists for the pair."""
        scorer = DraftQualityScorer()
        rate = scorer.get_acceptance_rate("d", "t")
        assert rate is None

    def test_returns_none_for_partially_recorded_pair(self) -> None:
        """Returns None when draft is recorded but target differs."""
        scorer = DraftQualityScorer()
        scorer.record("d", "target-x", accepted=4, total=8)
        assert scorer.get_acceptance_rate("d", "other-target") is None

    def test_returns_rate_after_multiple_records(self) -> None:
        """Returns the EMA rate after multiple records."""
        scorer = DraftQualityScorer()
        scorer.record("d", "t", accepted=4, total=8)
        scorer.record("d", "t", accepted=8, total=8)
        expected = 0.95 * 0.5 + 0.05 * 1.0  # = 0.525
        assert math.isclose(
            scorer.get_acceptance_rate("d", "t"), expected
        )


# ===================================================================
# DraftQualityScorer.select_best_draft
# ===================================================================


class TestSelectBestDraft:
    """Selecting the best draft model for a given target."""

    def test_selects_highest_rate(self) -> None:
        """Selects the draft with the highest acceptance rate."""
        scorer = DraftQualityScorer()
        # Give each draft enough calls to meet default min_calls=3
        for _ in range(3):
            scorer.record("draft-a", "target-x", accepted=4, total=8)  # 0.5
            scorer.record("draft-b", "target-x", accepted=7, total=8)  # 0.875

        best = scorer.select_best_draft("target-x", ["draft-a", "draft-b"])
        assert best == "draft-b"

    def test_returns_none_when_no_data(self) -> None:
        """Returns None when no drafts have been recorded for the target."""
        scorer = DraftQualityScorer()
        best = scorer.select_best_draft("target-x", ["draft-a", "draft-b"])
        assert best is None

    def test_returns_none_when_empty_available_list(self) -> None:
        """Returns None when the available drafts list is empty."""
        scorer = DraftQualityScorer()
        scorer.record("draft-a", "target-x", accepted=4, total=8)
        best = scorer.select_best_draft("target-x", [])
        assert best is None

    def test_min_calls_filtering(self) -> None:
        """Ignors drafts below the minimum call threshold."""
        scorer = DraftQualityScorer()
        scorer.record("draft-a", "target-x", accepted=4, total=8)   # 1 call, rate=0.5
        scorer.record("draft-a", "target-x", accepted=8, total=8)   # 2 calls
        scorer.record("draft-b", "target-x", accepted=7, total=8)   # 1 call, rate=0.875
        scorer.record("draft-b", "target-x", accepted=6, total=8)   # 2 calls
        scorer.record("draft-b", "target-x", accepted=5, total=8)   # 3 calls
        scorer.record("draft-b", "target-x", accepted=8, total=8)   # 4 calls

        # min_calls=3: draft-a has 2 (< 3), draft-b has 4 (>= 3)
        best = scorer.select_best_draft(
            "target-x", ["draft-a", "draft-b"], min_calls=3
        )
        assert best == "draft-b"

    def test_min_calls_one_accepts_single_record(self) -> None:
        """With min_calls=1, a single record is enough to be considered."""
        scorer = DraftQualityScorer()
        scorer.record("draft-a", "target-x", accepted=7, total=8)
        best = scorer.select_best_draft("target-x", ["draft-a"], min_calls=1)
        assert best == "draft-a"

    def test_skips_drafts_with_insufficient_calls(self) -> None:
        """When all drafts have fewer calls than min_calls, returns None."""
        scorer = DraftQualityScorer()
        scorer.record("draft-a", "target-x", accepted=4, total=8)  # 1 call
        best = scorer.select_best_draft(
            "target-x", ["draft-a", "draft-b"], min_calls=5
        )
        assert best is None

    def test_ignores_drafts_for_other_targets(self) -> None:
        """Drafts recorded for other targets are not considered."""
        scorer = DraftQualityScorer()
        # Give each draft enough calls to meet min_calls=3
        for _ in range(3):
            scorer.record("draft-a", "target-x", accepted=7, total=8)
            scorer.record("draft-a", "target-y", accepted=1, total=8)
            scorer.record("draft-b", "target-x", accepted=4, total=8)

        best = scorer.select_best_draft("target-x", ["draft-a", "draft-b"])
        assert best == "draft-a"

    def test_tie_breaker_first_in_list(self) -> None:
        """When two drafts have the same rate, the first listed wins."""
        scorer = DraftQualityScorer()
        # Both have identical rates
        scorer.record("draft-a", "target-x", accepted=4, total=8)
        scorer.record("draft-a", "target-x", accepted=4, total=8)
        scorer.record("draft-a", "target-x", accepted=4, total=8)
        scorer.record("draft-b", "target-x", accepted=4, total=8)
        scorer.record("draft-b", "target-x", accepted=4, total=8)
        scorer.record("draft-b", "target-x", accepted=4, total=8)

        best = scorer.select_best_draft("target-x", ["draft-a", "draft-b"])
        assert best == "draft-a"


# ===================================================================
# DraftQualityScorer.get_all_stats
# ===================================================================


class TestGetAllStats:
    """Retrieving all draft model statistics."""

    def test_returns_empty_list_when_no_stats(self) -> None:
        """Returns an empty list when no records exist."""
        scorer = DraftQualityScorer()
        assert scorer.get_all_stats() == []

    def test_returns_one_entry_after_single_record(self) -> None:
        """Returns a list with one dict entry after a single record."""
        scorer = DraftQualityScorer()
        scorer.record("draft-a", "target-x", accepted=4, total=8)

        stats_list = scorer.get_all_stats()
        assert len(stats_list) == 1
        entry = stats_list[0]
        assert entry["draft_model"] == "draft-a"
        assert entry["target_model"] == "target-x"
        assert entry["acceptance_rate"] == 0.5
        assert entry["total_accepted"] == 4
        assert entry["total_proposed"] == 8
        assert entry["total_calls"] == 1

    def test_returns_multiple_entries(self) -> None:
        """Returns entries for all recorded pairs."""
        scorer = DraftQualityScorer()
        scorer.record("d1", "t1", accepted=4, total=8)  # rate 0.5
        scorer.record("d2", "t1", accepted=6, total=8)  # rate 0.75

        stats_list = scorer.get_all_stats()
        assert len(stats_list) == 2

        # Build a lookup dict for easier assertion
        by_draft = {e["draft_model"]: e for e in stats_list}
        assert by_draft["d1"]["acceptance_rate"] == 0.5
        assert by_draft["d2"]["acceptance_rate"] == 0.75

    def test_acceptance_rate_is_rounded_to_three_decimals(self) -> None:
        """Acceptance rate is rounded to 3 decimal places."""
        scorer = DraftQualityScorer()
        # Use a rate that produces many decimal places
        scorer.record("d", "t", accepted=1, total=3)  # 0.333...

        stats_list = scorer.get_all_stats()
        assert len(stats_list) == 1
        entry = stats_list[0]
        # 1/3 = 0.333333... -> rounded to 3 decimal places -> 0.333
        assert entry["acceptance_rate"] == round(1.0 / 3.0, 3)


# ===================================================================
# DraftQualityScorer.get_leaderboard
# ===================================================================


class TestGetLeaderboard:
    """Retrieving ranked leaderboard for a target model."""

    def test_returns_empty_list_when_no_stats_for_target(self) -> None:
        """Returns empty list when no records exist for the target."""
        scorer = DraftQualityScorer()
        leaderboard = scorer.get_leaderboard("target-x")
        assert leaderboard == []

    def test_returns_only_entries_for_requested_target(self) -> None:
        """Filters out entries belonging to other targets."""
        scorer = DraftQualityScorer()
        scorer.record("d1", "target-x", accepted=4, total=8)
        scorer.record("d2", "target-x", accepted=7, total=8)
        scorer.record("d3", "target-y", accepted=8, total=8)  # filtered out

        leaderboard = scorer.get_leaderboard("target-x")
        assert len(leaderboard) == 2
        for entry in leaderboard:
            assert entry["draft_model"] in ("d1", "d2")

    def test_returns_sorted_by_rate_descending(self) -> None:
        """Entries are sorted by acceptance_rate descending."""
        scorer = DraftQualityScorer()
        scorer.record("low", "t", accepted=2, total=8)     # 0.25
        scorer.record("high", "t", accepted=7, total=8)    # 0.875
        scorer.record("medium", "t", accepted=4, total=8)  # 0.5

        leaderboard = scorer.get_leaderboard("t")
        rates = [e["acceptance_rate"] for e in leaderboard]
        assert rates == sorted(rates, reverse=True)
        assert leaderboard[0]["draft_model"] == "high"
        assert leaderboard[1]["draft_model"] == "medium"
        assert leaderboard[2]["draft_model"] == "low"

    def test_entry_includes_rate_and_calls(self) -> None:
        """Each entry has draft_model, acceptance_rate, and total_calls."""
        scorer = DraftQualityScorer()
        scorer.record("d1", "t", accepted=4, total=8)
        scorer.record("d1", "t", accepted=8, total=8)  # 2 calls

        leaderboard = scorer.get_leaderboard("t")
        assert len(leaderboard) == 1
        entry = leaderboard[0]
        assert set(entry.keys()) == {"draft_model", "acceptance_rate", "total_calls"}
        assert entry["draft_model"] == "d1"
        assert entry["total_calls"] == 2

    def test_empty_leaderboard_when_no_records_at_all(self) -> None:
        """Empty leaderboard when scorer has never recorded anything."""
        scorer = DraftQualityScorer()
        assert scorer.get_leaderboard("anything") == []


# ===================================================================
# Integration-style tests
# ===================================================================


class TestDraftQualityScorerIntegration:
    """Multiple steps combined to exercise the full API surface."""

    def test_full_workflow(self) -> None:
        """End-to-end: record, query, select, leaderboard."""
        scorer = DraftQualityScorer(decay_factor=0.95)

        # Record several outcomes for 2 drafts targeting the same model
        scorer.record("draft-7b", "target-70b", accepted=4, total=8)
        scorer.record("draft-7b", "target-70b", accepted=6, total=8)
        scorer.record("draft-7b", "target-70b", accepted=7, total=8)
        scorer.record("draft-1b", "target-70b", accepted=1, total=8)
        scorer.record("draft-1b", "target-70b", accepted=2, total=8)

        # Also record against another target (should be isolated)
        scorer.record("draft-7b", "target-7b", accepted=8, total=8)

        # Query individual rates
        rate_7b = scorer.get_acceptance_rate("draft-7b", "target-70b")
        rate_1b = scorer.get_acceptance_rate("draft-1b", "target-70b")
        rate_none = scorer.get_acceptance_rate("draft-x", "target-70b")

        assert rate_7b is not None and rate_7b > 0
        assert rate_1b is not None and rate_1b >= 0
        assert rate_none is None
        # draft-7b should have higher rate than draft-1b
        assert rate_7b > rate_1b

        # Select best
        best = scorer.select_best_draft("target-70b", ["draft-7b", "draft-1b"])
        assert best == "draft-7b"

        # Verify isolation: for target-7b, draft-7b is the only candidate
        # Give it 3 so it meets default min_calls=3
        scorer.record("draft-7b", "target-7b", accepted=8, total=8)
        scorer.record("draft-7b", "target-7b", accepted=8, total=8)
        best_7b = scorer.select_best_draft("target-7b", ["draft-7b", "draft-1b"])
        assert best_7b == "draft-7b"

        # Leaderboard
        board = scorer.get_leaderboard("target-70b")
        assert len(board) == 2
        assert board[0]["draft_model"] == "draft-7b"  # highest rate first

        # All stats
        all_stats = scorer.get_all_stats()
        assert len(all_stats) == 3  # 3 unique (draft, target) pairs

    def test_thread_safety(self) -> None:
        """Basic verification that the class has a lock attribute.

        We cannot easily test concurrent access deterministically in-process,
        but we verify the infrastructure is present.
        """
        scorer = DraftQualityScorer()
        assert hasattr(scorer, "_lock")
        # The lock was acquired and released during the record call
        scorer.record("d", "t", accepted=4, total=8)
        rate = scorer.get_acceptance_rate("d", "t")
        assert rate == 0.5
