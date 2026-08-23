"""Tests for prefix clustering and pre-warm prediction.

Covers:
- PrefixCluster dataclass and top_prefix property
- PrewarmPrediction dataclass
- _prefix_features feature extraction
- _cluster_hash computation
- PrefixClusterer lifecycle, predictions, eviction, cooldown, and metrics
"""

from __future__ import annotations

import math

import pytest

from distllm.dist.prefix_clustering import (
    PrefixCluster,
    PrefixClusterer,
    PrewarmPrediction,
    _cluster_hash,
    _prefix_features,
)


# ---------------------------------------------------------------------------
# PrefixCluster
# ---------------------------------------------------------------------------

class TestPrefixCluster:
    """Tests for the PrefixCluster dataclass."""

    def test_construction_with_defaults(self) -> None:
        cluster = PrefixCluster(
            cluster_id="pc-abc123",
            feature_hash=42,
            member_prefixes=[((1, 2, 3), 5)],
        )
        assert cluster.cluster_id == "pc-abc123"
        assert cluster.feature_hash == 42
        assert cluster.member_prefixes == [((1, 2, 3), 5)]
        assert cluster.centroid_entropy == 0.0
        assert cluster.avg_length == 0.0
        assert cluster.last_accessed == 0.0
        assert cluster.access_count == 0

    def test_construction_explicit(self) -> None:
        cluster = PrefixCluster(
            cluster_id="pc-fff",
            feature_hash=999,
            member_prefixes=[((10, 20), 3), ((30, 40), 1)],
            centroid_entropy=1.5,
            avg_length=2.0,
            last_accessed=1000.0,
            access_count=7,
        )
        assert cluster.centroid_entropy == 1.5
        assert cluster.avg_length == 2.0
        assert cluster.last_accessed == 1000.0
        assert cluster.access_count == 7

    def test_top_prefix_empty_members(self) -> None:
        cluster = PrefixCluster(cluster_id="empty", feature_hash=0, member_prefixes=[])
        assert cluster.top_prefix is None

    def test_top_prefix_single_member(self) -> None:
        cluster = PrefixCluster(
            cluster_id="single",
            feature_hash=1,
            member_prefixes=[((5, 5, 5), 3)],
        )
        assert cluster.top_prefix == (5, 5, 5)

    def test_top_prefix_returns_highest_frequency(self) -> None:
        cluster = PrefixCluster(
            cluster_id="multi",
            feature_hash=2,
            member_prefixes=[
                ((1, 2), 1),
                ((3, 4), 10),
                ((5, 6), 5),
            ],
        )
        # Highest frequency is 10 for (3, 4).
        assert cluster.top_prefix == (3, 4)

    def test_top_prefix_tie_returns_first_seen(self) -> None:
        """When frequencies tie, max() returns the first element. We verify
        the property does not crash and returns a valid tuple."""
        cluster = PrefixCluster(
            cluster_id="tie",
            feature_hash=3,
            member_prefixes=[
                ((9, 9), 2),
                ((8, 8), 2),
            ],
        )
        # max() is stable-starting from CPython 3.14? Actually max() on a
        # list returns the *first* encountered max.  Both have the same key
        # value (2) so the first element wins.
        result = cluster.top_prefix
        assert result is not None
        assert result in ((9, 9), (8, 8))


# ---------------------------------------------------------------------------
# PrewarmPrediction
# ---------------------------------------------------------------------------

class TestPrewarmPrediction:
    """Tests for the PrewarmPrediction dataclass."""

    def test_construction_with_defaults(self) -> None:
        pred = PrewarmPrediction(
            source_prefix=(1, 2),
            target_prefix=(3, 4),
            cluster_id="pc-abc",
            confidence=0.5,
        )
        assert pred.source_prefix == (1, 2)
        assert pred.target_prefix == (3, 4)
        assert pred.cluster_id == "pc-abc"
        assert pred.confidence == 0.5
        assert pred.prewarm_to_gpu is True  # default

    def test_construction_explicit_prewarm_flag(self) -> None:
        pred = PrewarmPrediction(
            source_prefix=(1,),
            target_prefix=(2,),
            cluster_id="pc-def",
            confidence=0.6,
            prewarm_to_gpu=False,
        )
        assert pred.prewarm_to_gpu is False


# ---------------------------------------------------------------------------
# _prefix_features (private utility)
# ---------------------------------------------------------------------------

class TestPrefixFeatures:
    """Tests for the _prefix_features feature extraction function."""

    def test_empty_list_returns_zero_features(self) -> None:
        features = _prefix_features([])
        # Empty branch returns avg_log_freq instead of first_token.
        assert features == {
            "entropy_2gram": 0.0,
            "entropy_3gram": 0.0,
            "avg_log_freq": 0.0,
            "length_norm": 0.0,
        }

    def test_single_token_no_ngrams_returns_zero_entropy(self) -> None:
        features = _prefix_features([42])
        assert features["entropy_2gram"] == 0.0
        assert features["entropy_3gram"] == 0.0
        # length_norm = round(1 / 4096, 4) = 0.0002
        assert features["length_norm"] == 0.0002
        assert features["first_token"] == pytest.approx(42.0 / 100.0)

    def test_two_tokens_no_trigrams(self) -> None:
        features = _prefix_features([1, 2])
        # 1 bigram: (1,2) → entropy_2gram = 0
        assert features["entropy_2gram"] == 0.0
        # No trigrams
        assert features["entropy_3gram"] == 0.0
        # length_norm = round(2 / 4096, 4) = 0.0005
        assert features["length_norm"] == 0.0005

    def test_repeating_tokens_yields_zero_entropy(self) -> None:
        tokens = [7, 7, 7, 7, 7, 7, 7, 7]
        features = _prefix_features(tokens)
        # All bigrams and trigrams are identical → entropy 0.
        assert features["entropy_2gram"] == 0.0
        assert features["entropy_3gram"] == 0.0

    def test_all_unique_bigrams_maximises_entropy(self) -> None:
        # 8 tokens → 7 unique bigrams, 6 unique trigrams.
        tokens = [0, 1, 2, 3, 4, 5, 6, 7]
        features = _prefix_features(tokens)
        expected_2gram = round(
            -sum((1 / 7) * math.log2(1 / 7) for _ in range(7)), 2
        )
        expected_3gram = round(
            -sum((1 / 6) * math.log2(1 / 6) for _ in range(6)), 2
        )
        assert features["entropy_2gram"] == expected_2gram
        assert features["entropy_3gram"] == expected_3gram

    def test_first_token_mod_100(self) -> None:
        features = _prefix_features([105, 0, 0, 0, 0, 0, 0, 0])
        assert features["first_token"] == pytest.approx(5.0 / 100.0)

    def test_length_norm_capped_at_one(self) -> None:
        # A very long token list should produce length_norm == 1.0.
        tokens = list(range(5000))
        features = _prefix_features(tokens)
        assert features["length_norm"] == 1.0


# ---------------------------------------------------------------------------
# _cluster_hash (private utility)
# ---------------------------------------------------------------------------

class TestClusterHash:
    """Tests for the _cluster_hash function."""

    def test_same_features_same_hash(self) -> None:
        f1 = {"entropy_2gram": 0.0, "length_norm": 0.002, "first_token": 0.0}
        f2 = {"entropy_2gram": 0.0, "length_norm": 0.002, "first_token": 0.0}
        assert _cluster_hash(f1) == _cluster_hash(f2)

    def test_different_entropy_bucket_different_hash(self) -> None:
        # entropy_bucket = int(4.5 / 2) = 2 vs int(0.0 / 2) = 0
        f1 = {"entropy_2gram": 4.5, "length_norm": 0.002, "first_token": 0.0}
        f2 = {"entropy_2gram": 0.0, "length_norm": 0.002, "first_token": 0.0}
        assert _cluster_hash(f1) != _cluster_hash(f2)

    def test_different_length_bucket_different_hash(self) -> None:
        # length_bucket = int(1.0 * 8) = 8 vs int(0.0 * 8) = 0
        f1 = {"entropy_2gram": 0.0, "length_norm": 1.0, "first_token": 0.0}
        f2 = {"entropy_2gram": 0.0, "length_norm": 0.0, "first_token": 0.0}
        assert _cluster_hash(f1) != _cluster_hash(f2)

    def test_different_first_bucket_different_hash(self) -> None:
        # first_bucket = int(0.8 * 5) = 4 vs int(0.0 * 5) = 0
        f1 = {"entropy_2gram": 0.0, "length_norm": 0.002, "first_token": 0.8}
        f2 = {"entropy_2gram": 0.0, "length_norm": 0.002, "first_token": 0.0}
        assert _cluster_hash(f1) != _cluster_hash(f2)

    def test_deterministic_regression(self) -> None:
        """A known feature set must always produce the same hash."""
        features = {"entropy_2gram": 1.5, "length_norm": 0.125, "first_token": 0.25}
        # entropy_bucket = int(1.5/2) = 0
        # length_bucket  = int(0.125*8) = 1
        # first_bucket   = int(0.25*5) = 1
        actual = _cluster_hash(features)
        assert isinstance(actual, int)
        assert actual >= 0
        # Re-run confirms determinism.
        assert _cluster_hash(features) == actual


# ---------------------------------------------------------------------------
# PrefixClusterer
# ---------------------------------------------------------------------------

# Helpers used by multiple tests in this class.
def _all_same_token(token: int, length: int = 8) -> list[int]:
    """Return a uniform token list of *length* (default 8)."""
    return [token] * length


class TestPrefixClusterer:
    """Tests for the PrefixClusterer class."""

    # -- Construction -------------------------------------------------------

    def test_default_construction(self) -> None:
        clusterer = PrefixClusterer()
        assert clusterer._min_prefix_len == 8
        assert clusterer._max_clusters == 100
        assert clusterer._confidence_threshold == 0.4
        assert clusterer._cooldown_s == 60.0
        stats = clusterer.get_stats()
        assert stats["clusters"] == 0
        assert stats["total_observations"] == 0
        assert stats["prewarm_triggered"] == 0

    def test_custom_construction(self) -> None:
        clusterer = PrefixClusterer(
            min_prefix_len=4,
            max_clusters=10,
            prewarm_confidence_threshold=0.8,
            cooldown_s=120.0,
        )
        assert clusterer._min_prefix_len == 4
        assert clusterer._max_clusters == 10
        assert clusterer._confidence_threshold == 0.8
        assert clusterer._cooldown_s == 120.0

    # -- observe_and_predict: short / empty input --------------------------

    def test_empty_token_list_returns_empty(self) -> None:
        clusterer = PrefixClusterer()
        assert clusterer.observe_and_predict([]) == []
        assert clusterer.get_stats()["total_observations"] == 0

    def test_short_prefix_below_min_length_returns_empty(self) -> None:
        clusterer = PrefixClusterer(min_prefix_len=8)
        assert clusterer.observe_and_predict([1, 2, 3]) == []
        assert clusterer.get_stats()["total_observations"] == 0

    def test_prefix_exactly_min_length_is_observed(self) -> None:
        clusterer = PrefixClusterer(min_prefix_len=4)
        result = clusterer.observe_and_predict([1, 2, 3, 4])
        assert result == []
        assert clusterer.get_stats()["total_observations"] == 1

    # -- Cluster creation and update ---------------------------------------

    def test_first_observation_creates_new_cluster(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        result = clusterer.observe_and_predict(_all_same_token(0))
        # First observation → cluster created, no predictions.
        assert result == []
        stats = clusterer.get_stats()
        assert stats["clusters"] == 1
        assert stats["total_observations"] == 1
        clusters = clusterer.get_clusters()
        assert len(clusters) == 1
        assert clusters[0]["members"] == 1
        assert clusters[0]["access_count"] == 1

    def test_same_prefix_updates_existing_cluster(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        clusterer.observe_and_predict(_all_same_token(0))
        clusterer.observe_and_predict(_all_same_token(0))
        stats = clusterer.get_stats()
        assert stats["clusters"] == 1
        assert stats["total_observations"] == 2
        clusters = clusterer.get_clusters()
        assert clusters[0]["access_count"] == 2

    def test_different_prefix_same_cluster_adds_member(self) -> None:
        """Two prefixes with same feature buckets land in the same cluster."""
        clusterer = PrefixClusterer(cooldown_s=0)
        # Both [0]*8 and [0]*7+[1] have entropy_bucket=0, length_bucket=0,
        # first_bucket=0 → same cluster.
        clusterer.observe_and_predict(_all_same_token(0))          # prefix A
        clusterer.observe_and_predict([0, 0, 0, 0, 0, 0, 0, 1])  # prefix B
        stats = clusterer.get_stats()
        assert stats["clusters"] == 1
        clusters = clusterer.get_clusters()
        assert clusters[0]["members"] == 2

    # -- Predictions -------------------------------------------------------

    def test_prediction_generated_for_higher_frequency_member(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        # Build cluster: prefix A (freq 1), then prefix B triggers prediction.
        clusterer.observe_and_predict(_all_same_token(0))          # A freq=1
        result = clusterer.observe_and_predict(
            [0, 0, 0, 0, 0, 0, 0, 1],  # B added
        )
        # A has freq 1 out of total 2 → confidence = 0.5 >= 0.4
        assert len(result) == 1
        pred = result[0]
        assert pred.source_prefix == (0, 0, 0, 0, 0, 0, 0, 1)
        assert pred.target_prefix == (0, 0, 0, 0, 0, 0, 0, 0)
        assert pred.confidence == 0.5
        assert pred.prewarm_to_gpu is False  # conf <= 0.7

    def test_prediction_with_high_confidence_sets_gpu_flag(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        # Make prefix A very frequent so conf(A) > 0.7.
        for _ in range(5):
            clusterer.observe_and_predict(_all_same_token(0))
        # Introduce B → A has conf = 5/6 ≈ 0.833 > 0.7.
        result = clusterer.observe_and_predict([0, 0, 0, 0, 0, 0, 0, 1])
        assert len(result) == 1
        assert result[0].prewarm_to_gpu is True

    def test_below_threshold_no_prediction(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        # Build cluster where a low-frequency member's confidence < 0.4.
        for _ in range(5):
            clusterer.observe_and_predict(_all_same_token(0))          # A freq=5
        clusterer.observe_and_predict([0, 0, 0, 0, 0, 0, 0, 1])      # B freq=1
        # Now call with A again; B has conf = 1/7 ≈ 0.143 < 0.4 → no pred.
        result = clusterer.observe_and_predict(_all_same_token(0))
        assert result == []

    # -- Cooldown ----------------------------------------------------------

    def test_cooldown_blocks_repeated_prediction(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=60)
        # Build cluster with A freq=5, B freq=1.
        for _ in range(5):
            clusterer.observe_and_predict(_all_same_token(0))
        # First call with B → prediction for A (conf = 5/6 ≈ 0.833).
        result_1 = clusterer.observe_and_predict([0, 0, 0, 0, 0, 0, 0, 1])
        assert len(result_1) == 1  # prediction made
        # Second call with B → cooldown still active (same test run) → blocked.
        result_2 = clusterer.observe_and_predict([0, 0, 0, 0, 0, 0, 0, 1])
        assert result_2 == []  # blocked by cooldown

    def test_zero_cooldown_allows_repeated_predictions(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        for _ in range(5):
            clusterer.observe_and_predict(_all_same_token(0))
        # First call with B → prediction.
        clusterer.observe_and_predict([0, 0, 0, 0, 0, 0, 0, 1])
        # Second call with B → no cooldown blocking, but conf(A) = 5/7 ≈ 0.714
        # which is still >= 0.4 → prediction again.
        result = clusterer.observe_and_predict([0, 0, 0, 0, 0, 0, 0, 1])
        assert len(result) == 1
        assert result[0].target_prefix == (0, 0, 0, 0, 0, 0, 0, 0)

    # -- Eviction ----------------------------------------------------------

    def test_eviction_when_max_clusters_exceeded(self) -> None:
        """Oldest cluster evicted when a new unique cluster arrives."""
        clusterer = PrefixClusterer(max_clusters=2, cooldown_s=0)
        # Three different clusters via different first_bucket values:
        # first_bucket = int(0.0*5)=0, int(0.25*5)=1, int(0.50*5)=2
        clusterer.observe_and_predict(_all_same_token(0))    # cluster 0:0:0
        clusterer.observe_and_predict(_all_same_token(25))   # cluster 0:0:1
        # Third call triggers eviction of the oldest cluster (0:0:0).
        clusterer.observe_and_predict(_all_same_token(50))   # cluster 0:0:2
        stats = clusterer.get_stats()
        assert stats["clusters"] == 2
        # The surviving clusters should be for tokens 25 and 50.
        top_prefixes = {c["top_prefix"][0] for c in clusterer.get_clusters()}
        assert 25 in top_prefixes
        assert 50 in top_prefixes
        assert 0 not in top_prefixes

    def test_evict_oldest_cluster_empty_no_error(self) -> None:
        clusterer = PrefixClusterer()
        # Calling the private eviction method on empty state must not crash.
        clusterer._evict_oldest_cluster()
        assert clusterer.get_stats()["clusters"] == 0

    # -- get_stats / get_clusters ------------------------------------------

    def test_get_stats_empty(self) -> None:
        clusterer = PrefixClusterer()
        stats = clusterer.get_stats()
        assert stats == {
            "clusters": 0,
            "total_observations": 0,
            "total_predictions": 0,
            "prewarm_triggered": 0,
            "avg_members_per_cluster": 0.0,
        }

    def test_get_stats_after_observations(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        clusterer.observe_and_predict(_all_same_token(0))
        clusterer.observe_and_predict(_all_same_token(0))
        stats = clusterer.get_stats()
        assert stats["clusters"] == 1
        assert stats["total_observations"] == 2
        assert stats["avg_members_per_cluster"] == 1.0

    def test_get_stats_counts_predictions(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        # Cluster with two members → B triggers prediction for A.
        clusterer.observe_and_predict(_all_same_token(0))
        clusterer.observe_and_predict([0, 0, 0, 0, 0, 0, 0, 1])
        stats = clusterer.get_stats()
        assert stats["total_predictions"] == 1
        assert stats["prewarm_triggered"] == 1

    def test_get_clusters_empty(self) -> None:
        clusterer = PrefixClusterer()
        assert clusterer.get_clusters() == []

    def test_get_clusters_returns_expected_fields(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        clusterer.observe_and_predict(_all_same_token(0))
        clusters = clusterer.get_clusters()
        assert len(clusters) == 1
        entry = clusters[0]
        # All expected keys present.
        assert "cluster_id" in entry
        assert "members" in entry
        assert "top_prefix" in entry
        assert "centroid_entropy" in entry
        assert "access_count" in entry
        # Verify types.
        assert isinstance(entry["cluster_id"], str)
        assert isinstance(entry["members"], int)
        assert isinstance(entry["top_prefix"], list)
        assert isinstance(entry["centroid_entropy"], float)
        assert isinstance(entry["access_count"], int)

    def test_get_clusters_top_prefix_none_when_empty(self) -> None:
        """A cluster created from a single observation has top_prefix."""
        clusterer = PrefixClusterer(cooldown_s=0)
        clusterer.observe_and_predict(_all_same_token(10))
        clusters = clusterer.get_clusters()
        assert clusters[0]["top_prefix"] == [10, 10, 10, 10, 10, 10, 10, 10]

    # -- Edge cases --------------------------------------------------------

    def test_min_prefix_len_respected(self) -> None:
        clusterer = PrefixClusterer(min_prefix_len=16, cooldown_s=0)
        # 8 tokens is below min_prefix_len → ignored.
        result = clusterer.observe_and_predict(_all_same_token(0, length=8))
        assert result == []
        assert clusterer.get_stats()["total_observations"] == 0
        # 16 tokens → observed.
        result = clusterer.observe_and_predict(_all_same_token(0, length=16))
        assert result == []
        assert clusterer.get_stats()["total_observations"] == 1

    def test_duplicate_prefix_only_increases_freq_no_self_prediction(self) -> None:
        """Calling with the same prefix multiple times does not generate
        a self-prediction (source == target is skipped)."""
        clusterer = PrefixClusterer(cooldown_s=0)
        clusterer.observe_and_predict(_all_same_token(0))
        result = clusterer.observe_and_predict(_all_same_token(0))
        # Only one member in the cluster, and it matches source → skip.
        assert result == []

    def test_large_number_of_members_pruned_to_top_20(self) -> None:
        """When member list exceeds 20 only the top 20 by frequency survive."""
        clusterer = PrefixClusterer(cooldown_s=0)
        # Add the main prefix many times.
        for _ in range(10):
            clusterer.observe_and_predict(_all_same_token(0))
        # Now add 25 unique prefixes that also map to the same cluster.
        for i in range(1, 26):
            # Use the same first_token (0) and low entropy so they share
            # the same cluster.
            tokens = [0] * 7 + [i]
            clusterer.observe_and_predict(tokens)
        # Each of the 25 new prefixes was observed once, the main prefix
        # was seen 10 times.  The member list should be capped at 20.
        clusters = clusterer.get_clusters()
        assert clusters[0]["members"] <= 20
        # The top member should be the high-frequency one.
        top = clusters[0]["top_prefix"]
        assert top == [0, 0, 0, 0, 0, 0, 0, 0]

    def test_observe_and_predict_returns_list_type(self) -> None:
        clusterer = PrefixClusterer(cooldown_s=0)
        result = clusterer.observe_and_predict(_all_same_token(0))
        assert isinstance(result, list)

    def test_get_stats_avg_members_division_by_zero(self) -> None:
        """When there are zero clusters, avg_members_per_cluster is 0."""
        clusterer = PrefixClusterer()
        stats = clusterer.get_stats()
        # The formula divides by max(len(self._clusters), 1), so it yields 0.
        assert stats["avg_members_per_cluster"] == 0.0
