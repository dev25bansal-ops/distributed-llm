"""Comprehensive tests for the recovery system.

Covers:
- NodeRecoveryManager: callbacks, checkpoints, TTL, disk persistence
- Layer redistribution computation
- Predictive failure detection
- Progressive degradation tiers
- ReputationSystem: lock safety, score bounds
- StragglerDetector: baseline reset
"""

from __future__ import annotations

import os
import tempfile
import time

from distllm.core.graceful_degradation import (
    RECOVERY_DEGRADATION_TIERS,
    get_recovery_tier,
)
from distllm.core.predictive_failure import (
    PredictiveFailureDetector,
)
from distllm.dist.recovery import (
    NodeRecoveryManager,
    NodeRecoveryPlan,
    SequenceCheckpoint,
    compute_redistributions,
)
from distllm.dist.reputation import ReputationSystem
from distllm.dist.straggler import DetectionMethod, StragglerDetector

# ── RecoveryManager: Callbacks ──────────────────────────────────────────────


class TestRecoveryManagerCallbacks:

    def test_drain_callback_fired(self):
        mgr = NodeRecoveryManager()
        cb = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mgr.set_drain_callback(cb)
        mgr.on_node_failure("node-1")
        cb.assert_called_once_with("node-1")

    def test_mark_dead_callback_fired(self):
        mgr = NodeRecoveryManager()
        cb = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mgr.set_mark_dead_callback(cb)
        mgr.on_node_failure("node-1")
        cb.assert_called_once_with("node-1")

    def test_redistribute_callback_fired(self):
        mgr = NodeRecoveryManager()
        # Redistribution planning needs a topology: without layer
        # assignments there are no survivors to rebalance onto and the
        # callback can never fire.
        mgr.set_layer_assignments({
            "node-1": (0, 11),
            "node-2": (12, 21),
        })
        cb = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mgr.set_redistribute_layers_callback(cb)
        mgr.on_node_failure("node-1")
        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "node-1"
        assert isinstance(args[1], NodeRecoveryPlan)

    def test_recover_callback_fired_with_checkpoints(self):
        from unittest.mock import MagicMock
        mgr = NodeRecoveryManager()
        cb = MagicMock(return_value=["snap1"])
        mgr.set_recover_sequences_callback(cb)
        mgr.save_checkpoint("req-1", {}, [1], [2], "node-1")
        mgr.on_node_failure("node-1")
        cb.assert_called_once_with("node-1", ["req-1"])

    def test_no_recover_callback_counts_lost(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("req-1", {}, [1], [2], "node-1")
        plan = mgr.on_node_failure("node-1")
        assert plan.total_sequences_lost == 1
        assert plan.recovered_sequences == []

    def test_recover_callback_exception_counts_lost(self):
        from unittest.mock import MagicMock
        mgr = NodeRecoveryManager()
        cb = MagicMock(side_effect=RuntimeError("fail"))
        mgr.set_recover_sequences_callback(cb)
        mgr.save_checkpoint("req-1", {}, [1], [2], "node-1")
        plan = mgr.on_node_failure("node-1")
        assert plan.total_sequences_lost == 1


# ── RecoveryManager: Checkpoints ────────────────────────────────────────────


class TestRecoveryManagerCheckpoints:

    def test_save_and_get(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("r1", {"k": "v"}, [1, 2], [3], "n0")
        ckpt = mgr.get_checkpoint("r1")
        assert ckpt is not None
        assert ckpt.request_id == "r1"
        assert ckpt.node_id == "n0"

    def test_get_checkpoints_for_node(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("r1", {}, [1], [], "n0")
        mgr.save_checkpoint("r2", {}, [2], [], "n0")
        mgr.save_checkpoint("r3", {}, [3], [], "n1")
        ckpts = mgr.get_checkpoints_for_node("n0")
        assert len(ckpts) == 2

    def test_drop_checkpoint(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("r1", {}, [1], [], "n0")
        mgr.drop_checkpoint("r1")
        assert mgr.get_checkpoint("r1") is None

    def test_checkpoint_overwrite(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("r1", {}, [1], [2], "n0")
        mgr.save_checkpoint("r1", {}, [3], [4], "n0")
        ckpt = mgr.get_checkpoint("r1")
        assert ckpt.prompt_tokens == [3]

    def test_checkpoint_size_bytes_dict(self):
        import torch
        ckpt = SequenceCheckpoint(
            request_id="r1",
            kv_cache={"layer0": torch.zeros(2, 4)},
            prompt_tokens=[1],
            generated_tokens=[2],
            node_id="n0",
        )
        assert ckpt.size_bytes() > 0

    def test_checkpoint_size_bytes_list(self):
        import torch
        ckpt = SequenceCheckpoint(
            request_id="r1",
            kv_cache=[torch.zeros(2, 4), torch.zeros(3, 5)],
            prompt_tokens=[1],
            generated_tokens=[2],
            node_id="n0",
        )
        assert ckpt.size_bytes() > 0

    def test_checkpoint_size_bytes_none(self):
        ckpt = SequenceCheckpoint(
            request_id="r1", kv_cache=None,
            prompt_tokens=[1], generated_tokens=[2], node_id="n0",
        )
        assert ckpt.size_bytes() == 0


# ── RecoveryManager: TTL ────────────────────────────────────────────────────


class TestRecoveryManagerTTL:

    def test_evict_stale_checkpoints(self):
        mgr = NodeRecoveryManager(checkpoint_ttl_s=0.1)
        mgr.save_checkpoint("r1", {}, [1], [], "n0")
        time.sleep(0.15)
        evicted = mgr.evict_stale_checkpoints()
        assert evicted == 1
        assert mgr.get_checkpoint("r1") is None

    def test_fresh_checkpoints_not_evicted(self):
        mgr = NodeRecoveryManager(checkpoint_ttl_s=60.0)
        mgr.save_checkpoint("r1", {}, [1], [], "n0")
        evicted = mgr.evict_stale_checkpoints()
        assert evicted == 0
        assert mgr.get_checkpoint("r1") is not None


# ── RecoveryManager: Disk Persistence ───────────────────────────────────────


class TestRecoveryManagerDiskPersistence:

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr1 = NodeRecoveryManager()
            mgr1.save_checkpoint("r1", {}, [1, 2], [3, 4], "n0")
            mgr1.save_checkpoint("r2", {}, [5], [6], "n1")
            assert mgr1.save_to_disk(path) is True

            mgr2 = NodeRecoveryManager()
            assert mgr2.load_from_disk(path) is True
            assert mgr2.get_checkpoint("r1") is not None
            assert mgr2.get_checkpoint("r2") is not None
            assert mgr2.get_checkpoint("r1").prompt_tokens == [1, 2]
        finally:
            os.unlink(path)

    def test_load_nonexistent_returns_false(self):
        mgr = NodeRecoveryManager()
        assert mgr.load_from_disk("/nonexistent/path.json") is False


# ── RecoveryManager: Recovery History ───────────────────────────────────────


class TestRecoveryManagerHistory:

    def test_recovery_history_recorded(self):
        mgr = NodeRecoveryManager()
        mgr.on_node_failure("node-1")
        history = mgr.get_recovery_history()
        assert len(history) == 1
        assert history[0]["node_id"] == "node-1"

    def test_metrics_after_failure(self):
        mgr = NodeRecoveryManager()
        mgr.on_node_failure("node-1")
        m = mgr.get_metrics()
        assert m["failed_nodes"] == 1
        assert m["recoveries"] == 1


# ── Layer Redistribution ────────────────────────────────────────────────────


class TestLayerRedistribution:

    def test_even_split(self):
        """Head-edge failure with a single adjacent survivor.

        C3 fix: only the ADJACENT survivor absorbs; the pre-fix code
        split across all survivors, producing overlapping ranges.
        """
        redistributions = compute_redistributions(
            failed_start_layer=0,
            failed_end_layer=5,
            surviving_nodes={"n1": (6, 11), "n2": (12, 17)},
        )
        assert [r.surviving_node_id for r in redistributions] == ["n1"]
        r = redistributions[0]
        assert (r.added_start_layer, r.added_end_layer) == (0, 5)
        assert (r.new_start_layer, r.new_end_layer) == (0, 11)

    def test_uneven_split_with_remainder(self):
        """Adjacent survivor absorbs the whole orphan when it is the only
        neighbor on that side of the failed range."""
        redistributions = compute_redistributions(
            failed_start_layer=0,
            failed_end_layer=4,
            surviving_nodes={"n1": (5, 9), "n2": (10, 14)},
        )
        assert [r.surviving_node_id for r in redistributions] == ["n1"]
        total_added = sum(
            r.added_end_layer - r.added_start_layer + 1
            for r in redistributions
        )
        assert total_added == 5

    def test_single_survivor(self):
        redistributions = compute_redistributions(
            failed_start_layer=0,
            failed_end_layer=5,
            surviving_nodes={"n1": (6, 11)},
        )
        assert len(redistributions) == 1
        assert redistributions[0].added_start_layer == 0
        assert redistributions[0].added_end_layer == 5

    def test_no_survivors(self):
        redistributions = compute_redistributions(0, 5, {})
        assert redistributions == []

    def test_new_range_covers_original(self):
        redistributions = compute_redistributions(
            failed_start_layer=4,
            failed_end_layer=7,
            surviving_nodes={"n1": (0, 3), "n2": (8, 11)},
        )
        for r in redistributions:
            # New range should include both original and added layers
            assert r.new_start_layer <= r.added_start_layer
            assert r.new_end_layer >= r.added_end_layer


# ── Predictive Failure Detection ────────────────────────────────────────────


class TestPredictiveFailure:

    def test_no_signals_healthy(self):
        det = PredictiveFailureDetector()
        pred = det.check_gpu_health("n0", {})
        assert pred.failure_probability == 0.0
        assert pred.recommendation == "ok"

    def test_ecc_uncorrectable_immediate_drain(self):
        det = PredictiveFailureDetector()
        pred = det.check_gpu_health("n0", {"ecc_uncorrectable": 1})
        assert pred.failure_probability >= 0.9
        assert pred.recommendation == "immediate_drain"

    def test_thermal_throttle_monitored(self):
        det = PredictiveFailureDetector()
        pred = det.check_gpu_health("n0", {"thermal_throttle": True})
        assert pred.failure_probability > 0.0
        assert pred.recommendation in ("monitor", "preemptive_drain", "immediate_drain")

    def test_multiple_signals_compound(self):
        det = PredictiveFailureDetector()
        pred = det.check_gpu_health("n0", {
            "thermal_throttle": True,
            "memory_fragmentation_pct": 90,
            "clock_throttle_pct": 30,
        })
        assert pred.failure_probability > 0.3

    def test_history_tracked(self):
        det = PredictiveFailureDetector()
        det.check_gpu_health("n0", {"thermal_throttle": True})
        det.check_gpu_health("n0", {"thermal_throttle": True})
        assert len(det.get_history("n0")) == 2

    def test_trending_nodes(self):
        det = PredictiveFailureDetector()
        for _ in range(5):
            det.check_gpu_health("n0", {
                "thermal_throttle": True,
                "memory_fragmentation_pct": 90,
                "clock_throttle_pct": 30,
            })
        trending = det.get_trending_nodes(threshold=0.1)
        assert "n0" in trending


# ── Progressive Degradation Tiers ───────────────────────────────────────────


class TestProgressiveDegradation:

    def test_full_tier_at_100_percent(self):
        tier = get_recovery_tier(4, 4)
        assert tier["name"] == "full"
        assert tier["batch_size"] == 4

    def test_reduced_tier_at_75_percent(self):
        tier = get_recovery_tier(3, 4)
        assert tier["name"] == "reduced"

    def test_minimal_tier_at_50_percent(self):
        tier = get_recovery_tier(2, 4)
        assert tier["name"] == "minimal"

    def test_emergency_tier_at_25_percent(self):
        tier = get_recovery_tier(1, 4)
        assert tier["name"] == "emergency"

    def test_tier_count(self):
        assert len(RECOVERY_DEGRADATION_TIERS) == 4

    def test_zero_nodes_total(self):
        tier = get_recovery_tier(0, 0)
        assert tier["name"] == "full"


# ── ReputationSystem: Lock Safety ───────────────────────────────────────────


class TestReputationLockSafety:

    def test_get_scores_no_deadlock(self):
        """get_scores() should not deadlock with RLock."""
        rep = ReputationSystem()
        rep.record_success("n1", latency_ms=10.0, tokens=100)
        rep.record_success("n2", latency_ms=20.0, tokens=200)
        scores = rep.get_scores()
        assert "n1" in scores
        assert "n2" in scores

    def test_score_bounds(self):
        rep = ReputationSystem()
        rep.record_success("n1", latency_ms=10.0, tokens=100)
        score = rep.get_score("n1")
        assert 0.0 <= score <= 1.0

    def test_persistently_failing_low_score(self):
        rep = ReputationSystem()
        for _ in range(10):
            rep.record_failure("n1")
        score = rep.get_score("n1")
        assert score <= 0.2

    def test_unknown_node_neutral_score(self):
        rep = ReputationSystem()
        assert rep.get_score("unknown") == 0.5


# ── StragglerDetector: Baseline Reset ───────────────────────────────────────


class TestStragglerBaselineReset:

    def test_reset_baseline_clears_state(self):
        det = StragglerDetector(detection_method=DetectionMethod.THRESHOLD, consecutive_threshold=1)
        for _ in range(10):
            det.record_latency("fast", 10.0)
            det.record_latency("slow", 200.0)
        det.check()
        det.reset_baseline("slow")
        s = det.stats()
        assert s["nodes"]["slow"]["consecutive_slow"] == 0
        assert s["nodes"]["slow"]["is_straggler"] is False

    def test_reset_baseline_unknown_node_no_error(self):
        det = StragglerDetector()
        det.reset_baseline("nonexistent")
