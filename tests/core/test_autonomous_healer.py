"""Tests for Autonomous GPU Cluster Healing.

Tests GPUHealthState enum values, GPUHeartbeat dataclass health scoring,
FailurePredictor heuristic fallback (no sklearn), GPUResetManager dry-run
mode, and AutonomousHealer state machine transitions.

Run: python -m pytest tests/core/test_autonomous_healer.py -v

All tests run without GPU, sklearn, or subprocess.
"""

from unittest.mock import MagicMock, patch

import pytest

from distllm.core.autonomous_healer import (
    GPUHealthState,
    GPUHeartbeat,
    FailurePredictor,
    GPUResetManager,
    AutonomousHealer,
)


class TestGPUHealthState:
    """Verify all GPU health state enum values exist."""

    def test_all_values_exist(self):
        assert GPUHealthState.HEALTHY.value == "healthy"
        assert GPUHealthState.DEGRADED.value == "degraded"
        assert GPUHealthState.DRAINING.value == "draining"
        assert GPUHealthState.RECOVERING.value == "recovering"
        assert GPUHealthState.SHADOW.value == "shadow"
        assert GPUHealthState.OFFLINE.value == "offline"

    def test_members_count(self):
        assert len(GPUHealthState) == 6


class TestGPUHeartbeat:
    """Dataclass defaults and composite health_score property."""

    def test_defaults(self):
        hb = GPUHeartbeat(node_id="gpu-0")
        assert hb.node_id == "gpu-0"
        assert hb.ecc_corrected_total == 0
        assert hb.ecc_uncorrected_total == 0
        assert hb.ecc_corrected_rate == 0.0
        assert hb.gpu_temp_c == 0.0
        assert hb.memory_temp_c == 0.0
        assert hb.thermal_throttling is False
        assert hb.power_limit_throttling is False
        assert hb.nvlink_crc_errors == 0
        assert hb.nvlink_crc_rate == 0.0
        assert hb.pcie_replay_count == 0
        assert hb.pcie_link_speed_current == 0.0
        assert hb.pcie_link_speed_max == 0.0
        assert hb.memory_used_mb == 0.0
        assert hb.memory_total_mb == 0.0
        assert hb.memory_retired_pages == 0
        assert hb.memory_retired_pending == 0
        assert hb.gpu_util_pct == 0.0
        assert hb.memory_util_pct == 0.0
        assert hb.pcie_bandwidth_util_pct == 0.0

    def test_health_score_returns_one_for_clean_heartbeat(self):
        hb = GPUHeartbeat(node_id="gpu-0")
        assert hb.health_score == 1.0

    def test_health_score_ecc_uncorrected(self):
        hb = GPUHeartbeat(node_id="gpu-0", ecc_uncorrected_total=1)
        assert hb.health_score == 0.5  # 1.0 - 0.5

    def test_health_score_ecc_corrected_rate_high(self):
        hb = GPUHeartbeat(node_id="gpu-0", ecc_corrected_rate=20.0)
        assert hb.health_score == 0.7  # 1.0 - 0.3

    def test_health_score_ecc_corrected_rate_medium(self):
        hb = GPUHeartbeat(node_id="gpu-0", ecc_corrected_rate=5.0)
        assert hb.health_score == 0.9  # 1.0 - 0.1

    def test_health_score_thermal_throttling(self):
        hb = GPUHeartbeat(node_id="gpu-0", thermal_throttling=True)
        assert hb.health_score == 0.7  # 1.0 - 0.3

    def test_health_score_power_limit_throttling(self):
        hb = GPUHeartbeat(node_id="gpu-0", power_limit_throttling=True)
        assert hb.health_score == 0.7  # 1.0 - 0.3

    def test_health_score_high_temperature(self):
        hb = GPUHeartbeat(node_id="gpu-0", gpu_temp_c=90.0)
        assert hb.health_score == 0.8  # 1.0 - 0.2

    def test_health_score_moderate_temperature(self):
        hb = GPUHeartbeat(node_id="gpu-0", gpu_temp_c=80.0)
        assert hb.health_score == 0.9  # 1.0 - 0.1

    def test_health_score_nvlink_crc_rate_high(self):
        hb = GPUHeartbeat(node_id="gpu-0", nvlink_crc_rate=10.0)
        assert hb.health_score == 0.7  # 1.0 - 0.3

    def test_health_score_nvlink_crc_rate_medium(self):
        hb = GPUHeartbeat(node_id="gpu-0", nvlink_crc_rate=3.0)
        assert hb.health_score == 0.9  # 1.0 - 0.1

    def test_health_score_pcie_replay(self):
        hb = GPUHeartbeat(node_id="gpu-0", pcie_replay_count=20)
        assert hb.health_score == 0.8  # 1.0 - 0.2

    def test_health_score_memory_retired_pending(self):
        hb = GPUHeartbeat(node_id="gpu-0", memory_retired_pending=1)
        assert hb.health_score == 0.7  # 1.0 - 0.3

    def test_health_score_memory_retired_pages(self):
        hb = GPUHeartbeat(node_id="gpu-0", memory_retired_pages=1)
        assert hb.health_score == 0.9  # 1.0 - 0.1

    def test_health_score_bounded_below_zero(self):
        """Multiple severe issues clamp at 0.0 rather than going negative."""
        hb = GPUHeartbeat(
            node_id="gpu-0",
            ecc_uncorrected_total=1,      # -0.5
            thermal_throttling=True,       # -0.3
            nvlink_crc_rate=10.0,          # -0.3
            memory_retired_pending=1,      # -0.3
        )                                   # 1.0 - 1.4 -> 0.0
        assert hb.health_score == 0.0


class TestFailurePredictor:
    """FailurePredictor with heuristic fallback (no sklearn)."""

    def test_is_trained_false_on_init(self):
        predictor = FailurePredictor()
        assert predictor.is_trained is False

    def test_predict_clean_returns_low_risk(self):
        predictor = FailurePredictor()
        hb = GPUHeartbeat(node_id="gpu-0")
        risk = predictor.predict(hb)
        assert risk < 0.3
        assert risk == 0.0

    def test_predict_bad_returns_high_risk(self):
        """Predict with ecc_uncorrected_total > 0 returns high risk."""
        predictor = FailurePredictor()
        hb = GPUHeartbeat(
            node_id="gpu-0",
            ecc_uncorrected_total=1,   # +0.4
            nvlink_crc_rate=20.0,      # +0.2  (> 10)
        )
        risk = predictor.predict(hb)
        assert risk > 0.5
        assert risk == pytest.approx(0.6)

    def test_record_outcome_stores_samples(self):
        predictor = FailurePredictor()
        hb = GPUHeartbeat(node_id="gpu-0")
        predictor.record_outcome(hb, failed=False)
        assert len(predictor._samples) == 1

    def test_train_returns_false_without_sklearn(self):
        """train() returns False when sklearn is not available."""
        predictor = FailurePredictor(cold_start_threshold=1)
        hb = GPUHeartbeat(node_id="gpu-0")
        predictor.record_outcome(hb, failed=False)
        assert len(predictor._samples) >= 1
        result = predictor.train()
        assert result is False
        assert predictor.is_trained is False

    def test_predict_uses_heuristic_when_not_trained(self):
        predictor = FailurePredictor()
        clean = GPUHeartbeat(node_id="gpu-0")
        assert predictor.predict(clean) == 0.0

        bad = GPUHeartbeat(
            node_id="gpu-0",
            ecc_uncorrected_total=2,      # +0.4
            ecc_corrected_rate=60.0,       # +0.3  (> 50)
            thermal_throttling=True,       # +0.3
        )
        assert predictor.predict(bad) == 1.0  # 0.4 + 0.3 + 0.3 = 1.0

    def test_heuristic_clamps_to_one(self):
        predictor = FailurePredictor()
        hb = GPUHeartbeat(
            node_id="gpu-0",
            ecc_uncorrected_total=1,         # +0.4
            ecc_corrected_rate=100.0,        # +0.3  (> 50)
            thermal_throttling=True,          # +0.3
            gpu_temp_c=95.0,                  # +0.25 (> 90)
            nvlink_crc_rate=20.0,             # +0.2  (> 10)
            memory_retired_pending=1,          # +0.3
            pcie_replay_count=200,             # +0.2  (> 100)
        )  # sum = 1.95 -> clamped to 1.0
        assert predictor.predict(hb) == 1.0

    def test_temperature_heuristic(self):
        predictor = FailurePredictor()
        hb = GPUHeartbeat(node_id="gpu-0", gpu_temp_c=95.0)
        assert predictor.predict(hb) == 0.25

        hb2 = GPUHeartbeat(node_id="gpu-0", gpu_temp_c=85.0)
        assert predictor.predict(hb2) == 0.1

        hb3 = GPUHeartbeat(node_id="gpu-0", gpu_temp_c=75.0)
        assert predictor.predict(hb3) == 0.0

    def test_ecc_corrected_rate_heuristic(self):
        predictor = FailurePredictor()
        hb = GPUHeartbeat(node_id="gpu-0", ecc_corrected_rate=60.0)
        assert predictor.predict(hb) == 0.3  # > 50

        hb2 = GPUHeartbeat(node_id="gpu-0", ecc_corrected_rate=20.0)
        assert predictor.predict(hb2) == 0.15  # > 10

        hb3 = GPUHeartbeat(node_id="gpu-0", ecc_corrected_rate=5.0)
        assert predictor.predict(hb3) == 0.0  # <= 10

    def test_extract_features(self):
        predictor = FailurePredictor()
        hb = GPUHeartbeat(
            node_id="gpu-0",
            ecc_corrected_rate=12.0,
            ecc_uncorrected_total=3,
            gpu_temp_c=80.0,
            thermal_throttling=True,
            nvlink_crc_rate=4.0,
            pcie_replay_count=50,
            memory_retired_pages=2,
            memory_retired_pending=1,
        )
        feats = predictor.extract_features(hb)
        assert feats == [
            12.0,
            3,            # min(3, 100)
            0.8,          # 80.0 / 100.0
            1.0,          # thermal_throttling bool -> float
            4.0,
            0.5,          # min(50 / 100, 1.0)
            0.02,         # 2 / 100.0
            0.1,          # 1 / 10.0
        ]


class TestGPUResetManager:
    """GPU reset manager dry-run and stats behavior."""

    def test_dry_run_does_not_call_subprocess(self):
        """With dry_run=True, reset_gpu returns True without subprocess."""
        mgr = GPUResetManager(dry_run=True)
        result = mgr.reset_gpu("node-0", 0)
        assert result is True

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_dry_run_false_calls_subprocess(self, mock_run, mock_sleep):
        mock_run.return_value = MagicMock(returncode=0, stdout="0", stderr="")
        mgr = GPUResetManager(dry_run=False)
        result = mgr.reset_gpu("node-0", 0)
        assert result is True
        mock_run.assert_called()

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_reset_failure_returns_false(self, mock_run, mock_sleep):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        mgr = GPUResetManager(dry_run=False)
        result = mgr.reset_gpu("node-0", 0)
        assert result is False

    def test_stats_returns_expected_keys(self):
        mgr = GPUResetManager(dry_run=True)
        mgr.reset_gpu("node-0", 0)
        stats = mgr.stats
        assert "reset_count" in stats
        assert "recovery_count" in stats
        assert stats["reset_count"] == 1
        assert stats["recovery_count"] == 0

    def test_stats_after_multiple_resets(self):
        mgr = GPUResetManager(dry_run=True)
        mgr.reset_gpu("node-0", 0)
        mgr.reset_gpu("node-1", 1)
        stats = mgr.stats
        assert stats["reset_count"] == 2


class TestAutonomousHealer:
    """Autonomous healer state machine and integration tests."""

    def test_initial_state_empty(self):
        healer = AutonomousHealer()
        stats = healer.stats
        assert stats["state_counts"] == {}

    def test_record_heartbeat_stores_heartbeat(self):
        healer = AutonomousHealer()
        hb = GPUHeartbeat(node_id="gpu-0")
        healer.record_heartbeat(hb)
        assert healer._heartbeats["gpu-0"] is hb

    def test_check_all_returns_state_dict(self):
        healer = AutonomousHealer()
        hb = GPUHeartbeat(node_id="gpu-0")
        healer.record_heartbeat(hb)
        state = healer.check_all()
        assert isinstance(state, dict)
        assert state["gpu-0"] == GPUHealthState.HEALTHY

    def test_healthy_to_draining_transition(self):
        on_drain = MagicMock()
        healer = AutonomousHealer(
            on_drain_callback=on_drain,
            failure_threshold=0.3,
        )
        hb = GPUHeartbeat(node_id="gpu-0", ecc_uncorrected_total=1)
        healer.record_heartbeat(hb)
        state = healer.check_all()
        assert state["gpu-0"] == GPUHealthState.DRAINING
        on_drain.assert_called_once_with("gpu-0")

    def test_healthy_stays_healthy_for_low_risk(self):
        healer = AutonomousHealer(failure_threshold=0.5)
        hb = GPUHeartbeat(node_id="gpu-0")
        healer.record_heartbeat(hb)
        state = healer.check_all()
        assert state["gpu-0"] == GPUHealthState.HEALTHY

    def test_draining_to_shadow(self):
        """Check DRAINING transitions through RECOVERING to SHADOW."""
        healer = AutonomousHealer(
            dry_run=True,
            failure_threshold=0.3,
        )
        hb = GPUHeartbeat(node_id="gpu-0", ecc_uncorrected_total=1)
        healer.record_heartbeat(hb)

        # First call: HEALTHY -> DRAINING
        healer.check_all()

        # Second call: DRAINING -> RECOVERING -> SHADOW (dry_run succeeds)
        state = healer.check_all()
        assert state["gpu-0"] == GPUHealthState.SHADOW

    def test_reset_failure_goes_offline(self):
        healer = AutonomousHealer(
            dry_run=False,
            failure_threshold=0.3,
        )
        hb = GPUHeartbeat(node_id="gpu-0", ecc_uncorrected_total=1)
        healer.record_heartbeat(hb)

        # First call: HEALTHY -> DRAINING
        healer.check_all()

        # Mock reset_gpu to return False
        with patch.object(healer._reset_mgr, "reset_gpu", return_value=False):
            state = healer.check_all()
            assert state["gpu-0"] == GPUHealthState.OFFLINE

    def test_full_cycle(self):
        """HEALTHY -> DRAINING -> RECOVERING -> SHADOW -> HEALTHY."""
        on_drain = MagicMock()
        on_recover = MagicMock()

        healer = AutonomousHealer(
            on_drain_callback=on_drain,
            on_recover_callback=on_recover,
            failure_threshold=0.3,
            recovery_threshold=0.5,
            shadow_duration_s=0,    # complete shadow immediately
            dry_run=True,
        )

        # 1. Record bad heartbeat (risk >= 0.4 from ecc_uncorrected)
        bad_hb = GPUHeartbeat(node_id="gpu-0", ecc_uncorrected_total=1)
        healer.record_heartbeat(bad_hb)

        # 2. HEALTHY -> DRAINING
        state = healer.check_all()
        assert state["gpu-0"] == GPUHealthState.DRAINING
        on_drain.assert_called_once_with("gpu-0")

        # 3. DRAINING -> RECOVERING -> SHADOW
        state = healer.check_all()
        assert state["gpu-0"] == GPUHealthState.SHADOW

        # 4. Replace with clean heartbeat for recovery
        clean_hb = GPUHeartbeat(node_id="gpu-0")
        healer.record_heartbeat(clean_hb)

        # 5. SHADOW -> HEALTHY (risk=0.0 < 0.5, duration=0 satisfied)
        state = healer.check_all()
        assert state["gpu-0"] == GPUHealthState.HEALTHY
        on_recover.assert_called_once_with("gpu-0")

    def test_shadow_extends_when_risk_elevated(self):
        """SHADOW extends when risk >= recovery_threshold."""
        healer = AutonomousHealer(
            failure_threshold=0.3,
            recovery_threshold=0.15,
            shadow_duration_s=0,
            dry_run=True,
        )

        hb = GPUHeartbeat(node_id="gpu-0", ecc_uncorrected_total=1)
        healer.record_heartbeat(hb)

        # HEALTHY -> DRAINING
        healer.check_all()
        # DRAINING -> SHADOW
        healer.check_all()

        # Shadow check: risk=0.4 >= 0.15 -> extend, stay in SHADOW
        state = healer.check_all()
        assert state["gpu-0"] == GPUHealthState.SHADOW

    def test_stats_returns_expected_keys(self):
        healer = AutonomousHealer()
        hb = GPUHeartbeat(node_id="gpu-0")
        healer.record_heartbeat(hb)
        healer.check_all()
        stats = healer.stats
        assert "state_counts" in stats
        assert "predictor_trained" in stats
        assert "reset_count" in stats
        assert "recovery_count" in stats
        assert stats["predictor_trained"] is False
        assert stats["state_counts"] == {"healthy": 1}

    def test_multiple_gpus_tracked_independently(self):
        healer = AutonomousHealer(
            dry_run=True,
            failure_threshold=0.3,
        )
        healer.record_heartbeat(GPUHeartbeat(
            node_id="gpu-0", ecc_uncorrected_total=1,
        ))
        healer.record_heartbeat(GPUHeartbeat(node_id="gpu-1"))

        state = healer.check_all()
        assert state["gpu-0"] == GPUHealthState.DRAINING
        assert state["gpu-1"] == GPUHealthState.HEALTHY
