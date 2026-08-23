"""Tests for CalibrationResult and calibrate function.

Uses the import-helper pattern to load modules directly.
"""

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/calibration.py")
CalibrationResult = _mod.CalibrationResult
calibrate = _mod.calibrate
apply_to_scheduler = _mod.apply_to_scheduler
_measure_kv_bytes_per_token = _mod._measure_kv_bytes_per_token


class TestCalibrationResult:
    def test_defaults(self):
        r = CalibrationResult()
        assert r.kv_bytes_per_token == 0
        assert r.gpu_memory_total_mb == 0
        assert r.recommended_max_preempted == 4
        assert r.recommended_max_batch_size == 32
        assert r.recommended_max_tokens_per_batch == 32768
        assert r.recommended_chunk_size == 512


class TestMeasureKvBytesPerToken:
    def test_default_model(self):
        result = _measure_kv_bytes_per_token({})
        # Default: 2*32*32*(4096/32)*2 = 524288
        assert result == 524288

    def test_custom_model(self):
        info = {
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
        }
        result = _measure_kv_bytes_per_token(info)
        # 2*12*4*(768/12)*2 = 12288
        assert result == 12288

    def test_without_kv_heads_defaults_to_num_heads(self):
        info = {
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
        }
        result = _measure_kv_bytes_per_token(info)
        # Falls back: num_kv_heads = num_attention_heads = 32
        assert result == 524288

    def test_zero_heads(self):
        info = {
            "hidden_size": 0,
            "num_layers": 1,
            "num_attention_heads": 0,
        }
        result = _measure_kv_bytes_per_token(info)
        # head_dim = 128, num_heads=0 so num_kv_heads=0
        # 2*1*0*128*2 = 0
        assert result == 0


class TestCalibrateFunction:
    def test_calibrate_with_model_info(self):
        """calibrate with model info only (no GPU needed)."""
        info = {
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        result = calibrate(model_info=info)
        assert result.kv_bytes_per_token == 131072  # 2*32*8*128*2
        assert result.calibration_time_ms >= 0
        # Without GPU, gpu_memory will be 0 so compute-based recommendations
        # default to max_preempted=4 etc.
        assert isinstance(result.recommended_max_preempted, int)

    def test_calibrate_without_model_info(self):
        result = calibrate()
        assert result.kv_bytes_per_token == 524288
        assert result.calibration_time_ms >= 0

    def test_calibrate_returns_correct_type(self):
        result = calibrate()
        assert isinstance(result, CalibrationResult)


class StubScheduler:
    """Minimal scheduler stub without external dependencies."""
    def __init__(self):
        self.max_batch_size = 0
        self.max_tokens_per_batch = 0
        self._max_preempted = 0
        self._budget = StubBudget()


class StubBudget:
    max_batch_size = 0
    max_total_tokens = 0
    max_prefill_tokens = 0


class TestApplyToScheduler:
    def test_applies_all_fields(self):
        result = CalibrationResult(
            recommended_max_batch_size=16,
            recommended_max_tokens_per_batch=16384,
            recommended_max_prefill_tokens=4096,
            recommended_max_preempted=8,
        )
        scheduler = StubScheduler()
        apply_to_scheduler(scheduler, result)

        assert scheduler.max_batch_size == 16
        assert scheduler.max_tokens_per_batch == 16384
        assert scheduler._budget.max_batch_size == 16
        assert scheduler._budget.max_total_tokens == 16384
        assert scheduler._budget.max_prefill_tokens == 4096
        assert scheduler._max_preempted == 8
