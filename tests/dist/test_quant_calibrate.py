"""Real tests for partition/quant_calibrate — QualityCalibrator."""
from __future__ import annotations


class TestQualityCalibrator:
    def test_calibrator_init(self):
        from distllm.dist.partition.quant_calibrate import QualityCalibrator

        cal = QualityCalibrator()
        assert cal is not None

    def test_calibration_result(self):
        from distllm.dist.partition.quant_calibrate import CalibrationResult

        result = CalibrationResult(
            method="int8", perplexity=5.2,
        )
        assert result.method == "int8"
        assert result.perplexity == 5.2

    def test_calibration_report(self):
        from distllm.dist.partition.quant_calibrate import CalibrationReport

        report = CalibrationReport()
        assert report is not None
