"""Tests for QuantizationAutoTuner.

Target: 15+ tests.
"""

from __future__ import annotations

import pytest

from distllm.dist.partition.quantization_tuner import (
    NodeQuantRecommendation,
    QuantizationAutoTuner,
    QuantizationPlan,
    QuantMethod,
    QUANT_PROFILES,
)


class TestQuantMethod:
    def test_all_methods_exist(self):
        assert QuantMethod.NONE == "none"
        assert QuantMethod.BNB_4BIT == "bnb_4bit"
        assert QuantMethod.BNB_8BIT == "bnb_8bit"
        assert QuantMethod.GPTQ == "gptq"
        assert QuantMethod.AWQ == "awq"


class TestQuantProfiles:
    def test_none_no_quality_loss(self):
        assert QUANT_PROFILES[QuantMethod.NONE].quality_loss == 0.0

    def test_bnb_4bit_memory_reduction(self):
        assert QUANT_PROFILES[QuantMethod.BNB_4BIT].memory_reduction == 0.25

    def test_bnb_8bit_memory_reduction(self):
        assert QUANT_PROFILES[QuantMethod.BNB_8BIT].memory_reduction == 0.5

    def test_all_have_supported_hardware(self):
        for method, profile in QUANT_PROFILES.items():
            assert len(profile.supported_hardware) > 0


class TestQuantizationAutoTuner:
    def test_recommend_fits_no_quant(self):
        tuner = QuantizationAutoTuner()
        nodes = [{"node_id": "gpu-0", "device_type": "cuda", "total_memory_bytes": 80 * 1024**3}]
        plan = tuner.recommend(nodes, model_size_bytes=10 * 1024**3, num_layers=32)
        assert plan.recommendations[0].method == QuantMethod.NONE

    def test_recommend_needs_quant(self):
        tuner = QuantizationAutoTuner()
        nodes = [{"node_id": "gpu-0", "device_type": "cuda", "total_memory_bytes": 8 * 1024**3}]
        plan = tuner.recommend(nodes, model_size_bytes=40 * 1024**3, num_layers=32)
        assert plan.recommendations[0].method != QuantMethod.NONE

    def test_recommend_respects_quality_limit(self):
        tuner = QuantizationAutoTuner(max_quality_loss=0.0)
        nodes = [{"node_id": "gpu-0", "device_type": "cuda", "total_memory_bytes": 4 * 1024**3}]
        plan = tuner.recommend(nodes, model_size_bytes=40 * 1024**3, num_layers=32)
        assert plan.recommendations[0].quality_loss == 0.0

    def test_recommend_unsupported_hardware(self):
        tuner = QuantizationAutoTuner()
        nodes = [{"node_id": "gpu-0", "device_type": "cpu", "total_memory_bytes": 4 * 1024**3}]
        plan = tuner.recommend(nodes, model_size_bytes=40 * 1024**3, num_layers=32)
        assert plan.recommendations[0].method == QuantMethod.BNB_4BIT  # forced fallback

    def test_recommend_multiple_nodes(self):
        tuner = QuantizationAutoTuner()
        nodes = [
            {"node_id": "gpu-0", "device_type": "cuda", "total_memory_bytes": 80 * 1024**3},
            {"node_id": "gpu-1", "device_type": "cuda", "total_memory_bytes": 4 * 1024**3},
        ]
        plan = tuner.recommend(nodes, model_size_bytes=40 * 1024**3, num_layers=32)
        assert len(plan.recommendations) == 2
        assert plan.recommendations[0].method == QuantMethod.NONE
        assert plan.recommendations[1].method != QuantMethod.NONE


class TestQuantizationPlan:
    def test_methods_used(self):
        plan = QuantizationPlan(recommendations=[
            NodeQuantRecommendation("a", QuantMethod.NONE, 100, 100, 0, 0, 1.0, 0.0, ""),
            NodeQuantRecommendation("b", QuantMethod.BNB_4BIT, 100, 25, 75, 75, 1.1, 0.03, ""),
        ])
        assert plan.methods_used == {QuantMethod.NONE, QuantMethod.BNB_4BIT}

    def test_summary(self):
        plan = QuantizationPlan(
            recommendations=[
                NodeQuantRecommendation("a", QuantMethod.NONE, 100, 100, 0, 0, 1.0, 0.0, ""),
            ],
            strategy="No quantization needed",
        )
        s = plan.summary()
        assert "No quantization" in s
