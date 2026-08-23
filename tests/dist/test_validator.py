"""Real tests for partition/validator — PartitionValidator."""
from __future__ import annotations


class TestPartitionValidator:
    def test_validator_init(self):
        from distllm.dist.partition.validator import PartitionValidator
        from distllm.dist.partition.cost_model import PartitionCostModel

        cm = PartitionCostModel(gpu_profiles={}, layer_weights=[], topology=None)
        v = PartitionValidator(cost_model=cm)
        assert v is not None

    def test_validation_report(self):
        from distllm.dist.partition.validator import ValidationReport

        report = ValidationReport(
            is_valid=True, issues=[], warnings=[],
            simulation=None,
        )
        assert report.is_valid is True

    def test_what_if_scenario(self):
        from distllm.dist.partition.validator import WhatIfScenario

        w = WhatIfScenario(
            scenario="add one more GPU",
            original_throughput=1000.0, new_throughput=1200.0,
            throughput_change_pct=20.0,
            new_bottleneck="memory", impact_description="faster",
        )
        assert w.scenario == "add one more GPU"
