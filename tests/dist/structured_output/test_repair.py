"""Tests for structured output repair orchestrator."""

import json
from distllm.dist.structured_output.engine import RepairOrchestrator, RepairConfig


class TestValidateToken:
    def test_validate_token_no_schema(self):
        orch = RepairOrchestrator()
        assert orch.validate_token("{", "") is True

    def test_validate_token_with_schema(self):
        orch = RepairOrchestrator()
        schema = {"type": "object"}
        assert orch.validate_token('{"key": "value"}', "", schema) is True

    def test_get_valid_prefix(self):
        orch = RepairOrchestrator()
        orch.set_valid_prefix('{"valid": true}')
        assert orch.get_valid_prefix() == '{"valid": true}'


class TestRepairOutput:
    def test_repair_unclosed_brace(self):
        orch = RepairOrchestrator()
        result = orch.repair_output('{"key": "value"')
        data = json.loads(result)
        assert data["key"] == "value"

    def test_repair_valid_json(self):
        orch = RepairOrchestrator()
        result = orch.repair_output('{"key": "value"}')
        data = json.loads(result)
        assert data["key"] == "value"

    def test_repair_rate(self):
        config = RepairConfig(max_repair_attempts=5)
        orch = RepairOrchestrator(config)
        assert orch.repair_rate == 1.0
