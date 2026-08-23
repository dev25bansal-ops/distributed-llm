"""Tests for the benchmarks/evaluation_harness module."""
from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/benchmarks/evaluation_harness.py")
LMEvalConfig = _mod.LMEvalConfig
DistLLMModelAdapter = _mod.DistLLMModelAdapter
LMEvalRunner = _mod.LMEvalRunner


class TestLMEvalConfig:
    def test_defaults(self):
        cfg = LMEvalConfig()
        assert "mmlu" in cfg.tasks
        assert cfg.model == "local-completion"
        assert cfg.batch_size == "auto"

    def test_custom(self):
        cfg = LMEvalConfig(tasks=["hellaswag"], num_fewshot=5)
        assert cfg.tasks == ["hellaswag"]
        assert cfg.num_fewshot == 5


class TestDistLLMModelAdapter:
    def test_construction(self):
        adapter = DistLLMModelAdapter("http://test:8000", api_key="sk-test")
        assert adapter._base_url == "http://test:8000"
        assert "Bearer sk-test" in adapter._headers.get("Authorization", "")


class TestLMEvalRunner:
    def test_construction(self):
        runner = LMEvalRunner(coordinator_url="http://test:8000")
        assert runner.config is not None
        assert runner._adapter._base_url == "http://test:8000"

    def test_run_without_lm_eval(self):
        runner = LMEvalRunner()
        result = runner.run()
        assert "error" in result
        assert result["error"] == "lm_eval not installed"
