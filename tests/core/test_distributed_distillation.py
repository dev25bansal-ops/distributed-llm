"""Tests for DistillationConfig and DistributedDistillationEngine."""

from __future__ import annotations

import tempfile

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_dd_mod = load_module("distllm/core/distributed_distillation.py")
DistillationConfig = _dd_mod.DistillationConfig
DistributedDistillationEngine = _dd_mod.DistributedDistillationEngine


class TestDistillationConfig:
    """DistillationConfig dataclass -- construction and defaults."""

    def test_default_config(self):
        cfg = DistillationConfig()
        assert cfg.temperature == 2.0
        assert cfg.alpha == 0.5
        assert cfg.batch_size == 4
        assert cfg.max_samples == 10000
        assert cfg.max_length == 512
        assert cfg.learning_rate == 5e-5
        assert cfg.idle_only is True
        assert len(cfg.seed_prompts) == 7

    def test_custom_config(self):
        cfg = DistillationConfig(
            teacher_model="teacher/large",
            student_model_path="student/small",
            temperature=4.0,
            alpha=0.7,
            batch_size=8,
            max_samples=100,
        )
        assert cfg.teacher_model == "teacher/large"
        assert cfg.student_model_path == "student/small"
        assert cfg.temperature == 4.0
        assert cfg.alpha == 0.7
        assert cfg.batch_size == 8
        assert cfg.max_samples == 100

    def test_checkpoint_dir_default(self):
        cfg = DistillationConfig()
        assert cfg.checkpoint_dir == "/tmp/distllm-distillation"

    def test_seed_prompts_default_content(self):
        cfg = DistillationConfig()
        assert any("distributed computing" in p for p in cfg.seed_prompts)
        assert any("capital of France" in p for p in cfg.seed_prompts)


class TestDistributedDistillationEngineConstruction:
    """Engine construction with different configs."""

    def test_default_construction(self):
        cfg = DistillationConfig()
        engine = DistributedDistillationEngine(config=cfg)
        assert engine._config is cfg
        assert engine._teacher_forward is None
        assert engine._is_running is False
        assert engine._samples_generated == 0
        assert engine._steps_completed == 0

    def test_start_stop_state(self):
        cfg = DistillationConfig(checkpoint_dir=tempfile.mkdtemp())
        engine = DistributedDistillationEngine(config=cfg)
        # start succeeds (thread launches even without models)
        started = engine.start()
        assert started is True
        # The loop thread is running and will exit when reaching max_samples
        engine.stop()
        assert engine._is_running is False

    def test_stop_when_not_running(self):
        cfg = DistillationConfig(checkpoint_dir=tempfile.mkdtemp())
        engine = DistributedDistillationEngine(config=cfg)
        # stop should not crash when not running
        engine.stop()
        assert engine._is_running is False

    def test_stats_defaults(self):
        cfg = DistillationConfig(checkpoint_dir=tempfile.mkdtemp())
        engine = DistributedDistillationEngine(config=cfg)
        stats = engine.stats
        assert stats["is_running"] is False
        assert stats["steps_completed"] == 0
        assert stats["samples_generated"] == 0
        assert stats["avg_loss"] == 0.0
        assert stats["avg_kl"] == 0.0
        assert stats["avg_ce"] == 0.0

    def test_double_start_returns_false(self):
        cfg = DistillationConfig(checkpoint_dir=tempfile.mkdtemp())
        engine = DistributedDistillationEngine(config=cfg)
        engine._is_running = True
        result = engine.start()
        assert result is False


class TestDistributedDistillationEngineTeacherForward:
    """Custom teacher_forward callable."""

    def test_custom_teacher_forward_is_stored(self):
        cfg = DistillationConfig(checkpoint_dir=tempfile.mkdtemp())

        def fake_teacher(prompt: str, max_length: int) -> dict:
            return {"input_ids": None, "teacher_logits": None}

        engine = DistributedDistillationEngine(
            config=cfg, teacher_forward=fake_teacher,
        )
        assert engine._teacher_forward is fake_teacher

    def test_custom_teacher_forward_called(self):
        cfg = DistillationConfig(checkpoint_dir=tempfile.mkdtemp())
        calls = []

        def fake_teacher(prompt: str, max_length: int) -> dict:
            calls.append((prompt, max_length))
            return {"input_ids": None, "teacher_logits": None}

        engine = DistributedDistillationEngine(
            config=cfg, teacher_forward=fake_teacher,
        )
        result = engine._generate_teacher_targets("test prompt")
        assert len(calls) == 1
        assert calls[0][0] == "test prompt"
        assert result is not None
