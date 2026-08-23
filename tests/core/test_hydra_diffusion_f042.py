"""Regression: F-042 video pipeline must not feed random noise nor rebuild the
model per call, and must not combine DataParallel with cpu_offload.

These checks are environment-independent (no diffusers/torch required) and pin
the behavioral contract that fixed the finding.
"""

from __future__ import annotations

import inspect
import re


class TestVideoPipelineContract:
    def test_no_random_noise_input_in_generate(self):
        """generate() must never internally construct random-noise input."""
        from distllm.core.hydra_diffusion import VideoPipeline

        src = inspect.getsource(VideoPipeline.generate)
        assert "randn" not in src, "generate() must not create random noise input"
        assert "rand(" not in src

    def test_generate_requires_init_image(self):
        from distllm.core.hydra_diffusion import VideoPipeline

        src = inspect.getsource(VideoPipeline.generate)
        assert "init_image" in src, "generate() must accept a real init image"

    def test_pipe_loaded_once_and_cached(self):
        """generate() must not rebuild the model on every call."""
        from distllm.core.hydra_diffusion import VideoPipeline

        cls_src = inspect.getsource(VideoPipeline)
        assert "self._pipe" in cls_src, "pipeline must be cached on self"
        # The heaviest model load lives in load(), not in generate().
        gen_src = inspect.getsource(VideoPipeline.generate)
        assert "from_pretrained" not in gen_src, "generate() must not load the model"

    def test_is_loaded_gates_generate(self):
        from distllm.core.hydra_diffusion import VideoPipeline

        assert hasattr(VideoPipeline, "is_loaded")


class TestDiffusionLoadContract:
    def test_multi_gpu_does_not_conflict_offload_and_dataparallel(self):
        """load() must not execute enable_model_cpu_offload AND DataParallel."""
        from distllm.core.hydra_diffusion import DiffusionPipeline

        src = inspect.getsource(DiffusionPipeline.load)
        # Use rfind: the first occurrences are in a comment; the actual calls
        # are the LAST occurrences.
        offload_idx = src.rfind("enable_model_cpu_offload(")
        dp_idx = src.rfind("DataParallel(")
        if offload_idx == -1 or dp_idx == -1:
            raise AssertionError("expected both branches to exist")
        # DataParallel is in the `if` branch (first); the offload call is in the
        # `else` branch (second) — so offload must appear AFTER DataParallel,
        # and an `else:` must sit between them. They are mutually exclusive.
        assert offload_idx > dp_idx, "offload must be in the else branch (after DataParallel)"
        between = src[dp_idx:offload_idx]
        assert "else" in between, "an else must separate the two branches"