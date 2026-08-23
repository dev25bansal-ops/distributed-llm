"""Regression tests for HIGH fix C12: OOMError never raised.

CUDA out-of-memory during vLLM model load / forward was swallowed into a generic
``ModelLoadError`` / ``RuntimeError``. Now ``torch.cuda.OutOfMemoryError`` is
caught and re-raised as ``OOMError`` so callers can react (e.g. evict, retry,
degrade) instead of masking the failure.
"""

from __future__ import annotations

import builtins
import types

import pytest

torch = pytest.importorskip("torch")

from distllm.backends.vllm_backend import VLLMNodeAdapter
from distllm.errors.types import OOMError


def test_load_model_raises_oomerror(monkeypatch):
    # Build an adapter without a real model; make `from vllm import LLM`
    # resolve to a fake module whose LLM() raises CUDA OOM.
    adapter = object.__new__(VLLMNodeAdapter)
    adapter.model_name = "test/model"
    adapter._config = {}
    adapter._llm = None
    adapter._tokenizer = None
    adapter._is_pipeline_mode = False

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "vllm" or name.startswith("vllm."):
            vllm_mod = types.ModuleType("vllm")

            def _boom(*args, **kwargs):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")

            vllm_mod.LLM = _boom
            return vllm_mod
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(OOMError):
        adapter.load_model()


def test_forward_raises_oomerror(monkeypatch):
    adapter = object.__new__(VLLMNodeAdapter)
    adapter.model_name = "test/model"
    adapter._llm = types.SimpleNamespace()
    adapter._tokenizer = types.SimpleNamespace(decode=lambda *a, **k: "x")
    adapter._is_pipeline_mode = False
    adapter._model = types.SimpleNamespace()

    def boom_generate(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    adapter._llm.generate = boom_generate

    # `_forward_with_input_ids` does `from vllm import SamplingParams`;
    # provide a fake vllm module so we reach the generate() OOM path.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "vllm" or name.startswith("vllm."):
            vllm_mod = types.ModuleType("vllm")

            class _SamplingParams:
                def __init__(self, *args, **kwargs):
                    pass

            vllm_mod.SamplingParams = _SamplingParams
            return vllm_mod
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(OOMError):
        adapter.forward(input_ids=torch.zeros(1, 1, dtype=torch.long))
