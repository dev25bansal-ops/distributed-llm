"""Regression: local-model generation returns a string end-to-end.

Surfaced while preparing the investor demo: Coordinator.generate() leaked a
generator object (returning <generator ...> instead of a str) and the
_PromptLookupStrategy token loop crashed because token_gen.sample() returns a
(tensor, logprobs) tuple. Both are fixed so a real local model produces text.
"""

from __future__ import annotations

import os

import pytest

from distllm.core.coordinator import Coordinator, CoordinatorConfig


def test_prompt_lookup_strategy_generate_returns_str(monkeypatch, tmp_path):
    """The _PromptLookupStrategy must return a joined str, not a generator."""
    from distllm.core.inference_engine import InferenceEngine, _PromptLookupStrategy

    engine = InferenceEngine(model_name="dummy")
    # Minimal stubs so the strategy runs without a real model:
    engine.tokenizer = type(
        "T",
        (),
        {
            "encode": lambda *a, **kw: __import__("torch").tensor([[1, 2, 3]]),
            "decode": lambda *a, **kw: "tok,",
            "eos_token_id": 0,
        },
    )()
    import torch

    class _FakeModel:
        def __init__(self):
            self.logits = torch.ones((1, 10, 10))

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

        def __call__(self, *a, **k):
            class _O:
                logits = torch.ones((1, 10, 10))
            return _O()

    engine.local_partitioner = type("P", (), {"full_model": _FakeModel()})()
    engine._token_gen = type(
        "G",
        (),
        {
            "sample": lambda *a, **k: (torch.tensor([[5]]), None),  # (1,1) tensor
        },
    )()

    strat = _PromptLookupStrategy(engine)
    out = strat.generate("hi", 4, 0.0, 1.0, 0)
    assert isinstance(out, str), "generate() must return a string, not a generator"
    assert out == "tok,tok,tok,tok,"


def test_coordinator_has_agentic_router_attribute():
    """Coordinator must always define _agentic_router so generate() never
    AttributeErrors on the pre-routing check."""
    c = Coordinator(config=CoordinatorConfig(model_name="dummy"))
    assert c._agentic_router is None