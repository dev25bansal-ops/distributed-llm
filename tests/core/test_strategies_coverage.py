"""Coverage for the execution-strategy modules (M-05 decomposition).

``core/strategies/`` is 1,010 lines that ``inference_engine`` imports and calls
on every generation, but no test imported or executed any of it. The modules
define an ``EngineContext`` protocol for exactly this purpose — "any object
providing these attributes can host the strategies, which makes them testable
without a full InferenceEngine" — so the seam is exercised here with small
stand-ins rather than a real engine.

These tests do not assert generation quality; they pin the wiring: that each
strategy forwards the arguments it promises, and that the local streaming path
respects stop tokens and decodes what the engine yields.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="strategy modules import torch")

from distllm.core.strategies.local import LocalStrategy  # noqa: E402
from distllm.core.strategies.prompt_lookup import PromptLookupStrategy  # noqa: E402
from distllm.core.strategies.speculative import SpeculativeStrategy  # noqa: E402


class _FakeTokenizer:
    """Minimal tokenizer stand-in: encode -> a fixed tensor, decode -> str."""

    eos_token_id = 2

    def __init__(self, ids=None):
        self._ids = ids if ids is not None else [11, 12, 13]
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, text, return_tensors="pt"):
        self.encode_calls.append((text, return_tensors))
        return torch.tensor([self._ids], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        self.decode_calls.append((list(ids), skip_special_tokens))
        return "|".join(str(i) for i in ids)


class _FakePartitioner:
    def __init__(self):
        self.full_model = torch.nn.Linear(2, 2)


class _FakeEngine:
    """Implements just the EngineContext surface LocalStrategy touches."""

    def __init__(self, tokens=(5, 6)):
        self.tokenizer = _FakeTokenizer()
        self.local_partitioner = _FakePartitioner()
        self._tokens = tokens
        self.local_calls = []
        self.iter_calls = []
        self._stream_pool = None
        self._node_order_property = None

    def _generate_local(self, prompt, max_new_tokens, temperature, top_p, top_k, **kw):
        self.local_calls.append(
            dict(prompt=prompt, max_new=max_new_tokens, temperature=temperature,
                 top_p=top_p, top_k=top_k, **kw))
        return "generated"

    def _iter_local_tokens(self, input_ids, max_new_tokens, **kw):
        self.iter_calls.append(dict(input_ids=input_ids, max_new=max_new_tokens, **kw))
        for t in self._tokens:
            yield t


class TestLocalStrategyGenerate:
    def test_forwards_sampling_parameters(self):
        eng = _FakeEngine()
        out = LocalStrategy(eng).generate("hi", 8, 0.7, 0.9, 40)
        assert out == "generated"
        call = eng.local_calls[0]
        assert call["prompt"] == "hi"
        assert call["max_new"] == 8
        assert call["temperature"] == 0.7
        assert call["top_p"] == 0.9
        assert call["top_k"] == 40

    def test_forwards_optional_kwargs_it_recognises(self):
        eng = _FakeEngine()
        LocalStrategy(eng).generate(
            "hi", 4, 0.0, 1.0, -1,
            logit_bias={"a": 1}, stop_tokens=[9], constraint="json",
            ignored_kwarg="dropped",
        )
        call = eng.local_calls[0]
        assert call["logit_bias"] == {"a": 1}
        assert call["stop_tokens"] == [9]
        assert call["constraint"] == "json"
        # Unknown kwargs are not forwarded — the strategy's signature is the contract.
        assert "ignored_kwarg" not in call


class TestLocalStrategyStream:
    def test_yields_one_decoded_piece_per_token(self):
        eng = _FakeEngine(tokens=(5, 6))
        pieces = list(LocalStrategy(eng).generate_stream("hi", 2, 0.0, 1.0, -1))
        assert pieces == ["5", "6"]

    def test_encodes_with_tensors_and_moves_to_model_device(self):
        eng = _FakeEngine(tokens=(5,))
        list(LocalStrategy(eng).generate_stream("hello", 1, 0.0, 1.0, -1))
        assert eng.tokenizer.encode_calls[0] == ("hello", "pt")
        assert eng.iter_calls[0]["input_ids"].device.type in ("cpu", "cuda")

    def test_stop_token_ids_include_eos_and_explicit_stops(self):
        eng = _FakeEngine(tokens=(5,))
        list(LocalStrategy(eng).generate_stream("hi", 1, 0.0, 1.0, -1, stop_tokens=[9, 10]))
        stop_ids = eng.iter_calls[0]["stop_token_ids"]
        assert 9 in stop_ids and 10 in stop_ids
        assert _FakeTokenizer.eos_token_id in stop_ids

    def test_no_eos_token_id_does_not_crash(self):
        eng = _FakeEngine(tokens=(5,))
        eng.tokenizer.eos_token_id = None
        list(LocalStrategy(eng).generate_stream("hi", 1, 0.0, 1.0, -1, stop_tokens=[7]))
        assert eng.iter_calls[0]["stop_token_ids"] == {7}

    def test_empty_token_stream_yields_nothing(self):
        eng = _FakeEngine(tokens=())
        assert list(LocalStrategy(eng).generate_stream("hi", 1, 0.0, 1.0, -1)) == []

    def test_stream_forwards_sampling_and_constraint(self):
        eng = _FakeEngine(tokens=(5,))
        list(LocalStrategy(eng).generate_stream(
            "hi", 3, 0.4, 0.8, 20, logit_bias={"x": 2}, constraint="cfg"))
        call = eng.iter_calls[0]
        assert call["temperature"] == 0.4
        assert call["top_p"] == 0.8
        assert call["top_k"] == 20
        assert call["logit_bias"] == {"x": 2}
        assert call["constraint"] == "cfg"
        assert call["max_new"] == 3


class TestStrategyModulesImportAndExpose:
    """The modules are re-exported by inference_engine for backward compat."""

    def test_all_five_strategies_export_from_package(self):
        from distllm.core import strategies

        for name in ("LocalStrategy", "SpeculativeStrategy",
                     "DistributedStrategy", "DistributedSpeculativeStrategy",
                     "PromptLookupStrategy"):
            assert hasattr(strategies, name), f"{name} missing from package"
            assert hasattr(strategies, "_" + name), f"underscore alias {name} missing"

    def test_inference_engine_re_exports_same_classes(self):
        from distllm.core import inference_engine, strategies

        assert inference_engine._LocalStrategy is strategies.LocalStrategy
        assert inference_engine._PromptLookupStrategy is strategies.PromptLookupStrategy

    def test_strategies_construct_against_a_context_like_object(self):
        # The EngineContext protocol exists so a stand-in can host them.
        for cls in (LocalStrategy, SpeculativeStrategy, PromptLookupStrategy):
            inst = cls(_FakeEngine())
            assert hasattr(inst, "generate")


class TestDistributedStrategies:
    """The distributed variants forward to the engine and fail closed.

    Their streaming paths drive a real thread pool and node KV caches, which
    cannot be exercised without a cluster. What is testable here is the argument
    forwarding and the guard that refuses to stream with no nodes registered --
    the error a user actually hits when the pipeline is empty.
    """

    def test_distributed_generate_forwards_request_id_and_constraint(self):
        from distllm.core.strategies.distributed import DistributedStrategy

        class _Eng:
            tokenizer = _FakeTokenizer()
            _node_order = ["n1"]

            def __init__(self):
                self.calls = []

            def _generate_distributed(self, prompt, mx, temp, tp, tk, **kw):
                self.calls.append(dict(prompt=prompt, max_new=mx, **kw))
                return "dist-out"

        eng = _Eng()
        out = DistributedStrategy(eng).generate(
            "p", 5, 0.2, 0.9, 10, request_id="r1", constraint="j")
        assert out == "dist-out"
        assert eng.calls[0]["request_id"] == "r1"
        assert eng.calls[0]["constraint"] == "j"
        assert eng.calls[0]["max_new"] == 5

    def test_distributed_stream_without_nodes_raises_config_error(self):
        from distllm.core.strategies.distributed import DistributedStrategy
        from distllm.errors import ConfigError

        class _Eng:
            tokenizer = _FakeTokenizer()
            _node_order = []

        with pytest.raises(ConfigError):
            list(DistributedStrategy(_Eng()).generate_stream("p", 4, 0.0, 1.0, -1))

    def test_distributed_spec_generate_forwards_all_sampling_params(self):
        from distllm.core.strategies.distributed_spec import (
            DistributedSpeculativeStrategy,
        )

        class _Eng:
            tokenizer = _FakeTokenizer()
            _node_order = ["n1"]

            def __init__(self):
                self.calls = []

            def _generate_distributed_speculative(self, prompt, mx, temp, tp, tk):
                self.calls.append((prompt, mx, temp, tp, tk))
                return "spec-out"

        eng = _Eng()
        out = DistributedSpeculativeStrategy(eng).generate("q", 7, 0.3, 0.85, 25)
        assert out == "spec-out"
        assert eng.calls[0] == ("q", 7, 0.3, 0.85, 25)
