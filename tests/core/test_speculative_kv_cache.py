"""Speculative decoder generation invariants.

Tests that the SpeculativeDecoder produces correct output shapes,
respects configuration, and correctly verifies draft tokens.
"""

import importlib.util
import os

import torch


def _get_module():
    path = os.path.join("src", "distllm", "core", "speculative_decoder.py")
    spec = importlib.util.spec_from_file_location("speculative_decoder", path)
    mod = importlib.util.module_from_spec(spec)
    import types
    mod.torch = __import__("torch")
    mod.F = __import__("torch.nn.functional", fromlist=["nn"])
    mod.Any = __import__("typing").Any
    mod.Callable = __import__("typing").Callable
    logger = types.ModuleType("logger")
    logger.info = lambda *a, **kw: None
    logger.warning = lambda *a, **kw: None
    mod.logger = logger
    spec.loader.exec_module(mod)
    return mod


def _always_logit(tok: int):
    return torch.full((1, 1, 100), -10.0).scatter_(-1, torch.tensor([[[tok]]]), 10.0)


def _identity_target(input_ids, **kwargs):
    b, s = input_ids.shape
    logits = torch.full((b, s, 100), -10.0)
    for i in range(s - 1):
        t = input_ids[0, i + 1].item()
        if t < 100:
            logits[0, i, t] = 10.0
    lt = input_ids[0, -1].item()
    if lt < 100:
        logits[0, -1, lt] = 10.0
    return logits


class TestDraftModel:
    """Draft model integration with SpeculativeDecoder."""

    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.SD = cls.mod.SpeculativeDecoder

    def test_draft_generates_correct_count(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: _always_logit(42),
                     num_candidates=3, temperature=0)
        tokens = d._draft_forward(torch.tensor([[1, 2, 3]]), num_tokens=3)
        assert len(tokens[0]) == 3
        assert tokens[0, 0].item() == 42

    def test_draft_empty_when_no_tokens(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: _always_logit(42),
                     num_candidates=3, temperature=0)
        tokens = d._draft_forward(torch.tensor([[1, 2, 3]]), num_tokens=0)
        assert tokens.shape[1] == 0

    def test_draft_increases_input_length(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: _always_logit(7),
                     num_candidates=4, temperature=0)
        prefix = torch.tensor([[5]])
        tokens = d._draft_forward(prefix, num_tokens=2)
        assert tokens.shape[1] == 2
        assert tokens[0, 0].item() == 7


class TestVerification:
    """Verification invariants."""

    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.SD = cls.mod.SpeculativeDecoder

    def test_accept_all_advances_all(self):
        prefix = torch.tensor([[1, 2, 3]])
        draft = torch.tensor([[10, 20, 30]])
        full = torch.cat([prefix, draft], dim=1)
        target_logits = _identity_target(full)
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     temperature=0)
        num_acc = d._verify_tokens(prefix, full, draft, target_logits)
        assert num_acc == 3

    def test_reject_first_token_advances_none(self):
        def target_fn(input_ids, **kw):
            logits = torch.full((1, input_ids.shape[1], 100), -10.0)
            logits[0, :, 99] = 10.0
            return logits
        prefix = torch.tensor([[1, 2, 3]])
        draft = torch.tensor([[5, 6]])
        full = torch.cat([prefix, draft], dim=1)
        d = self.SD(target_forward=target_fn,
                     draft_forward=lambda x: _always_logit(5),
                     temperature=0)
        num_acc = d._verify_tokens(prefix, full, draft, target_fn(full))
        assert num_acc == 0

    def test_reject_at_second_token(self):
        """Target accepts first draft token, rejects second."""
        def target_fn(input_ids, **kw):
            b, s = input_ids.shape
            logits = torch.full((b, s, 100), -10.0)
            logits[0, 2, 10] = 10.0  # position 2 predicts draft token 10
            logits[0, 3, 99] = 10.0  # position 3 predicts 99 (not draft 20)
            logits[0, -1, 0] = 10.0
            return logits
        prefix = torch.tensor([[1, 2, 3]])
        draft = torch.tensor([[10, 20]])
        full = torch.cat([prefix, draft], dim=1)
        d = self.SD(target_forward=target_fn,
                     draft_forward=lambda x: _always_logit(10),
                     temperature=0)
        num_acc = d._verify_tokens(prefix, full, draft, target_fn(full))
        assert num_acc == 1

    def test_greedy_accept_all_matching(self):
        prefix = torch.tensor([[1, 2, 3]])
        draft = torch.tensor([[10, 20, 30, 40]])
        full = torch.cat([prefix, draft], dim=1)
        target_logits = _identity_target(full)
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     temperature=0)
        num_acc = d._verify_tokens(prefix, full, draft, target_logits)
        assert num_acc == 4

    def test_kv_advance_equals_accepted_count(self):
        """After accepting K drafts, only K+1 new tokens are added."""
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     num_candidates=4, temperature=0)
        prompt = torch.tensor([[1, 2, 3]])
        output = d.generate(prompt, max_new_tokens=5)
        total_new = output.shape[1] - prompt.shape[1]
        assert total_new == 5


class TestGeneration:
    """Full generation loop invariants."""

    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.SD = cls.mod.SpeculativeDecoder

    def test_output_grows_by_max_new_tokens(self):
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     temperature=0)
        out = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=8)
        assert out.shape[1] == 11

    def test_output_contains_prompt(self):
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     temperature=0)
        out = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
        assert out[0, 0].item() == 1
        assert out[0, 1].item() == 2
        assert out[0, 2].item() == 3

    def test_different_temperatures_work(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 5, 100),
                     draft_forward=lambda x: torch.randn(1, 1, 100),
                     temperature=1.0)
        out = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
        assert out.shape[1] == 6
