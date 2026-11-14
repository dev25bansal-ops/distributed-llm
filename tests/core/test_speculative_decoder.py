"""Tests for speculative decoding using the current SpeculativeDecoder API."""

import importlib.util
import os

import pytest
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


def _always_logit(token_id: int):
    return torch.full((1, 1, 100), -10.0).scatter_(-1, torch.tensor([[[token_id]]]), 10.0)


def _identity_target(input_ids, **kwargs):
    """Mock target: logits[pos] predicts the token AT input[pos].

    Convention: ``target_logits[:, pos, :]`` is the distribution over the
    token at position ``pos+1`` — the same convention used by
    ``SpeculativeDecoder._verify_tokens``.  The last position predicts the
    last token itself so greedy verification of a final draft token works.
    """
    batch, seq_len = input_ids.shape
    logits = torch.full((batch, seq_len, 100), -10.0)
    for i in range(seq_len - 1):
        t = input_ids[0, i + 1].item()
        if t < 100:
            logits[0, i, t] = 10.0
    lt = input_ids[0, -1].item()
    if lt < 100:
        logits[0, -1, lt] = 10.0
    return logits


class TestSpeculativeDecoder:
    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.SD = cls.mod.SpeculativeDecoder

    def test_init_defaults(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: torch.randn(1, 1, 100))
        assert d._num_candidates == 5
        assert d._top_k == 20
        assert d._temperature == 1.0

    def test_init_custom(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: torch.randn(1, 1, 100),
                     num_candidates=7, top_k=10, temperature=0.5)
        assert d._num_candidates == 7
        assert d._top_k == 10
        assert d._temperature == 0.5

    def test_draft_forward_generates_correct_count(self):
        def draft_fn(input_ids, **kwargs):
            return _always_logit(42)
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=draft_fn,
                     num_candidates=4, temperature=0)
        prefix = torch.tensor([[1, 2, 3]])
        tokens, logprobs = d._draft_forward(prefix, num_tokens=3)
        assert tokens.shape == (1, 3)
        assert tokens[0, 0].item() == 42
        assert len(logprobs) == 3

    def test_verify_tokens_all_accepted_greedy(self):
        prefix = torch.tensor([[1, 2, 3]])
        draft = torch.tensor([[10, 20]])
        full = torch.cat([prefix, draft], dim=1)
        target_logits = _identity_target(full)
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     temperature=0)
        # draft_logprobs: q=1.0 for token 10 (softmax of _always_logit(10))
        draft_logprobs = [1.0, 1.0]
        accepted = d._verify_tokens(prefix, full, draft, target_logits,
                                     draft_logprobs=draft_logprobs)
        assert accepted == 2

    def test_verify_tokens_none_accepted_greedy(self):
        def target_fn(input_ids, **kw):
            logits = torch.full((1, input_ids.shape[1], 100), -10.0)
            logits[0, :, 99] = 10.0
            return logits
        prefix = torch.tensor([[1, 2, 3]])
        draft = torch.tensor([[5]])
        full = torch.cat([prefix, draft], dim=1)
        d = self.SD(target_forward=target_fn,
                     draft_forward=lambda x: _always_logit(5),
                     temperature=0)
        accepted = d._verify_tokens(prefix, full, draft, target_fn(full))
        assert accepted == 0

    def test_generate_full_acceptance(self):
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     num_candidates=5, temperature=0)
        output = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=4)
        assert output.shape[1] == 7
        assert all(output[0, 3 + i].item() == 10 for i in range(4))

    def test_generate_respects_max_new_tokens(self):
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     num_candidates=10, temperature=0)
        output = d.generate(torch.tensor([[1]]), max_new_tokens=2)
        assert output.shape[1] == 3

    def test_stats_tracking(self):
        d = self.SD(target_forward=_identity_target,
                     draft_forward=lambda x: _always_logit(10),
                     num_candidates=3, temperature=0)
        d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=5)
        stats = d.stats
        assert stats["draft_calls"] > 0
        assert stats["target_calls"] > 0
        assert stats["accepted"] > 0
        assert stats["total_proposed"] > 0

    def test_sample_greedy(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: torch.randn(1, 1, 100),
                     temperature=0)
        logits = torch.full((1, 100), -10.0)
        logits[0, 42] = 10.0
        token = d._sample(logits)
        assert token.item() == 42

    def test_sample_temperature(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: torch.randn(1, 1, 100))
        logits = torch.randn(1, 100)
        token = d._sample(logits)
        assert token.shape == (1, 1)
        assert 0 <= token.item() < 100

    def test_generate_with_kwargs(self):
        def target_fn(input_ids, **kwargs):
            assert "extra" in kwargs
            return _identity_target(input_ids)
        def draft_fn(input_ids, **kwargs):
            assert "extra" in kwargs
            return _always_logit(10)
        d = self.SD(target_forward=target_fn,
                     draft_forward=draft_fn,
                     temperature=0)
        output = d.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2, extra="value")
        assert output.shape[1] == 5

    def test_init_device_conversion(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: torch.randn(1, 1, 100),
                     device="cpu")
        assert str(d._device) == "cpu"

    def test_stats_before_generate(self):
        d = self.SD(target_forward=lambda x: torch.randn(1, 1, 100),
                     draft_forward=lambda x: torch.randn(1, 1, 100))
        stats = d.stats
        assert stats["draft_calls"] == 0
        assert stats["target_calls"] == 0
        assert stats["accepted"] == 0


class TestDraftProbabilityMatchesSampler:
    """Finding 257: q must come from the top-k masked distribution.

    ``_draft_forward`` sampled the token from the top-k masked
    (renormalized) softmax but recorded q from the UNMASKED softmax, so
    rejection sampling ran min(1, p/q) against the wrong q and
    over-accepted (top_k=1 drafts were accepted as if q were the unmasked
    mass).
    """

    @classmethod
    def setup_class(cls):
        cls.mod = _get_module()
        cls.SD = cls.mod.SpeculativeDecoder

    def test_recorded_q_is_one_when_top_k_is_one(self):
        VOC = 8
        # Unmasked softmax: token 7 has mass 0.3 (spread over the rest).
        def draft_fn(input_ids, **kwargs):
            out = torch.zeros(1, input_ids.shape[1], VOC)
            out[0, -1, 7] = torch.log(torch.tensor(0.3))
            for t in range(7):
                out[0, -1, t] = torch.log(torch.tensor(0.7 / 7))
            return out

        d = self.SD(target_forward=lambda x: torch.randn(1, 1, VOC),
                     draft_forward=draft_fn,
                     num_candidates=1, top_k=1, temperature=1.0, device="cpu")
        tokens, logprobs = d._draft_forward(torch.tensor([[1, 2, 3]]), 1)

        # top_k=1 => the sampler ALWAYS proposes the argmax (token 7), so the
        # true proposal probability is 1.0; the pre-fix code recorded 0.3.
        assert tokens[0, 0].item() == 7
        assert logprobs[0] == pytest.approx(1.0, abs=1e-5)

    def test_acceptance_rate_matches_exact_sampling(self):
        """With q=1 (top_k=1), acceptance must be ~p_target, not 1.0."""
        VOC = 64
        p_target = 0.7

        def draft_fn(input_ids, **kwargs):
            out = torch.zeros(1, input_ids.shape[1], VOC)
            out[0, -1, 7] = 4.0  # always drafted (top_k=1)
            return out

        def target_fn(input_ids, **kwargs):
            # Direct log-space construction: softmax gives p(token 7) = 0.7.
            out = torch.full(
                (1, input_ids.shape[1], VOC),
                float(torch.log(torch.tensor((1.0 - p_target) / (VOC - 1)))),
            )
            out[0, :, 7] = float(torch.log(torch.tensor(p_target)))
            return out

        d = self.SD(target_forward=target_fn, draft_forward=draft_fn,
                     num_candidates=1, top_k=1, temperature=1.0, device="cpu")
        torch.manual_seed(0)
        accepts = 0
        trials = 500
        prompt = torch.tensor([[1, 2, 3]])
        for _ in range(trials):
            draft_tokens, draft_logprobs = d._draft_forward(prompt, 1)
            full = torch.cat([prompt, draft_tokens], dim=1)
            logits = target_fn(full)
            if d._verify_tokens(
                prompt, full, draft_tokens, logits,
                draft_logprobs=draft_logprobs,
            ) == 1:
                accepts += 1
        rate = accepts / trials
        # Exact speculative sampling: acceptance ~ min(1, p/q) = 0.7.
        assert 0.55 <= rate <= 0.85, (
            f"acceptance rate {rate:.2f} should track p=0.70, not 1.0"
        )
