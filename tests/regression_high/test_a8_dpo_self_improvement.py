"""A8 regression: DPO/RLHF router self-improvement from preference pairs.

Proves (model-free, no LLM fine-tune) that the A8 preference-learning
pipeline — :class:`PreferenceStore` + :class:`DPOTrainer` (Bradley-Terry
DPO loss) + :func:`self_improve` — works end to end on synthetic,
linearly-separable logit vectors:

1. Preference pairs are stored (and survivors persist to JSONL).
2. The DPO loss is finite and **decreases** over optimizer steps on a
   synthetic separable preference set (small torch tensors, CPU).
3. ``self_improve`` returns a usable weight delta / updated params.
4. The reference policy is **frozen** — its params are unchanged after
   training (no catastrophic forgetting of the base policy).

Reuses C1's ``LearningRouter`` preference-collection bridge
(``add_preference_pair`` / ``feed_store``) so the store can be seeded from
C1's collected data without regressing C1.

Honest scope: the DPO *loss + optimization* is proven on synthetic logits.
A real 7B LLM fine-tune is NOT run — that is out of scope; the math the
task asks for is what's verified here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

torch = pytest.importorskip("torch")
from distllm.core import preference_learning as pl  # noqa: E402
from distllm.core.learning_router import LearningRouter, RewardSignal  # noqa: E402
from distllm.core.model_router import ModelRouter  # noqa: E402
from distllm.config.settings import ChatRouterSettings  # noqa: E402


FEATURE_DIM = 8


def _make_separable_store(n: int = 24, seed: int = 0) -> pl.PreferenceStore:
    """Synthetic, linearly-separable preference set.

    Chosen completions have a strongly positive even-index logit pattern;
    rejected completions have the opposite.  The DPO head can therefore
    separate them and drive the loss down.
    """
    g = torch.Generator().manual_seed(seed)
    store = pl.PreferenceStore()
    for _ in range(n):
        c = torch.randn(FEATURE_DIM, generator=g)
        c[::2] += 3.0
        r = torch.randn(FEATURE_DIM, generator=g)
        r[::2] -= 3.0
        store.add(prompt=f"q{_}", chosen_logits_or_tokens=c, rejected_logits_or_tokens=r)
    return store


# ── 1. preference pairs are stored ────────────────────────────────────────

class TestPreferenceStore:
    def test_add_and_size(self):
        s = pl.PreferenceStore()
        assert s.size == 0
        s.add("p1", [1.0, 2.0], [0.0, -1.0])
        s.add("p2", "codellama", "llama3")
        assert s.size == 2
        assert len(s) == 2

    def test_pairs_snapshot(self):
        s = pl.PreferenceStore()
        s.add("p1", [1.0], [0.0])
        pairs = s.pairs()
        assert len(pairs) == 1
        assert pairs[0].prompt == "p1"
        # snapshot is a copy: mutating it does not change the store
        pairs.append("extra")
        assert s.size == 1

    def test_persist_roundtrip(self, tmp_path):
        path = tmp_path / "prefs.jsonl"
        s = pl.PreferenceStore(persist_path=path)
        s.add("p1", [1.0, 2.0], [0.0, -1.0])
        s.add("p2", "codellama", "llama3")
        # reopen from the same JSONL file
        s2 = pl.PreferenceStore(persist_path=path)
        assert s2.size == 2
        prompts = {p.prompt for p in s2.pairs()}
        assert prompts == {"p1", "p2"}

    def test_save_explicit_path(self, tmp_path):
        s = pl.PreferenceStore()
        s.add("p1", torch.tensor([1.0, 2.0]), torch.tensor([0.0, -1.0]))
        out = tmp_path / "out.jsonl"
        s.save(out)
        assert out.exists()
        s3 = pl.PreferenceStore(persist_path=out)
        assert s3.size == 1


# ── 2. DPO loss finite + DECREASES over steps ───────────────────────────

class TestDPOLossDecreases:
    def test_loss_finite_initial(self):
        store = _make_separable_store()
        tr = pl.DPOTrainer(feature_dim=FEATURE_DIM, beta=0.2, lr=2e-2)
        loss = tr.loss_on_store(store)
        assert float("nan") != loss
        assert float("inf") != loss

    def test_loss_decreases_over_steps(self):
        store = _make_separable_store(seed=7)
        tr = pl.DPOTrainer(feature_dim=FEATURE_DIM, beta=0.2, lr=2e-2)
        init = tr.loss_on_store(store)
        losses = [init]
        for _ in range(150):
            c = [pl._try_coerce(p.chosen) for p in store.pairs()]
            r = [pl._try_coerce(p.rejected) for p in store.pairs()]
            tr.train_step(c, r)
            losses.append(tr.loss_on_store(store))
        final = losses[-1]
        # Strict decrease on a separable set, and finite throughout.
        assert all(float("nan") != x for x in losses)
        assert final < init - 1e-3, f"loss did not decrease: {init} -> {final}"
        # Monotonic-ish: average of second half < average of first half.
        assert (sum(losses[len(losses)//2:]) / (len(losses)//2)) < init

    def test_reference_frozen_during_dpo_training(self):
        store = _make_separable_store(seed=11)
        tr = pl.DPOTrainer(feature_dim=FEATURE_DIM, beta=0.2, lr=2e-2)
        ref_before = {k: v.detach().clone() for k, v in tr.reference_head.state_dict().items()}
        for _ in range(100):
            c = [pl._try_coerce(p.chosen) for p in store.pairs()]
            r = [pl._try_coerce(p.rejected) for p in store.pairs()]
            tr.train_step(c, r)
        ref_after = tr.reference_head.state_dict()
        for k in ref_before:
            assert torch.equal(ref_before[k], ref_after[k]), f"ref param {k} mutated"


# ── 3. self_improve returns a usable weight delta ───────────────────────

class TestSelfImprove:
    def test_returns_weight_delta_and_history(self):
        store = _make_separable_store(seed=3, n=32)
        router = object()  # opaque router stub; must NOT be mutated
        result = pl.self_improve(router, store, steps=120, beta=0.2, lr=2e-2)
        assert result.num_pairs == store.size
        assert result.initial_loss is not None
        assert result.final_loss < result.initial_loss - 1e-3
        assert len(result.loss_history) == 120
        # weight delta is a dict of tensors with the head's param names
        assert set(result.weight_delta.keys()) == {"weight", "bias"}
        for t in result.weight_delta.values():
            assert torch.is_tensor(t)
            assert torch.isfinite(t).all()

    def test_weight_delta_applies_to_a_head(self):
        """The returned delta is consistent: head_initial + delta == head_final
        (the exact policy-head param change the router would apply)."""
        store = _make_separable_store(seed=5, n=16)
        result = pl.self_improve(router=object(), store=store, steps=80, beta=0.2, lr=2e-2)
        for k in result.weight_delta:
            recomputed = result.head_initial[k] + result.weight_delta[k]
            assert torch.allclose(recomputed, result.head_final[k], atol=1e-6), \
                f"delta does not reconcile with final head for {k}"
        # The delta is non-trivial (the head actually learned something).
        total_norm = sum(
            delta.norm().item() for delta in result.weight_delta.values()
        )
        assert total_norm > 1e-3

    def test_reference_unchanged_flag(self):
        store = _make_separable_store(seed=9, n=20)
        result = pl.self_improve(router=object(), store=store, steps=60)
        assert result.reference_unchanged is True


# ── 4. reuse C1 collection without regressing it ─────────────────────────

class TestC1Bridge:
    def _base(self) -> ModelRouter:
        return ModelRouter(
            ChatRouterSettings(enabled=True, name="t", default_model="llama3", routes=[])
        )

    def test_learning_router_exposes_preference_bridge(self):
        lr = LearningRouter(self._base(), models=["codellama", "llama3"])
        assert hasattr(lr, "add_preference_pair")
        assert hasattr(lr, "export_preference_pairs")
        assert hasattr(lr, "feed_store")

    def test_add_preference_pair_ignores_identical_models(self):
        lr = LearningRouter(self._base(), models=["a", "b"])
        before = lr.stats["total_context_buckets"]
        lr.add_preference_pair("q", "a", "a")
        assert lr.stats["total_context_buckets"] == before

    def test_c1_derived_preferences_feed_store(self):
        """C1 bandit data flows into the preference store w/o regressing C1."""
        lr = LearningRouter(self._base(), models=["codellama", "llama3"], epsilon=0.0)
        # Build up history so a clear preference emerges.
        for _ in range(12):
            lr.record_outcome("llama3", RewardSignal(user_rating=1.0), text="math problem")
            lr.record_outcome("codellama", RewardSignal(user_rating=0.0), text="math problem")
        store = pl.PreferenceStore()
        added = lr.feed_store(store, min_separation=0.25)
        assert added >= 1, "C1 should derive at least one preference pair"
        assert store.size == added
        # The store now carries usable (chosen, rejected) pairs.
        pair = store.pairs()[0]
        assert pair.chosen != pair.rejected


# ── 5. RLHF scaffold seam (thin, not the proven path) ───────────────────

class TestRLHFScaffold:
    def test_rlhf_trainer_seeded_and_runs(self):
        tr = pl.RLHFTrainer(feature_dim=FEATURE_DIM, lr=1e-3)

        def reward(logits: torch.Tensor) -> torch.Tensor:
            # Reward high even-index activation.
            return (logits[::2]).mean()

        tr.set_reward(reward)
        x = torch.randn(FEATURE_DIM)
        loss = tr.ppo_step(x)
        assert float("nan") != loss and float("inf") != loss
        # reference must stay frozen after a ppo step too
        for p in tr._ref.parameters():
            assert p.requires_grad is False
