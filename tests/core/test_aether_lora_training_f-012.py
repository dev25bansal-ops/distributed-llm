"""Regression tests for F-012: the LoRA path must actually train the adapter.

``Aether.start_finetuning(lora_config=...)`` used to create a LoRA adapter
but then train the FULL base weights; the adapter's random ``A`` / zero ``B``
were never updated, so ``merge()`` added ``(alpha/rank) * (0 @ A^T) == 0`` and
the returned weights were identical to plain full-weight training while
reporting ``used_lora=True``.

The trainer now runs federated rounds over composed base+LoRA effective
weights, converts aggregated gradients into LoRA parameter gradients, and
writes the learned A/B back into the adapter before merge.
"""

import pytest
import torch

from distllm.core.aether_federated import (
    Aether,
    FederatedConfig,
    FederatedTrainer,
    LoRAAdapterManager,
    LoRAConfig,
)

torch.manual_seed(0)


def _const_grad_fn(coef: float = 0.5):
    """Deterministic local_train_fn: constant non-zero grads and fixed loss."""

    def fn(model_weights, shard, local_epochs=1, learning_rate=1e-4):
        return [torch.full_like(w, coef) for w in model_weights.values()], 1.0

    return fn


# ---------------------------------------------------------------------------
# LoRAAdapterManager.update / adapter_scales (new write-back API)
# ---------------------------------------------------------------------------


def test_adapter_update_persists_trained_matrices():
    mgr = LoRAAdapterManager(device="cpu")
    base = {"w": torch.ones(3, 4)}
    aid = mgr.create(base, rank=2, alpha=4.0)
    a0, b0 = mgr.load(aid)["w"]

    mgr.update(aid, {"w": (a0 + 1.0, b0 + 2.0)})
    a1, b1 = mgr.load(aid)["w"]
    assert torch.allclose(a1, a0 + 1.0)
    assert torch.allclose(b1, b0 + 2.0)

    assert mgr.adapter_scales(aid) == {"w": 4.0 / 2.0}


def test_adapter_update_validates_keys_shapes_and_existence():
    mgr = LoRAAdapterManager(device="cpu")
    base = {"w": torch.ones(3, 4)}
    aid = mgr.create(base, rank=2, alpha=4.0)
    a, b = mgr.load(aid)["w"]

    with pytest.raises(ValueError, match="not part of adapter"):
        mgr.update(aid, {"nope": (a, b)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        mgr.update(aid, {"w": (torch.zeros(1, 1), b)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        mgr.update(aid, {"w": (a, torch.zeros(9, 9))})

    mgr.unload(aid)
    with pytest.raises(KeyError):
        mgr.update(aid, {"w": (a, b)})


# ---------------------------------------------------------------------------
# FederatedTrainer.train_lora -- exact SGD math on the adapter parameters
# ---------------------------------------------------------------------------


def test_train_lora_updates_only_adapter_and_matches_reference_math():
    torch.manual_seed(7)
    base = {"w": torch.ones(3, 4)}
    mgr = LoRAAdapterManager(device="cpu")
    aid = mgr.create(base, rank=2, alpha=4.0)  # scale s = alpha/rank = 2.0
    a0, b0 = mgr.load(aid)["w"]

    trainer = FederatedTrainer(
        config=FederatedConfig(num_rounds=2, learning_rate=0.1)
    )
    out = trainer.train_lora(
        adapter_manager=mgr,
        adapter_id=aid,
        base_model=base,
        dataset=["shard-1", "shard-2"],
        local_train_fn=_const_grad_fn(),
        num_rounds=2,
    )

    # Base weights are frozen: what comes back is untouched.
    assert torch.equal(out["w"], base["w"])

    # Reference: two rounds of SGD on (A, B) with constant grad G = 0.5.
    # Round 1: grad_B = s*(G@A0); grad_A = s*(G^T@B0) = 0 because B is zero-init.
    # Round 2 uses B1 for both gradients.
    s, lr = 2.0, 0.1
    g = torch.full((3, 4), 0.5)
    b1 = b0 - lr * s * (g @ a0)
    b2 = b1 - lr * s * (g @ a0)
    a2 = a0 - lr * s * (g.T @ b1)

    a_t, b_t = mgr.load(aid)["w"]
    assert torch.allclose(b_t, b2, atol=1e-6)
    assert torch.allclose(a_t, a2, atol=1e-6)
    # The adapter really moved away from its zero-init B.
    assert not torch.equal(b_t, b0)


def test_train_lora_records_round_stats():
    base = {"w": torch.ones(3, 4)}
    mgr = LoRAAdapterManager(device="cpu")
    aid = mgr.create(base, rank=2, alpha=4.0)
    trainer = FederatedTrainer(
        config=FederatedConfig(num_rounds=3, learning_rate=0.05)
    )
    trainer.train_lora(
        adapter_manager=mgr,
        adapter_id=aid,
        base_model=base,
        dataset=["s1", "s2"],
        local_train_fn=_const_grad_fn(),
        num_rounds=3,
    )
    stats = trainer.round_stats
    assert len(stats) == 3
    assert [st["round"] for st in stats] == [1, 2, 3]
    assert all(st["participants"] == 2 for st in stats)
    assert all(st["avg_loss"] == 1.0 for st in stats)
    assert trainer.best_loss == 1.0


def test_train_lora_rejects_too_few_clients():
    base = {"w": torch.ones(3, 4)}
    mgr = LoRAAdapterManager(device="cpu")
    aid = mgr.create(base, rank=2, alpha=4.0)
    trainer = FederatedTrainer(
        config=FederatedConfig(num_rounds=1, min_clients=3)
    )
    with pytest.raises(ValueError, match="at least 3 clients"):
        trainer.train_lora(
            adapter_manager=mgr,
            adapter_id=aid,
            base_model=base,
            dataset=["only-one"],
            local_train_fn=_const_grad_fn(),
        )


def test_train_lora_rejects_empty_adapter():
    base = {"w": torch.ones(3, 4)}
    mgr = LoRAAdapterManager(device="cpu")
    # target_modules matches nothing -> adapter has zero layers.
    aid = mgr.create(base, rank=2, alpha=4.0, target_modules=["does-not-match"])
    trainer = FederatedTrainer(config=FederatedConfig(num_rounds=1))
    with pytest.raises(ValueError, match="adapts no layers"):
        trainer.train_lora(
            adapter_manager=mgr,
            adapter_id=aid,
            base_model=base,
            dataset=["s1", "s2"],
            local_train_fn=_const_grad_fn(),
        )


# ---------------------------------------------------------------------------
# End-to-end via Aether.start_finetuning(lora_config=...)
# ---------------------------------------------------------------------------


def test_start_finetuning_lora_adapts_target_layer_only():
    torch.manual_seed(11)
    base = {
        "fc.weight": torch.randn(3, 5),
        "bias": torch.randn(3),  # 1-D: never adapted
        "other.weight": torch.randn(2, 2),  # excluded via target_modules
    }
    lora_cfg = LoRAConfig(rank=2, alpha=4.0, target_modules=["fc"])
    aether = Aether(config=FederatedConfig(num_rounds=2, learning_rate=0.1))

    result = aether.start_finetuning(
        base_model=base,
        dataset=["d1", "d2"],
        local_train_fn=_const_grad_fn(),
        num_rounds=2,
        lora_config=lora_cfg,
    )

    assert result["used_lora"] is True

    final = aether.get_global_model()
    # The adapted layer moved away from the base (training actually happened).
    delta = final["fc.weight"] - base["fc.weight"]
    assert delta.abs().max() > 0
    # Frozen layers are byte-identical to the base.
    assert torch.equal(final["bias"], base["bias"])
    assert torch.equal(final["other.weight"], base["other.weight"])
    # The delta is low-rank (<= r): it came from B @ A^T, not full training.
    # Explicit atol: float32 rounding noise must not count as rank.
    assert torch.linalg.matrix_rank(delta, atol=1e-4).item() <= lora_cfg.rank


def test_start_finetuning_lora_final_weights_match_trained_adapter_merge():
    """The merged output must equal base + s*(B@A^T) of the TRAINED adapter."""
    torch.manual_seed(123)
    base = {"fc.weight": torch.randn(3, 5)}
    rank, alpha, lr, rounds = 2, 4.0, 0.1, 2
    s = alpha / rank

    aether = Aether(config=FederatedConfig(num_rounds=rounds, learning_rate=lr))
    result = aether.start_finetuning(
        base_model=base,
        dataset=["d1", "d2"],
        local_train_fn=_const_grad_fn(),
        num_rounds=rounds,
        lora_config=LoRAConfig(rank=rank, alpha=alpha, target_modules=["fc"]),
    )
    assert result["used_lora"] is True
    final = aether.get_global_model()

    # Reproduce create()'s random init exactly: re-seeding and replaying the
    # base-weight draw aligns the global RNG with the state create() saw, so
    # re-drawing randn(in, r)*0.02 yields the same A0.
    torch.manual_seed(123)
    _ = torch.randn(3, 5)  # same draw that built base["fc.weight"]
    a0 = torch.randn(5, rank) * 0.02
    b0 = torch.zeros(3, rank)

    g = torch.full((3, 5), 0.5)  # fedavg of two identical constant grads
    b1 = b0 - lr * s * (g @ a0)
    b2 = b1 - lr * s * (g @ a0)
    a1 = a0 - lr * s * (g.T @ b0)
    a2 = a1 - lr * s * (g.T @ b1)

    expected_fc = base["fc.weight"] + s * (b2 @ a2.T)
    assert torch.allclose(final["fc.weight"], expected_fc, atol=1e-5)


def test_start_finetuning_without_lora_still_full_weight_training():
    """Regression guard: the non-LoRA path keeps updating every layer."""
    torch.manual_seed(21)
    base = {
        "fc.weight": torch.randn(3, 5),
        "bias": torch.randn(3),
    }
    aether = Aether(config=FederatedConfig(num_rounds=1, learning_rate=0.1))
    result = aether.start_finetuning(
        base_model=base,
        dataset=["d1", "d2"],
        local_train_fn=_const_grad_fn(),
        num_rounds=1,
    )

    assert result["used_lora"] is False
    final = aether.get_global_model()
    # Full-weight path applies grads to ALL layers: W <- W - lr * mean(G).
    expected_fc = base["fc.weight"] - 0.1 * torch.full_like(base["fc.weight"], 0.5)
    expected_bias = base["bias"] - 0.1 * torch.full_like(base["bias"], 0.5)
    assert torch.allclose(final["fc.weight"], expected_fc, atol=1e-6)
    assert torch.allclose(final["bias"], expected_bias, atol=1e-6)
