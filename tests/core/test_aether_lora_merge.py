"""Regression tests for B2: LoRA merge must apply alpha/rank scaling.

``LoRAAdapterManager.merge()`` previously computed ``delta = B @ A^T`` with no
``alpha / rank`` scaling (the default path), producing wrong weights.  rank and
alpha are now persisted at ``create()`` and applied when no ``scale`` is given.
"""

import torch

from distllm.core.aether_federated import LoRAAdapterManager


def test_merge_applies_alpha_rank_scaling():
    mgr = LoRAAdapterManager(device="cpu")
    base = {"layer.0": torch.ones(4, 4)}
    adapter_id = mgr.create(base, rank=8, alpha=16.0)  # alpha/rank = 2.0

    # Overwrite the (zero-init) B matrix so delta is non-trivial.
    a = torch.randn(4, 8) * 0.1
    b = torch.randn(4, 8)
    mgr._adapters[adapter_id]["layer.0"] = (a, b)

    merged = mgr.merge(adapter_id, base)
    expected = base["layer.0"] + (b @ a.T) * (16.0 / 8.0)
    assert torch.allclose(merged["layer.0"], expected)


def test_merge_scale_override_wins():
    mgr = LoRAAdapterManager(device="cpu")
    base = {"layer.0": torch.ones(4, 4)}
    adapter_id = mgr.create(base, rank=8, alpha=16.0)

    a = torch.randn(4, 8) * 0.1
    b = torch.randn(4, 8)
    mgr._adapters[adapter_id]["layer.0"] = (a, b)

    merged = mgr.merge(adapter_id, base, scale=0.5)
    assert torch.allclose(merged["layer.0"], base["layer.0"] + (b @ a.T) * 0.5)


def test_merge_legacy_adapter_unscaled():
    """Adapters created before metadata was persisted fall back to scale 1.0."""
    mgr = LoRAAdapterManager(device="cpu")
    base = {"layer.0": torch.zeros(4, 4)}
    adapter_id = mgr.create(base, rank=8, alpha=16.0)
    mgr._adapter_meta.pop(adapter_id)  # simulate a legacy adapter

    a = torch.randn(4, 8) * 0.1
    b = torch.randn(4, 8)
    mgr._adapters[adapter_id]["layer.0"] = (a, b)

    merged = mgr.merge(adapter_id, base)
    assert torch.allclose(merged["layer.0"], base["layer.0"] + b @ a.T)


def test_unload_cleans_metadata():
    mgr = LoRAAdapterManager(device="cpu")
    adapter_id = mgr.create({"layer.0": torch.ones(4, 4)}, rank=8, alpha=16.0)
    assert adapter_id in mgr._adapter_meta
    assert mgr.unload(adapter_id)
    assert adapter_id not in mgr._adapter_meta
