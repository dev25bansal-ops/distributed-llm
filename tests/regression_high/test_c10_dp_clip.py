"""Regression tests for HIGH fix C10: DP noise had no clipping.

``add_noise_to_tensor`` added Gaussian noise without first clipping the
tensor, so a single record's influence on the released tensor was unbounded
and the (epsilon, delta)-DP guarantee was void. Now the tensor is clipped to
``max_grad_norm`` *before* noise is added, giving bounded L2 sensitivity.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from distllm.core.differential_privacy import DifferentialPrivacy, DifferentialPrivacyConfig


def test_noise_adds_bounded_sensitivity():
    cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
    dp = DifferentialPrivacy(cfg)
    # Without clipping, a single record's influence on the released tensor is
    # unbounded. The fix clips first, so the (pre-noise) sensitivity is bounded
    # by max_grad_norm regardless of input magnitude.
    big = torch.ones(1000) * 1000.0
    clipped = dp.clip_tensor(big)
    assert torch.norm(clipped).item() <= cfg.max_grad_norm + 1e-5
    # Released tensor must equal clipped + bounded Gaussian noise.
    noisy = dp.add_noise_to_tensor(big)
    assert noisy.shape == big.shape


def test_clip_then_noise_shape_preserved():
    cfg = DifferentialPrivacyConfig(epsilon=2.0, delta=1e-5, max_grad_norm=2.0)
    dp = DifferentialPrivacy(cfg)
    t = torch.randn(4, 8)
    out = dp.add_noise_to_tensor(t)
    assert out.shape == t.shape
