"""Tests for DifferentialPrivacy — calibrated noise and PII anonymization.

Covers:
- DifferentialPrivacyConfig: epsilon, delta, sigma computation
- DifferentialPrivacy: add_noise_to_tensor, add_noise_to_kv_cache,
  clip_tensor, privacy_budget_used
- InputAnonymizer: anonymize, has_pii
"""

from __future__ import annotations

import math

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_dp = load_module("distllm/core/differential_privacy.py")
DifferentialPrivacy = _dp.DifferentialPrivacy
DifferentialPrivacyConfig = _dp.DifferentialPrivacyConfig
InputAnonymizer = _dp.InputAnonymizer


# ── DifferentialPrivacyConfig ─────────────────────────────────────────────────


class TestDifferentialPrivacyConfig:
    def test_default_values(self):
        cfg = DifferentialPrivacyConfig()
        assert cfg.epsilon == 1.0
        assert cfg.delta == 1e-5
        assert cfg.max_grad_norm == 1.0
        assert cfg.noise_multiplier == 0.0

    def test_sigma_auto_computed(self):
        cfg = DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5)
        expected = math.sqrt(2 * math.log(1.25 / 1e-5))
        assert cfg.sigma == pytest.approx(expected)

    def test_sigma_with_noise_multiplier(self):
        cfg = DifferentialPrivacyConfig(
            epsilon=1.0, delta=1e-5,
            max_grad_norm=2.0, noise_multiplier=0.5,
        )
        # sigma = max_grad_norm * noise_multiplier = 2.0 * 0.5
        assert cfg.sigma == 1.0

    def test_sigma_scales_with_epsilon(self):
        low_eps = DifferentialPrivacyConfig(epsilon=0.1).sigma
        high_eps = DifferentialPrivacyConfig(epsilon=10.0).sigma
        assert low_eps > high_eps  # Smaller epsilon = more noise

    def test_custom_values(self):
        cfg = DifferentialPrivacyConfig(epsilon=4.0, delta=1e-6, max_grad_norm=0.5)
        assert cfg.epsilon == 4.0
        assert cfg.delta == 1e-6
        assert cfg.max_grad_norm == 0.5


# ── DifferentialPrivacy ───────────────────────────────────────────────────────


class TestDifferentialPrivacyConstruction:
    def test_default_config(self):
        dp = DifferentialPrivacy()
        assert dp._config.epsilon == 1.0

    def test_custom_config(self):
        cfg = DifferentialPrivacyConfig(epsilon=4.0, delta=1e-6)
        dp = DifferentialPrivacy(config=cfg)
        assert dp._config.epsilon == 4.0


class TestDifferentialPrivacyAddNoise:
    def test_add_noise_to_tensor_changes_values(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        dp = DifferentialPrivacy()
        tensor = torch.ones(10)
        noisy = dp.add_noise_to_tensor(tensor)
        assert noisy.shape == tensor.shape
        # Should be different from original (with high probability)
        assert not torch.allclose(noisy, tensor)
        # Original should not be modified
        assert torch.allclose(tensor, torch.ones(10))

    def test_add_noise_to_tensor_zero_sigma(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        cfg = DifferentialPrivacyConfig(epsilon=1000.0, noise_multiplier=0.0)
        dp = DifferentialPrivacy(config=cfg)
        tensor = torch.ones(10)
        noisy = dp.add_noise_to_tensor(tensor)
        # Clip-then-noise: norm sqrt(10) > max_grad_norm 1.0 is clipped first,
        # then sigma ~0.0048 noise is added — tolerance must exceed sigma.
        clipped = dp.clip_tensor(tensor)
        assert torch.allclose(noisy, clipped, atol=0.05)

    def test_add_noise_to_kv_cache(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        dp = DifferentialPrivacy()
        kv_cache = [(torch.ones(4, 8), torch.ones(4, 8)) for _ in range(3)]
        noisy_cache = dp.add_noise_to_kv_cache(kv_cache)
        assert len(noisy_cache) == 3
        for (nk, nv), (ok, ov) in zip(noisy_cache, kv_cache):
            assert nk.shape == ok.shape
            assert nv.shape == ov.shape
            # Noise added (with high probability)
            assert not torch.allclose(nk, ok)
            assert not torch.allclose(nv, ov)

    def test_add_noise_to_kv_cache_empty(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        dp = DifferentialPrivacy()
        noisy_cache = dp.add_noise_to_kv_cache([])
        assert noisy_cache == []


class TestDifferentialPrivacyClipThenNoise:
    """Regression S2: the noise path must L2-clip to max_grad_norm BEFORE noise.

    Before the fix, add_noise_to_tensor added Gaussian noise to the raw
    tensor without clipping, so a tensor with norm >> max_grad_norm preserved
    its unbounded norm in the output and the Gaussian (ε, δ) guarantee did
    not hold.  clip_tensor existed but was never called from the noise path.
    """

    def test_add_noise_to_tensor_clips_before_noise(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        cfg = DifferentialPrivacyConfig(max_grad_norm=1.0, noise_multiplier=0.05)
        dp = DifferentialPrivacy(config=cfg)
        # L2 norm of ones(64) == 8.0, well above max_grad_norm == 1.0.
        tensor = torch.ones(64)
        noisy = dp.add_noise_to_tensor(tensor)

        # Clip-then-noise: the effective input is L2-clipped to max_grad_norm,
        # so the output norm ≈ 1.0 (clipped) + small noise (0.05 * sqrt(64) ≈ 0.4).
        # Without clipping the output norm would be ≈ 8.0.  Fails pre-fix.
        assert noisy.norm().item() < 3.0, (
            "output L2 norm not bounded by max_grad_norm: "
            "tensor was not clipped before noise was added"
        )

        # The noisy output is statistically different from the input: the
        # per-element differences are far from zero (dominated by the clip,
        # plus the calibrated noise).
        diff = noisy - tensor
        assert diff.norm().item() > 1.0, "noisy output is not different from input"

    def test_add_noise_to_kv_cache_clips_before_noise(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        cfg = DifferentialPrivacyConfig(max_grad_norm=1.0, noise_multiplier=0.05)
        dp = DifferentialPrivacy(config=cfg)
        # Each 4x8 ones tensor has L2 norm == sqrt(32) ≈ 5.66 >> max_grad_norm.
        kv_cache = [(torch.ones(4, 8), torch.ones(4, 8)) for _ in range(3)]
        noisy_cache = dp.add_noise_to_kv_cache(kv_cache)

        for layer_idx, ((nk, nv), (ok, ov)) in enumerate(zip(noisy_cache, kv_cache)):
            # Clip-then-noise: output norm bounded by max_grad_norm + noise.
            assert nk.norm().item() < 3.0, f"key of layer {layer_idx} not clipped before noise"
            assert nv.norm().item() < 3.0, f"value of layer {layer_idx} not clipped before noise"
            # Statistically different from the raw input (noise + clip applied).
            assert not torch.allclose(nk, ok)
            assert not torch.allclose(nv, ov)


class TestDifferentialPrivacyClipTensor:
    def test_clip_tensor_under_norm_unchanged(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        dp = DifferentialPrivacy(DifferentialPrivacyConfig(max_grad_norm=10.0))
        tensor = torch.ones(10)  # norm = sqrt(10) ~ 3.16
        clipped = dp.clip_tensor(tensor)
        # Norm under threshold, should be clone
        assert torch.allclose(clipped, tensor)

    def test_clip_tensor_over_norm(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        dp = DifferentialPrivacy(DifferentialPrivacyConfig(max_grad_norm=1.0))
        tensor = torch.ones(10) * 10  # norm > 1
        clipped = dp.clip_tensor(tensor)
        assert clipped.norm().item() <= 1.0

    def test_clip_tensor_immutable(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        dp = DifferentialPrivacy()
        tensor = torch.ones(10) * 5
        original = tensor.clone()
        dp.clip_tensor(tensor)
        assert torch.allclose(tensor, original)


class TestDifferentialPrivacyBudget:
    def test_privacy_budget_zero_queries(self):
        dp = DifferentialPrivacy()
        budget = dp.privacy_budget_used(0)
        assert budget["num_queries"] == 0
        assert budget["total_epsilon"] == 0.0

    def test_privacy_budget_positive_queries(self):
        dp = DifferentialPrivacy(DifferentialPrivacyConfig(epsilon=1.0, delta=1e-5))
        budget = dp.privacy_budget_used(10)
        assert budget["num_queries"] == 10
        assert budget["total_epsilon"] > 0
        assert "sigma" in budget

    def test_privacy_budget_increases_with_queries(self):
        dp = DifferentialPrivacy()
        b1 = dp.privacy_budget_used(1)["total_epsilon"]
        b10 = dp.privacy_budget_used(10)["total_epsilon"]
        assert b10 > b1

    def test_negative_queries_returns_zero(self):
        dp = DifferentialPrivacy()
        budget = dp.privacy_budget_used(-1)
        assert budget["num_queries"] == 0
        assert budget["total_epsilon"] == 0.0


# ── InputAnonymizer ───────────────────────────────────────────────────────────


class TestInputAnonymizer:
    def test_anonymize_email(self):
        text = "Contact me at user@example.com for details"
        result = InputAnonymizer.anonymize(text)
        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_anonymize_phone(self):
        text = "Call me at 555-123-4567"
        result = InputAnonymizer.anonymize(text)
        assert "[PHONE]" in result
        assert "555-123-4567" not in result

    def test_anonymize_ssn(self):
        text = "My SSN is 123-45-6789"
        result = InputAnonymizer.anonymize(text)
        assert "[SSN]" in result
        assert "123-45-6789" not in result

    def test_anonymize_ip(self):
        text = "Server IP is 192.168.1.1"
        result = InputAnonymizer.anonymize(text)
        assert "[IP]" in result
        assert "192.168.1.1" not in result

    def test_anonymize_clean_text(self):
        text = "Hello, this is a normal message with no PII"
        result = InputAnonymizer.anonymize(text)
        assert result == text

    def test_anonymize_empty_string(self):
        assert InputAnonymizer.anonymize("") == ""

    def test_has_pii_true(self):
        assert InputAnonymizer.has_pii("email me at test@example.com") is True

    def test_has_pii_false(self):
        assert InputAnonymizer.has_pii("Hello, how are you?") is False

    def test_has_pii_empty(self):
        assert InputAnonymizer.has_pii("") is False
