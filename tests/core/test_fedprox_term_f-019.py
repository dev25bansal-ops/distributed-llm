"""Regression tests for F-019 — FedProx proximal term math.

Bug: ``_apply_fedprox_term`` computed ``mu * (grad - global_param)``, mixing a
gradient tensor with a weight tensor elementwise.  That double-counted
``mu * grad`` and subtracted a weight from a gradient: the effective update was
``(1+mu)*grad - mu*w_global`` instead of the FedProx update

    grad + mu * (w_local - w_global)          (Li et al. 2020)

Fix: compute the proximal term from the pre-round local weights obtained via
the ``lora_adapter`` callback; parameters whose w_local is unavailable are
passed through unchanged instead of corrupted.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_fft = load_module("distllm/core/federated_finetuner.py")
FederatedFineTuner = _fft.FederatedFineTuner

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


def _make_fft(mu: float, global_params, local_params=None):
    adapter = (lambda: local_params) if local_params is not None else None
    return FederatedFineTuner(
        node_id="n1",
        algorithm="fedprox",
        fedprox_mu=mu,
        lora_adapter=adapter,
        global_model_params=global_params,
    )


class TestFedProxMath:
    def test_correct_form_grad_plus_mu_wdiff(self):
        """result == grad + mu*(w_local - w_global), not (1+mu)*grad - mu*w_g."""
        w_local = torch.tensor([1.0, 2.0])
        w_global = torch.tensor([0.5, 0.5])
        grad = torch.tensor([3.0, 4.0])
        fft = _make_fft(0.1, [w_global], [w_local])

        result = fft._apply_fedprox_term([grad])
        expected = grad + 0.1 * (w_local - w_global)
        assert torch.allclose(result[0], expected)

    def test_no_gradient_rescaling_when_local_equals_global(self):
        """When w_local == w_global the proximal term vanishes: result == grad."""
        w = torch.tensor([0.7, -1.2])
        grad = torch.tensor([3.0, 4.0])  # deliberately != 2*grad of old bug
        fft = _make_fft(1.0, [w], [w])

        result = fft._apply_fedprox_term([grad])
        assert torch.allclose(result[0], grad)

    def test_grad_not_double_counted(self):
        """Old bug returned (1+mu)*grad - mu*w_g; verify mu*grad is NOT added."""
        w_local = torch.tensor([0.0])
        w_global = torch.tensor([0.0])
        grad = torch.tensor([2.0])
        fft = _make_fft(0.5, [w_global], [w_local])

        result = fft._apply_fedprox_term([grad])
        # Old buggy code would give 1.5*grad == 3.0 here.
        assert torch.allclose(result[0], torch.tensor([2.0]))

    def test_stats_counter_incremented(self):
        fft = _make_fft(1.0, [torch.zeros(2)], [torch.ones(2)])
        fft._apply_fedprox_term([torch.ones(2)])
        assert fft._stats["fedprox_proximal_terms"] == 1


class TestFedProxFallbacks:
    def test_no_local_params_passthrough(self):
        """Without any way to read w_local, grads must pass through untouched."""
        grad = torch.tensor([3.0, 4.0])
        fft = _make_fft(1.0, [torch.tensor([9.9])], local_params=None)

        result = fft._apply_fedprox_term([grad])
        assert torch.allclose(result[0], grad)
        assert fft._stats["fedprox_proximal_terms"] == 0

    def test_adapter_failure_passthrough(self):
        """A raising lora_adapter must not corrupt gradients."""
        def broken_adapter():
            raise RuntimeError("adapter gone")

        fft = FederatedFineTuner(
            node_id="n1", algorithm="fedprox", fedprox_mu=1.0,
            lora_adapter=broken_adapter,
            global_model_params=[torch.tensor([1.0])],
        )
        grad = torch.tensor([3.0])
        result = fft._apply_fedprox_term([grad])
        assert torch.allclose(result[0], grad)

    def test_shorter_local_params_partial_coverage(self):
        """Indices beyond len(w_local) keep their raw gradient."""
        w_local = [torch.tensor([1.0])]
        grads = [torch.tensor([1.0]), torch.tensor([2.0])]
        fft = _make_fft(1.0, [torch.tensor([0.0]), torch.tensor([0.0])], w_local)

        result = fft._apply_fedprox_term(grads)
        assert torch.allclose(result[0], torch.tensor([2.0]))   # 1.0 + 1*(1-0)
        assert torch.allclose(result[1], grads[1])               # passthrough

    def test_mu_zero_noop(self):
        fft = _make_fft(0.0, [torch.tensor([1.0])], [torch.tensor([5.0])])
        grad = torch.tensor([2.0])
        result = fft._apply_fedprox_term([grad])
        assert torch.allclose(result[0], grad)


class TestFedProxEndToEnd:
    def test_average_gradients_uses_correct_proximal_term(self):
        """_average_gradients with fedprox adds mu*(w_local-w_global) to the
        LOCAL gradient only, then averages with peer grads as usual."""
        w_local = torch.tensor([2.0])
        w_global = torch.tensor([0.0])
        local_grad = torch.tensor([4.0])
        peer_grad = torch.tensor([6.0])

        fft = FederatedFineTuner(
            node_id="n1",
            algorithm="fedprox",
            fedprox_mu=0.5,
            lora_adapter=lambda: [w_local],
            global_model_params=[w_global],
            apply_gradients=lambda *_: None,
        )
        fft.add_peer("peer-1")
        fft._received_grads["peer-1"] = [peer_grad]

        merged = fft._average_gradients([local_grad])
        # local becomes 4.0 + 0.5*(2.0-0.0) = 5.0, then avg(5.0, 6.0) = 5.5
        assert torch.allclose(merged[0], torch.tensor([5.5]))

    def test_train_round_applies_valid_update(self):
        """Full round: apply_gradients receives grad + mu*(w_local-w_global)."""
        seen = {}

        def apply_fn(grads, lr):
            seen["g"] = grads[0].clone()

        fft = FederatedFineTuner(
            node_id="n1",
            algorithm="fedprox",
            fedprox_mu=2.0,
            dp_epsilon=float("inf"),  # disable DP noise/clipping
            lora_adapter=lambda: [torch.tensor([1.0])],
            global_model_params=[torch.tensor([0.25])],
            apply_gradients=apply_fn,
        )
        local_grad = torch.tensor([1.0])
        # A peer contribution is required so _average_gradients takes the
        # multi-source path where the FedProx branch lives.
        fft.add_peer("peer-1")
        fft._received_grads["peer-1"] = [local_grad.clone()]
        fft.train_round(lambda steps: [local_grad.clone()])
        proximal_local = local_grad + 2.0 * (torch.tensor([1.0]) - torch.tensor([0.25]))
        # merged = average(proximal_local, peer_grad)
        expected = (proximal_local + local_grad) / 2
        assert torch.allclose(seen["g"], expected)
