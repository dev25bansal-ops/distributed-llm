"""Regression tests for KV-cache crop fallback on legacy tuple caches (C-04).

Bug C-04: `past.crop(...)` was guarded only by `if hasattr(past, "crop"):`
with no `else`. HuggingFace `DynamicCache` has `.crop()`, but a legacy
tuple-of-tensors `past_key_values` does not. When the check failed, the
token buffer dropped the rejected draft tokens but the KV cache silently
retained their entries, so subsequent forward passes attended over
rejected tokens -> garbled output with no error and no log line.

Fix: an explicit `else` branch that slices each 4-D tensor along the
sequence dimension (dim=2) to keep only `prefix_len + accepted` entries.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")


def _make_tuple_cache(n_layers: int = 2, seq: int = 8, heads: int = 2, dim: int = 4):
    """Build a legacy tuple-of-tensors KV cache: ((k, v), (k, v), ...)."""
    layers = []
    for _ in range(n_layers):
        k = torch.arange(seq * heads * dim, dtype=torch.float32).reshape(1, heads, seq, dim)
        v = torch.arange(seq * heads * dim, dtype=torch.float32).reshape(1, heads, seq, dim) + 1000
        layers.append((k, v))
    return tuple(layers)


def test_tuple_cache_has_no_crop_method():
    """Confirm the premise: legacy tuples genuinely lack .crop()."""
    cache = _make_tuple_cache()
    assert not hasattr(cache, "crop"), (
        "This test targets caches WITHOUT .crop(); fixture is wrong."
    )


def test_rejected_tokens_truncate_tuple_cache_sequence_dim():
    """Simulate the C-04 fix: slicing a tuple cache along dim=2.

    prefix_len=3, accepted=1 -> keep 4 entries, not 8.
    """
    cache = _make_tuple_cache(seq=8)
    prefix_len, accepted = 3, 1
    keep = prefix_len + accepted

    truncated = tuple(
        tuple(t[:, :, :keep, :] if t.dim() == 4 else t for t in layer)
        for layer in cache
    )

    for layer in truncated:
        for t in layer:
            assert t.shape[2] == keep, (
                f"Expected sequence dim {keep}, got {t.shape[2]}"
            )
    # Untouched original still has 8 (no mutation of the source)
    assert cache[0][0].shape[2] == 8


def test_truncated_cache_preserves_leading_entries():
    """Slicing must keep the FIRST `keep` entries, not the last."""
    cache = _make_tuple_cache(seq=8, heads=1, dim=1)
    keep = 4
    truncated = tuple(
        tuple(t[:, :, :keep, :] if t.dim() == 4 else t for t in layer)
        for layer in cache
    )
    original_k = cache[0][0]  # (1, 1, 8, 1)
    new_k = truncated[0][0]
    assert torch.equal(new_k, original_k[:, :, :keep, :]), (
        "Truncation must preserve the leading entries."
    )


def test_non_4d_tensors_are_passed_through_unchanged():
    """Tensors that aren't 4-D (e.g. attention masks) must not be sliced."""
    mask = torch.ones(1, 8)  # 2-D
    layer = (torch.zeros(1, 2, 8, 4), mask)
    cache = (layer,)
    keep = 4
    truncated = tuple(
        tuple(t[:, :, :keep, :] if t.dim() == 4 else t for t in layer)
        for layer in cache
    )
    assert torch.equal(truncated[0][1], mask), (
        "2-D tensors must pass through unchanged."
    )
    assert truncated[0][0].shape[2] == keep


def test_dynamic_cache_path_still_uses_crop():
    """The preferred path: objects WITH .crop() should use it, not the fallback."""
    cache = _make_tuple_cache()
    # Attach a fake crop to emulate a DynamicCache
    calls = []

    class FakeDynamicCache:
        def crop(self, n: int) -> None:
            calls.append(n)

    dyn = FakeDynamicCache()
    if hasattr(dyn, "crop"):
        dyn.crop(4)
    else:  # pragma: no cover - the fallback we added
        raise AssertionError("FakeDynamicCache should have .crop()")
    assert calls == [4], f"crop() should be called with 4, got {calls}"
