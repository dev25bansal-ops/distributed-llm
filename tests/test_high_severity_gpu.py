"""Real-GPU regression tests for High-severity findings H7 (shared_layer_pool
from __future__ import annotations
fingerprint collision) and H8 (gpu_resource_manager free/used swap).

These require a working CUDA device. They are skipped automatically when CUDA
is unavailable, so the suite stays portable — but on a CUDA host they exercise
the fixes against real VRAM / real tensors.

H8: ``GPUResourceManager.snapshot()`` previously reported ``free_mb`` and
``used_mb`` swapped. After allocating a big tensor on the GPU, ``used_mb`` must
rise and ``free_mb`` must fall, and ``used + free`` must equal ``total``.

H7: ``_fingerprint_layer`` hashed only the first 1KB, so two distinct
same-shape layers that share leading bytes but differ later produced identical
fingerprints -> wrongly aliased weights. The fixed full-content hash must give
these two tensors DIFFERENT fingerprints, while two identical tensors share one.
"""


import pytest

try:
    import torch
    _ = torch.float16  # canary: real torch always has this; pollution replaces torch with an empty stub
except (ModuleNotFoundError, ImportError, AttributeError) as _e:
    pytest.skip(f"requires working torch / distllm (GPU) (not available): {_e}", allow_module_level=True)


import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required"
)


# ── H8: snapshot free/used must not be swapped ────────────────────────────

def test_gpu_snapshot_free_used_not_swapped():
    from distllm.core.gpu_resource_manager import GPUResourceManager

    mgr = GPUResourceManager()
    mgr.register_device(0)
    before = mgr.snapshot(0)
    assert before is not None, "snapshot() returned None on a CUDA host"

    # Sanity: total = used + free (within rounding), and free <= total.
    assert before.free_mb <= before.total_mb + 1.0
    assert before.used_mb <= before.total_mb + 1.0
    assert abs((before.used_mb + before.free_mb) - before.total_mb) < 2.0, (
        f"used+free ({before.used_mb}+{before.free_mb}) != total "
        f"({before.total_mb})"
    )

    # Allocate ~256MB on the GPU and hold a reference.
    n = (256 * 1024 * 1024) // 4  # float32 elements ≈ 256 MB
    big = torch.empty(n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    try:
        after = mgr.snapshot(0)
        assert after is not None
        # used must INCREASE and free must DECREASE after allocating.
        assert after.used_mb > before.used_mb + 100, (
            f"used_mb did not rise after 256MB alloc: "
            f"{before.used_mb} -> {after.used_mb} (free/used likely swapped)"
        )
        assert after.free_mb < before.free_mb - 100, (
            f"free_mb did not fall after 256MB alloc: "
            f"{before.free_mb} -> {after.free_mb} (free/used likely swapped)"
        )
        # used should be well below total (we only grabbed 256MB of 8GB).
        assert after.used_mb < after.total_mb, "used_mb >= total_mb (swapped?)"
    finally:
        del big
        torch.cuda.empty_cache()


# ── H7: fingerprint must not collide on same-shape distinct layers ────────

def test_fingerprint_distinguishes_same_shape_layers():
    from distllm.core.shared_layer_pool import SharedLayerPool

    pool = SharedLayerPool()
    shape = (512, 512)  # > 1KB leading region shared below

    # Two tensors with IDENTICAL first 1KB but DIFFERENT tails.
    a = torch.zeros(shape, dtype=torch.float32)
    b = torch.zeros(shape, dtype=torch.float32)
    # First 1024 bytes = first 256 float32 values -> keep identical (all zeros).
    # Differ only far past the 1KB window.
    b[-1, -1] = 12345.0

    fa = pool._fingerprint_layer("a", a)
    fb = pool._fingerprint_layer("b", b)

    assert fa.param_hash != fb.param_hash, (
        "H7: same-shape layers sharing leading 1KB got identical fingerprints "
        "-> would be wrongly aliased in the pool"
    )


def test_fingerprint_matches_identical_layers():
    from distllm.core.shared_layer_pool import SharedLayerPool

    pool = SharedLayerPool()
    a = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
    b = a.clone()
    fa = pool._fingerprint_layer("a", a)
    fb = pool._fingerprint_layer("b", b)
    assert fa.param_hash == fb.param_hash, (
        "identical tensors must share a fingerprint (legitimate dedup)"
    )


# ── H6: decode steps (past_key_values present) must NOT replay stale graph ─

class _KVSensitiveModel(torch.nn.Module):
    """Fake causal LM whose logits depend on whether the KV cache is threaded
    through. If replay() ignores past_key_values (the H6 bug), the output will
    match the no-KV path and be wrong for a decode step.
    """

    def __init__(self, vocab: int = 16):
        super().__init__()
        self.vocab = vocab
        self.calls: list[bool] = []  # records whether past_key_values was passed

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=True, **kw):
        had_kv = past_key_values is not None
        self.calls.append(had_kv)
        bsz = input_ids.shape[0]
        seq = input_ids.shape[1]
        logits = torch.zeros(bsz, seq, self.vocab, device=input_ids.device)
        # Encode "did we see a KV cache" into the argmax token so the test can
        # detect whether the KV was honored: token 7 with KV, token 3 without.
        logits[:, -1, 7 if had_kv else 3] = 10.0

        class _Out:
            pass
        o = _Out()
        o.logits = logits
        return o


def test_cuda_graph_decode_falls_back_to_eager_with_kv():
    from distllm.core.cuda_graph import CudaGraphCapture

    model = _KVSensitiveModel().cuda()
    runner = CudaGraphCapture(model, batch_sizes=[1], max_seq_len=8)
    runner.capture(device="cuda")

    # A real decode step supplies a past_key_values cache. The fix must route
    # this to eager execution (honoring the KV), NOT replay the stale graph.
    input_ids = torch.randint(0, 16, (1, 1), device="cuda")
    fake_kv = [(torch.zeros(1, 1, 1, 1, device="cuda"),
                torch.zeros(1, 1, 1, 1, device="cuda"))]

    model.calls.clear()
    logits = runner.replay(input_ids, past_key_values=fake_kv)

    # The eager path must have been called WITH the KV cache.
    assert any(model.calls), (
        "H6: replay() ignored past_key_values (replayed stale graph) — "
        "decode would use a stale KV cache and produce wrong logits"
    )
    # And the returned logits must reflect the KV-aware path (token 7).
    assert int(logits.argmax(dim=-1)[0].item()) == 7, (
        "H6: decode logits did not reflect the KV cache (stale-KV replay)"
    )
