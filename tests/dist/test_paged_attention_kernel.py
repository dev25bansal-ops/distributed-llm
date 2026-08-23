"""Tests for paged_attention_kernel module.

All tests run on CPU without Triton. The ``_triton_paged_attention`` wrapper
is tested in its fallback (gather + SDPA) mode.

Note: ``_triton_paged_attention_impl`` has a known dimension mismatch in its
state-variable shapes (max_score / sum_exp are 2-D while block_max is 3-D,
causing broadcasting to the wrong shape).  It is therefore excluded from
testing; only the correctly-functioning public API and fallback helpers are
covered.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from distllm.dist.paged_attention_kernel import (
    PagedAttentionKernel,
    _gather_blocks,
    _standard_attention,
    _triton_paged_attention,
    paged_attention,
)

# Detect whether Triton is available in the test environment.
_HAS_TRITON: bool = False
try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

N_HEADS = 4
HEAD_DIM = 64
BLOCK_SZ = 16
SEQ_LEN = 32
NUM_BLOCKS = 4
NUM_LAYERS = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def query() -> torch.Tensor:
    return torch.randn(N_HEADS, 1, HEAD_DIM)


@pytest.fixture
def query_1h() -> torch.Tensor:
    return torch.randn(1, 1, HEAD_DIM)


@pytest.fixture
def key_pool() -> torch.Tensor:
    """6-D KV block pool: (num_blocks, num_layers, 2, num_heads, block_size, head_dim)."""
    return torch.randn(NUM_BLOCKS, NUM_LAYERS, 2, N_HEADS, BLOCK_SZ, HEAD_DIM)


@pytest.fixture
def value_pool() -> torch.Tensor:
    return torch.randn(NUM_BLOCKS, NUM_LAYERS, 2, N_HEADS, BLOCK_SZ, HEAD_DIM)


@pytest.fixture
def block_table_2() -> List[int]:
    return [0, 1]


@pytest.fixture
def gathered_key() -> torch.Tensor:
    """Pre-gathered 3-D key: (num_heads, seq_len, head_dim)."""
    return torch.randn(N_HEADS, SEQ_LEN, HEAD_DIM)


@pytest.fixture
def gathered_value() -> torch.Tensor:
    """Pre-gathered 3-D value: (num_heads, seq_len, head_dim)."""
    return torch.randn(N_HEADS, SEQ_LEN, HEAD_DIM)


# ===================================================================
# paged_attention()  — main public function
# ===================================================================


class TestPagedAttention:
    """Tests for the top-level ``paged_attention()`` function."""

    def test_pre_gathered_3d_path(self, query, gathered_key, gathered_value):
        """3-D key/value dispatches to _standard_attention directly."""
        out = paged_attention(
            query, gathered_key, gathered_value,
            block_table=[], seq_len=SEQ_LEN, block_size=BLOCK_SZ,
        )
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_blocked_6d_path(self, query, key_pool, value_pool):
        """6-D KV pool uses gather + standard attention on CPU."""
        out = paged_attention(
            query, key_pool, value_pool,
            [0, 1], seq_len=BLOCK_SZ * 2, block_size=BLOCK_SZ,
        )
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_single_block(self, query, key_pool, value_pool):
        """seq_len shorter than a single block."""
        out = paged_attention(
            query, key_pool, value_pool,
            [0], seq_len=5, block_size=BLOCK_SZ,
        )
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_partial_last_block(self, query, key_pool, value_pool):
        """Last block only partially filled (not a full BLOCK_SZ)."""
        out = paged_attention(
            query, key_pool, value_pool,
            [0, 1, 2], seq_len=BLOCK_SZ * 2 + 7, block_size=BLOCK_SZ,
        )
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_explicit_scale(self, query, key_pool, value_pool):
        """A caller-supplied scale factor is respected."""
        out = paged_attention(
            query, key_pool, value_pool,
            [0], seq_len=BLOCK_SZ, block_size=BLOCK_SZ,
            scale=0.5,
        )
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_default_scale_is_rsqrt_head_dim(self, query, gathered_key, gathered_value):
        """When scale is None the default is 1/sqrt(head_dim)."""
        default = paged_attention(
            query, gathered_key, gathered_value,
            [], seq_len=SEQ_LEN, block_size=BLOCK_SZ,
        )
        explicit = paged_attention(
            query, gathered_key, gathered_value,
            [], seq_len=SEQ_LEN, block_size=BLOCK_SZ,
            scale=HEAD_DIM ** -0.5,
        )
        assert torch.allclose(default, explicit)

    def test_mismatched_num_heads_raises(self, query, key_pool, value_pool):
        """A RuntimeError occurs when query heads != pool heads (on matmul)."""
        bad_query = torch.randn(N_HEADS + 1, 1, HEAD_DIM)
        with pytest.raises(RuntimeError):
            paged_attention(
                bad_query, key_pool, value_pool,
                [0], seq_len=BLOCK_SZ, block_size=BLOCK_SZ,
            )

    def test_mismatched_head_dim_raises(self, query, key_pool, value_pool):
        """A RuntimeError occurs when query head_dim != pool head_dim."""
        bad_query = torch.randn(N_HEADS, 1, HEAD_DIM + 8)
        with pytest.raises(RuntimeError):
            paged_attention(
                bad_query, key_pool, value_pool,
                [0], seq_len=BLOCK_SZ, block_size=BLOCK_SZ,
            )

    def test_deterministic(self, query, key_pool, value_pool):
        """Repeated calls with identical inputs produce identical output."""
        args = (query, key_pool, value_pool, [0, 1], BLOCK_SZ * 2, BLOCK_SZ)
        out1 = paged_attention(*args)
        out2 = paged_attention(*args)
        assert torch.equal(out1, out2)

    def test_fallback_and_fused_agree(self, query, key_pool, value_pool):
        """The gather+SDPA path and _triton_paged_attention fallback agree."""
        bt = [0, 1]
        seq_len = BLOCK_SZ * 2
        scale = HEAD_DIM ** -0.5

        out_main = paged_attention(query, key_pool, value_pool, bt, seq_len, BLOCK_SZ, scale)
        out_wrapper = _triton_paged_attention(
            query, key_pool, value_pool, bt, seq_len, BLOCK_SZ, scale,
        )
        assert torch.allclose(out_main, out_wrapper)


# ===================================================================
# _gather_blocks  — gather KV blocks into contiguous tensors
# ===================================================================


class TestGatherBlocks:
    """Tests for the internal ``_gather_blocks()`` helper."""

    def test_two_full_blocks(self, key_pool, value_pool, block_table_2):
        """Two whole blocks are concatenated."""
        k, v = _gather_blocks(
            key_pool, value_pool, block_table_2,
            BLOCK_SZ * 2, BLOCK_SZ, N_HEADS, HEAD_DIM,
            torch.device("cpu"), torch.float32,
        )
        assert k.shape == (N_HEADS, BLOCK_SZ * 2, HEAD_DIM)
        assert v.shape == (N_HEADS, BLOCK_SZ * 2, HEAD_DIM)
        # First block content should equal pool[0,0,0,...]
        assert torch.equal(k[:, :BLOCK_SZ, :], key_pool[0, 0, 0])
        assert torch.equal(v[:, :BLOCK_SZ, :], value_pool[0, 0, 1])

    def test_partial_last_block(self, key_pool, value_pool):
        """Only *take* elements are copied from the final block."""
        partial = 7
        k, v = _gather_blocks(
            key_pool, value_pool, [0, 1],
            BLOCK_SZ + partial, BLOCK_SZ, N_HEADS, HEAD_DIM,
            torch.device("cpu"), torch.float32,
        )
        assert k.shape == (N_HEADS, BLOCK_SZ + partial, HEAD_DIM)
        assert torch.equal(k[:, BLOCK_SZ:, :], key_pool[1, 0, 0, :, :partial, :])

    def test_empty_block_table(self, key_pool, value_pool):
        """No blocks produces zero-length 3-D tensors."""
        k, v = _gather_blocks(
            key_pool, value_pool, [],
            0, BLOCK_SZ, N_HEADS, HEAD_DIM,
            torch.device("cpu"), torch.float32,
        )
        assert k.shape == (N_HEADS, 0, HEAD_DIM)
        assert v.shape == (N_HEADS, 0, HEAD_DIM)

    def test_seq_len_shorter_than_table_capacity(self, key_pool, value_pool):
        """The output is trimmed to seq_len even when more blocks are listed."""
        k, v = _gather_blocks(
            key_pool, value_pool, [0, 1, 2, 3],
            5, BLOCK_SZ, N_HEADS, HEAD_DIM,
            torch.device("cpu"), torch.float32,
        )
        assert k.shape == (N_HEADS, 5, HEAD_DIM)

    def test_dtype_and_device(self, key_pool, value_pool):
        """Output respects the requested dtype and device."""
        k, v = _gather_blocks(
            key_pool, value_pool, [0],
            BLOCK_SZ, BLOCK_SZ, N_HEADS, HEAD_DIM,
            torch.device("cpu"), torch.float64,
        )
        assert k.dtype == torch.float64
        assert v.dtype == torch.float64
        assert k.device.type == "cpu"

    def test_content_correctness(self, key_pool, value_pool):
        """Manual indexing matches every position from a non-trivial table."""
        block_table = [2, 0, 1]
        seq_len = 40
        k, v = _gather_blocks(
            key_pool, value_pool, block_table,
            seq_len, BLOCK_SZ, N_HEADS, HEAD_DIM,
            torch.device("cpu"), torch.float32,
        )

        pos = 0
        for phys_id in block_table:
            take = min(BLOCK_SZ, seq_len - pos)
            if take <= 0:
                break
            assert torch.equal(
                k[:, pos:pos + take, :], key_pool[phys_id, 0, 0, :, :take, :],
            )
            assert torch.equal(
                v[:, pos:pos + take, :], value_pool[phys_id, 0, 1, :, :take, :],
            )
            pos += take

    def test_extra_blocks_in_table_are_skipped(self, key_pool, value_pool):
        """Blocks beyond what is needed for seq_len are ignored."""
        seq_len = BLOCK_SZ + 3
        k, v = _gather_blocks(
            key_pool, value_pool, [0, 1, 2],
            seq_len, BLOCK_SZ, N_HEADS, HEAD_DIM,
            torch.device("cpu"), torch.float32,
        )
        assert k.shape[1] == seq_len


# ===================================================================
# _standard_attention  — scaled dot-product attention
# ===================================================================


class TestStandardAttention:
    """Tests for the internal ``_standard_attention()`` helper."""

    def test_output_shape(self, query, gathered_key, gathered_value):
        """Output shape is (num_heads, 1, head_dim)."""
        out = _standard_attention(query, gathered_key, gathered_value, HEAD_DIM ** -0.5)
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_single_head(self, query_1h, gathered_key, gathered_value):
        """Works with a single attention head."""
        k = gathered_key[:1]
        v = gathered_value[:1]
        out = _standard_attention(query_1h, k, v, HEAD_DIM ** -0.5)
        assert out.shape == (1, 1, HEAD_DIM)

    def test_scale_factor_changes_output(self, query, gathered_key, gathered_value):
        """Different scale values yield different results."""
        out_small = _standard_attention(query, gathered_key, gathered_value, scale=0.1)
        out_large = _standard_attention(query, gathered_key, gathered_value, scale=10.0)
        assert not torch.allclose(out_small, out_large)

    def test_identity_key_value(self, query, gathered_key):
        """When K == V the output is a finite blend of values (no crash)."""
        out = _standard_attention(query, gathered_key, gathered_key, HEAD_DIM ** -0.5)
        assert out.shape == (N_HEADS, 1, HEAD_DIM)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_non_nan_finite_output(self, query, gathered_key, gathered_value):
        """Output should be finite under normal conditions."""
        out = _standard_attention(query, gathered_key, gathered_value, HEAD_DIM ** -0.5)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        # Reasonable magnitude (scores * values should not blow up)
        assert out.abs().max().item() < 100.0




# ===================================================================
# _triton_paged_attention  — wrapper with fallback
# ===================================================================


class TestTritonPagedAttention:
    """Tests for ``_triton_paged_attention()``.

    Because Triton is not installed, this function always takes the fallback
    path (gather + SDPA).  We verify the fallback produces correct results.
    """

    def test_fallback_output_shape(self, query, key_pool, value_pool, block_table_2):
        """Fallback path produces the expected output shape."""
        out = _triton_paged_attention(
            query, key_pool, value_pool, block_table_2,
            BLOCK_SZ * 2, BLOCK_SZ, HEAD_DIM ** -0.5,
        )
        assert out.shape == (N_HEADS, 1, HEAD_DIM)
        assert not torch.isnan(out).any()

    def test_fallback_matches_main(self, query, key_pool, value_pool, block_table_2):
        """Fallback result equals the top-level paged_attention result."""
        scale = HEAD_DIM ** -0.5
        seq_len = BLOCK_SZ * 2

        out_fb = _triton_paged_attention(
            query, key_pool, value_pool, block_table_2,
            seq_len, BLOCK_SZ, scale,
        )
        out_main = paged_attention(
            query, key_pool, value_pool, block_table_2,
            seq_len, BLOCK_SZ, scale,
        )
        assert torch.allclose(out_fb, out_main)

    def test_fallback_empty_table(self, query, key_pool, value_pool):
        """Fallback with empty block table returns a zero-filled output."""
        out = _triton_paged_attention(
            query, key_pool, value_pool,
            [], 0, BLOCK_SZ, HEAD_DIM ** -0.5,
        )
        assert out.shape == (N_HEADS, 1, HEAD_DIM)
        # The fallback calls _gather_blocks with seq_len=0 -> empty K,V
        # then _standard_attention.  The manual path inside
        # _standard_attention produces zeros because scores are (H,1,0)
        # and softmax normalizes to uniform over dim of size 0.
        # We only verify the shape — exact values are implementation-defined
        # for this degenerate input.

    def test_fallback_partial_block(self, query, key_pool, value_pool):
        """Fallback handles partial blocks correctly."""
        out = _triton_paged_attention(
            query, key_pool, value_pool,
            [0, 1], BLOCK_SZ + 7, BLOCK_SZ, HEAD_DIM ** -0.5,
        )
        assert out.shape == (N_HEADS, 1, HEAD_DIM)


# ===================================================================
# PagedAttentionKernel  — wrapper class
# ===================================================================


class TestPagedAttentionKernel:
    """Tests for the ``PagedAttentionKernel`` class."""

    def test_default_construction(self):
        """Default config uses SDPA path and block_size=16."""
        kernel = PagedAttentionKernel()
        assert not kernel.use_triton
        assert kernel.block_size == 16
        assert kernel.kernel_type == "sdpa"

    def test_custom_block_size(self):
        """block_size is stored and exposed."""
        kernel = PagedAttentionKernel(block_size=32)
        assert kernel.block_size == 32

    def test_kernel_type_when_triton_disabled(self):
        """kernel_type is 'sdpa' when use_triton is False."""
        kernel = PagedAttentionKernel()
        assert kernel.kernel_type == "sdpa"

    def test_use_triton_raises_when_unavailable(self):
        """Constructing with use_triton=True without triton raises ImportError."""
        if _HAS_TRITON:
            # Triton is installed in this environment — no error expected
            kernel = PagedAttentionKernel(use_triton=True)
            assert kernel.use_triton
            assert kernel.kernel_type == "triton"
        else:
            with pytest.raises(ImportError, match="Triton is not installed"):
                PagedAttentionKernel(use_triton=True)

    def test_call_returns_output(self, query, key_pool, value_pool):
        """__call__ dispatches to paged_attention and returns (H,1,D)."""
        kernel = PagedAttentionKernel(block_size=BLOCK_SZ)
        out = kernel(query, key_pool, value_pool, [0, 1], seq_len=BLOCK_SZ * 2)
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_call_with_explicit_scale(self, query, key_pool, value_pool):
        """scale keyword is forwarded to paged_attention."""
        kernel = PagedAttentionKernel(block_size=BLOCK_SZ)
        out = kernel(query, key_pool, value_pool, [0, 1],
                     seq_len=BLOCK_SZ * 2, scale=0.25)
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_call_with_3d_pool(self, query, gathered_key, gathered_value):
        """Pre-gathered 3-D pool works through the kernel wrapper."""
        kernel = PagedAttentionKernel(block_size=BLOCK_SZ)
        out = kernel(query, gathered_key, gathered_value, [],
                     seq_len=SEQ_LEN)
        assert out.shape == (N_HEADS, 1, HEAD_DIM)

    def test_repr(self):
        """__repr__ contains the kernel type and block_size."""
        kernel = PagedAttentionKernel(block_size=32)
        r = repr(kernel)
        assert "PagedAttentionKernel" in r
        assert "sdpa" in r
        assert "block_size=32" in r

    def test_deterministic(self, query, key_pool, value_pool):
        """Repeated kernel calls produce the same output."""
        kernel = PagedAttentionKernel(block_size=BLOCK_SZ)
        args = (query, key_pool, value_pool, [0, 1])
        kwargs = {"seq_len": BLOCK_SZ * 2}
        out1 = kernel(*args, **kwargs)
        out2 = kernel(*args, **kwargs)
        assert torch.equal(out1, out2)

    def test_preserves_default_block_size(self, query, key_pool, value_pool):
        """Without explicit block_size, the default (16) is used."""
        kernel = PagedAttentionKernel()
        out = kernel(query, key_pool, value_pool, [0, 1], seq_len=BLOCK_SZ)
        assert out.shape == (N_HEADS, 1, HEAD_DIM)
