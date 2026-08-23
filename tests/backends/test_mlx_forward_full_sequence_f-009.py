"""Regression test for audit finding F-009.

MLXNodeAdapter._forward_input_ids used to loop over tokens and call
self._model(mx.array([token_id]).reshape(1, 1)) once per token with no
KV-state carry-over, no attention_mask and no position_ids.  Every token
was scored as an isolated length-1 sequence, so tokens after the first
got no causal attention over prior context — the concatenated "logits"
were silently wrong for any seq_len > 1.  It also read only the first
batch row (.tolist()[0]), silently dropping batch rows 1..N-1.

Fix: the full sequence is evaluated in a SINGLE model call (shape
(1, seq_len)), per batch row; tuple returns from newer mlx-lm versions
are unpacked.  These tests use a recording fake model + a stub ``mx``
module because MLX itself is macOS-only and not importable here.
"""

import inspect

import numpy as np
import torch

from distllm.backends.mlx_backend import MLXNodeAdapter

VOCAB = 32


class _StubMX:
    """Stand-in for mlx.core: array() just wraps input in numpy."""

    @staticmethod
    def array(x):
        return np.asarray(x)


class _PrefixCumsumModel:
    """Fake MLX model whose position-i logits encode cumsum(tokens[:i+1]).

    With this function, output at position i can only be correct if the
    model saw the WHOLE prefix — impossible under the old per-token
    (1,1)-shaped calls.
    """

    def __init__(self):
        self.calls: list[np.ndarray] = []

    def __call__(self, mlx_input):
        ids = np.asarray(mlx_input)
        self.calls.append(ids.copy())
        bsz, seq_len = ids.shape[0], ids.shape[-1]
        out = np.zeros((bsz, seq_len, VOCAB), dtype=np.float32)
        for row in range(bsz):
            cum = np.cumsum(ids[row])
            out[row, :, 0] = cum
            out[row, :, 1] = ids[row]  # raw token id at each position
        return out


class _TupleReturnModel(_PrefixCumsumModel):
    """Newer mlx-lm versions return (logits, cache) tuples."""

    def __call__(self, mlx_input):
        logits = super().__call__(mlx_input)
        return logits, []


def _make_adapter(model) -> MLXNodeAdapter:
    adapter = MLXNodeAdapter("fake/mlx-model")
    adapter._model = model  # bypass load_model(); no MLX runtime needed
    return adapter


class TestFullSequenceSingleCall:
    """The core defect: tokens must be scored with causal context."""

    def test_one_model_call_per_sequence_not_per_token(self):
        model = _PrefixCumsumModel()
        adapter = _make_adapter(model)
        logits, kv = adapter._forward_input_ids(
            torch.tensor([[5, 7, 9]]), _StubMX,
        )
        assert len(model.calls) == 1, (
            f"expected ONE full-sequence call, got {len(model.calls)}"
        )
        assert model.calls[0].shape == (1, 3)
        assert kv == []

    def test_position_logits_depend_on_full_prefix(self):
        """logits[tokens[i]] must change when an EARLIER token changes."""
        adapter = _make_adapter(_PrefixCumsumModel())
        base, _ = adapter._forward_input_ids(torch.tensor([[5, 7, 9]]), _StubMX)
        mutated, _ = adapter._forward_input_ids(torch.tensor([[5, 8, 9]]), _StubMX)
        # Position 2 sees prefix [5,7,9] vs [5,8,9]: cumsum 21 vs 22.
        assert not torch.allclose(base[0, 2], mutated[0, 2])
        assert base[0, 2, 0].item() == 21.0
        assert mutated[0, 2, 0].item() == 22.0
        # Position 0 is identical in both inputs.
        assert torch.allclose(base[0, 0], mutated[0, 0])

    def test_output_shape_is_batch_seq_vocab(self):
        adapter = _make_adapter(_PrefixCumsumModel())
        logits, _ = adapter._forward_input_ids(torch.tensor([[5, 7, 9]]), _StubMX)
        assert logits.shape == (1, 3, VOCAB)

    def test_single_token_input_still_works(self):
        model = _PrefixCumsumModel()
        adapter = _make_adapter(model)
        logits, _ = adapter._forward_input_ids(torch.tensor([[42]]), _StubMX)
        assert logits.shape == (1, 1, VOCAB)
        assert logits[0, 0, 1].item() == 42.0


class TestBatchRowsPreserved:
    """Old code silently evaluated only .tolist()[0] — row 0 alone."""

    def test_both_batch_rows_forwarded(self):
        model = _PrefixCumsumModel()
        adapter = _make_adapter(model)
        logits, _ = adapter._forward_input_ids(
            torch.tensor([[5, 7], [10, 20]]), _StubMX,
        )
        assert logits.shape == (2, 2, VOCAB)
        # One full-sequence call PER ROW, both rows seen.
        assert sorted(c.shape[0] for c in model.calls) == [1, 1]
        seen_tokens = {int(c[0][0]) for c in model.calls}
        assert seen_tokens == {5, 10}

    def test_row_values_not_swapped_or_dropped(self):
        adapter = _make_adapter(_PrefixCumsumModel())
        logits, _ = adapter._forward_input_ids(
            torch.tensor([[1, 2], [3, 4]]), _StubMX,
        )
        assert logits[1, 1, 0].item() == 7.0  # cumsum of row 1, not row 0
        assert logits[0, 1, 0].item() == 3.0


class TestMlxLmTupleApi:
    """Newer mlx-lm returns (logits, cache); must unpack, not concat tuples."""

    def test_tuple_return_unpacked(self):
        adapter = _make_adapter(_TupleReturnModel())
        logits, kv = adapter._forward_input_ids(torch.tensor([[5, 7, 9]]), _StubMX)
        assert logits.shape == (1, 3, VOCAB)
        assert logits[0, 2, 0].item() == 21.0
        assert kv == []


class TestFullModelPathRoutedThroughFix:
    """_forward_full_model (hidden_states -> argmax) shares the fixed path."""

    def test_hidden_states_argmax_gets_full_sequence(self):
        model = _PrefixCumsumModel()
        adapter = _make_adapter(model)
        hidden = torch.randn(1, 4, VOCAB)
        hidden[..., :] = torch.eye(VOCAB)[torch.tensor([3, 1, 4, 2])]
        logits, _ = adapter._forward_full_model(hidden, _StubMX)
        assert logits.shape == (1, 4, VOCAB)
        assert len(model.calls) == 1
        assert model.calls[0].shape == (1, 4)


class TestPerTokenLoopIsGone:
    """Source-level guard: the isolated reshape(1, 1) loop must not return."""

    def test_no_per_token_reshape_1_1_loop(self):
        src = inspect.getsource(MLXNodeAdapter)
        assert "mx.array([token_id]).reshape(1, 1)" not in src
        assert ".tolist()[0]" not in src
