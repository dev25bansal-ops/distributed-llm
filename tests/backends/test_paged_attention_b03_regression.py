"""B-03 regression: allocate_sequence(0) + prefill block.num_tokens seeding.

Covers paged_attention.py:273 (allocate_sequence), 388-390 (append_token
block-full check): before the fix, allocate_sequence(seq, 0) returned []
and the first append_token() raised IndexError; and prefill allocations
left every block.num_tokens == 0, so append_token() never advanced to
a new block at the boundary.
"""

from distllm.backends.paged_attention import PagedAttentionManager


def _make(**kwargs):
    defaults = dict(
        num_blocks=16, block_size=4, num_layers=1,
        num_heads=2, head_dim=4, device="cpu",
    )
    defaults.update(kwargs)
    return PagedAttentionManager(**defaults)


def test_allocate_sequence_zero_then_append_does_not_crash():
    pam = _make()
    bids = pam.allocate_sequence("zero", 0)
    assert len(bids) == 1  # empty seq still gets one writable block
    bid = pam.append_token("zero")  # would IndexError before the fix
    assert bid == bids[0]
    assert pam._seq_blocks["zero"].num_tokens == 1
    assert pam._blocks[bid].num_tokens == 1


def test_prefill_seeds_block_num_tokens():
    pam = _make(block_size=4)
    pam.allocate_sequence("pre", 10)
    assert pam._seq_blocks["pre"].num_tokens == 10
    table = pam.get_block_table("pre")
    assert len(table) == 3
    assert [pam._blocks[b].num_tokens for b in table] == [4, 4, 2]


def test_prefill_full_last_block_appends_to_new_block():
    pam = _make(block_size=4)
    pam.allocate_sequence("pre", 8)  # exactly 2 full blocks
    table = pam.get_block_table("pre")
    assert [pam._blocks[b].num_tokens for b in table] == [4, 4]
    new_bid = pam.append_token("pre")  # last block is full -> new block
    assert new_bid not in table
    assert len(pam.get_block_table("pre")) == 3
    assert pam._blocks[new_bid].num_tokens == 1


def test_copy_on_write_of_zero_token_sequence_append():
    pam = _make()
    pam.allocate_sequence("src", 0)
    pam.copy_on_write("src", "dst")
    bid = pam.append_token("dst")
    assert bid in pam.get_block_table("dst")
