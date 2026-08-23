import torch

from distllm.core.hybrid_cache import ContiguousKVBuffer


def test_append_increments_num_tokens_and_preserves_sequence():
    num_layers = 2
    num_heads = 4
    head_dim = 8
    max_tokens = 64
    device = "cpu"
    dtype = torch.float32

    buf = ContiguousKVBuffer(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_tokens=max_tokens,
        dtype=dtype,
        device=device,
    )

    total = 0
    expected_keys = []
    expected_values = []
    for i in range(50):
        n = 1
        key = torch.full((num_heads, n, head_dim), float(i), dtype=dtype)
        value = torch.full((num_heads, n, head_dim), float(i * 10), dtype=dtype)
        buf.append(layer_idx=0, key=key, value=value)
        total += n
        expected_keys.append(float(i))
        expected_values.append(float(i * 10))

        assert buf.num_tokens == total

    k, v = buf.get(layer_idx=0, seq_len=buf.num_tokens)
    assert k.shape == (num_heads, total, head_dim)
    assert v.shape == (num_heads, total, head_dim)

    flat_k = k[:, :, 0].mean(dim=0).tolist()
    flat_v = v[:, :, 0].mean(dim=0).tolist()
    assert flat_k == expected_keys, f"keys corrupted: {flat_k}"
    assert flat_v == expected_values, f"values corrupted: {flat_v}"


def test_append_multi_token_block():
    num_layers = 1
    num_heads = 2
    head_dim = 4
    max_tokens = 64
    device = "cpu"
    dtype = torch.float32

    buf = ContiguousKVBuffer(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_tokens=max_tokens,
        dtype=dtype,
        device=device,
    )

    n = 5
    block_key = torch.arange(n, dtype=dtype).view(1, n, 1).expand(num_heads, n, head_dim)
    block_val = (-torch.arange(n, dtype=dtype)).view(1, n, 1).expand(num_heads, n, head_dim)
    buf.append(layer_idx=0, key=block_key, value=block_val)
    assert buf.num_tokens == n

    k, v = buf.get(layer_idx=0, seq_len=buf.num_tokens)
    assert k[:, :, 0].mean(dim=0).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert v[:, :, 0].mean(dim=0).tolist() == [0.0, -1.0, -2.0, -3.0, -4.0]
