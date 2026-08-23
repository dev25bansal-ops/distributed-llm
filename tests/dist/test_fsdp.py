"""Tests for FSDP-style weight sharding across nodes.

Covers:
- FSDPConfig construction and defaults
- FSDPShard construction and input validation
- Sharding logic with world_size > 1 (no distributed required)
- No-op behavior when world_size = 1
- free() restoration after gather (state flag set manually)
- Mixed precision config storage
- Helper functions: _ceil_div, _resolve_device, _get_param_by_name
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from distllm.dist.fsdp import (
    FSDPConfig,
    FSDPShard,
    _ceil_div,
    _get_param_by_name,
    _resolve_device,
)


# ── Test Model Fixtures ────────────────────────────────────────────────


class _LinearStack(nn.Module):
    """Simple model with layers of varying sizes for FSDP testing.

    - fc1: 128 x 128 (16384 weight elements, 128 bias)
    - fc2: 64 x 64 (4096 weight elements, 64 bias)
    - small: 4 x 4 (16 weight elements, 4 bias) — below default min_param_size
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(64, 64)
        self.small = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)


class _EmptyModule(nn.Module):
    """Module with no parameters."""


class _FewParamModule(nn.Module):
    """Module with a single small parameter for edge-case tests."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(20))


class _ModuleWithList(nn.Module):
    """Module containing ModuleList for nested name resolution tests."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])


# ── FSDPConfig Tests ───────────────────────────────────────────────────


class TestFSDPConfig:
    """FSDPConfig dataclass construction and defaults."""

    def test_default_values(self) -> None:
        """Default FSDPConfig matches expected defaults."""
        config = FSDPConfig()
        assert config.world_size == 1
        assert config.rank == 0
        assert config.min_param_size == 1024
        assert config.cpu_offload is False
        assert config.mixed_precision is None

    def test_custom_values(self) -> None:
        """All fields can be set at construction."""
        config = FSDPConfig(
            world_size=8,
            rank=3,
            min_param_size=2048,
            cpu_offload=True,
            mixed_precision=torch.float16,
        )
        assert config.world_size == 8
        assert config.rank == 3
        assert config.min_param_size == 2048
        assert config.cpu_offload is True
        assert config.mixed_precision is torch.float16

    def test_mixed_precision_none(self) -> None:
        """mixed_precision can be explicitly None."""
        config = FSDPConfig(mixed_precision=None)
        assert config.mixed_precision is None

    def test_min_param_size_zero(self) -> None:
        """min_param_size can be 0 -- all parameters eligible for sharding."""
        config = FSDPConfig(min_param_size=0)
        assert config.min_param_size == 0


# ── FSDPShard Construction Tests ───────────────────────────────────────


class TestFSDPShardConstruction:
    """FSDPShard constructor validation."""

    def test_default_config(self) -> None:
        """Constructed with only module uses default FSDPConfig."""
        model = _LinearStack()
        fsdp = FSDPShard(model)
        assert fsdp._config.world_size == 1
        assert fsdp._config.rank == 0

    def test_explicit_world_size_and_rank(self) -> None:
        """Explicit world_size/rank override defaults."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=4, rank=2)
        assert fsdp._config.world_size == 4
        assert fsdp._config.rank == 2

    def test_config_object(self) -> None:
        """Passing an FSDPConfig object is used directly."""
        model = _LinearStack()
        config = FSDPConfig(world_size=2, rank=1)
        fsdp = FSDPShard(model, config=config)
        assert fsdp._config.world_size == 2
        assert fsdp._config.rank == 1
        assert fsdp._config is config

    def test_explicit_args_override_config(self) -> None:
        """Explicit world_size/rank take precedence over config fields."""
        model = _LinearStack()
        config = FSDPConfig(world_size=2, rank=0)
        fsdp = FSDPShard(model, world_size=8, rank=3, config=config)
        assert fsdp._config.world_size == 8
        assert fsdp._config.rank == 3

    def test_invalid_world_size_zero(self) -> None:
        """world_size=0 raises ValueError."""
        model = _LinearStack()
        with pytest.raises(ValueError, match="world_size"):
            FSDPShard(model, world_size=0)

    def test_invalid_world_size_negative(self) -> None:
        """Negative world_size raises ValueError."""
        model = _LinearStack()
        with pytest.raises(ValueError, match="world_size"):
            FSDPShard(model, world_size=-1)

    def test_invalid_rank_negative(self) -> None:
        """Negative rank raises ValueError."""
        model = _LinearStack()
        with pytest.raises(ValueError, match="rank"):
            FSDPShard(model, world_size=4, rank=-1)

    def test_invalid_rank_too_high(self) -> None:
        """rank >= world_size raises ValueError."""
        model = _LinearStack()
        with pytest.raises(ValueError, match="rank"):
            FSDPShard(model, world_size=4, rank=4)

    def test_rank_zero_is_valid(self) -> None:
        """rank=0 is always valid for world_size >= 1."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=1, rank=0)
        assert fsdp._config.rank == 0

    def test_last_rank_is_valid(self) -> None:
        """rank=world_size-1 is valid (edge of the valid range)."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=8, rank=7)
        assert fsdp._config.rank == 7

    def test_empty_module_constructs(self) -> None:
        """Constructing with a parameterless module works."""
        model = _EmptyModule()
        fsdp = FSDPShard(model, world_size=1)
        assert fsdp._module is model


# ── FSDPShard: world_size=1 (no-op mode) ───────────────────────────────


class TestFSDPShardWorldSizeOne:
    """All public methods are no-ops when world_size <= 1."""

    def test_shard_noop(self) -> None:
        """shard() does not modify parameters or populate _sharded_params."""
        model = _LinearStack()
        original_weight = model.fc1.weight.data.clone()
        fsdp = FSDPShard(model, world_size=1)
        fsdp.shard()
        assert not fsdp._sharded_params
        assert torch.equal(model.fc1.weight.data, original_weight)

    def test_gather_noop(self) -> None:
        """gather() is a no-op; _gathered stays False."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=1)
        fsdp.gather()
        assert fsdp._gathered is False

    def test_free_noop(self) -> None:
        """free() is a no-op; _gathered stays False."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=1)
        fsdp.free()
        assert fsdp._gathered is False

    def test_forward_passthrough(self) -> None:
        """forward() delegates directly to the module when world_size=1."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=1)
        x = torch.randn(2, 128)
        output = fsdp.forward(x)
        expected = model(x)
        assert torch.equal(output, expected)

    def test_shard_gather_free_cycle(self) -> None:
        """Full shard-gather-free cycle does not alter parameters."""
        model = _LinearStack()
        original = {n: p.data.clone() for n, p in model.named_parameters()}
        fsdp = FSDPShard(model, world_size=1)
        fsdp.shard()
        fsdp.gather()
        fsdp.free()
        for n, p in model.named_parameters():
            assert torch.equal(p.data, original[n])


# ── FSDPShard: sharding logic (world_size > 1, no dist) ───────────────


class TestFSDPShardSharding:
    """Shard logic without a distributed process group.

    shard() does not call ``torch.distributed`` -- it only splits tensors
    locally.  These tests verify the split is correct.
    """

    WORLD_SIZE = 4
    RANK = 0

    def test_shard_populates_sharded_params(self) -> None:
        """Large parameters are recorded in ``_sharded_params``."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()
        # fc1.weight has 16384 >= 1024 -> sharded
        assert "fc1.weight" in fsdp._sharded_params
        # fc2.weight has 4096 >= 1024 -> sharded
        assert "fc2.weight" in fsdp._sharded_params

    def test_shard_skips_small_params(self) -> None:
        """Parameters below ``min_param_size`` (1024) are not sharded."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()
        # small.weight has 16 elements < 1024
        assert "small.weight" not in fsdp._sharded_params
        # All biases are < 1024
        assert "fc1.bias" not in fsdp._sharded_params
        assert "fc2.bias" not in fsdp._sharded_params
        assert "small.bias" not in fsdp._sharded_params

    def test_shard_skips_params_below_world_size(self) -> None:
        """Parameters with numel < world_size are not sharded.

        Uses world_size=32, min_param_size=16, and a 20-element param.
        The param passes the min_param_size check (20 >= 16) but
        fails the world_size check (20 < 32).
        """
        model = _FewParamModule()
        fsdp = FSDPShard(
            model,
            world_size=32,
            rank=0,
            config=FSDPConfig(min_param_size=16),
        )
        fsdp.shard()
        assert not fsdp._sharded_params

    @pytest.mark.parametrize(
        ("rank", "numel_expected"),
        [
            (0, 4096),
            (1, 4096),
            (2, 4096),
            (3, 4096),
        ],
    )
    def test_shard_correct_chunk_size(self, rank: int, numel_expected: int) -> None:
        """Each rank receives the expected number of elements.

        fc1.weight has 16384 elements.  With world_size=4:
        chunk_size = ceil(16384 / 4) = 4096 elements per rank.
        """
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=rank)
        fsdp.shard()
        # Grab the first (and any) sharded parameter to verify chunk size.
        name, (local_shard, *_rest) = next(
            iter(fsdp._sharded_params.items())
        )
        assert local_shard.numel() == numel_expected, (
            f"rank {rank} expected {numel_expected} elements, "
            f"got {local_shard.numel()} for {name}"
        )

    def test_shard_chunk_content_rank_zero(self) -> None:
        """Rank 0 receives the first chunk of the flattened parameter."""
        model = _LinearStack()
        full_weight = model.fc1.weight.data.flatten().clone()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=0)
        fsdp.shard()
        local_shard, *_rest = fsdp._sharded_params["fc1.weight"]
        expected_chunk = full_weight[:4096]
        assert torch.equal(local_shard, expected_chunk)

    def test_shard_chunk_content_rank_last(self) -> None:
        """Last rank receives the final chunk of the flattened parameter."""
        model = _LinearStack()
        full_weight = model.fc1.weight.data.flatten().clone()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=3)
        fsdp.shard()
        local_shard, *_rest = fsdp._sharded_params["fc1.weight"]
        expected_chunk = full_weight[12288:]  # 3 * 4096 = 12288
        assert torch.equal(local_shard, expected_chunk)

    def test_shard_replaces_param_data(self) -> None:
        """After shard(), param.data is the flat local chunk, not the full tensor."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()
        # fc1.weight was [128, 128]; after shard it is flat [4096]
        assert model.fc1.weight.data.numel() == 4096
        assert model.fc1.weight.data.dtype == torch.float32

    def test_shard_small_params_unchanged(self) -> None:
        """Non-sharded parameters retain their original shapes and values."""
        model = _LinearStack()
        original = model.small.weight.data.clone()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()
        # small.weight was [4, 4] and should remain so
        assert model.small.weight.data.shape == (4, 4)
        assert torch.equal(model.small.weight.data, original)
        assert model.small.bias.data.shape == (4,)

    def test_shard_twice_does_not_raise(self) -> None:
        """Calling shard() a second time does not raise, even though it
        operates on already-sharded parameters (further splitting them)."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()
        # Second call operates on flat shards (smaller tensors), which
        # may be further split -- the important thing is it does not crash.
        fsdp.shard()

    def test_shard_empty_module(self) -> None:
        """Sharding an empty module does not raise."""
        model = _EmptyModule()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()
        assert not fsdp._sharded_params

    def test_shard_preserves_metadata(self) -> None:
        """Each ``_sharded_params`` entry stores correct original metadata."""
        model = _LinearStack()
        original_shape = model.fc1.weight.data.shape
        original_dtype = model.fc1.weight.data.dtype
        original_device = model.fc1.weight.data.device

        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()

        _local_shard, orig_shape, orig_dtype, orig_device = fsdp._sharded_params[
            "fc1.weight"
        ]
        assert orig_shape == original_shape
        assert orig_dtype == original_dtype
        assert orig_device == original_device

    def test_shard_with_min_param_size_zero(self) -> None:
        """Setting min_param_size=0 shards all parameters >= world_size."""
        model = _LinearStack()
        config = FSDPConfig(min_param_size=0)
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK, config=config)
        fsdp.shard()
        # fc1.weight (16384), fc2.weight (4096), fc1.bias (128), fc2.bias (64),
        # small.weight (16), small.bias (4) should all be sharded
        # (all >= min_param_size=0 AND >= world_size=4, since numel >= ws
        #  means the guard "numel < ws" is False)
        assert "fc1.weight" in fsdp._sharded_params
        assert "fc2.weight" in fsdp._sharded_params
        assert "fc1.bias" in fsdp._sharded_params
        assert "fc2.bias" in fsdp._sharded_params
        assert "small.weight" in fsdp._sharded_params
        assert "small.bias" in fsdp._sharded_params


# ── FSDPShard: free() restoration ──────────────────────────────────────


class TestFSDPShardFreeAfterShard:
    """free() behaviour when ``_gathered`` is controlled directly.

    ``gather()`` requires a live ``torch.distributed`` process group which
    these unit tests do not set up.  We set ``_gathered = True`` directly
    on the object to exercise the free() body and verify local-shard
    restoration.
    """

    WORLD_SIZE = 4
    RANK = 0

    def test_free_early_return_when_not_gathered(self) -> None:
        """free() returns immediately when ``_gathered`` is False."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()
        assert fsdp._gathered is False
        fsdp.free()
        # _gathered must remain False (free() exited early)
        assert fsdp._gathered is False

    def test_free_early_return_when_no_sharded_params(self) -> None:
        """free() returns immediately when ``_sharded_params`` is empty."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        # No shard() called -- _sharded_params is empty
        fsdp._gathered = True  # artificially set to True
        fsdp.free()
        # free() exits early because ``not self._sharded_params``
        assert fsdp._gathered is True  # unchanged

    def test_free_restores_local_shard(self) -> None:
        """free() restores each sharded parameter to its local chunk."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()

        # Record the local shard values as they were after shard()
        local_values: dict[str, torch.Tensor] = {}
        for name, (local_shard, *_rest) in fsdp._sharded_params.items():
            local_values[name] = local_shard.clone()

        # Simulate that gather completed
        fsdp._gathered = True
        fsdp.free()

        # After free(), param.data must equal the local shard
        assert fsdp._gathered is False
        for name in fsdp._sharded_params:
            param = dict(model.named_parameters())[name]
            assert torch.equal(param.data, local_values[name])

    def test_free_resets_gathered_flag(self) -> None:
        """free() sets ``_gathered`` back to False."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=self.WORLD_SIZE, rank=self.RANK)
        fsdp.shard()
        fsdp._gathered = True
        fsdp.free()
        assert fsdp._gathered is False

    def test_free_with_cpu_offload_config(self) -> None:
        """free() works unaffected when cpu_offload config is set."""
        config = FSDPConfig(world_size=4, rank=0, cpu_offload=True)
        model = _LinearStack()
        fsdp = FSDPShard(model, config=config)
        fsdp.shard()
        fsdp._gathered = True
        fsdp.free()
        assert fsdp._gathered is False

    def test_free_with_nonzero_rank(self) -> None:
        """free() restores local shard correctly on a non-zero rank."""
        model = _LinearStack()
        fsdp = FSDPShard(model, world_size=4, rank=2)
        fsdp.shard()

        local_values: dict[str, torch.Tensor] = {}
        for name, (local_shard, *_rest) in fsdp._sharded_params.items():
            local_values[name] = local_shard.clone()

        fsdp._gathered = True
        fsdp.free()

        assert fsdp._gathered is False
        for name in fsdp._sharded_params:
            param = dict(model.named_parameters())[name]
            assert torch.equal(param.data, local_values[name])


# ── FSDPShard: mixed_precision config ──────────────────────────────────


class TestFSDPShardMixedPrecision:
    """Mixed precision configuration storage (applied during gather only)."""

    def test_config_stores_dtype(self) -> None:
        """mixed_precision dtype is stored in the config."""
        config = FSDPConfig(mixed_precision=torch.float16)
        assert config.mixed_precision is torch.float16

    def test_shard_with_mixed_precision_no_error(self) -> None:
        """shard() does not use mixed_precision -- does not raise."""
        model = _LinearStack()
        config = FSDPConfig(world_size=2, rank=0, mixed_precision=torch.float16)
        fsdp = FSDPShard(model, config=config)
        fsdp.shard()
        assert "fc1.weight" in fsdp._sharded_params
        # shard() stores the local chunk in its original dtype
        local_shard, _orig_shape, orig_dtype, _orig_device = fsdp._sharded_params[
            "fc1.weight"
        ]
        assert local_shard.dtype == orig_dtype


# ── Helper: _ceil_div ──────────────────────────────────────────────────


class TestCeilDiv:
    """Ceiling division helper."""

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (10, 5, 2),   # exact division
            (10, 3, 4),   # round up
            (5, 5, 1),    # equal
            (3, 5, 1),    # numerator smaller
            (0, 5, 0),    # numerator zero
            (1, 1, 1),    # both one
            (100, 1, 100),  # denominator one
            (7, 2, 4),    # typical non-exact
        ],
    )
    def test_ceil_div(self, a: int, b: int, expected: int) -> None:
        assert _ceil_div(a, b) == expected


# ── Helper: _resolve_device ────────────────────────────────────────────


class TestResolveDevice:
    """Device resolution helper."""

    def test_module_with_params(self) -> None:
        """Returns the device of the first parameter in the module."""
        model = _LinearStack()
        device = _resolve_device(model)
        assert device == torch.device("cpu")

    def test_empty_module_fallback(self) -> None:
        """Returns cuda:0 when CUDA is available, CPU otherwise for empty module."""
        model = _EmptyModule()
        device = _resolve_device(model)
        expected = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert device == expected


# ── Helper: _get_param_by_name ─────────────────────────────────────────


class TestGetParamByName:
    """Parameter name resolution helper."""

    def test_simple_weight(self) -> None:
        """Resolves ``'fc1.weight'`` correctly."""
        model = _LinearStack()
        param = _get_param_by_name(model, "fc1.weight")
        assert param is not None
        assert isinstance(param, nn.Parameter)
        assert param.shape == (128, 128)

    def test_simple_bias(self) -> None:
        """Resolves ``'fc1.bias'`` correctly."""
        model = _LinearStack()
        param = _get_param_by_name(model, "fc1.bias")
        assert param is not None
        assert isinstance(param, nn.Parameter)

    def test_nonexistent_param(self) -> None:
        """Returns None for a nonexistent top-level parameter name."""
        model = _LinearStack()
        assert _get_param_by_name(model, "nonexistent") is None

    def test_nonexistent_subparam(self) -> None:
        """Returns None for a nonexistent sub-parameter name."""
        model = _LinearStack()
        assert _get_param_by_name(model, "fc1.nonexistent") is None

    def test_module_not_param(self) -> None:
        """Returns None when the resolved object is a Module, not Parameter."""
        model = _LinearStack()
        assert _get_param_by_name(model, "fc1") is None

    def test_empty_string(self) -> None:
        """An empty name string returns None."""
        model = _LinearStack()
        assert _get_param_by_name(model, "") is None

    def test_module_list_index(self) -> None:
        """Resolves ``'layers.0.weight'`` through a ModuleList."""
        model = _ModuleWithList()
        param = _get_param_by_name(model, "layers.0.weight")
        assert param is not None
        assert isinstance(param, nn.Parameter)
        assert param.shape == (10, 10)

    def test_module_list_invalid_index(self) -> None:
        """Returns None for an out-of-range ModuleList index."""
        model = _ModuleWithList()
        assert _get_param_by_name(model, "layers.99.weight") is None

    def test_module_list_non_numeric(self) -> None:
        """Returns None for a non-numeric ModuleList index."""
        model = _ModuleWithList()
        assert _get_param_by_name(model, "layers.abc") is None

    def test_nonexistent_module(self) -> None:
        """Returns None when an intermediate module does not exist."""
        model = _LinearStack()
        assert _get_param_by_name(model, "missing_layer.weight") is None
