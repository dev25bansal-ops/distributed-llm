"""Tests for ParamUpdateChannel -- mid-stream parameter updates.

Covers:
- GenerationParams dataclass defaults
- GenerationParams.update modifies fields
- Construction with empty channels
- register and get
- update pushes changes
- unregister removes channel
- list_requests and __contains__ and __len__

No MagicMock -- real dicts and threading.Lock.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/param_update_channel.py")
ParamUpdateChannel = _mod.ParamUpdateChannel
GenerationParams = _mod.GenerationParams


class TestGenerationParams:
    """GenerationParams dataclass."""

    def test_defaults(self) -> None:
        p = GenerationParams()
        assert p.temperature == 0.7
        assert p.top_p == 0.9
        assert p.top_k == 0
        assert p.max_tokens == 128

    def test_custom_values(self) -> None:
        p = GenerationParams(temperature=0.1, top_p=0.5, top_k=50, max_tokens=256)
        assert p.temperature == 0.1
        assert p.top_k == 50

    def test_update_existing_field(self) -> None:
        p = GenerationParams()
        p.update(temperature=0.3)
        assert p.temperature == 0.3

    def test_update_ignores_unknown(self) -> None:
        p = GenerationParams()
        p.update(unknown_param=42)  # should not raise
        assert not hasattr(p, "unknown_param")

    def test_update_multiple(self) -> None:
        p = GenerationParams(temperature=0.7, top_p=0.9)
        p.update(temperature=0.5, top_p=0.8, max_tokens=512)
        assert p.temperature == 0.5
        assert p.top_p == 0.8
        assert p.max_tokens == 512


class TestParamUpdateChannelConstruction:
    """Construction and initial state."""

    def test_default_construction(self) -> None:
        channel = ParamUpdateChannel()
        assert channel._channels == {}
        assert len(channel) == 0

    def test_new_channel_empty(self) -> None:
        channel = ParamUpdateChannel()
        assert channel.list_requests() == []


class TestParamUpdateChannelRegister:
    """Registration."""

    def test_register_with_defaults(self) -> None:
        channel = ParamUpdateChannel()
        channel.register("req-1")
        params = channel.get("req-1")
        assert params is not None
        assert params.temperature == 0.7

    def test_register_with_custom_params(self) -> None:
        channel = ParamUpdateChannel()
        p = GenerationParams(temperature=0.1)
        channel.register("req-1", p)
        assert channel.get("req-1") is p

    def test_get_returns_none_for_unknown(self) -> None:
        channel = ParamUpdateChannel()
        assert channel.get("nonexistent") is None


class TestParamUpdateChannelUpdate:
    """Mid-stream updates."""

    def test_update_existing_request(self) -> None:
        channel = ParamUpdateChannel()
        channel.register("req-1")
        channel.update("req-1", temperature=0.2)
        params = channel.get("req-1")
        assert params is not None
        assert params.temperature == 0.2

    def test_update_nonexistent_does_nothing(self) -> None:
        channel = ParamUpdateChannel()
        channel.update("nonexistent", temperature=0.5)  # should not raise
        assert channel.get("nonexistent") is None

    def test_update_multiple_params(self) -> None:
        channel = ParamUpdateChannel()
        channel.register("req-1")
        channel.update("req-1", temperature=0.3, top_k=100)
        params = channel.get("req-1")
        assert params is not None
        assert params.temperature == 0.3
        assert params.top_k == 100


class TestParamUpdateChannelUnregister:
    """Unregistration."""

    def test_unregister(self) -> None:
        channel = ParamUpdateChannel()
        channel.register("req-1")
        channel.unregister("req-1")
        assert channel.get("req-1") is None

    def test_unregister_nonexistent(self) -> None:
        channel = ParamUpdateChannel()
        channel.unregister("nonexistent")  # should not raise

    def test_list_requests(self) -> None:
        channel = ParamUpdateChannel()
        channel.register("req-1")
        channel.register("req-2")
        requests = channel.list_requests()
        assert "req-1" in requests
        assert "req-2" in requests

    def test_contains(self) -> None:
        channel = ParamUpdateChannel()
        channel.register("req-1")
        assert "req-1" in channel
        assert "nonexistent" not in channel

    def test_len(self) -> None:
        channel = ParamUpdateChannel()
        assert len(channel) == 0
        channel.register("req-1")
        assert len(channel) == 1
        channel.register("req-2")
        assert len(channel) == 2
