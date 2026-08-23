"""Tests for protocol types (ErrorCode, etc.) in distllm.core.types.

Covers:
- ErrorCode enum values
- ErrorCode name lookup
- ErrorCode comparison

No MagicMock -- pure enum.
"""

from __future__ import annotations

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_mod = load_module("distllm/core/types.py")
ErrorCode = _mod.ErrorCode


class TestErrorCode:
    """ErrorCode IntEnum."""

    def test_enum_values(self) -> None:
        assert ErrorCode.UNKNOWN == 0
        assert ErrorCode.MODEL_ERROR == 1
        assert ErrorCode.OOM == 2
        assert ErrorCode.TIMEOUT == 3
        assert ErrorCode.INVALID_INPUT == 4
        assert ErrorCode.NODE_UNREACHABLE == 5
        assert ErrorCode.CIRCUIT_BREAKER_OPEN == 6

    def test_enum_members(self) -> None:
        assert ErrorCode(0) == ErrorCode.UNKNOWN
        assert ErrorCode(3) == ErrorCode.TIMEOUT
        assert ErrorCode(6) == ErrorCode.CIRCUIT_BREAKER_OPEN

    def test_enum_names(self) -> None:
        assert ErrorCode.UNKNOWN.name == "UNKNOWN"
        assert ErrorCode.TIMEOUT.name == "TIMEOUT"
        assert ErrorCode.CIRCUIT_BREAKER_OPEN.name == "CIRCUIT_BREAKER_OPEN"

    def test_enum_ordering(self) -> None:
        codes = sorted([ErrorCode.TIMEOUT, ErrorCode.UNKNOWN, ErrorCode.OOM])
        assert codes == [ErrorCode.UNKNOWN, ErrorCode.OOM, ErrorCode.TIMEOUT]

    def test_enum_is_int(self) -> None:
        assert isinstance(ErrorCode.OOM, int)
        assert int(ErrorCode.TIMEOUT) == 3

    def test_all_members_present(self) -> None:
        expected_names = {
            "UNKNOWN", "MODEL_ERROR", "OOM", "TIMEOUT",
            "INVALID_INPUT", "NODE_UNREACHABLE", "CIRCUIT_BREAKER_OPEN",
        }
        actual_names = {m.name for m in ErrorCode}
        assert actual_names == expected_names
        assert len(ErrorCode) == 7
