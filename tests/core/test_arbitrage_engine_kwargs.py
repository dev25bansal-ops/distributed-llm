"""Tests for the ArbitrageEngine startup-kwargs compatibility shim.

The coordinator wires ArbitrageEngine with ``min_savings_pct`` and
``check_interval_s`` kwargs.  These tests exercise the helper
:func:`distllm.core.subsystem_kwargs.validate_subsystem_kwargs` (the
3rd-regression-of-the-same-shape guard) and the underlying ArbitrageEngine
constructor to confirm:

1. The two existing kwargs (``min_savings_pct``, ``check_interval_s``)
   still construct the engine successfully and apply to the thresholds.
2. An unknown kwarg (``foo=42``) now logs a warning instead of raising
   ``TypeError`` — and the engine still constructs.
3. The validator strips unknown kwargs and leaves valid ones intact.
"""

from __future__ import annotations

import pytest

from distllm.core.arbitrage_engine import ArbitrageEngine
from distllm.core.subsystem_kwargs import validate_subsystem_kwargs


@pytest.fixture
def captured_warnings() -> list[dict]:
    """Capture loguru records — pytest's caplog cannot see loguru output."""
    from loguru import logger as loguru_logger

    records: list[dict] = []

    def _sink(message) -> None:
        records.append(message.record)

    handler_id = loguru_logger.add(_sink, level="WARNING")
    yield records
    loguru_logger.remove(handler_id)


class TestArbitrageEngineKwargs:
    """Constructor compatibility for the kwargs the coordinator passes."""

    def test_min_savings_pct_applies_to_thresholds(self) -> None:
        """`min_savings_pct` is a backward-compat alias for the three thresholds."""
        engine = ArbitrageEngine(min_savings_pct=33.0)
        assert engine._price_drop_threshold == 33.0
        assert engine._region_savings_threshold == 33.0
        assert engine._provider_savings_threshold == 33.0

    def test_check_interval_s_is_accepted(self) -> None:
        """`check_interval_s` is a backward-compat alias for the scan interval."""
        engine = ArbitrageEngine(check_interval_s=12.5)
        assert engine._check_interval_s == 12.5

    def test_min_savings_pct_and_check_interval_s_together(self) -> None:
        """Both kwargs together still construct cleanly."""
        engine = ArbitrageEngine(min_savings_pct=20.0, check_interval_s=30.0)
        assert engine._price_drop_threshold == 20.0
        assert engine._check_interval_s == 30.0

    def test_check_interval_s_none_falls_back_to_default(self) -> None:
        """`check_interval_s=None` (the default) keeps the 60s default."""
        engine = ArbitrageEngine()
        assert engine._check_interval_s == 60.0


class TestValidateSubsystemKwargs:
    """The validator strips unknown kwargs with a warning."""

    def test_known_kwargs_pass_through(self) -> None:
        """Known kwargs are returned unchanged in the pruned dict."""
        pruned = validate_subsystem_kwargs(
            ArbitrageEngine,
            {"min_savings_pct": 10.0, "check_interval_s": 5.0},
            subsystem_name="arbitrage_engine",
        )
        assert pruned == {"min_savings_pct": 10.0, "check_interval_s": 5.0}

    def test_unknown_kwarg_is_dropped_with_warning(self, captured_warnings: list[dict]) -> None:
        """An unknown kwarg is dropped and a warning is logged."""
        pruned = validate_subsystem_kwargs(
            ArbitrageEngine,
            {"min_savings_pct": 10.0, "foo": 42},
            subsystem_name="arbitrage_engine",
        )
        # 'foo' is dropped, 'min_savings_pct' is preserved
        assert pruned == {"min_savings_pct": 10.0}
        # The warning mentions the dropped kwarg and the subsystem name
        assert any(
            "foo" in r["message"] and "arbitrage_engine" in r["message"]
            for r in captured_warnings
        )

    def test_unknown_kwarg_alone_does_not_raise(self, captured_warnings: list[dict]) -> None:
        """An unknown kwarg alone does NOT raise — it's stripped."""
        # Without the validator this would raise TypeError.  With the
        # validator it returns an empty dict and logs a warning.
        pruned = validate_subsystem_kwargs(
            ArbitrageEngine,
            {"foo": 42, "bar": "baz"},
            subsystem_name="arbitrage_engine",
        )
        assert pruned == {}
        # Both unknown kwargs warned
        msgs = [r["message"] for r in captured_warnings]
        assert any("foo" in m for m in msgs)
        assert any("bar" in m for m in msgs)

    def test_validator_does_not_mutate_input(self) -> None:
        """The input dict is not mutated (callers may reuse it)."""
        kwargs = {"min_savings_pct": 7.0, "foo": 42}
        original = dict(kwargs)
        validate_subsystem_kwargs(ArbitrageEngine, kwargs, subsystem_name="arbitrage_engine")
        assert kwargs == original

    def test_engine_constructs_through_validator_with_unknown_kwarg(self) -> None:
        """Full integration: validator output constructs the engine successfully.

        This is the regression scenario from earlier today — a new
        release passes ``foo=42`` (a kwarg the engine doesn't accept),
        and the validator drops it so the engine still starts.
        """
        pruned = validate_subsystem_kwargs(
            ArbitrageEngine,
            {"min_savings_pct": 10.0, "check_interval_s": 7.5, "foo": 42},
            subsystem_name="arbitrage_engine",
        )
        engine = ArbitrageEngine(**pruned)
        assert engine._price_drop_threshold == 10.0
        assert engine._check_interval_s == 7.5

    def test_empty_kwargs_returns_empty_dict(self) -> None:
        """Empty input returns an empty dict (no warnings, no errors)."""
        pruned = validate_subsystem_kwargs(ArbitrageEngine, {}, subsystem_name="x")
        assert pruned == {}

    def test_default_subsystem_name_uses_class_name(self, captured_warnings: list[dict]) -> None:
        """Without subsystem_name, the class name is used in the warning."""
        validate_subsystem_kwargs(ArbitrageEngine, {"unknown_kwarg": 1})
        assert any("ArbitrageEngine" in r["message"] for r in captured_warnings)