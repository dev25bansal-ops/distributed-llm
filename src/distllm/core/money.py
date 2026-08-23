"""Decimal-based money helpers for the billing ledger.

M1: ledger math was done in float and ``tenant.total_cost`` was re-rounded on
every ``+=`` (``round(x + y, 6)``), which accumulates binary-float drift at
billing volume. These helpers keep running monetary sums as ``Decimal`` and
quantize exactly once, at display time, to a fixed number of decimal places.

Public API stays float-compatible: callers that want a number get ``float``,
callers that want exact display get a quantized ``Decimal`` / string.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

# Two decimal places is the smallest meaningful billing unit (cents); the
# accumulator itself keeps full precision and is quantized only on read.
QUANTIZE = Decimal("0.01")


class Money:
    """An exact monetary accumulator backed by Decimal.

    Addition is exact (no per-``+=`` rounding). Use :meth:`value` /
    :meth:`as_float` to read a display/compatible representation, quantized
    once to ``QUANTIZE``.
    """

    __slots__ = ("_value",)

    def __init__(self, amount: Decimal | float | int | str = 0) -> None:
        self._value = Decimal(str(amount)) if not isinstance(amount, Decimal) else amount

    def add(self, amount: Decimal | float | int | str) -> Money:
        self._value += Decimal(str(amount)) if not isinstance(amount, Decimal) else amount
        return self

    def __iadd__(self, amount: Decimal | float | int | str) -> Money:
        return self.add(amount)

    def value(self) -> Decimal:
        """Exact (unrounded) Decimal value of the accumulator."""
        return self._value

    def quantized(self) -> Decimal:
        """Value quantized once to the display unit (no intermediate rounding)."""
        return self._value.quantize(QUANTIZE, rounding=ROUND_HALF_UP)

    def as_float(self) -> float:
        """Float-compatible view (for callers expecting float)."""
        return float(self.quantized())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Money({self._value})"
