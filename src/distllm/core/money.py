"""Exact money math for billing accumulators (M1).

The cost ledgers used plain ``float`` +=, which drifts at billing volume
(10,000 x $0.0001 must be exactly $1.00). All accumulation goes through
:class:`MoneyAccumulator`, which keeps a Decimal total; the public header
API keeps returning float strings unchanged.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal


class Money:
    """Exact additive money value with one-shot cent quantization.

    Unlike repeated ``round(x + y, 6)`` float accumulation (which truncates
    and drifts), :meth:`add` keeps an exact Decimal total; :meth:`as_float`
    quantizes to cents ONCE at the boundary.
    """

    __slots__ = ("_total",)

    def __init__(self, start: float | Decimal | int = 0.0) -> None:
        self._total = start if isinstance(start, Decimal) else Decimal(str(start))

    def add(self, amount: float | Decimal | int) -> None:
        if isinstance(amount, Decimal):
            self._total += amount
        else:
            # str() preserves the decimal literal instead of inheriting
            # binary-float representation error.
            self._total += Decimal(str(amount))

    def value(self) -> Decimal:
        return self._total

    def as_float(self) -> float:
        return float(self._total.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))

    def __float__(self) -> float:
        return float(self._total)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Money({self._total})"


class MoneyAccumulator(Money):
    """Accumulator alias of :class:`Money` (used under the ledger's lock)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"MoneyAccumulator({self._total})"
