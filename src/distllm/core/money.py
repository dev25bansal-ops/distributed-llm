"""Exact money math for billing accumulators (M1).

The cost ledgers used plain ``float`` +=, which drifts at billing volume
(10,000 x $0.0001 must be exactly $1.00). All accumulation goes through
:class:`MoneyAccumulator`, which keeps a Decimal total; the public header
API keeps returning float strings unchanged.
"""

from __future__ import annotations

from decimal import Decimal


class MoneyAccumulator:
    """Thread-safe-enough additive accumulator backed by Decimal.

    Instances are used under the owning ledger's lock; no internal lock.
    """

    __slots__ = ("_total",)

    def __init__(self) -> None:
        self._total = Decimal(0)

    def add(self, amount: float | Decimal | int) -> None:
        if isinstance(amount, Decimal):
            self._total += amount
        else:
            # str() preserves the decimal literal instead of inheriting
            # binary-float representation error.
            self._total += Decimal(str(amount))

    def value(self) -> Decimal:
        return self._total

    def __float__(self) -> float:
        return float(self._total)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"MoneyAccumulator({self._total})"
