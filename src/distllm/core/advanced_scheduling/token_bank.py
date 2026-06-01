"""Token credit system for usage tracking and billing."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TokenCredit:
    """A token credit allocation."""
    user_id: str
    balance: int = 0
    total_earned: int = 0
    total_spent: int = 0
    last_updated: float = field(default_factory=time.time)


class TokenBank:
    """Manages token credits for users."""

    def __init__(self):
        self._credits: dict[str, TokenCredit] = {}
        self._lock = threading.Lock()

    def get_balance(self, user_id: str) -> int:
        with self._lock:
            credit = self._credits.get(user_id)
            return credit.balance if credit else 0

    def add_credits(self, user_id: str, amount: int) -> TokenCredit:
        with self._lock:
            if user_id not in self._credits:
                self._credits[user_id] = TokenCredit(user_id=user_id)
            credit = self._credits[user_id]
            credit.balance += amount
            credit.total_earned += amount
            credit.last_updated = time.time()
            return credit

    def spend_credits(self, user_id: str, amount: int) -> bool:
        with self._lock:
            credit = self._credits.get(user_id)
            if credit is None or credit.balance < amount:
                return False
            credit.balance -= amount
            credit.total_spent += amount
            credit.last_updated = time.time()
            return True

    def get_summary(self, user_id: str) -> dict:
        with self._lock:
            credit = self._credits.get(user_id)
            if credit is None:
                return {"user_id": user_id, "balance": 0}
            return {
                "user_id": credit.user_id,
                "balance": credit.balance,
                "total_earned": credit.total_earned,
                "total_spent": credit.total_spent,
            }
