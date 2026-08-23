"""Tests for TokenCredit and TokenBank."""

from __future__ import annotations

from tests._import_helper import bootstrap_fake_packages, load_module

bootstrap_fake_packages()

_tb = load_module("distllm/core/advanced_scheduling/token_bank.py")
TokenCredit = _tb.TokenCredit
TokenBank = _tb.TokenBank


class TestTokenCredit:
    """Test suite for TokenCredit dataclass."""

    def test_default_construction(self) -> None:
        credit = TokenCredit(user_id="user-1")
        assert credit.user_id == "user-1"
        assert credit.balance == 0
        assert credit.total_earned == 0
        assert credit.total_spent == 0
        assert isinstance(credit.last_updated, float)

    def test_custom_values(self) -> None:
        import time
        ts = time.time()
        credit = TokenCredit(
            user_id="user-1",
            balance=100,
            total_earned=500,
            total_spent=400,
            last_updated=ts,
        )
        assert credit.balance == 100
        assert credit.total_earned == 500
        assert credit.total_spent == 400
        assert credit.last_updated == ts


class TestTokenBank:
    """Test suite for TokenBank."""

    def test_default_construction(self) -> None:
        bank = TokenBank()
        assert bank._credits == {}

    def test_get_balance_unknown_user_returns_zero(self) -> None:
        bank = TokenBank()
        assert bank.get_balance("nonexistent") == 0

    def test_add_credits_creates_new_user(self) -> None:
        bank = TokenBank()
        credit = bank.add_credits("user-1", 100)
        assert credit.user_id == "user-1"
        assert credit.balance == 100
        assert credit.total_earned == 100

    def test_add_credits_existing_user(self) -> None:
        bank = TokenBank()
        bank.add_credits("user-1", 100)
        credit = bank.add_credits("user-1", 50)
        assert credit.balance == 150
        assert credit.total_earned == 150
        assert credit.total_spent == 0

    def test_get_balance_after_add(self) -> None:
        bank = TokenBank()
        bank.add_credits("user-1", 200)
        assert bank.get_balance("user-1") == 200

    def test_spend_credits_success(self) -> None:
        bank = TokenBank()
        bank.add_credits("user-1", 100)
        assert bank.spend_credits("user-1", 30) is True
        assert bank.get_balance("user-1") == 70
        assert bank._credits["user-1"].total_spent == 30

    def test_spend_credits_insufficient_balance(self) -> None:
        bank = TokenBank()
        bank.add_credits("user-1", 10)
        assert bank.spend_credits("user-1", 20) is False
        # Balance unchanged
        assert bank.get_balance("user-1") == 10

    def test_spend_credits_unknown_user(self) -> None:
        bank = TokenBank()
        assert bank.spend_credits("unknown", 5) is False

    def test_spend_credits_exact_balance(self) -> None:
        bank = TokenBank()
        bank.add_credits("user-1", 50)
        assert bank.spend_credits("user-1", 50) is True
        assert bank.get_balance("user-1") == 0

    def test_get_summary_unknown_user(self) -> None:
        bank = TokenBank()
        summary = bank.get_summary("unknown")
        assert summary == {"user_id": "unknown", "balance": 0}

    def test_get_summary_known_user(self) -> None:
        bank = TokenBank()
        bank.add_credits("user-1", 500)
        bank.spend_credits("user-1", 150)

        summary = bank.get_summary("user-1")
        assert summary["user_id"] == "user-1"
        assert summary["balance"] == 350
        assert summary["total_earned"] == 500
        assert summary["total_spent"] == 150

    def test_multiple_users_independent(self) -> None:
        bank = TokenBank()
        bank.add_credits("alice", 100)
        bank.add_credits("bob", 200)

        assert bank.get_balance("alice") == 100
        assert bank.get_balance("bob") == 200

        bank.spend_credits("alice", 50)
        assert bank.get_balance("alice") == 50
        # Bob's balance is unaffected
        assert bank.get_balance("bob") == 200
