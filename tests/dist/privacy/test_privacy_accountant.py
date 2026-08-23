"""Tests for the privacy accountant."""
from distllm.dist.privacy.privacy_accountant import EpsilonBudget, PrivacyAccountant, DPInferenceConfig


class TestEpsilonBudget:
    def test_budget_creation(self):
        budget = EpsilonBudget(user_id="user-1", total_epsilon=10.0)
        assert budget.user_id == "user-1"
        assert budget.remaining == 10.0

    def test_budget_spend(self):
        budget = EpsilonBudget(user_id="user-1", total_epsilon=10.0)
        assert budget.can_serve(request_epsilon=3.0) is True
        budget.spend(request_epsilon=3.0)
        assert budget.spent_epsilon == 3.0
        assert budget.remaining == 7.0

    def test_budget_exhausted(self):
        budget = EpsilonBudget(user_id="user-1", total_epsilon=5.0)
        budget.spend(request_epsilon=5.0)
        assert budget.can_serve(request_epsilon=1.0) is False
        assert budget.remaining == 0.0


class TestPrivacyAccountant:
    def test_get_budget(self):
        accountant = PrivacyAccountant()
        budget = accountant.get_budget("user-42")
        assert isinstance(budget, EpsilonBudget)
        assert budget.user_id == "user-42"

    def test_check_and_record(self):
        accountant = PrivacyAccountant()
        assert accountant.check_request("user-42", request_epsilon=5.0) is True
        accountant.record_spend("user-42", 5.0, "test-model", 100)
        budget = accountant.get_budget("user-42")
        assert budget.spent_epsilon == 5.0

    def test_check_rejected(self):
        accountant = PrivacyAccountant()
        accountant.record_spend("user-42", 10.0, "test-model", 100)
        assert accountant.check_request("user-42", request_epsilon=1.0) is False

    def test_summary(self):
        accountant = PrivacyAccountant()
        accountant.record_spend("alice", 2.0, "model-a", 50)
        accountant.record_spend("bob", 3.0, "model-b", 100)
        summary = accountant.summary()
        assert isinstance(summary, dict)


class TestDPInferenceConfig:
    def test_default_config(self):
        config = DPInferenceConfig()
        assert config.mechanism == "gaussian"
        assert config.epsilon == 8.0
        assert config.injection_point == "logits"

    def test_custom_config(self):
        config = DPInferenceConfig(mechanism="laplace", epsilon=1.0, injection_point="embeddings")
        assert config.mechanism == "laplace"
        assert config.epsilon == 1.0
