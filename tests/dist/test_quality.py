"""Tests for distllm.dist.quality — precision-aware quality SLA enforcement."""

import torch
import pytest

from distllm.dist.quality import (
    QualityTier,
    SLAPolicy,
    PRECISION_RANK,
    QualitySLA,
)


class TestQualityTier:
    """Enum with three tiers: high, medium, low."""

    def test_members(self) -> None:
        assert QualityTier.HIGH.value == "high"
        assert QualityTier.MEDIUM.value == "medium"
        assert QualityTier.LOW.value == "low"

    def test_is_str_enum(self) -> None:
        assert QualityTier.HIGH.value == "high"


class TestSLAPolicy:
    """Dataclass that carries tier, min_precision, max_quality_loss, request_class."""

    def test_for_tier_high(self) -> None:
        policy = SLAPolicy.for_tier(QualityTier.HIGH)
        assert policy.tier == QualityTier.HIGH
        assert policy.min_precision == "bfloat16"
        assert policy.max_quality_loss == pytest.approx(0.01)
        assert policy.request_class == "creative"

    def test_for_tier_medium(self) -> None:
        policy = SLAPolicy.for_tier(QualityTier.MEDIUM)
        assert policy.tier == QualityTier.MEDIUM
        assert policy.min_precision == "float16"
        assert policy.max_quality_loss == pytest.approx(0.05)
        assert policy.request_class == "general"

    def test_for_tier_low(self) -> None:
        policy = SLAPolicy.for_tier(QualityTier.LOW)
        assert policy.tier == QualityTier.LOW
        assert policy.min_precision == "int8"
        assert policy.max_quality_loss == pytest.approx(0.2)
        assert policy.request_class == "classification"

    def test_default_request_class(self) -> None:
        policy = SLAPolicy(tier=QualityTier.MEDIUM, min_precision="float16", max_quality_loss=0.05)
        assert policy.request_class == ""


class TestPRECISION_RANK:
    """Module-level dict mapping precision strings to numeric ranks."""

    def test_all_entries_present(self) -> None:
        assert PRECISION_RANK["int4"] == 0
        assert PRECISION_RANK["int8"] == 1
        assert PRECISION_RANK["float8_e4m3fn"] == 2
        assert PRECISION_RANK["float8_e5m2"] == 2
        assert PRECISION_RANK["float16"] == 3
        assert PRECISION_RANK["bfloat16"] == 4
        assert PRECISION_RANK["float32"] == 5

    def test_unknown_precision_defaults_zero(self) -> None:
        assert PRECISION_RANK.get("fp64", 0) == 0
        assert PRECISION_RANK.get("", 0) == 0


class TestQualitySLA:
    """Static-method utility class for precision selection, SLA checks, quality eval, tier inference."""

    # -- select_precision_for_request ----------------------------------------

    @pytest.mark.parametrize("tier,expected", [
        (QualityTier.HIGH, torch.bfloat16),
        (QualityTier.MEDIUM, torch.float16),
        (QualityTier.LOW, torch.int8),
    ])
    def test_select_precision_for_request(self, tier: QualityTier, expected: torch.dtype) -> None:
        assert QualitySLA.select_precision_for_request(tier) is expected

    def test_select_precision_default_tier(self) -> None:
        assert QualitySLA.select_precision_for_request() is torch.float16

    # -- check_precision_meets_sla -------------------------------------------

    @pytest.mark.parametrize("actual,required,expected", [
        ("float32", QualityTier.HIGH, True),
        ("bfloat16", QualityTier.HIGH, True),
        ("float16", QualityTier.HIGH, False),
        ("int8", QualityTier.HIGH, False),
        ("int4", QualityTier.HIGH, False),
        ("bfloat16", QualityTier.MEDIUM, True),
        ("float16", QualityTier.MEDIUM, True),
        ("int8", QualityTier.MEDIUM, False),
        ("int8", QualityTier.LOW, True),
        ("int4", QualityTier.LOW, False),
        ("float32", QualityTier.LOW, True),
    ])
    def test_check_precision_meets_sla(
        self, actual: str, required: QualityTier, expected: bool,
    ) -> None:
        assert QualitySLA.check_precision_meets_sla(actual, required) is expected

    def test_check_precision_case_insensitive(self) -> None:
        assert QualitySLA.check_precision_meets_sla("BFLOAT16", QualityTier.HIGH) is True
        assert QualitySLA.check_precision_meets_sla("Int8", QualityTier.HIGH) is False

    def test_check_precision_unknown_actual_defaults_low(self) -> None:
        assert QualitySLA.check_precision_meets_sla("unknown_dtype", QualityTier.HIGH) is False

    # -- evaluate_quality ----------------------------------------------------

    def test_evaluate_quality_identical_logits(self) -> None:
        gold = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        candidate = gold.clone()
        kl = QualitySLA.evaluate_quality(gold, candidate)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_evaluate_quality_different_logits(self) -> None:
        gold = torch.tensor([[2.0, 0.0, 0.0]])
        candidate = torch.tensor([[0.0, 2.0, 0.0]])
        kl = QualitySLA.evaluate_quality(gold, candidate)
        assert kl > 0.0

    def test_evaluate_quality_shape_mismatch_raises(self) -> None:
        gold = torch.tensor([[1.0, 0.0]])
        candidate = torch.tensor([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Logit shapes must match"):
            QualitySLA.evaluate_quality(gold, candidate)

    def test_evaluate_quality_single_element(self) -> None:
        gold = torch.tensor([[0.0]])
        candidate = torch.tensor([[0.5]])
        kl = QualitySLA.evaluate_quality(gold, candidate)
        assert isinstance(kl, float)
        assert kl >= 0.0

    def test_evaluate_quality_batched(self) -> None:
        gold = torch.randn(4, 10)
        candidate = torch.randn(4, 10)
        kl = QualitySLA.evaluate_quality(gold, candidate)
        assert isinstance(kl, float)
        assert 0.0 <= kl <= 10.0  # reasonable KL range

    def test_evaluate_quality_three_dimensional(self) -> None:
        gold = torch.randn(2, 3, 5)
        candidate = torch.randn(2, 3, 5)
        kl = QualitySLA.evaluate_quality(gold, candidate)
        assert isinstance(kl, float)

    # -- infer_quality_tier_from_request -------------------------------------

    @pytest.mark.parametrize("prompt,expected", [
        ("write a story about AI", QualityTier.HIGH),
        ("Write code for a fibonacci function", QualityTier.HIGH),
        ("create a new project", QualityTier.HIGH),
        ("generate image description", QualityTier.HIGH),
        ("compose an email", QualityTier.HIGH),
        ("def add(a, b):", QualityTier.HIGH),
        ("class MyModel:", QualityTier.HIGH),
        ("function calculate()", QualityTier.HIGH),
        ("classify this text", QualityTier.LOW),
        ("what category does this belong to", QualityTier.LOW),
        ("label the following items", QualityTier.LOW),
        ("answer yes or no", QualityTier.LOW),
        ("is this true or false", QualityTier.LOW),
        ("extract the names", QualityTier.LOW),
        ("sentiment analysis", QualityTier.LOW),
        ("translate hello world", QualityTier.MEDIUM),
        ("summarize this article", QualityTier.MEDIUM),
        ("", QualityTier.MEDIUM),
        ("What is the weather today?", QualityTier.MEDIUM),
    ])
    def test_infer_quality_tier_from_request(
        self, prompt: str, expected: QualityTier,
    ) -> None:
        assert QualitySLA.infer_quality_tier_from_request(prompt) is expected

    def test_infer_tier_high_takes_precedence_over_low(self) -> None:
        prompt = "classify and write a story"
        assert QualitySLA.infer_quality_tier_from_request(prompt) is QualityTier.HIGH

    def test_infer_tier_empty_string(self) -> None:
        assert QualitySLA.infer_quality_tier_from_request("") is QualityTier.MEDIUM

    def test_infer_tier_case_insensitive(self) -> None:
        assert QualitySLA.infer_quality_tier_from_request("WRITE A STORY") is QualityTier.HIGH
        assert QualitySLA.infer_quality_tier_from_request("CLASSIFY") is QualityTier.LOW
