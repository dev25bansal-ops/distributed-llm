"""Quality SLA enforcement for precision-aware serving.

Defines per-request quality budgets that map to minimum precision
requirements, preventing INT4 serving for high-quality requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class QualityTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SLAPolicy:
    tier: QualityTier
    min_precision: str
    max_quality_loss: float
    request_class: str = ""

    @classmethod
    def for_tier(cls, tier: QualityTier) -> "SLAPolicy":
        policies = {
            QualityTier.HIGH: cls(
                tier=tier,
                min_precision="bfloat16",
                max_quality_loss=0.01,
                request_class="creative",
            ),
            QualityTier.MEDIUM: cls(
                tier=tier,
                min_precision="float16",
                max_quality_loss=0.05,
                request_class="general",
            ),
            QualityTier.LOW: cls(
                tier=tier,
                min_precision="int8",
                max_quality_loss=0.2,
                request_class="classification",
            ),
        }
        return policies[tier]


PRECISION_RANK = {
    "int4": 0,
    "int8": 1,
    "float8_e4m3fn": 2,
    "float8_e5m2": 2,
    "float16": 3,
    "bfloat16": 4,
    "float32": 5,
}


class QualitySLA:
    @staticmethod
    def select_precision_for_request(
        quality_tier: QualityTier = QualityTier.MEDIUM,
    ) -> torch.dtype:
        policy = SLAPolicy.for_tier(quality_tier)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "int8": torch.int8,
        }
        return dtype_map.get(policy.min_precision, torch.float16)

    @staticmethod
    def check_precision_meets_sla(
        actual_precision: str,
        required_tier: QualityTier,
    ) -> bool:
        policy = SLAPolicy.for_tier(required_tier)
        actual_rank = PRECISION_RANK.get(actual_precision.lower(), 0)
        required_rank = PRECISION_RANK.get(policy.min_precision.lower(), 0)
        return actual_rank >= required_rank

    @staticmethod
    def evaluate_quality(
        gold_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
    ) -> float:
        if gold_logits.shape != candidate_logits.shape:
            raise ValueError("Logit shapes must match")

        gold_probs = torch.softmax(gold_logits.float(), dim=-1)
        candidate_probs = torch.softmax(candidate_logits.float(), dim=-1)

        eps = 1e-8
        kl_div = torch.sum(gold_probs * torch.log((gold_probs + eps) / (candidate_probs + eps)), dim=-1)

        return kl_div.mean().item()

    @staticmethod
    def infer_quality_tier_from_request(prompt: str) -> QualityTier:
        prompt_lower = prompt.lower()

        high_keywords = ["write a story", "write code", "create", "generate", "compose",
                         "def ", "class ", "function "]
        if any(kw in prompt_lower for kw in high_keywords):
            return QualityTier.HIGH

        low_keywords = ["classify", "category", "label", "yes or no", "true or false",
                        "extract", "sentiment"]
        if any(kw in prompt_lower for kw in low_keywords):
            return QualityTier.LOW

        return QualityTier.MEDIUM
