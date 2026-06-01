"""Canary rollout, versioning, cost, and tenant configuration classes."""

from pydantic import BaseModel, Field, field_validator, SecretStr

__all__ = [
    "RolloutStageModel",
    "CanarySettings",
    "VersionSettings",
    "CostSettings",
    "TenantSettings",
]


class RolloutStageModel(BaseModel):
    """Single stage in a canary rollout."""
    weight_pct: float
    analysis_duration_s: int = 300


class CanarySettings(BaseModel):
    """Automated canary deployment configuration."""
    enabled: bool = False
    stable_version: str = "stable"
    canary_version: str = "canary"
    rollback_threshold: float = 0.05
    stages: list[RolloutStageModel] = Field(default_factory=lambda: [
        RolloutStageModel(weight_pct=5, analysis_duration_s=300),
        RolloutStageModel(weight_pct=25, analysis_duration_s=600),
        RolloutStageModel(weight_pct=50, analysis_duration_s=600),
        RolloutStageModel(weight_pct=75, analysis_duration_s=300),
        RolloutStageModel(weight_pct=100, analysis_duration_s=300),
    ])

    @field_validator("rollback_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"rollback_threshold must be 0.0-1.0, got {v}")
        return v

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, v: list[RolloutStageModel]) -> list[RolloutStageModel]:
        if not v:
            raise ValueError("stages must not be empty")
        for stage in v:
            if not (0 < stage.weight_pct <= 100):
                raise ValueError(f"stage weight_pct must be 0-100, got {stage.weight_pct}")
        return v


class VersionSettings(BaseModel):
    """Model versioning and A/B testing configuration."""
    enabled: bool = False
    max_versions: int = 4
    shadow_enabled: bool = False
    shadow_pct: float = 0.0  # Percentage of traffic to shadow (0-100)
    blue_green_enabled: bool = False
    ab_testing_enabled: bool = False
    ab_test_split: float = 50.0  # Percentage for variant B (0-100)
    auto_promote_enabled: bool = False
    min_samples: int = 100  # Minimum samples before statistical test
    significance_level: float = 0.05  # p-value threshold


class CostSettings(BaseModel):
    """Cost-aware scheduling configuration."""
    enabled: bool = False
    budget_per_hour: float = 0.0
    spot_preference: float = 0.8

    @field_validator("budget_per_hour")
    @classmethod
    def validate_budget(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"budget_per_hour must be >= 0, got {v}")
        return v

    @field_validator("spot_preference")
    @classmethod
    def validate_preference(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"spot_preference must be 0.0-1.0, got {v}")
        return v


class TenantSettings(BaseModel):
    """Multi-tenant SaaS configuration."""
    enabled: bool = False
    default_tier: str = "free"
    admin_api_key: SecretStr | None = Field(default=None, description="Admin API key for tenant management. Set via DISTLLM__TENANT__ADMIN_API_KEY env var.")
