"""Monitoring, alerting, and chaos engineering configuration classes."""

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "MonitoringSettings",
    "AlertingSettings",
    "ChaosSettings",
]


class MonitoringSettings(BaseModel):
    """System monitoring configuration."""
    enabled: bool = True


class AlertingSettings(BaseModel):
    """Prometheus alerting rules configuration."""
    enabled: bool = False
    prometheus_url: str = "http://localhost:9090"
    rule_file: str | None = None

    @field_validator("prometheus_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"prometheus_url must start with http:// or https://, got '{v}'")
        return v


class ChaosSettings(BaseModel):
    """Chaos engineering fault injection configuration."""
    enabled: bool = False
    allowed_scenarios: list[str] = Field(default_factory=lambda: ["kill_node", "add_latency", "drop_message", "corrupt_data"])
    max_latency_ms: int = 5000

    @field_validator("max_latency_ms")
    @classmethod
    def validate_latency(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_latency_ms must be >= 1, got {v}")
        return v
