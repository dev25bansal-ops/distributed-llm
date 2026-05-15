"""K8s CRD extensions for canary deployments.

Extends the existing K8s operator CRD with canary-specific fields.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CanarySpec:
    """Canary deployment specification for K8s CRD."""
    enabled: bool = False
    canary_version: str = "canary"
    canary_weight: int = 0
    analysis_duration_s: int = 300
    rollback_threshold: float = 0.05
    stages: List[Dict] = field(default_factory=lambda: [
        {"weight_pct": 5, "analysis_duration_s": 300},
        {"weight_pct": 25, "analysis_duration_s": 600},
        {"weight_pct": 50, "analysis_duration_s": 600},
        {"weight_pct": 75, "analysis_duration_s": 300},
        {"weight_pct": 100, "analysis_duration_s": 300},
    ])

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "canary_version": self.canary_version,
            "canary_weight": self.canary_weight,
            "analysis_duration_s": self.analysis_duration_s,
            "rollback_threshold": self.rollback_threshold,
            "stages": self.stages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CanarySpec":
        default_stages = [
            {"weight_pct": 5, "analysis_duration_s": 300},
            {"weight_pct": 25, "analysis_duration_s": 600},
            {"weight_pct": 50, "analysis_duration_s": 600},
            {"weight_pct": 75, "analysis_duration_s": 300},
            {"weight_pct": 100, "analysis_duration_s": 300},
        ]
        return cls(
            enabled=data.get("enabled", False),
            canary_version=data.get("canary_version", "canary"),
            canary_weight=data.get("canary_weight", 0),
            analysis_duration_s=data.get("analysis_duration_s", 300),
            rollback_threshold=data.get("rollback_threshold", 0.05),
            stages=data.get("stages", default_stages),
        )


@dataclass
class NodePoolWithCanary:
    """Extended node pool specification with canary fields.

    Adds to the existing K8s operator node pool schema:
    - canary_weight: percentage of traffic to route to canary version
    - canary_version: version string for the canary
    """
    host: str
    port_range: str
    start_layer: int
    end_layer: int
    node_role: str = "auto"
    replicas: int = 1
    canary_weight: int = 0
    canary_version: str = ""

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port_range": self.port_range,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "node_role": self.node_role,
            "replicas": self.replicas,
            "canary_weight": self.canary_weight,
            "canary_version": self.canary_version,
        }
