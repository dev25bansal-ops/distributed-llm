"""Edge routing logic: decide whether to serve locally or fall back to cloud."""

from enum import Enum
from typing import Optional

from loguru import logger

from distllm.edge.models import EdgeConfig


class EdgeRouteDecision(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


class EdgeRouter:
    """Decides whether to route requests to edge or cloud.

    Uses configurable thresholds for active requests, queue depth,
    and model availability to make routing decisions.
    """

    def __init__(self, config: EdgeConfig):
        self.config = config
        self._consecutive_overloads = 0
        self._force_cloud = False
        self._force_cloud_until = 0.0

    def decide(self, body: dict, active_requests: int = 0) -> EdgeRouteDecision:
        """Determine where to route the request."""
        import time

        # Model not deployed locally
        model = body.get("model", "")
        if model and model not in self.config.models:
            logger.debug(f"Model {model} not deployed on edge, routing to cloud")
            return EdgeRouteDecision.CLOUD

        # Overload protection
        if active_requests >= self.config.max_concurrent_requests:
            self._consecutive_overloads += 1
            if self._consecutive_overloads >= 3:
                self._force_cloud = True
                self._force_cloud_until = time.time() + 30.0
                logger.warning("Edge overloaded, forcing cloud fallback for 30s")
            return EdgeRouteDecision.CLOUD
        else:
            self._consecutive_overloads = max(0, self._consecutive_overloads - 1)

        if self._force_cloud:
            if time.time() < self._force_cloud_until:
                return EdgeRouteDecision.CLOUD
            self._force_cloud = False
            logger.info("Edge recovered, resuming local serving")

        return EdgeRouteDecision.EDGE

    def reset(self) -> None:
        self._consecutive_overloads = 0
        self._force_cloud = False
        self._force_cloud_until = 0.0
