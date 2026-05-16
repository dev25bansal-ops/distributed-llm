"""Spot instance interruption handler."""

import time
from typing import Callable, Dict, List, Optional

from loguru import logger


class SpotHandler:
    """Handles spot instance lifecycle: interruption detection, draining, fallback.

    In production, this would poll cloud provider metadata endpoints or accept
    webhooks. Here we provide the detection and response framework.
    """

    # AWS spot instance gives 2-minute warning
    INTERRUPTION_WARNING_SECONDS = 120

    def __init__(self, cost_tracker, drain_timeout: float = 30.0):
        self.cost_tracker = cost_tracker
        self.drain_timeout = drain_timeout
        self._interrupted_nodes: Dict[str, float] = {}  # node_id -> timestamp
        self._drain_callback: Optional[Callable] = None
        self._fallback_callback: Optional[Callable] = None

    def set_drain_callback(self, callback: Callable) -> None:
        """Set callback to drain active requests from a node.

        Callback signature: drain_callback(node_id: str) -> None
        """
        self._drain_callback = callback

    def set_fallback_callback(self, callback: Callable) -> None:
        """Set callback to route to fallback nodes.

        Callback signature: fallback_callback(node_id: str) -> List[str]
        Returns list of fallback node IDs.
        """
        self._fallback_callback = callback

    def handle_interruption_notice(self, node_id: str) -> List[str]:
        """Handle a spot instance interruption notice.

        Args:
            node_id: The node receiving the interruption notice.

        Returns:
            List of fallback node IDs to route to.
        """
        logger.warning(f"[Spot] Interruption notice for {node_id}, draining...")
        self._interrupted_nodes[node_id] = time.time()
        self.cost_tracker.record_spot_interruption()

        # Drain active requests
        if self._drain_callback:
            try:
                self._drain_callback(node_id)
            except Exception as e:
                logger.error(f"[Spot] Error draining {node_id}: {e}")

        # Get fallback nodes
        fallback_nodes = []
        if self._fallback_callback:
            try:
                fallback_nodes = self._fallback_callback(node_id)
            except Exception as e:
                logger.error(f"[Spot] Error getting fallback for {node_id}: {e}")

        logger.info(f"[Spot] {node_id} drained, routing to {fallback_nodes}")
        return fallback_nodes

    def is_interrupted(self, node_id: str) -> bool:
        """Check if a node has received an interruption notice."""
        return node_id in self._interrupted_nodes

    def get_interrupted_nodes(self) -> List[str]:
        """Get list of all interrupted nodes."""
        return list(self._interrupted_nodes.keys())

    def check_interruption_metadata(self) -> Optional[str]:
        """Poll cloud provider metadata endpoint for interruption notices.

        In production, this polls the AWS instance metadata service:
        http://169.254.169.254/latest/meta-data/spot/termination-time

        Returns:
            node_id if interruption detected, None otherwise.
        """
        try:
            import urllib.request
            url = "http://169.254.169.254/latest/meta-data/spot/termination-time"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    # This is a simplified check - in production, parse the response
                    return "spot-node"
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
            logger.debug(f"Spot metadata check failed: {e}")
        return None

    def poll_interruptions(self, node_ids: List[str]) -> Dict[str, List[str]]:
        """Poll for interruptions across multiple spot nodes.

        Args:
            node_ids: List of spot node IDs to check.

        Returns:
            Dict mapping interrupted node_id to list of fallback nodes.
        """
        results = {}
        interruption = self.check_interruption_metadata()
        if interruption and interruption in node_ids:
            fallbacks = self.handle_interruption_notice(interruption)
            results[interruption] = fallbacks
        return results

    def clear_interruption(self, node_id: str) -> None:
        """Clear interruption status for a node (e.g., after replacement)."""
        self._interrupted_nodes.pop(node_id, None)
