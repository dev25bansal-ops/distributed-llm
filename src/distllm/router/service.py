"""Router service for distributed-llm.

Routes incoming requests to coordinator instances using consistent hashing
for sticky sessions, with health-aware failover.
"""

import hashlib

from loguru import logger

from distllm.router.consistent_hash import ConsistentHashRing
from distllm.router.discovery import CoordinatorDiscovery, CoordinatorInfo


class RouterService:
    """Routes requests to coordinators with sticky sessions."""

    def __init__(self):
        self._ring = ConsistentHashRing()
        self._discovery = CoordinatorDiscovery()
        self._discovery.on_change(self._on_coordinator_change)

    async def start(self, mode: str = "static", config: dict | None = None) -> None:
        """Start the router with the given discovery mode."""
        config = config or {}
        await self._discovery.start(config)
        logger.info(f"RouterService started (discovery mode: {mode})")

    async def stop(self) -> None:
        await self._discovery.stop()
        logger.info("RouterService stopped")

    def get_coordinator(self, session_key: str) -> CoordinatorInfo | None:
        """Get the coordinator for a session key.

        Uses consistent hashing for sticky routing, with fallback
        to the next healthy node if the primary is unhealthy.
        """
        healthy = {
            c.node_id for c in self._discovery.get_healthy()
        }
        node_id = self._ring.get_node_with_fallback(session_key, healthy)
        if node_id is None:
            return None
        return self._discovery.get_all().get(node_id)

    def update_health(self, node_id: str, healthy: bool) -> None:
        self._discovery.set_health(node_id, healthy)

    @property
    def ring(self) -> ConsistentHashRing:
        return self._ring

    @property
    def discovery(self) -> CoordinatorDiscovery:
        return self._discovery

    def _on_coordinator_change(self, node_id: str, added: bool) -> None:
        if added:
            self._ring.add_node(node_id)
            logger.info(f"Coordinator {node_id} added to routing ring")
        else:
            self._ring.remove_node(node_id)
            logger.info(f"Coordinator {node_id} removed from routing ring")


def compute_session_key(request_data: dict, client_host: str | None = None) -> str:
    """Compute a sticky session key from request data.

    Uses user identifier, client host, or content hash as fallback.
    """
    # Try to extract user/client identifier
    if request_data.get("user"):
        return str(request_data["user"])
    if client_host:
        return client_host

    # Fallback: hash the last message content
    messages = request_data.get("messages", [])
    if messages:
        content = messages[-1].get("content", "")
        return hashlib.md5(content.encode()).hexdigest()

    # Ultimate fallback: random
    import uuid
    return str(uuid.uuid4())
