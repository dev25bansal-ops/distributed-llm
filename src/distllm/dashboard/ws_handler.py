"""WebSocket handler for real-time dashboard updates.

Broadcasts node health, metrics, and request stats to connected clients.
"""

import asyncio
import json
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        data = json.dumps(message)
        disconnected = set()
        for conn in self.active_connections:
            try:
                await conn.send_text(data)
            except Exception:
                disconnected.add(conn)
        # Clean up disconnected clients
        self.active_connections -= disconnected

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()


async def metrics_broadcaster(coordinator, interval: float = 1.0):
    """Periodically broadcast metrics to all connected WebSocket clients.

    Runs as a background task. Stops when coordinator is None.
    """
    if coordinator is None:
        return

    while True:
        try:
            data = {
                "type": "metrics",
                "data": {
                    "model": getattr(coordinator, "model_name", "unknown"),
                    "nodes": len(getattr(coordinator, "nodes", {})),
                },
            }

            # Add scheduler stats
            try:
                if coordinator.scheduler:
                    data["data"]["scheduler"] = coordinator.scheduler.stats()
            except Exception:
                pass

            # Add node health
            try:
                nodes = {}
                for node_id, reg in getattr(coordinator, "nodes", {}).items():
                    nodes[node_id] = {
                        "healthy": reg.healthy,
                        "host": reg.host,
                        "layers": f"{reg.start_layer}-{reg.end_layer}",
                    }
                data["data"]["nodes"] = nodes
            except Exception:
                pass

            await manager.broadcast(data)
        except Exception as e:
            logger.debug(f"Broadcast error: {e}")

        await asyncio.sleep(interval)
