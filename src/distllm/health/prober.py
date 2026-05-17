"""gRPC health probing for distributed-llm nodes."""

import time


async def probe_node(
    client,
    timeout: float = 10.0,
) -> tuple[bool, float, dict]:
    """Probe a node via its gRPC client and measure latency.

    Args:
        client: NodeClient instance with a health_check method.
        timeout: Maximum seconds to wait for response.

    Returns:
        (success, latency_ms, health_data)
    """
    start = time.perf_counter()
    try:
        health = client.health_check()
        elapsed_ms = (time.perf_counter() - start) * 1000

        return True, elapsed_ms, {
            "memory_used": health.memory_used,
            "memory_total": health.memory_total,
            "gpu_utilization": getattr(health, "gpu_utilization", 0.0),
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return False, elapsed_ms, {"error": str(e)}
