"""Locust gRPC load tests for distributed-llm nodes.

Tests gRPC endpoints: ForwardPass, HealthCheck, RegistrationUser

Usage:
    locust -f tests/load/grpc_locust/locustfile.py --config tests/load/grpc_locust/locust.conf
    locust -f tests/load/grpc_locust/locustfile.py --headless -u 20 -r 5 -t 5m
"""

import time
import struct
from unittest.mock import MagicMock

from locust import HttpUser, task, between, events
from locust.user.wait_time import constant_throughput


class GrpcForwardPassUser(HttpUser):
    """Simulates gRPC ForwardPass requests to a node."""

    wait_time = between(0.01, 0.1)  # 10-100ms between requests
    host = "http://localhost:50051"  # Default gRPC node address

    @task(8)
    def forward_pass_small(self):
        """Small batch forward pass (typical inference)."""
        # Note: This uses HTTP as a proxy for gRPC in locust.
        # For real gRPC, use grpcio + locust's custom client pattern.
        with self.client.post(
            "/forward",
            json={
                "request_id": f"req-{int(time.time() * 1000)}",
                "input_ids": [1, 2, 3, 4, 5],
                "batch_size": 1,
                "seq_len": 5,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Forward pass failed: {response.status_code}")

    @task(2)
    def forward_pass_large(self):
        """Large batch forward pass (batched inference)."""
        with self.client.post(
            "/forward",
            json={
                "request_id": f"req-{int(time.time() * 1000)}",
                "input_ids": list(range(128)),
                "batch_size": 4,
                "seq_len": 32,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Large forward pass failed: {response.status_code}")


class HealthCheckUser(HttpUser):
    """Simulates health check polling."""

    wait_time = constant_throughput(1.0)  # 1 health check per second
    host = "http://localhost:50051"

    @task(1)
    def health_check(self):
        """Poll node health."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class RegistrationUser(HttpUser):
    """Simulates node registration traffic."""

    wait_time = between(1.0, 5.0)
    host = "http://localhost:50051"

    @task(1)
    def register_node(self):
        """Register a new node."""
        with self.client.post(
            "/register",
            json={
                "node_info": {
                    "node_id": f"node-{int(time.time())}",
                    "host": "localhost",
                    "port": 50052,
                    "total_memory": 16_000_000_000,
                    "available_memory": 12_000_000_000,
                    "device_type": "cuda",
                },
                "num_layers": 12,
            },
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Registration failed: {response.status_code}")


# --- Events: Custom stats ---

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Starting gRPC load test...")
    print(f"Target: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.runner.stats
    print("\n--- Load Test Summary ---")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    if stats.total.num_requests > 0:
        print(f"Success rate: {(1 - stats.total.num_failures / stats.total.num_requests) * 100:.1f}%")
