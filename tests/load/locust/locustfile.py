"""Locust load test for Distributed LLM API.

Usage:
    locust -f tests/load/locust/locustfile.py --host http://localhost:8000

    Headless:
    locust -f tests/load/locust/locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 2 --run-time 10m

    With API key:
    locust -f tests/load/locust/locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 2 --run-time 10m \
        --config tests/load/locust/locust.conf
"""

import json
import os
import random

from locust import HttpUser, between, events, task

# --- Configuration ---

API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "distributed-llm")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "128"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

PROMPTS = [
    "Explain the concept of pipeline parallelism in large language model inference.",
    "Write a short Python function that implements a consistent hash ring with virtual nodes.",
    "What are the key differences between data parallelism, tensor parallelism, "
    "and pipeline parallelism for distributed training?",
    "Describe the architecture of the Transformer model, including self-attention, "
    "multi-head attention, and positional encoding.",
    "How does KV caching improve autoregressive text generation efficiency? "
    "Explain with time complexity analysis.",
    "Compare and contrast vLLM, SGLang, and TensorRT-LLM for optimizing "
    "LLM inference throughput.",
    "Explain speculative decoding and how it accelerates autoregressive generation.",
    "What are the challenges of running LLMs across multiple machines? "
    "Discuss network latency, bandwidth, and fault tolerance.",
]


def get_headers():
    """Build request headers with optional API key."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def random_prompt():
    """Select a random prompt from the list."""
    return random.choice(PROMPTS)


# --- Event Hooks ---


@events.test_start.add_context
def on_test_start(environment, **kwargs):
    """Log test configuration at start."""
    print(f"\n{'='*60}")
    print("Distributed LLM Load Test")
    print(f"Target: {environment.host}")
    print(f"Model: {MODEL}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"API Key: {'set' if API_KEY else 'not set'}")
    print(f"{'='*60}\n")


@events.test_stop.add_context
def on_test_stop(environment, **kwargs):
    """Print summary statistics at end."""
    stats = environment.runner.stats
    print(f"\n{'='*60}")
    print("Load Test Summary")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    if stats.total.num_requests > 0:
        print(f"Failure rate: {stats.total.num_failures / stats.total.num_requests * 100:.2f}%")
        print(f"Average response time: {stats.total.avg_response_time:.0f}ms")
        print(f"Median response time: {stats.total.median_response_time:.0f}ms")
        print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
        print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print(f"{'='*60}\n")


# --- User Classes ---


class ChatUser(HttpUser):
    """Simulates a user making chat completion requests."""

    wait_time = between(1, 3)

    @task(weight=7)
    def chat_completion(self):
        """Non-streaming chat completion."""
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": random_prompt()},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stream": False,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=get_headers(),
            catch_response=True,
            name="/v1/chat/completions",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("choices") and len(data["choices"]) > 0:
                        response.success()
                    else:
                        response.failure("Empty choices in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 429:
                response.failure("Rate limited (429)")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(weight=2)
    def streaming_chat_completion(self):
        """Streaming chat completion (SSE)."""
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": random_prompt()},
            ],
            "max_tokens": min(MAX_TOKENS, 64),
            "temperature": TEMPERATURE,
            "stream": True,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=get_headers(),
            catch_response=True,
            name="/v1/chat/completions [stream]",
        ) as response:
            if response.status_code == 200:
                # For streaming, just check we got a response
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(weight=1)
    def list_models(self):
        """List available models."""
        with self.client.get(
            "/v1/models",
            headers=get_headers(),
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class HealthCheckUser(HttpUser):
    """Simulates monitoring/health check traffic."""

    wait_time = between(5, 15)

    @task(3)
    def health_check(self):
        """Health check endpoint."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def metrics(self):
        """Prometheus metrics endpoint."""
        with self.client.get(
            "/metrics",
            catch_response=True,
            name="/metrics",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class StressUser(HttpUser):
    """Aggressive load generator for stress testing."""

    wait_time = between(0, 0.5)

    @task
    def heavy_chat(self):
        """Chat completion with max tokens."""
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": random_prompt()},
            ],
            "max_tokens": 256,
            "temperature": TEMPERATURE,
            "stream": False,
        }

        self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=get_headers(),
        )
