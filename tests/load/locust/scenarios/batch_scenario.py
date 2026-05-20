"""Batch processing load test scenario.

Sends multiple prompts in a single batch request to test batch inference throughput.
"""

import json
import os
import random
import time

from locust import HttpUser, between, events, task

API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "distributed-llm")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
MAX_TOKENS = int(os.environ.get("BATCH_MAX_TOKENS", "128"))

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing.",
    "Write a haiku about coding.",
    "What is 2+2?",
    "Define machine learning.",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What is an API?",
    "Explain TCP/IP.",
    "What is Docker?",
]


class BatchUser(HttpUser):
    wait_time = between(2.0, 8.0)

    @task
    def batch_generation(self):
        prompts = random.sample(PROMPTS, min(BATCH_SIZE, len(PROMPTS)))
        batch_payload = {
            "model": MODEL,
            "prompts": prompts,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }
        start = time.time()
        with self.client.post("/v1/batch/completions", json=batch_payload, headers=HEADERS, catch_response=True) as resp:
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                resp.success()
                events.request.fire(
                    request_type="BATCH",
                    name="batch_generation",
                    response_time=elapsed,
                    response_length=len(resp.text),
                )
            else:
                resp.failure(f"Status {resp.status_code}")
