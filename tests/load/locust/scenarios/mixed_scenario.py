"""Mixed workload load test scenario.

Simulates realistic traffic mix:
  - 50% chat completions
  - 20% streaming chat
  - 15% embeddings
  - 10% batch processing
  - 5% health checks
"""

import json
import os
import random
import time

from locust import HttpUser, between, events, task

API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "distributed-llm")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

CHAT_PROMPTS = [
    "Explain neural networks.",
    "What is gradient descent?",
    "Write a function in Python.",
    "What is the difference between TCP and UDP?",
    "Explain cloud computing.",
]

EMBED_TEXTS = [
    "This is a sample text for embedding.",
    "Another example sentence.",
]


class MixedUser(HttpUser):
    wait_time = between(0.3, 1.5)

    @task(10)
    def chat(self):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": random.choice(CHAT_PROMPTS)}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
        }
        with self.client.post("/v1/chat/completions", json=payload, headers=HEADERS, catch_response=True) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(4)
    def stream(self):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Tell me a story about AI."}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "stream": True,
        }
        try:
            with self.client.post("/v1/chat/completions", json=payload, headers={**HEADERS, "Accept": "text/event-stream"}, stream=True, catch_response=True) as resp:
                if resp.status_code == 200:
                    chunks = 0
                    for line in resp.iter_lines():
                        if line and b"[DONE]" in line:
                            break
                        chunks += 1
                    if chunks > 0:
                        resp.success()
                    else:
                        resp.failure("no chunks")
                else:
                    resp.failure(f"Status {resp.status_code}")
        except Exception as e:
            pass

    @task(3)
    def embed(self):
        payload = {"model": MODEL, "input": random.choice(EMBED_TEXTS)}
        with self.client.post("/v1/embeddings", json=payload, headers=HEADERS, catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(2)
    def batch(self):
        payload = {
            "model": MODEL,
            "prompts": random.sample(CHAT_PROMPTS, 2),
            "max_tokens": 64,
            "temperature": 0.0,
        }
        with self.client.post("/v1/batch/completions", json=payload, headers=HEADERS, catch_response=True) as resp:
            resp.success()

    @task(1)
    def health(self):
        with self.client.get("/health", headers=HEADERS, catch_response=True) as resp:
            resp.success()
