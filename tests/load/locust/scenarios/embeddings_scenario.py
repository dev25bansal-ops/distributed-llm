"""Embeddings load test scenario.

Tests the embedding generation endpoint for RAG pipelines.
"""

import json
import os
import random
import time

from locust import HttpUser, between, events, task

API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "distributed-llm")

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Transformers have revolutionized NLP.",
    "Distributed computing enables large-scale model training.",
    "Attention is all you need.",
    "Embeddings capture semantic meaning of text.",
    "Vector databases enable similarity search.",
    "The GPU memory hierarchy affects performance.",
]


class EmbeddingsUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def create_embedding(self):
        text = random.choice(TEXTS)
        payload = {
            "model": MODEL,
            "input": text,
        }
        start = time.time()
        with self.client.post("/v1/embeddings", json=payload, headers=HEADERS, catch_response=True) as resp:
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                resp.success()
                events.request.fire(
                    request_type="EMBED",
                    name="create_embedding",
                    response_time=elapsed,
                    response_length=len(resp.text),
                )
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def batch_embeddings(self):
        payload = {
            "model": MODEL,
            "input": random.sample(TEXTS, 3),
        }
        with self.client.post("/v1/embeddings", json=payload, headers=HEADERS, catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")
