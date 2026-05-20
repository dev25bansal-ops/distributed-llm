"""Chat completion load test scenario.

Usage:
    locust -f tests/load/locust/scenarios/chat_scenario.py --host http://localhost:8000
    locust -f tests/load/locust/scenarios/chat_scenario.py --host http://localhost:8000 --headless -u 10 -r 2 --run-time 5m
"""

import json
import os
import random
import time

from locust import HttpUser, between, events, task

API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "distributed-llm")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

PROMPTS = [
    "Explain the concept of pipeline parallelism in large language model inference.",
    "Write a short poem about distributed computing.",
    "What are the key differences between GPT-4 and LLaMA architectures?",
    "Describe how attention mechanisms work in transformer models.",
    "Explain the benefits of quantization for model deployment.",
    "What is speculative decoding and how does it improve inference speed?",
    "Compare batch vs streaming inference for LLM serving.",
    "How does KV caching work during autoregressive generation?",
    "What are the challenges of serving LLMs at scale?",
    "Explain the role of NCCL in multi-GPU training and inference.",
]

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"


class ChatUser(HttpUser):
    wait_time = between(0.5, 3.0)

    @task(3)
    def chat_completion(self):
        prompt = random.choice(PROMPTS)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        start = time.time()
        with self.client.post("/v1/chat/completions", json=payload, headers=HEADERS, catch_response=True) as resp:
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                resp.success()
                events.request.fire(
                    request_type="CHAT",
                    name="chat_completion",
                    response_time=elapsed,
                    response_length=len(resp.text),
                )
            elif resp.status_code == 429:
                resp.success()  # rate limited is acceptable
            else:
                resp.failure(f"Status {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def short_prompt(self):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        with self.client.post("/v1/chat/completions", json=payload, headers=HEADERS, catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")
