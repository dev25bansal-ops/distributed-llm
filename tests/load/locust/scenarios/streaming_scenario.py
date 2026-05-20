"""Streaming chat completion load test scenario.

Usage:
    locust -f tests/load/locust/scenarios/streaming_scenario.py --host http://localhost:8000 --headless -u 5 -r 1 --run-time 5m
"""

import json
import os
import random
import time

from locust import HttpUser, between, events, task

API_KEY = os.environ.get("API_KEY", "")
MODEL = os.environ.get("MODEL", "distributed-llm")
MAX_TOKENS = int(os.environ.get("STREAMING_MAX_TOKENS", "512"))

HEADERS = {"Content-Type": "application/json", "Accept": "text/event-stream"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

PROMPTS = [
    "Tell me a long story about AI in the future.",
    "Explain the entire history of computing in detail.",
    "Write a comprehensive guide to machine learning.",
    "Describe the architecture of a distributed LLM system step by step.",
]


class StreamingUser(HttpUser):
    wait_time = between(1.0, 5.0)

    @task
    def stream_chat(self):
        prompt = random.choice(PROMPTS)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "stream": True,
        }
        start = time.time()
        chunks = 0
        full_text = ""
        try:
            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers=HEADERS,
                stream=True,
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"Status {resp.status_code}")
                    return
                for line in resp.iter_lines():
                    if line:
                        line = line.decode("utf-8", errors="replace")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    full_text += delta
                                chunks += 1
                            except json.JSONDecodeError:
                                pass
                elapsed = (time.time() - start) * 1000
                events.request.fire(
                    request_type="STREAM",
                    name="stream_chat",
                    response_time=elapsed,
                    response_length=len(full_text),
                )
                if chunks > 0:
                    resp.success()
                else:
                    resp.failure("No chunks received")
        except Exception as e:
            events.request.fire(
                request_type="STREAM",
                name="stream_chat",
                response_time=(time.time() - start) * 1000,
                response_length=0,
                exception=e,
            )
