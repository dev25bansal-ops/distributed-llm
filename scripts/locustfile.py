"""Locust load test for DistLLM: POST /v1/chat/completions.

Run against a live server:

    export API_KEY=<your key>
    locust -f scripts/locustfile.py --headless -u 10 -r 2 --run-time 60s \
        --host http://127.0.0.1:8000

Or use the all-in-one in-process harness (mock backend included):

    python scripts/load_test_runner.py

Notes
-----
* Every request carries a unique message payload so the server's request
  deduplication middleware never collapses concurrent requests into cached
  responses — each hit exercises the full pipeline.
* Requests are non-streaming (`"stream": false`). Streaming SSE responses are
  not measured by this file.
"""

from __future__ import annotations

import itertools
import os
import uuid

from locust import HttpUser, task, between

_counter = itertools.count()
MODEL = os.environ.get("DISTLLM_LOADTEST_MODEL", "distributed-llm")
MAX_TOKENS = int(os.environ.get("DISTLLM_LOADTEST_MAX_TOKENS", "64"))


class ChatCompletionsUser(HttpUser):
    """Simulated client hammering the OpenAI-compatible chat endpoint."""

    # No think time: keep 10 connections saturated to measure server-side
    # capacity rather than client pacing.
    wait_time = between(0, 0)

    def on_start(self) -> None:
        # API key auth (the same key the server was started with).
        api_key = os.environ.get("API_KEY", "")
        if api_key:
            self.client.headers["Authorization"] = f"Bearer {api_key}"

    @task
    def chat_completion(self) -> None:
        seq = next(_counter)
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    # Unique content per request defeats response dedup.
                    "content": f"Load test request {seq}-{uuid.uuid4().hex[:8]}: reply briefly.",
                }
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.7,
            "stream": False,
        }
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            name="POST /v1/chat/completions",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}: {resp.text[:200]}")
                return
            data = resp.json()
            choices = data.get("choices") or []
            if not choices or "message" not in choices[0]:
                resp.failure(f"malformed response body: {str(data)[:200]}")
