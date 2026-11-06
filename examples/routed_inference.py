"""Demonstrates DistLLM as a multi-provider inference control plane.

Usage:
    export OPENAI_API_KEY=sk-...
    export TOGETHER_API_KEY=...
    python examples/routed_inference.py
"""

import os
import openai

client = openai.Client(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("DISTLLM_API_KEY", "dev-key"),
)


def route_cheapest():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        extra_body={"provider": "cheapest"},
    )
    return response.choices[0].message.content


def route_fastest():
    response = client.chat.completions.create(
        model="claude-3-haiku",
        messages=[{"role": "user", "content": "Explain quantum computing in 3 sentences."}],
        extra_body={"provider": "fastest"},
    )
    return response.choices[0].message.content


def route_with_fallback():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Write a haiku about AI."}],
        extra_body={"provider_fallbacks": ["together", "fireworks"]},
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Cheapest route:", route_cheapest())
    print("Fastest route:", route_fastest())
    print("With fallback:", route_with_fallback())
