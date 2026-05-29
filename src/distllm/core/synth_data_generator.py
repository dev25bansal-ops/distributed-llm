"""Synthetic Data Generation Pipeline — generate datasets using the distributed cluster.

Routes generation through pipeline with configurable prompts, formats,
and output parsing.  Useful for data augmentation, fine-tuning dataset
creation, and stress-testing.

Usage::

    gen = SynthDataGenerator(pipeline=coord._pipeline, tokenizer=coord.tokenizer)
    dataset = gen.generate(
        prompts=["Write a story about", "Explain quantum computing"],
        max_samples=100,
        output_format="jsonl",
    )
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger


class SynthDataGenerator:
    """Generates synthetic datasets using the distributed pipeline.

    Each prompt template is expanded with random variations and fed through
    the pipeline.  Outputs are collected into a structured dataset.
    """

    def __init__(
        self,
        generate_fn: Any = None,
        tokenizer: Any = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ):
        self._generate = generate_fn
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

        self._stats = {"total_generated": 0, "total_tokens": 0, "errors": 0}

    def set_generate_fn(self, fn: Any) -> None:
        self._generate = fn

    def generate(
        self,
        prompts: list[str],
        max_samples: int = 100,
        output_format: str = "jsonl",
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Generate synthetic data from a list of prompt templates.

        Args:
            prompts: List of prompt strings.  Each generates one sample.
            max_samples: Maximum total samples to generate.
            output_format: Output format ("jsonl", "json", "text").
            system_prompt: Optional system prompt prepended to each.

        Returns:
            List of dicts with "prompt" and "response" keys.
        """
        dataset = []
        for prompt in prompts[:max_samples]:
            try:
                full_prompt = f"{system_prompt}\n{prompt}".strip() if system_prompt else prompt
                response = self._call_generate(full_prompt)
                token_count = len(self._tokenizer.encode(response)) if self._tokenizer else len(response.split())

                record = {
                    "prompt": prompt,
                    "response": response,
                    "system_prompt": system_prompt,
                    "timestamp": time.time(),
                    "token_count": token_count,
                }
                dataset.append(record)
                self._stats["total_generated"] += 1
                self._stats["total_tokens"] += token_count

            except Exception as e:
                logger.warning(f"Generation failed for prompt: {prompt[:50]}... ({e})")
                self._stats["errors"] += 1

            if len(dataset) >= max_samples:
                break

        logger.info(f"Synthetic data generated: {len(dataset)} samples, "
                     f"{self._stats['total_tokens']} tokens")
        return dataset

    def _call_generate(self, prompt: str) -> str:
        """Call the underlying generation function."""
        if self._generate is None:
            return f"Simulated response to: {prompt[:50]}..."
        return self._generate(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )

    def save(self, dataset: list[dict], path: str, fmt: str = "jsonl") -> None:
        """Save a generated dataset to disk."""
        with open(path, "w", encoding="utf-8") as f:
            if fmt == "jsonl":
                for record in dataset:
                    f.write(json.dumps(record) + "\n")
            elif fmt == "json":
                json.dump(dataset, f, indent=2)
            else:
                for record in dataset:
                    f.write(f"Prompt: {record['prompt']}\nResponse: {record['response']}\n\n")
        logger.info(f"Dataset saved to {path} ({len(dataset)} samples)")

    @property
    def stats(self) -> dict[str, Any]:
        return dict(self._stats)
