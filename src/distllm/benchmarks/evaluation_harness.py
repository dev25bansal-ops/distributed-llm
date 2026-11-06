"""Integration with EleutherAI lm-evaluation-harness."""
from __future__ import annotations

import json

from dataclasses import dataclass, field

import httpx

try:
    import lm_eval
    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False


@dataclass
class LMEvalConfig:
    tasks: list[str] = field(default_factory=lambda: ["mmlu", "gsm8k", "humaneval"])
    model: str = "local-completion"
    batch_size: str = "auto"
    num_fewshot: int = 0
    limit: int | None = None


class DistLLMModelAdapter:
    def __init__(self, coordinator_url="http://localhost:8000", api_key=""):
        self._base_url = coordinator_url
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def loglikelihood(self, requests):
        results = []
        for req in requests:
            resp = httpx.post(f"{self._base_url}/v1/completions", json={"prompt": req[0], "max_tokens": 1, "logprobs": 1}, headers=self._headers, timeout=30.0)
            data = resp.json()
            results.append(data.get("choices", [{}])[0].get("logprobs", 0))
        return results

    def generate(self, requests):
        results = []
        for req in requests:
            resp = httpx.post(f"{self._base_url}/v1/completions", json={"prompt": req[0], "max_tokens": req[1]}, headers=self._headers, timeout=60.0)
            data = resp.json()
            results.append(data.get("choices", [{}])[0].get("text", ""))
        return results


class LMEvalRunner:
    def __init__(self, config=None, coordinator_url="http://localhost:8000"):
        self.config = config or LMEvalConfig()
        self._adapter = DistLLMModelAdapter(coordinator_url)

    def run(self):
        if not _LM_EVAL_AVAILABLE:
            return {"error": "lm_eval not installed", "tasks": self.config.tasks}
        results = {}
        for task in self.config.tasks:
            try:
                results[task] = self._run_task(task)
            except Exception as e:
                # CQ-08: failure is recorded in the output value (visible to the caller); not propagated.
                results[task] = {"error": str(e)}
        return results

    def _run_task(self, task_name):
        # Honest-contract (hygiene pass 2026-11-06): the previous body returned
        # a *random* accuracy — fabricated benchmark numbers. Until the real
        # lm-eval wiring lands, fail loudly instead of inventing data.
        raise NotImplementedError(
            "LMEvalRunner._run_task is not implemented against lm_eval yet; "
            "it previously returned random.uniform(0.5, 0.8) as 'accuracy', "
            "which is not a measurement. Wire lm_eval.simple_evaluate() with "
            "DistLLMModelAdapter before using this runner."
        )

    def save_results(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.run(), f, indent=2)
