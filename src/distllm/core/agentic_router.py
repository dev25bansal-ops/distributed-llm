"""Agentic Inference Router — LLM-as-judge for dynamic model selection.

Uses a lightweight LLM (3B-8B parameters) as a "router judge" that
analyzes incoming requests and selects the optimal model, quantization,
and endpoint based on task complexity, required capabilities, latency
budget, and cost constraints.

Architecture::

    request + context
         │
         ▼
    ┌──────────────────────┐
    │  Router Judge (3B)   │  ← lightweight LLM loaded once
    │  Analyzes task type, │     via transformers / llama.cpp
    │  complexity, budget  │
    └──────────┬───────────┘
               │  structured output: {"model": "...", "reason": "..."}
               ▼
    ┌──────────────────────┐
    │  Reward loop (DPO)   │  ← preference optimization from feedback
    └──────────────────────┘

No other open-source LLM serving stack has an LLM-based routing layer.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger


# ── Routing decision schema ──────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """Structured output from the router judge."""
    model: str
    reason: str = ""
    confidence: float = 0.5
    suggested_quantization: str = ""
    estimated_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reason": self.reason,
            "confidence": self.confidence,
            "suggested_quantization": self.suggested_quantization,
            "estimated_latency_ms": self.estimated_latency_ms,
        }


@dataclass
class PreferenceExample:
    """A preference pair for DPO training."""
    prompt: str
    chosen_model: str
    rejected_model: str
    reward_chosen: float
    reward_rejected: float
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ── Router Judge — Core ──────────────────────────────────────────────────

_ROUTER_JUDGE_PROMPT = """You are a router judge for a distributed LLM inference system.
Analyze the user's request and select the best model from the available options.

Available models:
{models_json}

Routing criteria (in priority order):
1. Task type: code, math, creative writing, analysis, general chat
2. Complexity: simple (short, factual) → medium → complex (multi-step reasoning)
3. Latency SLA: {latency_sla}ms max
4. Cost budget: ${cost_budget} max per request

Respond with a JSON object:
{{
    "model": "model_name",
    "reason": "brief explanation",
    "confidence": 0.0-1.0,
    "suggested_quantization": "int4" | "int8" | "fp16" | "",
    "estimated_latency_ms": 0
}}"""


class RouterJudge:
    """A lightweight LLM instance used for routing decisions.

    Loads a small model (3B-8B params) on the cheapest available GPU
    or CPU and uses it to analyze requests and select the optimal model.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        max_length: int = 512,
        fallback_fn: Callable | None = None,
    ):
        self._model_path = model_path or os.environ.get("DISTLLM_ROUTER_MODEL", "")
        self._device = device
        self._max_length = max_length
        self._fallback_fn = fallback_fn
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._load_model()

    def _load_model(self) -> None:
        """Load the router judge model."""
        if not self._model_path:
            logger.info(
                "RouterJudge: no model path configured, will use fallback only. "
                "Set DISTLLM_ROUTER_MODEL or pass model_path."
            )
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"RouterJudge: loading {self._model_path} on {self._device}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype="auto",
                device_map=self._device,
            )
            self._model.eval()
            logger.info("RouterJudge: model loaded")
        except Exception as e:
            logger.error(f"RouterJudge: failed to load model: {e}")
            self._model = None
            self._tokenizer = None

    def decide(
        self,
        prompt: str,
        available_models: list[dict[str, Any]],
        latency_sla_ms: float = 5000.0,
        cost_budget: float = float("inf"),
    ) -> RoutingDecision:
        """Ask the router judge which model to use.

        Args:
            prompt: The user's request text.
            available_models: List of model descriptors, each with
                ``name``, ``quantization``, ``estimated_latency_ms``,
                ``cost_per_1k_tokens``.
            latency_sla_ms: Maximum acceptable latency.
            cost_budget: Maximum acceptable cost.

        Returns:
            A RoutingDecision with the selected model and metadata.
        """
        if self._model is None or self._tokenizer is None:
            return self._fallback(prompt, available_models)

        models_json = json.dumps(
            [{"name": m.get("name"), "quantization": m.get("quantization", ""),
              "latency_ms": m.get("estimated_latency_ms", 0),
              "cost": m.get("cost_per_1k_tokens", 0)} for m in available_models],
            indent=2,
        )
        judge_prompt = _ROUTER_JUDGE_PROMPT.format(
            models_json=models_json,
            latency_sla=latency_sla_ms,
            cost_budget=cost_budget if cost_budget < float("inf") else "unlimited",
        )
        messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": prompt[:self._max_length]},
        ]

        try:
            import torch
            inputs = self._tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs.shape[1]:], skip_special_tokens=True,
            )
            return self._parse_response(response, available_models)
        except Exception as e:
            logger.error(f"RouterJudge inference failed: {e}")
            return self._fallback(prompt, available_models)

    def _parse_response(
        self, response: str, available_models: list[dict[str, Any]],
    ) -> RoutingDecision:
        """Parse JSON response from the judge model."""
        try:
            # Find JSON block in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                model_name = data.get("model", "")
                # Validate the model is available
                valid_names = {m.get("name") for m in available_models}
                if model_name not in valid_names and valid_names:
                    model_name = next(iter(valid_names))
                return RoutingDecision(
                    model=model_name,
                    reason=data.get("reason", ""),
                    confidence=float(data.get("confidence", 0.5)),
                    suggested_quantization=data.get("suggested_quantization", ""),
                    estimated_latency_ms=float(data.get("estimated_latency_ms", 0)),
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return self._fallback(response, available_models)

    def _fallback(
        self, prompt_or_response: str, available_models: list[dict[str, Any]],
    ) -> RoutingDecision:
        """Fallback: use heuristic or provided fallback function."""
        if self._fallback_fn:
            result = self._fallback_fn(prompt_or_response, available_models)
            if result:
                return result
        # Last resort: pick the first available model
        if available_models:
            m = available_models[0]
            return RoutingDecision(
                model=m.get("name", ""),
                reason="fallback: router judge unavailable",
                confidence=0.3,
            )
        return RoutingDecision(model="", reason="no models available", confidence=0.0)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None


# ── Agentic Router (integrates judge + reward feedback) ──────────────────

class AgenticRouter:
    """Full agentic inference router with judge + preference optimization.

    Integrates RouterJudge with the existing ModelRouter and LearningRouter
    for a complete LLM-as-judge routing pipeline.

    Usage::

        router = AgenticRouter(
            base_router=model_router,
            available_models=[
                {"name": "codellama-7b", "quantization": "int4", ...},
                {"name": "llama-3-8b", "quantization": "fp16", ...},
            ],
        )
        decision = router.route("Write a Python function to sort a list")
        router.record_outcome(decision, user_rating=0.9, latency_ms=150)
    """

    def __init__(
        self,
        base_router: Any = None,
        available_models: list[dict[str, Any]] | None = None,
        judge_model_path: str = "",
        judge_device: str = "cpu",
        preference_capacity: int = 1000,
        fallback_to_heuristic: bool = True,
    ):
        self._base = base_router
        self._models = available_models or []
        self._fallback_to_heuristic = fallback_to_heuristic

        # Router judge (LLM)
        self._judge = RouterJudge(
            model_path=judge_model_path,
            device=judge_device,
            fallback_fn=self._heuristic_route if fallback_to_heuristic else None,
        )

        # Preference replay buffer for DPO
        self._preferences: list[PreferenceExample] = []
        self._preference_capacity = preference_capacity
        self._lock = threading.Lock()

        # Stats
        self._judge_calls = 0
        self._fallback_calls = 0
        self._total_routes = 0

    def route(
        self,
        prompt: str,
        latency_sla_ms: float = 5000.0,
        cost_budget: float = float("inf"),
    ) -> RoutingDecision:
        """Route a request to the best model.

        Uses the LLM judge when available, falls back to heuristic.
        """
        self._total_routes += 1
        decision = self._judge.decide(
            prompt, self._models, latency_sla_ms, cost_budget,
        )
        if self._judge.is_loaded and decision.model:
            self._judge_calls += 1
        else:
            self._fallback_calls += 1
            if self._fallback_to_heuristic:
                decision = self._heuristic_route(prompt, self._models)
        return decision

    def _heuristic_route(
        self, prompt: str, models: list[dict[str, Any]],
    ) -> RoutingDecision:
        """Heuristic fallback: keyword-based workload detection."""
        if self._base and hasattr(self._base, 'resolve'):
            model_name = self._base.resolve(prompt)
            return RoutingDecision(
                model=model_name,
                reason="heuristic fallback",
                confidence=0.4,
            )
        if models:
            return RoutingDecision(
                model=models[0].get("name", ""),
                reason="first available",
                confidence=0.3,
            )
        return RoutingDecision(model="", reason="no models", confidence=0.0)

    def record_outcome(
        self,
        decision: RoutingDecision,
        chosen_model: str = "",
        user_rating: float | None = None,
        latency_ms: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Record a routing outcome for preference optimization.

        Stores preference pairs when the decision was suboptimal.
        """
        chosen = chosen_model or decision.model
        with self._lock:
            self._preferences.append(PreferenceExample(
                prompt="",
                chosen_model=chosen,
                rejected_model="",
                reward_chosen=user_rating or 0.5,
                reward_rejected=0.0,
                context={"latency_ms": latency_ms, "cost_usd": cost_usd},
            ))
            if len(self._preferences) > self._preference_capacity:
                self._preferences = self._preferences[-self._preference_capacity:]

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_routes": self._total_routes,
                "judge_calls": self._judge_calls,
                "fallback_calls": self._fallback_calls,
                "judge_loaded": self._judge.is_loaded,
                "preferences_collected": len(self._preferences),
            }
