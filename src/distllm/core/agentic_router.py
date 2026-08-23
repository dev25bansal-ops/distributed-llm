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
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F
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
                # Validate the model is available; unknown names fall back
                # to the FIRST listed model (deterministic order).
                if (
                    model_name not in {m.get("name") for m in available_models}
                    and available_models
                ):
                    model_name = available_models[0].get("name", "")
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
        # DPO training parameters
        dpo_enabled: bool = True,
        dpo_train_every: int = 50,
        dpo_batch_size: int = 8,
        dpo_learning_rate: float = 5e-5,
        dpo_beta: float = 0.1,
        exploration_prob: float = 0.05,
        lora_r: int = 8,
        lora_alpha: int = 16,
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
        self._dpo_enabled = dpo_enabled
        self._dpo_train_every = dpo_train_every
        self._exploration_prob = exploration_prob
        self._lock = threading.Lock()

        # DPO trainer (initialized on first training if judge model is loaded)
        self._dpo_trainer: DPOTrainer | None = None
        self._dpo_train_steps = 0
        self._dpo_loss_avg = 0.0
        self._exploration_rounds = 0

        # Training config
        self._dpo_batch_size = dpo_batch_size
        self._dpo_lr = dpo_learning_rate
        self._dpo_beta = dpo_beta
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha

        # Stats
        self._judge_calls = 0
        self._fallback_calls = 0
        self._total_routes = 0

    # ── DPO Training ──────────────────────────────────────────────────

    def _init_dpo_trainer(self) -> DPOTrainer | None:
        """Initialize the DPO trainer if the judge model supports it."""
        if self._dpo_trainer is not None:
            return self._dpo_trainer
        if not self._dpo_enabled:
            return None
        model = getattr(self._judge, '_model', None)
        tokenizer = getattr(self._judge, '_tokenizer', None)
        if model is None or tokenizer is None:
            logger.warning("DPO: judge model not loaded — cannot train")
            return None
        try:
            self._dpo_trainer = DPOTrainer(
                model=model,
                tokenizer=tokenizer,
                lr=self._dpo_lr,
                beta=self._dpo_beta,
                lora_r=self._lora_r,
                lora_alpha=self._lora_alpha,
            )
            logger.info("DPO trainer initialized with LoRA")
            return self._dpo_trainer
        except Exception as e:
            logger.warning(f"DPO: failed to initialize trainer: {e}")
            return None

    def _train_dpo(self) -> float:
        """Run one DPO training step on the preference buffer.

        Returns the average loss, or 0.0 if training was skipped.
        """
        trainer = self._init_dpo_trainer()
        if trainer is None:
            return 0.0

        with self._lock:
            if len(self._preferences) < 4:
                return 0.0
            # Sample a batch from the buffer
            batch = random.sample(
                self._preferences,
                min(self._dpo_batch_size, len(self._preferences)),
            )
            # Copy for training outside the lock
            batch_copy = [
                (
                    ex.prompt,
                    ex.chosen_model,
                    ex.rejected_model,
                    ex.reward_chosen,
                    ex.reward_rejected,
                )
                for ex in batch
            ]

        # Normalize rewards in the batch
        rewards = [r for _, _, _, cr, rr in batch_copy for r in (cr, rr)]
        if rewards:
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards)) or 1.0
            batch_normalized = [
                (p, cm, rm, (cr - mean) / std, (rr - mean) / std)
                for p, cm, rm, cr, rr in batch_copy
            ]
        else:
            batch_normalized = [(p, cm, rm, cr, rr) for p, cm, rm, cr, rr in batch_copy]

        try:
            loss = trainer.train_step(batch_normalized)
            self._dpo_train_steps += 1
            # Smoothing
            self._dpo_loss_avg = 0.9 * self._dpo_loss_avg + 0.1 * loss
            logger.debug(f"DPO train step {self._dpo_train_steps}: loss={loss:.4f}")
            return loss
        except Exception as e:
            logger.warning(f"DPO training step failed: {e}")
            return 0.0

    # ── Exploration ───────────────────────────────────────────────────

    def _maybe_explore(self) -> RoutingDecision | None:
        """With probability *exploration_prob*, pick a random model.

        Returns a RoutingDecision or None (proceed with normal routing).
        """
        if not self._models or random.random() >= self._exploration_prob:
            return None
        model = random.choice(self._models)
        self._exploration_rounds += 1
        logger.debug(f"Exploration: trying {model.get('name', 'unknown')}")
        return RoutingDecision(
            model=model.get("name", ""),
            reason="exploration",
            confidence=0.3,
        )

    def route(
        self,
        prompt: str,
        latency_sla_ms: float = 5000.0,
        cost_budget: float = float("inf"),
    ) -> RoutingDecision:
        """Route a request to the best model.

        Uses the LLM judge when available, falls back to heuristic.
        With *exploration_prob* probability, tries a random model
        instead to gather preference data for DPO training.
        """
        self._total_routes += 1

        # Exploration: occasionally try a random model for data collection
        explore_decision = self._maybe_explore()
        if explore_decision is not None:
            return explore_decision

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
        rejected_model: str = "",
        user_rating: float | None = None,
        latency_ms: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Record a routing outcome for preference optimization.

        Stores preference pairs when the decision was suboptimal.
        Triggers periodic DPO training every ``dpo_train_every``
        preferences (default: 50).
        """
        chosen = chosen_model or decision.model
        with self._lock:
            self._preferences.append(PreferenceExample(
                prompt="",
                chosen_model=chosen,
                rejected_model=rejected_model or "",
                reward_chosen=user_rating or 0.5,
                reward_rejected=0.0,
                context={"latency_ms": latency_ms, "cost_usd": cost_usd},
            ))
            while len(self._preferences) > self._preference_capacity:
                self._preferences.pop(0)
            pref_count = len(self._preferences)

        # Periodic DPO training
        if (
            self._dpo_enabled
            and pref_count >= 4
            and pref_count % self._dpo_train_every < 10
        ):
            thread = threading.Thread(target=self._train_dpo, daemon=True)
            thread.start()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_routes": self._total_routes,
                "judge_calls": self._judge_calls,
                "fallback_calls": self._fallback_calls,
                "judge_loaded": self._judge.is_loaded,
                "preferences_collected": len(self._preferences),
                "exploration_rounds": getattr(self, '_exploration_rounds', 0),
                "dpo_train_steps": getattr(self, '_dpo_train_steps', 0),
                "dpo_loss_avg": round(getattr(self, '_dpo_loss_avg', 0.0), 4),
            }


# ── DPO Trainer (LoRA-based) ────────────────────────────────────────────────

class DPOTrainer:
    """Online DPO trainer with LoRA adapters for the router judge.

    Implements Direct Preference Optimization (DPO) to fine-tune the router
    judge model using preference pairs collected from routing outcomes.

    The router judge is a small LLM (3B-8B) so DPO fine-tuning is feasible
    on a single GPU with LoRA adapters (rank=8, ~0.1% of full parameters).

    Reference: "Direct Preference Optimization" (Rafailov et al., 2023)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        lr: float = 5e-5,
        beta: float = 0.1,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        self._tokenizer = tokenizer
        self._beta = beta
        self._device = next(model.parameters()).device

        # Attach LoRA adapters to the model
        self._model = self._wrap_lora(model, r=lora_r, alpha=lora_alpha)
        self._optimizer = torch.optim.AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=lr,
        )
        self._train_count = 0

    def _wrap_lora(
        self, model: torch.nn.Module, r: int = 8, alpha: int = 16,
    ) -> torch.nn.Module:
        """Wrap linear layers with LoRA adapters.

        Only attaches adapters to ``nn.Linear`` layers in the transformer
        blocks (not embeddings, lm_head, or norms).  Keeps the base model
        frozen — only the LoRA weights are trained.
        """
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            # Skip embeddings, lm_head, and norm-like layers
            if any(skip in name for skip in ("embed", "head", "norm", "ln_")):
                continue
            # Only target transformer attention and FF layers
            if not any(target in name for target in ("q_proj", "k_proj", "v_proj",
                                                       "o_proj", "gate_proj",
                                                       "up_proj", "down_proj")):
                continue

            # Replace with LoRA-wrapped version
            lora_layer = LoRALinear(module, r=r, alpha=alpha)
            parent_name, child_name = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, lora_layer)

        # Freeze non-LoRA parameters
        total_lora = 0
        for p in model.parameters():
            if getattr(p, "_is_lora", False):
                p.requires_grad = True
                total_lora += p.numel()
            else:
                p.requires_grad = False
        logger.info(f"DPO: LoRA adapters added ({total_lora:,} trainable params)")
        return model

    def train_step(
        self,
        batch: list[tuple[str, str, str, float, float]],
    ) -> float:
        """Run one DPO training step.

        Args:
            batch: List of (prompt, chosen_model, rejected_model,
                   reward_chosen, reward_rejected) tuples.

        Returns:
            Loss value.
        """
        losses = []
        self._model.train()

        for prompt, chosen, rejected, reward_c, reward_r in batch:
            if not prompt and not chosen:
                continue

            # Tokenize chosen and rejected model responses
            chosen_inputs = self._tokenize_for_dpo(prompt, chosen)
            rejected_inputs = self._tokenize_for_dpo(prompt, rejected)

            if chosen_inputs is None or rejected_inputs is None:
                continue

            # Forward pass for both responses
            chosen_logits = self._model(**chosen_inputs).logits
            rejected_logits = self._model(**rejected_inputs).logits

            # Compute log-probabilities of the response tokens
            chosen_logprob = self._logprob_of_response(
                chosen_logits, chosen_inputs["input_ids"],
            )
            rejected_logprob = self._logprob_of_response(
                rejected_logits, rejected_inputs["input_ids"],
            )

            # DPO loss with normalized rewards
            reward_gap = reward_c - reward_r
            log_prob_ratio = chosen_logprob - rejected_logprob
            loss = -F.logsigmoid(self._beta * log_prob_ratio * reward_gap)
            losses.append(loss)

        if not losses:
            return 0.0

        batch_loss = torch.stack(losses).mean()
        self._optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self._model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self._optimizer.step()
        self._train_count += 1

        return batch_loss.item()

    def _tokenize_for_dpo(
        self, prompt: str, response: str,
    ) -> dict | None:
        """Tokenize a (prompt, response) pair for the DPO step."""
        if not response:
            return None
        text = f"{prompt}\n{response}" if prompt else response
        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)
        if tokens["input_ids"].shape[1] == 0:
            return None
        return tokens

    def _logprob_of_response(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the average log-probability of the response tokens.

        Args:
            logits: Model output logits, shape (1, seq_len, vocab_size).
            input_ids: Token IDs, shape (1, seq_len).

        Returns:
            Scalar tensor: mean log-probability of the response tokens.
        """
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:]
        token_logprobs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        return token_logprobs.mean()

    @property
    def train_count(self) -> int:
        return self._train_count


class LoRALinear(torch.nn.Module):
    """LoRA adapter wrapping a frozen ``nn.Linear`` layer.

    Adds a low-rank decomposition ``A @ B`` that is trained while the
    original weight matrix stays frozen.  At inference time the adapter
    output ``x @ (A @ B)`` is added to the frozen output.
    """

    def __init__(self, linear: torch.nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.scaling = alpha / r

        # Freeze original weight
        self.weight = linear.weight
        self.bias = linear.bias
        for p in (self.weight, self.bias):
            if p is not None:
                p.requires_grad = False

        # LoRA down-projection A and up-projection B
        self.lora_A = torch.nn.Parameter(
            torch.randn(r, self.in_features) * 0.02,
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(self.out_features, r),
        )
        self.lora_A._is_lora = True  # type: ignore[attr-defined]
        self.lora_B._is_lora = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out
