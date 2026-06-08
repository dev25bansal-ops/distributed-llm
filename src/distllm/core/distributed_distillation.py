"""Inference-Aware Model Distillation — use the distributed cluster as a teacher.

During idle periods, the distributed cluster runs the teacher model to
generate soft targets (logits), which are used to train a smaller student
model. This produces a compact model that approximates the teacher's
output at a fraction of the inference cost.

Architecture:
    IdleDetector → Cluster is idle → Teacher generates logits
                                     → Student trains via KL divergence
                                     → Checkpoint saved to disk
                                     → Student model ready for deployment

No unlabeled data needed — the teacher generates its own training data
from random prompt seeds or cached user prompts.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from distllm.core.adaptive_compression import IdleDetector


@dataclass
class DistillationConfig:
    """Configuration for distributed distillation.

    Attributes:
        teacher_model: HuggingFace model ID for the teacher (runs on cluster).
        student_model_path: Path or HF ID for the student model.
        temperature: Softmax temperature for distillation (higher = softer targets).
        alpha: Weight for KL divergence loss vs CE loss (0=only CE, 1=only KL).
        batch_size: Tokens per training step.
        max_samples: Maximum generated samples before stopping.
        max_length: Maximum sequence length for teacher generation.
        learning_rate: Adam learning rate.
        idle_only: Only run when cluster is idle.
        checkpoint_dir: Directory to save/load distillation checkpoints.
        seed_prompts: List of seed prompts for data generation. If empty,
            uses built-in defaults.
    """
    teacher_model: str = ""
    student_model_path: str = ""
    temperature: float = 2.0
    alpha: float = 0.5
    batch_size: int = 4
    max_samples: int = 10000
    max_length: int = 512
    learning_rate: float = 5e-5
    idle_only: bool = True
    checkpoint_dir: str = "/tmp/distllm-distillation"
    seed_prompts: list[str] = field(default_factory=lambda: [
        "Explain how distributed computing works.",
        "What is the capital of France?",
        "Write a poem about artificial intelligence.",
        "Compare Python and Rust for systems programming.",
        "How does attention work in transformers?",
        "Describe the water cycle.",
        "What are the benefits of open source software?",
    ])


class DistributedDistillationEngine:
    """Distills a teacher model into a student model using the cluster.

    Uses idle cluster time to:
    1. Generate soft targets from the teacher model
    2. Train the student model via KL divergence + CE loss
    3. Save checkpointed student weights

    Thread-safe: uses threading.Lock for state transitions.
    """

    def __init__(
        self,
        config: DistillationConfig,
        teacher_forward: Callable | None = None,
        idle_detector: IdleDetector | None = None,
        utilization_fn: Callable[[], float] | None = None,
    ):
        self._config = config
        self._teacher_forward = teacher_forward  # Custom teacher callable
        self._idle_detector = idle_detector

        # State
        self._student_model: torch.nn.Module | None = None
        self._student_tokenizer: Any = None
        self._teacher_model: torch.nn.Module | None = None
        self._teacher_tokenizer: Any = None
        self._optimizer: torch.optim.Optimizer | None = None

        self._samples_generated = 0
        self._steps_completed = 0
        self._is_running = False
        self._lock = threading.Lock()
        self._should_stop = threading.Event()

        # Stats
        self._total_loss = 0.0
        self._total_kl = 0.0
        self._total_ce = 0.0
        self._teacher_tok_s = 0.0
        self._student_tok_s = 0.0

        # Ensure checkpoint dir exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def start(self) -> bool:
        """Start the distillation background thread.

        Loads teacher and student models, then begins the distillation
        loop in a background thread. Returns True if started successfully.
        """
        with self._lock:
            if self._is_running:
                logger.warning("Distillation already running")
                return False
            self._is_running = True
            self._should_stop.clear()

        # Load models
        try:
            self._load_models()
        except Exception as e:
            logger.error(f"Failed to load distillation models: {e}")
            with self._lock:
                self._is_running = False
            return False

        # Start background thread
        thread = threading.Thread(target=self._run_loop, daemon=True, name="distillation")
        thread.start()
        logger.info("Distillation engine started")
        return True

    def stop(self) -> None:
        """Signal the distillation loop to stop and save checkpoint."""
        self._should_stop.set()
        self._save_checkpoint()
        with self._lock:
            self._is_running = False
        logger.info(f"Distillation stopped after {self._steps_completed} steps")

    def _load_models(self) -> None:
        """Load teacher and student models."""
        cfg = self._config

        # Teacher (only if no custom forward provided)
        if self._teacher_forward is None and cfg.teacher_model:
            logger.info(f"Loading teacher model: {cfg.teacher_model}")
            self._teacher_model = AutoModelForCausalLM.from_pretrained(
                cfg.teacher_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._teacher_model.eval()
            self._teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model)

        # Student
        if cfg.student_model_path:
            logger.info(f"Loading student model: {cfg.student_model_path}")
            # Try loading from a checkpoint first
            ckpt_path = os.path.join(cfg.checkpoint_dir, "student.pt")
            loaded_from_checkpoint = False

            self._student_model = AutoModelForCausalLM.from_pretrained(
                cfg.student_model_path,
                torch_dtype=torch.float32,
            )
            self._student_tokenizer = AutoTokenizer.from_pretrained(cfg.student_model_path)
            if self._student_tokenizer.pad_token is None:
                self._student_tokenizer.pad_token = self._student_tokenizer.eos_token

            # Vocab size compatibility check
            if self._teacher_model is not None:
                teacher_vocab = self._teacher_model.config.vocab_size
                student_vocab = self._student_model.config.vocab_size
                if teacher_vocab != student_vocab:
                    logger.warning(
                        f"Teacher vocab size ({teacher_vocab}) != "
                        f"Student vocab size ({student_vocab}). "
                        "Distillation may produce misaligned logits."
                    )

            if os.path.exists(ckpt_path):
                try:
                    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    self._student_model.load_state_dict(state["model_state_dict"])
                    self._steps_completed = state.get("steps", 0)
                    self._samples_generated = state.get("samples", 0)
                    loaded_from_checkpoint = True
                    logger.info(f"Resumed from checkpoint ({self._steps_completed} steps)")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}")

            if not loaded_from_checkpoint:
                logger.info("Starting fresh distillation (no checkpoint found)")

            # Optimizer
            self._optimizer = torch.optim.AdamW(
                self._student_model.parameters(),
                lr=cfg.learning_rate,
            )

            # Move student to device
            if torch.cuda.is_available():
                self._student_model = self._student_model.cuda()

    def _generate_teacher_targets(self, prompt: str) -> dict | None:
        """Generate soft targets (logits) from the teacher model.

        Returns dict with 'input_ids', 'teacher_logits', 'attention_mask'
        or None on failure.
        """
        cfg = self._config

        if self._teacher_forward is not None:
            # Custom teacher callable (e.g., distributed cluster)
            return self._teacher_forward(prompt, max_length=cfg.max_length)

        if self._teacher_model is None:
            return None

        try:
            tokenizer = self._teacher_tokenizer
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._teacher_model(**inputs)
                logits = outputs.logits[:, :-1, :]  # Shift for next-token prediction

            return {
                "input_ids": inputs["input_ids"][:, 1:],  # Shifted targets
                "teacher_logits": logits.cpu().float(),
                "attention_mask": inputs.get("attention_mask", None),
            }
        except Exception as e:
            logger.warning(f"Teacher forward failed: {e}")
            return None

    def _train_step(self, target_data: dict) -> float:
        """Single training step on the student model.

        Computes KL divergence + CE loss against teacher logits.

        Returns the total loss value.
        """
        if self._student_model is None or self._optimizer is None:
            return 0.0

        cfg = self._config
        input_ids = target_data["input_ids"]
        teacher_logits = target_data["teacher_logits"]

        # Truncate to same length
        min_len = min(input_ids.shape[1], teacher_logits.shape[1])
        input_ids = input_ids[:, :min_len]
        teacher_logits = teacher_logits[:, :min_len, :]

        # Student forward
        student_outputs = self._student_model(input_ids)
        student_logits = student_outputs.logits

        # Temperature scaling
        teacher_soft = torch.softmax(teacher_logits / cfg.temperature, dim=-1)
        student_log = torch.log_softmax(student_logits / cfg.temperature, dim=-1)

        # KL divergence
        kl_loss = torch.nn.functional.kl_div(
            student_log, teacher_soft,
            reduction="batchmean",
            log_target=False,
        ) * (cfg.temperature ** 2)

        # Cross-entropy with hard targets
        ce_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            input_ids.view(-1),
            ignore_index=-100,
        )

        loss = cfg.alpha * kl_loss + (1 - cfg.alpha) * ce_loss

        # Backward
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._student_model.parameters(), 1.0)
        self._optimizer.step()

        self._total_loss += loss.item()
        self._total_kl += kl_loss.item()
        self._total_ce += ce_loss.item()
        self._steps_completed += 1

        return loss.item()

    def _run_loop(self) -> None:
        """Main distillation loop — runs in background thread."""
        cfg = self._config
        logger.info(f"Distillation loop started (max {cfg.max_samples} samples)")

        prompt_pool = list(cfg.seed_prompts)

        while not self._should_stop.is_set():
            # Check idle condition
            if cfg.idle_only and self._idle_detector and not self._idle_detector.is_idle:
                time.sleep(5)
                continue

            if self._samples_generated >= cfg.max_samples:
                logger.info(f"Reached max samples ({cfg.max_samples}), stopping")
                self._save_checkpoint()
                break

            # Generate sample
            prompt = prompt_pool[self._samples_generated % len(prompt_pool)]
            t0 = time.monotonic()
            target_data = self._generate_teacher_targets(prompt)
            elapsed = time.monotonic() - t0

            if target_data is None:
                time.sleep(1)
                continue

            # Track teacher throughput
            if elapsed > 0:
                num_tokens = target_data.get("input_ids", torch.tensor([[0]])).shape[1]
                self._teacher_tok_s = (self._teacher_tok_s * 0.9 + (num_tokens / elapsed) * 0.1)

            # Train student (multiple steps per sample for efficiency)
            for _ in range(min(4, cfg.batch_size)):
                loss = self._train_step(target_data)
                if self._should_stop.is_set():
                    break

            self._samples_generated += 1

            # Periodic logging
            if self._steps_completed % 10 == 0:
                avg_loss = self._total_loss / max(self._steps_completed, 1)
                logger.info(
                    f"Distillation: step={self._steps_completed} "
                    f"samples={self._samples_generated} "
                    f"loss={avg_loss:.4f} "
                    f"kl={self._total_kl / max(self._steps_completed, 1):.4f} "
                    f"ce={self._total_ce / max(self._steps_completed, 1):.4f} "
                    f"teacher={self._teacher_tok_s:.1f} tok/s"
                )

            # Periodic checkpoint
            if self._steps_completed % 50 == 0:
                self._save_checkpoint()

        logger.info(f"Distillation loop finished: {self._steps_completed} steps")

    def _save_checkpoint(self) -> None:
        """Save student model checkpoint to disk."""
        if self._student_model is None:
            return
        try:
            path = os.path.join(self._config.checkpoint_dir, "student.pt")
            torch.save({
                "model_state_dict": self._student_model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer else None,
                "steps": self._steps_completed,
                "samples": self._samples_generated,
                "loss": self._total_loss / max(self._steps_completed, 1),
            }, path)
            logger.debug(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    # ── Status ───────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "is_running": self._is_running,
                "steps_completed": self._steps_completed,
                "samples_generated": self._samples_generated,
                "avg_loss": round(self._total_loss / max(self._steps_completed, 1), 4),
                "avg_kl": round(self._total_kl / max(self._steps_completed, 1), 4),
                "avg_ce": round(self._total_ce / max(self._steps_completed, 1), 4),
                "teacher_throughput": round(self._teacher_tok_s, 1),
                "teacher_model": self._config.teacher_model,
                "student_model": self._config.student_model_path,
                "checkpoint_dir": self._config.checkpoint_dir,
            }
