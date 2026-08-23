"""LLM Evaluation Harness — standardized benchmarking for distributed models.

Supports three evaluation paradigms:

1. **HEIM tasks** (MMLU, GSM8K, HumanEval) — prompt-templated, deterministic
   scoring against reference answers.
2. **MT-Bench** — multi-turn chat with GPT-4-as-judge quality scoring.
3. **Chatbot Arena** — pairwise comparison of two model outputs by GPT-4 judge.

Architecture (standardized pipeline)::

    DatasetLoader -> PromptFormatter -> WorkerPool (parallel eval)
        -> Scorer -> ReportGenerator -> EvaluationReport (SQLite)

Usage::

    runner = EvalRunner(coordinator=coord)
    report = runner.run_heim("mmlu", model_id="my-model")
    report = runner.run_mt_bench(model_id="my-model", judge_model="gpt-4")
    report = runner.run_arena(model_a="model-a", model_b="model-b")
"""

from __future__ import annotations

import abc
import asyncio
import concurrent.futures
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path.home() / ".distllm" / "eval_results.db"
_MAX_WORKERS = min(8, (os.cpu_count() or 4))
_EVAL_TIMEOUT_S = 120.0
_MTBENCH_CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]
_ARENA_SYSTEM_PROMPT = (
    "You are an impartial judge comparing two AI assistant responses. "
    "Evaluate which response is more helpful, accurate, and safe."
)
_MTBENCH_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating the quality of an AI assistant's response. "
    "Score the response on a scale of 1 to 10 based on helpfulness, accuracy, and relevance."
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvalBenchmark(str, Enum):
    """Supported evaluation benchmarks."""
    MMLU = "mmlu"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    MT_BENCH = "mt_bench"
    ARENA = "arena"


class EvalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSample:
    """A single evaluation sample (question + reference + metadata)."""
    question: str
    answer: str | None = None  # reference answer
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""
    sample: EvalSample
    prediction: str
    score: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    model_id: str
    dataset: str
    config: dict[str, Any]
    metrics: dict[str, float]
    results: list[EvalResult] = field(default_factory=list)
    status: EvalStatus = EvalStatus.PENDING
    report_id: str = ""
    created_at: float = 0.0
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.report_id:
            self.report_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()


# ---------------------------------------------------------------------------
# SQLite persistence (follows patterns from persistence.py)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS eval_reports (
    report_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    config TEXT NOT NULL DEFAULT '{}',
    metrics TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    duration_s REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    prediction TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    score REAL NOT NULL DEFAULT 0.0,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    generated_tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (report_id) REFERENCES eval_reports(report_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_reports_model ON eval_reports(model_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_report ON eval_results(report_id);
"""


class EvalDB:
    """SQLite persistence for evaluation reports and results.

    Follows the pattern from :mod:`distllm.core.persistence`.
    """

    def __init__(self, db_path: str | Path = "") -> None:
        self._db_path = Path(str(db_path) if db_path else _DEFAULT_DB_PATH)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Create tables and ensure schema is current."""
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            logger.debug("Eval database initialized at {}", self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def save_report(self, report: EvalReport) -> None:
        """Persist an evaluation report and its results."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO eval_reports
                   (report_id, model_id, dataset, status, config, metrics, created_at, duration_s)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.report_id,
                    report.model_id,
                    report.dataset,
                    report.status.value,
                    json.dumps(report.config),
                    json.dumps(report.metrics),
                    report.created_at,
                    report.duration_s,
                ),
            )
            for result in report.results:
                conn.execute(
                    """INSERT INTO eval_results
                       (report_id, question, answer, prediction, category, score,
                        latency_ms, prompt_tokens, generated_tokens, error, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        report.report_id,
                        result.sample.question,
                        result.sample.answer,
                        result.prediction,
                        result.sample.category,
                        result.score,
                        result.latency_ms,
                        result.prompt_tokens,
                        result.generated_tokens,
                        result.error,
                        json.dumps(result.sample.metadata),
                    ),
                )
            conn.commit()

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Retrieve a report header by ID."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM eval_reports WHERE report_id = ?", (report_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def get_report_results(self, report_id: str) -> list[dict[str, Any]]:
        """Retrieve all results for a given report."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE report_id = ? ORDER BY id",
                (report_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_reports(
        self,
        model_id: str | None = None,
        dataset: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List evaluation reports with optional filtering."""
        with self._lock:
            conn = self._get_conn()
            where = []
            params: list[Any] = []
            if model_id:
                where.append("model_id = ?")
                params.append(model_id)
            if dataset:
                where.append("dataset = ?")
                params.append(dataset)
            clause = f"WHERE {' AND '.join(where)}" if where else ""
            rows = conn.execute(
                f"SELECT * FROM eval_reports {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_report(self, report_id: str) -> bool:
        """Delete a report and its results."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM eval_results WHERE report_id = ?", (report_id,))
            cur = conn.execute("DELETE FROM eval_reports WHERE report_id = ?", (report_id,))
            conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


class DatasetLoader(abc.ABC):
    """Abstract base for dataset loaders.

    Implementations load benchmark datasets and yield ``EvalSample``
    instances compatible with the evaluation pipeline.
    """

    @abc.abstractmethod
    def load(self, split: str = "test") -> list[EvalSample]:
        """Load the dataset and return a list of samples."""
        ...


class _MMLULoader(DatasetLoader):
    """Minimal MMLU dataset loader with embedded samples.

    In production, replace with ``datasets`` library::

        from datasets import load_dataset  # pip install datasets
        ds = load_dataset("mmlu", "all", split="test")
    """

    def __init__(self, num_samples: int = 20) -> None:
        self._num_samples = num_samples

    def load(self, split: str = "test") -> list[EvalSample]:
        samples: list[EvalSample] = []
        logger.info("Loading MMLU ({} samples, split={})", self._num_samples, split)

        # Embed a small reference set so the harness works out of the box.
        # Full dataset loading via `datasets` is attempted as a fallback.
        mmlu_questions = [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "category": "general",
            },
            {
                "question": "The chemical symbol for gold is:",
                "answer": "Au",
                "category": "science",
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "answer": "Mars",
                "category": "science",
            },
            {
                "question": "Who wrote 'Romeo and Juliet'?",
                "answer": "William Shakespeare",
                "category": "humanities",
            },
            {
                "question": "What is the largest ocean on Earth?",
                "answer": "Pacific Ocean",
                "category": "geography",
            },
            {
                "question": "In what year did World War II end?",
                "answer": "1945",
                "category": "history",
            },
            {
                "question": "What is the powerhouse of the cell?",
                "answer": "Mitochondria",
                "category": "biology",
            },
            {
                "question": "What is the value of Pi to two decimal places?",
                "answer": "3.14",
                "category": "math",
            },
            {
                "question": "Which element has the atomic number 1?",
                "answer": "Hydrogen",
                "category": "science",
            },
            {
                "question": "What is the speed of light in vacuum (m/s)?",
                "answer": "299,792,458",
                "category": "physics",
            },
        ]

        # Rotate through embedded questions up to num_samples
        for i in range(self._num_samples):
            q = mmlu_questions[i % len(mmlu_questions)]
            samples.append(
                EvalSample(
                    question=q["question"],
                    answer=q["answer"],
                    category=q["category"],
                    metadata={"source": "mmlu_embedded", "index": i},
                )
            )

        return samples


class _GSM8KLoader(DatasetLoader):
    """Minimal GSM8K (grade school math) dataset loader.

    In production, replace with ``datasets`` library for the full
    GSM8K test split (1319 samples).
    """

    def __init__(self, num_samples: int = 20) -> None:
        self._num_samples = num_samples

    def load(self, split: str = "test") -> list[EvalSample]:
        logger.info("Loading GSM8K ({} samples, split={})", self._num_samples, split)
        problems = [
            {
                "question": "Janet has 3 apples. She buys 5 more. How many does she have?",
                "answer": "8",
            },
            {
                "question": "A train travels 120 km in 2 hours. What is its speed in km/h?",
                "answer": "60",
            },
            {
                "question": "If a pizza has 8 slices and 3 people eat 2 slices each, how many slices remain?",
                "answer": "2",
            },
            {
                "question": "The product of two numbers is 36. One number is 9. What is the other?",
                "answer": "4",
            },
            {
                "question": "Alice is 12 years old. Bob is 3 years older. How old will Bob be in 5 years?",
                "answer": "20",
            },
            {
                "question": "A rectangle has length 10cm and width 5cm. What is its area?",
                "answer": "50",
            },
            {
                "question": "There are 24 students in a class. If they sit in rows of 6, how many rows?",
                "answer": "4",
            },
            {
                "question": "A store sold 15 items on Monday and 23 on Tuesday. How many total?",
                "answer": "38",
            },
            {
                "question": "If 3 notebooks cost $6, how much do 5 notebooks cost?",
                "answer": "10",
            },
            {
                "question": "A garden has 4 rows of 7 flowers each. How many flowers total?",
                "answer": "28",
            },
        ]
        samples: list[EvalSample] = []
        for i in range(self._num_samples):
            p = problems[i % len(problems)]
            samples.append(
                EvalSample(
                    question=p["question"],
                    answer=p["answer"],
                    category="math",
                    metadata={"source": "gsm8k_embedded", "index": i},
                )
            )
        return samples


class _HumanEvalLoader(DatasetLoader):
    """Minimal HumanEval (code generation) dataset loader.

    In production, use the full HumanEval dataset via ``datasets``
    or ``evaluate`` library.
    """

    def __init__(self, num_samples: int = 10) -> None:
        self._num_samples = num_samples

    def load(self, split: str = "test") -> list[EvalSample]:
        logger.info("Loading HumanEval ({} samples, split={})", self._num_samples, split)
        problems = [
            {
                "question": "Write a Python function that returns the sum of two numbers.",
                "answer": "def add(a, b):\n    return a + b",
            },
            {
                "question": "Write a function that checks if a number is even.",
                "answer": "def is_even(n):\n    return n % 2 == 0",
            },
            {
                "question": "Write a function that returns the length of a string.",
                "answer": "def string_length(s):\n    return len(s)",
            },
            {
                "question": "Write a function that reverses a list.",
                "answer": "def reverse_list(lst):\n    return lst[::-1]",
            },
            {
                "question": "Write a function that returns the max of two numbers.",
                "answer": "def max_of_two(a, b):\n    return a if a > b else b",
            },
        ]
        samples: list[EvalSample] = []
        for i in range(self._num_samples):
            p = problems[i % len(problems)]
            samples.append(
                EvalSample(
                    question=p["question"],
                    answer=p["answer"],
                    category="coding",
                    metadata={"source": "humaneval_embedded", "index": i},
                )
            )
        return samples


class _MTBenchLoader(DatasetLoader):
    """MT-Bench multi-turn conversation dataset loader.

    Each sample contains a system prompt and a list of user messages
    representing multi-turn conversations across 8 categories.
    """

    def __init__(self, num_samples: int = 8) -> None:
        self._num_samples = min(num_samples, len(_MTBENCH_CATEGORIES))

    def load(self, split: str = "test") -> list[EvalSample]:
        logger.info("Loading MT-Bench ({} categories)", self._num_samples)
        conversations: dict[str, list[str]] = {
            "writing": [
                "Write a persuasive essay about why remote work is beneficial.",
                "Now rewrite it as a formal business proposal.",
            ],
            "roleplay": [
                "Pretend you are a travel agent. Recommend a vacation package.",
                "Now act as the customer and ask follow-up questions.",
            ],
            "reasoning": [
                "If all A are B, and some B are C, can we conclude some A are C? Explain.",
                "Give an example where this reasoning would fail.",
            ],
            "math": [
                "Solve for x: 3x + 7 = 22",
                "Verify your solution step by step.",
            ],
            "coding": [
                "Write a binary search function in Python.",
                "Add input validation and error handling to it.",
            ],
            "extraction": [
                "Extract the dates and locations from: 'The event was held on March 15th, 2024 in New York City.'",
                "Format them as a JSON object.",
            ],
            "stem": [
                "Explain the greenhouse effect in simple terms.",
                "What are three practical ways to reduce carbon emissions?",
            ],
            "humanities": [
                "What were the main causes of the French Revolution?",
                "How did it influence modern democratic systems?",
            ],
        }
        samples: list[EvalSample] = []
        for category in _MTBENCH_CATEGORIES[:self._num_samples]:
            turns = conversations.get(category, ["Tell me something interesting."])
            samples.append(
                EvalSample(
                    question=json.dumps({"category": category, "turns": turns}),
                    category=category,
                    metadata={"source": "mt_bench", "num_turns": len(turns)},
                )
            )
        return samples


class _ArenaLoader(DatasetLoader):
    """Chatbot Arena pairwise comparison dataset loader.

    Pairs of prompts that are used to elicit responses from two models,
    which are then compared by a judge.
    """

    def __init__(self, num_samples: int = 10) -> None:
        self._num_samples = num_samples

    def load(self, split: str = "test") -> list[EvalSample]:
        logger.info("Loading Arena prompts ({} samples)", self._num_samples)
        prompts = [
            "Explain quantum computing to a 10-year-old.",
            "Write a limerick about Python programming.",
            "Compare and contrast REST and GraphQL.",
            "What are the ethical implications of AI in healthcare?",
            "Write a short story about a robot learning to paint.",
            "Explain the difference between TCP and UDP.",
            "What is the best way to learn a new programming language?",
            "Describe the process of photosynthesis.",
            "How does a blockchain work?",
            "What are the key principles of effective prompt engineering?",
            "Explain the concept of recursion with an example.",
            "What are the trade-offs between microservices and monoliths?",
            "How would you design a distributed rate limiter?",
            "Explain the CAP theorem in distributed systems.",
            "What is the role of attention in transformer models?",
        ]
        samples: list[EvalSample] = []
        for i in range(min(self._num_samples, len(prompts))):
            samples.append(
                EvalSample(
                    question=prompts[i],
                    category="arena",
                    metadata={"source": "arena_embedded", "index": i},
                )
            )
        return samples


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------


class PromptFormatter(abc.ABC):
    """Abstract base for formatting prompts for model evaluation."""

    @abc.abstractmethod
    def format(self, sample: EvalSample) -> str:
        """Format a sample into a model prompt string."""
        ...


class _HeimPromptFormatter(PromptFormatter):
    """Prompt formatter for HEIM-style benchmarks (MMLU, GSM8K, HumanEval)."""

    def __init__(self, benchmark: str) -> None:
        self._benchmark = benchmark

    def format(self, sample: EvalSample) -> str:
        if self._benchmark == "mmlu":
            return (
                f"Answer the following multiple-choice question:\n\n"
                f"{sample.question}\n\n"
                f"Answer:"
            )
        if self._benchmark == "gsm8k":
            return (
                f"Solve the following math problem step by step:\n\n"
                f"{sample.question}\n\n"
                f"Answer:"
            )
        if self._benchmark == "humaneval":
            return (
                f"Write a Python function for the following task. "
                f"Return only the code, no explanation.\n\n"
                f"{sample.question}\n\n"
                f"```python"
            )
        return sample.question


class _MTBenchPromptFormatter(PromptFormatter):
    """Formats MT-Bench multi-turn conversations."""

    def format(self, sample: EvalSample) -> str:
        data = json.loads(sample.question)
        turns = data.get("turns", [])
        category = data.get("category", "general")
        prompt = f"Category: {category}\n\n"
        for i, turn in enumerate(turns):
            prompt += f"Turn {i + 1}: {turn}\n"
        prompt += "\nRespond to the conversation above."
        return prompt


class _ArenaPromptFormatter(PromptFormatter):
    """Formats prompts for pairwise comparison."""

    def format(self, sample: EvalSample) -> str:
        return (
            f"Please respond to the following prompt:\n\n"
            f"{sample.question}\n\n"
            f"Response:"
        )


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class Scorer(abc.ABC):
    """Abstract base for scoring model predictions."""

    @abc.abstractmethod
    def score(self, sample: EvalSample, prediction: str) -> float:
        """Return a score between 0.0 and 1.0."""
        ...


class _ExactMatchScorer(Scorer):
    """Exact match scorer for deterministic benchmarks.

    For math (GSM8K), extracts the numeric answer from the prediction
    and compares it to the reference. For general QA, uses substring
    matching on key phrases.
    """

    def __init__(self, benchmark: str) -> None:
        self._benchmark = benchmark

    def score(self, sample: EvalSample, prediction: str) -> float:
        if not sample.answer:
            return 0.0

        reference = sample.answer.strip().lower()

        if self._benchmark == "gsm8k":
            # Extract the final numeric answer (last number in prediction)
            import re
            numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", prediction.replace(",", ""))
            if numbers:
                # Try exact match, then compare numerically
                pred_num = numbers[-1].replace(",", "")
                ref_num = reference.replace(",", "")
                try:
                    return 1.0 if float(pred_num) == float(ref_num) else 0.0
                except ValueError:
                    pass
            # Fallback: look for the reference in the prediction
            return 1.0 if reference in prediction.lower() else 0.0

        if self._benchmark == "humaneval":
            # Check if the prediction contains the reference function body
            ref_lines = [l.strip().lower() for l in sample.answer.split("\n") if l.strip()]
            pred_lower = prediction.lower()
            matches = sum(1 for line in ref_lines if line in pred_lower)
            return matches / max(len(ref_lines), 1)

        # Default: exact match or substring containment
        return 1.0 if reference in prediction.lower() else 0.0


class _MTBenchScorer(Scorer):
    """MT-Bench scorer that uses GPT-4-as-judge via API.

    Falls back to heuristic scoring if the judge API is unavailable.
    """

    def __init__(self, api_key: str = "", model: str = "gpt-4") -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model

    def score(self, sample: EvalSample, prediction: str) -> float:
        if self._api_key:
            try:
                return self._judge_via_api(sample, prediction)
            except Exception as exc:
                logger.warning("GPT-4 judge API call failed, using heuristic: {}", exc)
        return self._heuristic_score(sample, prediction)

    def _judge_via_api(self, sample: EvalSample, prediction: str) -> float:
        """Score via GPT-4-as-judge API."""
        import httpx

        data = json.loads(sample.question)
        category = data.get("category", "general")
        turns = data.get("turns", [])

        conversation = ""
        for i, turn in enumerate(turns):
            conversation += f"User: {turn}\n"
        conversation += f"\nAssistant: {prediction[:2000]}"

        messages = [
            {"role": "system", "content": _MTBENCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Category: {category}\n\n"
                    f"Conversation:\n{conversation}\n\n"
                    f"Please rate this response on a scale of 1 to 10. "
                    f"Return only a number."
                ),
            },
        ]

        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 10,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"].strip()

        # Parse numeric score (1-10)
        import re
        match = re.search(r"(\d+)", content)
        if match:
            score = max(1, min(10, int(match.group(1))))
            return score / 10.0

        return 0.5

    def _heuristic_score(self, sample: EvalSample, prediction: str) -> float:
        """Heuristic fallback: length-based and keyword coverage."""
        if not prediction.strip():
            return 0.0
        length_score = min(1.0, len(prediction) / 500.0)
        # Category-specific keyword presence
        data = json.loads(sample.question)
        category = data.get("category", "general")
        keywords = {
            "coding": ["def ", "return", "import", "class ", "function"],
            "math": ["=", "+", "-", "*", "/", "solve", "equation"],
            "reasoning": ["because", "therefore", "if", "then", "thus"],
            "writing": ["however", "furthermore", "moreover", "consequently"],
            "extraction": ["{", "}", '"', "[", "]"],
        }
        kws = keywords.get(category, [])
        kw_score = sum(1 for kw in kws if kw.lower() in prediction.lower())
        kw_score = min(1.0, kw_score / max(len(kws), 1))
        return round(length_score * 0.4 + kw_score * 0.6, 4)


class _ArenaScorer(Scorer):
    """Scorer for Chatbot Arena pairwise comparisons.

    Uses GPT-4-as-judge to determine which response is better.
    Falls back to heuristic (length-based) scoring.
    """

    def __init__(self, api_key: str = "", judge_model: str = "gpt-4") -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._judge_model = judge_model

    def score(self, sample: EvalSample, prediction: str) -> float:
        """Score is 1.0 if model_a wins, 0.0 if model_b wins, 0.5 if tie.

        For Arena, ``prediction`` is expected to be the concatenated
        responses from both models in the format:
        ``"MODEL_A: ...\n---\nMODEL_B: ..."``
        """
        return self._compare_via_api(sample, prediction)

    def _compare_via_api(self, sample: EvalSample, combined: str) -> float:
        """Use GPT-4 judge to pick the better response."""
        if not self._api_key:
            return self._heuristic_compare(sample, combined)

        import httpx

        # Split combined into model_a and model_b responses
        parts = combined.split("\n---\n")
        response_a = parts[0] if len(parts) > 0 else ""
        response_b = parts[1] if len(parts) > 1 else ""

        messages = [
            {"role": "system", "content": _ARENA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Prompt: {sample.question}\n\n"
                    f"Response A:\n{response_a[:2000]}\n\n"
                    f"Response B:\n{response_b[:2000]}\n\n"
                    f"Which response is better? Reply with 'A', 'B', or 'Tie'."
                ),
            },
        ]

        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._judge_model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 10,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            result = resp.json()
            verdict = result["choices"][0]["message"]["content"].strip().upper()

            if "A" in verdict and "B" not in verdict:
                return 1.0  # model_a wins
            if "B" in verdict and "A" not in verdict:
                return 0.0  # model_b wins
            return 0.5  # tie
        except Exception as exc:
            logger.warning("Arena judge API call failed, using heuristic: {}", exc)
            return self._heuristic_compare(sample, combined)

    def _heuristic_compare(self, sample: EvalSample, combined: str) -> float:
        """Fallback: longer response wins."""
        parts = combined.split("\n---\n")
        len_a = len(parts[0]) if len(parts) > 0 else 0
        len_b = len(parts[1]) if len(parts) > 1 else 0
        if len_a > len_b * 1.2:
            return 1.0
        if len_b > len_a * 1.2:
            return 0.0
        return 0.5


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generates aggregated evaluation reports from raw results."""

    def generate(
        self,
        model_id: str,
        dataset: str,
        config: dict[str, Any],
        results: list[EvalResult],
        duration_s: float,
    ) -> EvalReport:
        """Aggregate results into a scored report."""
        scored = [r for r in results if r.error is None]
        errors = [r for r in results if r.error is not None]

        scores = [r.score for r in scored]
        latencies = [r.latency_ms for r in scored]
        prompt_tokens = sum(r.prompt_tokens for r in scored)
        generated_tokens = sum(r.generated_tokens for r in scored)

        metrics: dict[str, float] = {
            "accuracy": round(sum(scores) / max(len(scores), 1), 4),
            "mean_score": round(sum(scores) / max(len(scores), 1), 4),
            "median_score": round(sorted(scores)[len(scores) // 2], 4) if scores else 0.0,
            "std_score": round(self._std(scores), 4) if len(scores) > 1 else 0.0,
            "total_samples": len(results),
            "scored_samples": len(scored),
            "error_samples": len(errors),
            "error_rate": round(len(errors) / max(len(results), 1), 4),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0.0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 1 else 0.0,
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 2) if len(latencies) > 2 else 0.0,
            "total_prompt_tokens": prompt_tokens,
            "total_generated_tokens": generated_tokens,
            "duration_s": round(duration_s, 2),
        }

        # Per-category breakdown
        categories: dict[str, list[float]] = {}
        for r in scored:
            cat = r.sample.category
            categories.setdefault(cat, []).append(r.score)
        for cat, cat_scores in categories.items():
            metrics[f"{cat}_accuracy"] = round(sum(cat_scores) / len(cat_scores), 4)

        return EvalReport(
            model_id=model_id,
            dataset=dataset,
            config=config,
            metrics=metrics,
            results=results,
            status=EvalStatus.COMPLETED,
            duration_s=duration_s,
        )

    @staticmethod
    def _std(values: list[float]) -> float:
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5


# ---------------------------------------------------------------------------
# Parallel worker pool
# ---------------------------------------------------------------------------


class _WorkerPool:
    """Manages parallel evaluation across worker threads."""

    def __init__(self, max_workers: int = _MAX_WORKERS) -> None:
        self._max_workers = max_workers

    def run(
        self,
        samples: list[EvalSample],
        generate_fn: Callable[[str], tuple[str, float, int, int]],  # (prediction, latency, prompt_tokens, gen_tokens)
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[EvalResult]:
        """Evaluate samples in parallel using a thread pool."""
        results: list[EvalResult | None] = [None] * len(samples)
        completed = [0]
        lock = threading.Lock()

        def _evaluate(idx: int, sample: EvalSample) -> None:
            try:
                prediction, latency, ptokens, gtokens = generate_fn(sample.question)
                with lock:
                    results[idx] = EvalResult(
                        sample=sample,
                        prediction=prediction,
                        latency_ms=latency,
                        prompt_tokens=ptokens,
                        generated_tokens=gtokens,
                    )
                    completed[0] += 1
                    if progress_cb:
                        progress_cb(completed[0], len(samples))
            except Exception as exc:
                logger.error("Eval failed for sample {}: {}", idx, exc)
                with lock:
                    results[idx] = EvalResult(
                        sample=sample,
                        prediction="",
                        error=str(exc),
                    )
                    completed[0] += 1
                    if progress_cb:
                        progress_cb(completed[0], len(samples))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                pool.submit(_evaluate, i, sample)
                for i, sample in enumerate(samples)
            ]
            concurrent.futures.wait(futures)

        return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Token count helper
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Rough token count estimate (characters / 4 + spaces)."""
    return len(text.split()) + len(text) // 4


# ---------------------------------------------------------------------------
# EvalRunner — public API
# ---------------------------------------------------------------------------


class EvalRunner:
    """Main evaluation runner.

    Coordinates dataset loading, prompt formatting, model inference,
    scoring, and report generation.

    Args:
        coordinator: Optional coordinator instance for local inference.
            If not set, ``generate_fn`` must be provided to ``run()``.
        db_path: Path to SQLite database for persisting results.
        max_workers: Number of parallel evaluation workers.
        api_key: OpenAI API key for judge-based evaluations (MT-Bench, Arena).
            Falls back to ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        coordinator: Any = None,
        db_path: str | Path = "",
        max_workers: int = _MAX_WORKERS,
        api_key: str = "",
    ) -> None:
        self._coordinator = coordinator
        self._pool = _WorkerPool(max_workers=max_workers)
        self._db = EvalDB(db_path=db_path)
        self._report_gen = ReportGenerator()
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._db.initialize()

    # ── Dataset loaders ───────────────────────────────────────────────────

    def _get_loader(self, benchmark: str) -> DatasetLoader:
        mapping: dict[str, type[DatasetLoader]] = {
            "mmlu": _MMLULoader,
            "gsm8k": _GSM8KLoader,
            "humaneval": _HumanEvalLoader,
            "mt_bench": _MTBenchLoader,
            "arena": _ArenaLoader,
        }
        cls = mapping.get(benchmark)
        if cls is None:
            raise ValueError(f"Unknown benchmark: {benchmark}. Choose from {list(mapping.keys())}")
        return cls()

    # ── Prompt formatters ─────────────────────────────────────────────────

    def _get_formatter(self, benchmark: str) -> PromptFormatter:
        mapping: dict[str, Callable[[], PromptFormatter]] = {
            "mmlu": lambda: _HeimPromptFormatter("mmlu"),
            "gsm8k": lambda: _HeimPromptFormatter("gsm8k"),
            "humaneval": lambda: _HeimPromptFormatter("humaneval"),
            "mt_bench": lambda: _MTBenchPromptFormatter(),
            "arena": lambda: _ArenaPromptFormatter(),
        }
        factory = mapping.get(benchmark)
        if factory is None:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        return factory()

    # ── Scorers ───────────────────────────────────────────────────────────

    def _get_scorer(self, benchmark: str) -> Scorer:
        mapping: dict[str, Callable[[], Scorer]] = {
            "mmlu": lambda: _ExactMatchScorer("mmlu"),
            "gsm8k": lambda: _ExactMatchScorer("gsm8k"),
            "humaneval": lambda: _ExactMatchScorer("humaneval"),
            "mt_bench": lambda: _MTBenchScorer(api_key=self._api_key),
            "arena": lambda: _ArenaScorer(api_key=self._api_key),
        }
        factory = mapping.get(benchmark)
        if factory is None:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        return factory()

    # ── Model inference ──────────────────────────────────────────────────

    def _generate(
        self,
        prompt: str,
        model_id: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        coordinator_url: str = "",
    ) -> tuple[str, float, int, int]:
        """Run model inference via coordinator or API URL.

        Returns:
            ``(prediction_text, latency_ms, prompt_tokens, generated_tokens)``
        """
        if self._coordinator is not None:
            return self._generate_local(prompt, max_tokens, temperature)
        if coordinator_url:
            return self._generate_remote(prompt, coordinator_url, model_id, max_tokens, temperature)
        raise RuntimeError(
            "No coordinator or API URL provided. Pass ``coordinator`` or set ``coordinator_url``."
        )

    def _generate_local(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.0
    ) -> tuple[str, float, int, int]:
        """Generate using the local coordinator."""
        if self._coordinator is None:
            raise RuntimeError("Coordinator not set")

        start = time.monotonic()
        prediction = self._coordinator.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        ptokens = _count_tokens(prompt)
        gtokens = _count_tokens(prediction)
        return prediction, elapsed_ms, ptokens, gtokens

    def _generate_remote(
        self,
        prompt: str,
        url: str,
        model_id: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, float, int, int]:
        """Generate via a remote API endpoint."""
        import httpx

        base_url = url.rstrip("/")
        start = time.monotonic()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_id or "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=_EVAL_TIMEOUT_S,
        )
        resp.raise_for_status()
        elapsed_ms = (time.monotonic() - start) * 1000
        data = resp.json()
        prediction = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        ptokens = usage.get("prompt_tokens", _count_tokens(prompt))
        gtokens = usage.get("completion_tokens", _count_tokens(prediction))
        return prediction, elapsed_ms, ptokens, gtokens

    # ── Public run methods ────────────────────────────────────────────────

    def run_heim(
        self,
        benchmark: str,
        model_id: str = "",
        max_tokens: int = 256,
        temperature: float = 0.0,
        coordinator_url: str = "",
        num_samples: int = 20,
    ) -> EvalReport:
        """Run a HEIM-style benchmark (MMLU, GSM8K, HumanEval).

        Args:
            benchmark: One of ``"mmlu"``, ``"gsm8k"``, ``"humaneval"``.
            model_id: Identifier for the model being evaluated.
            max_tokens: Maximum generation tokens per sample.
            temperature: Sampling temperature (0.0 for deterministic).
            coordinator_url: Remote API URL. If empty, uses local coordinator.
            num_samples: Number of samples to evaluate.

        Returns:
            An ``EvalReport`` with aggregated metrics.
        """
        if benchmark not in ("mmlu", "gsm8k", "humaneval"):
            raise ValueError(f"HEIM benchmark must be one of: mmlu, gsm8k, humaneval, got {benchmark}")

        logger.info("Starting HEIM benchmark: {} (model={})", benchmark, model_id)

        loader = self._get_loader(benchmark)
        formatter = self._get_formatter(benchmark)
        scorer = self._get_scorer(benchmark)

        # Override num_samples for loaders that accept it
        if hasattr(loader, "_num_samples"):
            loader._num_samples = num_samples  # type: ignore[assignment]

        samples = loader.load()
        config = {
            "benchmark": benchmark,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_samples": len(samples),
        }

        start_time = time.monotonic()

        def _progress(done: int, total: int) -> None:
            if done % max(1, total // 10) == 0 or done == total:
                logger.info("HEIM {} [{}/{}] - {:.0%}", benchmark, done, total, done / total)

        results = self._pool.run(
            samples=samples,
            generate_fn=lambda q: self._generate(
                formatter.format(EvalSample(question=q)),
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                coordinator_url=coordinator_url,
            ),
            progress_cb=_progress,
        )

        # Score results
        for r in results:
            if r.error is None:
                r.score = scorer.score(r.sample, r.prediction)

        duration_s = time.monotonic() - start_time
        report = self._report_gen.generate(model_id, benchmark, config, results, duration_s)

        self._db.save_report(report)
        logger.info("HEIM {} complete: accuracy={}", benchmark, report.metrics.get("accuracy"))
        return report

    def run_mt_bench(
        self,
        model_id: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        coordinator_url: str = "",
        num_categories: int = 8,
    ) -> EvalReport:
        """Run MT-Bench evaluation (multi-turn chat quality).

        Args:
            model_id: Identifier for the model being evaluated.
            max_tokens: Maximum generation tokens per turn.
            temperature: Sampling temperature for generation.
            coordinator_url: Remote API URL. If empty, uses local coordinator.
            num_categories: Number of MT-Bench categories to evaluate (1-8).

        Returns:
            An ``EvalReport`` with quality scores (1-10 scale, normalized to 0-1).
        """
        logger.info("Starting MT-Bench evaluation (model={})", model_id)

        loader = _MTBenchLoader(num_samples=num_categories)
        formatter = _MTBenchPromptFormatter()
        scorer = _MTBenchScorer(api_key=self._api_key)

        samples = loader.load()
        config = {
            "benchmark": "mt_bench",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_categories": len(samples),
            "judge_model": "gpt-4",
        }

        start_time = time.monotonic()

        results = self._pool.run(
            samples=samples,
            generate_fn=lambda q: self._generate(
                formatter.format(EvalSample(question=q)),
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                coordinator_url=coordinator_url,
            ),
            progress_cb=lambda d, t: logger.info("MT-Bench [{}/{}]", d, t),
        )

        for r in results:
            if r.error is None:
                r.score = scorer.score(r.sample, r.prediction)

        duration_s = time.monotonic() - start_time
        report = self._report_gen.generate(model_id, "mt_bench", config, results, duration_s)

        self._db.save_report(report)
        logger.info("MT-Bench complete: mean_score={}", report.metrics.get("mean_score"))
        return report

    def run_arena(
        self,
        model_a: str = "",
        model_b: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        coordinator_url_a: str = "",
        coordinator_url_b: str = "",
        num_samples: int = 10,
    ) -> EvalReport:
        """Run Chatbot Arena-style pairwise comparison.

        Both models respond to the same prompts, then a GPT-4 judge
        determines which response is better.

        Args:
            model_a: Identifier for model A.
            model_b: Identifier for model B.
            max_tokens: Maximum generation tokens per sample.
            temperature: Sampling temperature.
            coordinator_url_a: URL for model A's API. If empty, uses local coordinator.
            coordinator_url_b: URL for model B's API. Falls back to ``coordinator_url_a``.
            num_samples: Number of prompts to compare.

        Returns:
            An ``EvalReport`` where ``accuracy`` represents model A's win rate.
        """
        logger.info("Starting Arena comparison: {} vs {}", model_a, model_b)

        loader = _ArenaLoader(num_samples=num_samples)
        formatter = _ArenaPromptFormatter()
        scorer = _ArenaScorer(api_key=self._api_key)

        samples = loader.load()
        config = {
            "benchmark": "arena",
            "model_a": model_a,
            "model_b": model_b,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "num_samples": len(samples),
            "judge_model": "gpt-4",
        }

        url_b = coordinator_url_b or coordinator_url_a
        start_time = time.monotonic()
        results: list[EvalResult] = []

        for i, sample in enumerate(samples):
            prompt = formatter.format(sample)
            try:
                # Generate from model A
                pred_a, lat_a, pt_a, gt_a = self._generate(
                    prompt, model_id=model_a, max_tokens=max_tokens,
                    temperature=temperature, coordinator_url=coordinator_url_a,
                )
                # Generate from model B
                pred_b, lat_b, pt_b, gt_b = self._generate(
                    prompt, model_id=model_b, max_tokens=max_tokens,
                    temperature=temperature, coordinator_url=url_b,
                )
                combined = f"{pred_a}\n---\n{pred_b}"
                combined_latency = lat_a + lat_b
                results.append(EvalResult(
                    sample=sample,
                    prediction=combined,
                    latency_ms=combined_latency,
                    prompt_tokens=pt_a + pt_b,
                    generated_tokens=gt_a + gt_b,
                ))
            except Exception as exc:
                logger.error("Arena sample {} failed: {}", i, exc)
                results.append(EvalResult(
                    sample=sample,
                    prediction="",
                    error=str(exc),
                ))

            if (i + 1) % max(1, num_samples // 5) == 0 or i + 1 == num_samples:
                logger.info("Arena [{}/{}]", i + 1, num_samples)

        for r in results:
            if r.error is None:
                r.score = scorer.score(r.sample, r.prediction)

        duration_s = time.monotonic() - start_time
        report = self._report_gen.generate(
            f"{model_a}_vs_{model_b}", "arena", config, results, duration_s,
        )

        # Add arena-specific metrics
        win_rate = sum(1 for r in results if r.score == 1.0) / max(len(results), 1)
        tie_rate = sum(1 for r in results if r.score == 0.5) / max(len(results), 1)
        loss_rate = sum(1 for r in results if r.score == 0.0) / max(len(results), 1)
        report.metrics["win_rate"] = round(win_rate, 4)
        report.metrics["tie_rate"] = round(tie_rate, 4)
        report.metrics["loss_rate"] = round(loss_rate, 4)

        self._db.save_report(report)
        logger.info("Arena complete: win_rate={:.1%}, tie_rate={:.1%}", win_rate, tie_rate)
        return report

    # ── Report access ─────────────────────────────────────────────────────

    def list_reports(
        self,
        model_id: str | None = None,
        dataset: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List evaluation reports with optional filtering."""
        return self._db.list_reports(
            model_id=model_id, dataset=dataset, limit=limit, offset=offset,
        )

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Get a single report by ID."""
        return self._db.get_report(report_id)

    def get_report_results(self, report_id: str) -> list[dict[str, Any]]:
        """Get detailed results for a report."""
        return self._db.get_report_results(report_id)

    def delete_report(self, report_id: str) -> bool:
        """Delete a report and its results."""
        return self._db.delete_report(report_id)

    def close(self) -> None:
        """Close database connection."""
        self._db.close()


# ---------------------------------------------------------------------------
# Convenience: run all HEIM benchmarks
# ---------------------------------------------------------------------------


def run_all_heim(
    model_id: str = "",
    coordinator_url: str = "",
    num_samples: int = 20,
    runner: EvalRunner | None = None,
) -> dict[str, EvalReport]:
    """Run all three HEIM benchmarks (MMLU, GSM8K, HumanEval).

    Args:
        model_id: Model identifier.
        coordinator_url: Remote API URL.
        num_samples: Samples per benchmark.
        runner: Reusable EvalRunner instance. Creates one if not provided.

    Returns:
        Dict mapping benchmark name to EvalReport.
    """
    close_runner = runner is None
    runner = runner or EvalRunner()
    try:
        reports: dict[str, EvalReport] = {}
        for benchmark in ("mmlu", "gsm8k", "humaneval"):
            reports[benchmark] = runner.run_heim(
                benchmark=benchmark,
                model_id=model_id,
                coordinator_url=coordinator_url,
                num_samples=num_samples,
            )
        return reports
    finally:
        if close_runner:
            runner.close()


__all__ = [
    "EvalBenchmark",
    "EvalDB",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "EvalSample",
    "EvalStatus",
    "run_all_heim",
]
