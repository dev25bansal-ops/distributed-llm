"""Dataset loaders for the LLM evaluation harness.

Provides loaders for HEIM benchmarks (MMLU, GSM8K, HumanEval),
MT-Bench, and Chatbot Arena.
"""

from __future__ import annotations

import abc
import json

from loguru import logger

from distllm.core.evaluation.constants import _MTBENCH_CATEGORIES
from distllm.core.evaluation.models import EvalSample


# ---------------------------------------------------------------------------
# Base class
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


# ---------------------------------------------------------------------------
# HEIM loaders
# ---------------------------------------------------------------------------


class _MMLULoader(DatasetLoader):
    """MMLU dataset loader.

    Attempts to load the full MMLU dataset via ``datasets`` library.
    Falls back to embedded sample questions if ``datasets`` is unavailable.

    Install: ``pip install datasets``
    """

    def __init__(self, num_samples: int = 20) -> None:
        self._num_samples = num_samples

    def load(self, split: str = "test") -> list[EvalSample]:
        logger.info("Loading MMLU ({} samples, split={})", self._num_samples, split)

        try:
            from datasets import load_dataset
            ds = load_dataset("mmlu", "all", split=split)
            # Shuffle and truncate to requested sample count
            ds = ds.shuffle(seed=42).select(range(min(self._num_samples, len(ds))))
            samples = []
            for item in ds:
                samples.append(EvalSample(
                    question=item["question"],
                    answer=item["answer"],
                    category=item.get("subject", "general"),
                    metadata={"source": "mmlu_datasets"},
                ))
            logger.info("MMLU: loaded {} real samples from datasets library", len(samples))
            return samples
        except ImportError:
            logger.warning(
                "datasets library not available — using embedded MMLU samples. "
                "Install with: pip install datasets"
            )
        except Exception as e:
            logger.warning("MMLU dataset load failed ({}), using embedded fallback", e)

        # Fallback: 10 embedded questions rotated to fill num_samples
        embedded = [
            {"q": "What is the capital of France?", "a": "Paris", "c": "geography"},
            {"q": "The chemical symbol for gold is:", "a": "Au", "c": "science"},
            {"q": "Which planet is known as the Red Planet?", "a": "Mars", "c": "science"},
            {"q": "Who wrote 'Romeo and Juliet'?", "a": "William Shakespeare", "c": "humanities"},
            {"q": "What is the largest ocean on Earth?", "a": "Pacific Ocean", "c": "geography"},
            {"q": "In what year did World War II end?", "a": "1945", "c": "history"},
            {"q": "What is the powerhouse of the cell?", "a": "Mitochondria", "c": "biology"},
            {"q": "What is the value of Pi to two decimal places?", "a": "3.14", "c": "math"},
            {"q": "Which element has the atomic number 1?", "a": "Hydrogen", "c": "science"},
            {"q": "What is the speed of light in vacuum (m/s)?", "a": "299,792,458", "c": "physics"},
        ]
        samples = []
        for i in range(self._num_samples):
            q = embedded[i % len(embedded)]
            samples.append(EvalSample(
                question=q["q"], answer=q["a"], category=q["c"],
                metadata={"source": "mmlu_embedded", "index": i},
            ))
        return samples


class _GSM8KLoader(DatasetLoader):
    """GSM8K (grade school math) dataset loader.

    Attempts to load the full GSM8K test split via ``datasets`` library.
    Falls back to embedded problems if unavailable.
    """

    def __init__(self, num_samples: int = 20) -> None:
        self._num_samples = num_samples

    def load(self, split: str = "test") -> list[EvalSample]:
        logger.info("Loading GSM8K ({} samples, split={})", self._num_samples, split)

        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split=split)
            ds = ds.shuffle(seed=42).select(range(min(self._num_samples, len(ds))))
            samples = []
            for item in ds:
                samples.append(EvalSample(
                    question=item["question"],
                    answer=item["answer"],
                    category="math",
                    metadata={"source": "gsm8k_datasets"},
                ))
            logger.info("GSM8K: loaded {} real samples from datasets library", len(samples))
            return samples
        except ImportError:
            logger.warning("datasets library not available — using embedded GSM8K samples")
        except Exception as e:
            logger.warning("GSM8K dataset load failed ({}), using embedded fallback", e)

        # Fallback: embedded problems
        embedded = [
            {"q": "Janet has 3 apples. She buys 5 more. How many does she have?", "a": "8"},
            {"q": "A train travels 120 km in 2 hours. What is its speed in km/h?", "a": "60"},
            {"q": "If a pizza has 8 slices and 3 people eat 2 slices each, how many slices remain?", "a": "2"},
            {"q": "The product of two numbers is 36. One number is 9. What is the other?", "a": "4"},
            {"q": "Alice is 12 years old. Bob is 3 years older. How old will Bob be in 5 years?", "a": "20"},
            {"q": "A rectangle has length 10cm and width 5cm. What is its area?", "a": "50"},
            {"q": "There are 24 students in a class. If they sit in rows of 6, how many rows?", "a": "4"},
            {"q": "A store sold 15 items on Monday and 23 on Tuesday. How many total?", "a": "38"},
            {"q": "If 3 notebooks cost $6, how much do 5 notebooks cost?", "a": "10"},
            {"q": "A garden has 4 rows of 7 flowers each. How many flowers total?", "a": "28"},
        ]
        samples = []
        for i in range(self._num_samples):
            p = embedded[i % len(embedded)]
            samples.append(EvalSample(question=p["q"], answer=p["a"], category="math",
                                      metadata={"source": "gsm8k_embedded", "index": i}))
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


# ---------------------------------------------------------------------------
# MT-Bench loader
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Arena loader
# ---------------------------------------------------------------------------


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


__all__ = [
    "DatasetLoader",
    "_MMLULoader",
    "_GSM8KLoader",
    "_HumanEvalLoader",
    "_MTBenchLoader",
    "_ArenaLoader",
]
