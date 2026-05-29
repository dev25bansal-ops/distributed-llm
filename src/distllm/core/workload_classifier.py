"""Workload Classifier — classify prompts for speculative method selection.

Analyzes prompt text to determine the workload type, which influences
which speculative decoding method performs best:

- CODE: high code_ratio → ngram works well (repetitive syntax)
- INSTRUCTION: question/explanation patterns → eagle works well
- REPETITIVE: low entropy, repeated patterns → ngram excels
- DIVERSE: high entropy, varied content → eagle/draft_model
- UNKNOWN: insufficient signal

Usage::

    from distllm.core.workload_classifier import classify, classify_features

    wt = classify("def foo(): return 42")  # WorkloadType.CODE
    features = classify_features("def foo(): return 42")
    # {"code_ratio": 0.6, "entropy_3gram": 2.1, ...}
"""

from __future__ import annotations

import math
import re
from collections import Counter
from enum import Enum


class WorkloadType(str, Enum):
    CODE = "code"
    INSTRUCTION = "instruction"
    REPETITIVE = "repetitive"
    DIVERSE = "diverse"
    UNKNOWN = "unknown"


# Patterns that indicate code
_CODE_PATTERNS = [
    re.compile(r"^\s*(def |class |import |from |if |for |while |return |async def )", re.M),
    re.compile(r"[{}\[\]();]"),
    re.compile(r"print\(|console\.log|System\.out"),
    re.compile(r"^\s*//|^\s*/\*|^\s*#", re.M),
    re.compile(r"=>|->|::|&&|\|\|"),
]

# Patterns that indicate instruction/question
_INSTRUCTION_PATTERNS = [
    re.compile(r"\b(please|explain|summarize|describe|how|what|why|when|where|can you)\b", re.I),
    re.compile(r"\?"),
    re.compile(r"\b(step by step|in detail|briefly|overview)\b", re.I),
]


def classify(text: str) -> WorkloadType:
    """Classify a prompt into a workload type."""
    if not text or not text.strip():
        return WorkloadType.UNKNOWN

    features = classify_features(text)

    # Code detection: high code_ratio
    if features["code_ratio"] > 0.15:
        return WorkloadType.CODE

    # Instruction detection: question marks + instruction words
    if features["instruction_score"] > 0.3:
        return WorkloadType.INSTRUCTION

    # Repetitive detection: low entropy
    if features["entropy_3gram"] < 2.0 and features["repetition_ratio"] > 0.3:
        return WorkloadType.REPETITIVE

    # Diverse detection: high entropy
    if features["entropy_3gram"] > 4.0:
        return WorkloadType.DIVERSE

    # Default: if it has some code signals, classify as code
    if features["code_ratio"] > 0.05:
        return WorkloadType.CODE

    return WorkloadType.UNKNOWN


def classify_features(text: str) -> dict[str, float]:
    """Extract features used for workload classification.

    Returns a dict with:
    - code_ratio: fraction of lines matching code patterns
    - instruction_score: fraction of words matching instruction patterns
    - entropy_3gram: Shannon entropy of 3-gram distribution
    - repetition_ratio: fraction of repeated 3-grams
    """
    if not text:
        return {
            "code_ratio": 0.0,
            "instruction_score": 0.0,
            "entropy_3gram": 0.0,
            "repetition_ratio": 0.0,
        }

    lines = text.split("\n")

    # Code ratio
    code_lines = sum(
        1 for line in lines
        if any(p.search(line) for p in _CODE_PATTERNS)
    )
    code_ratio = code_lines / max(len(lines), 1)

    # Instruction score
    instruction_hits = sum(
        1 for p in _INSTRUCTION_PATTERNS
        if p.search(text)
    )
    instruction_score = instruction_hits / max(len(_INSTRUCTION_PATTERNS), 1)

    # 3-gram entropy
    trigrams = [text[i:i + 3] for i in range(max(len(text) - 2, 0))]
    trigram_counts = Counter(trigrams)
    total = sum(trigram_counts.values())
    entropy = 0.0
    if total > 0:
        for count in trigram_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

    # Repetition ratio
    repeated = sum(1 for c in trigram_counts.values() if c > 1)
    repetition_ratio = repeated / max(len(trigram_counts), 1)

    return {
        "code_ratio": round(code_ratio, 4),
        "instruction_score": round(instruction_score, 4),
        "entropy_3gram": round(entropy, 4),
        "repetition_ratio": round(repetition_ratio, 4),
    }
