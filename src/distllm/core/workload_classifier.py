"""Workload classifier for auto-speculative selection.

Classifies request text into workload types using heuristic analysis
(ngram entropy, keyword matching, structural patterns).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from enum import Enum


class WorkloadType(str, Enum):
    """Classified workload types for speculative method selection."""
    CODE = "code"
    REPETITIVE = "repetitive"
    DIVERSE = "diverse"
    INSTRUCTION = "instruction"
    UNKNOWN = "unknown"


# Keyword patterns for workload detection
_CODE_KEYWORDS = {
    "def ", "class ", "import ", "from ", "return ", "if ", "for ", "while ",
    "function ", "const ", "let ", "var ", "fn ", "pub ", "async ", "await ",
    "struct ", "impl ", "trait ", "interface ", "type ", "enum ",
    "print(", "console.", "self.", "this.", "@", "#include", "package ",
    "=>", "->", "::", "..",
}

_INSTRUCTION_KEYWORDS = {
    "please", "can you", "could you", "write", "explain", "summarize",
    "translate", "generate", "create", "describe", "analyze", "compare",
    "what is", "how to", "why does", "tell me",
}

# Code language markers
_CODE_MARKERS = {"```", ">>>", "... ", "$ ", "# ", "//", "/*", "*/"}


def _ngram_entropy(tokens: list[str], n: int = 3) -> float:
    """Calculate n-gram entropy of a token sequence.

    Low entropy = repetitive text; High entropy = diverse text.
    """
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    total = len(ngrams)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def _ngram_repetition_ratio(tokens: list[str], n: int = 3) -> float:
    """Return the fraction of n-grams beyond their first occurrence."""
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def _code_ratio(text: str) -> float:
    """Estimate the ratio of code-like content in the text."""
    if not text:
        return 0.0

    lines = text.split("\n")
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Check for code markers
        if any(marker in stripped for marker in _CODE_MARKERS):
            code_lines += 1
            continue
        # Check for code keywords
        if any(kw in text.lower() for kw in _CODE_KEYWORDS):
            code_lines += 1
            continue
        # Lines ending with semicolons or braces are likely code
        if stripped.endswith((";", "{", "}", ")", "]", ",")):
            code_lines += 1

    return code_lines / max(len(lines), 1)


def _keyword_density(text: str, keywords: set[str]) -> float:
    """Calculate the density of specific keywords in text."""
    text_lower = text.lower()
    if not text_lower:
        return 0.0
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches / len(keywords)


def classify(text: str) -> WorkloadType:
    """Classify request text into a workload type.

    Uses heuristics:
    1. Code detection: keyword matching + structural patterns
    2. Instruction detection: natural language instruction patterns
    3. Repetitive detection: low n-gram entropy
    4. Diverse: high entropy, no clear pattern

    Args:
        text: The input text to classify.

    Returns:
        Classified WorkloadType.
    """
    if not text:
        return WorkloadType.UNKNOWN

    # Tokenize: split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())

    # Check for code patterns
    code_ratio = _code_ratio(text)
    code_keyword_density = _keyword_density(text, _CODE_KEYWORDS)

    if code_ratio > 0.3 or code_keyword_density > 0.15:
        return WorkloadType.CODE

    # Check for instruction patterns
    instruction_density = _keyword_density(text, _INSTRUCTION_KEYWORDS)
    if instruction_density > 0.1 or _keyword_density(text, {"please", "can you", "how to", "what is"}) > 0.5:
        return WorkloadType.INSTRUCTION

    # Check entropy for repetitive vs diverse
    entropy = _ngram_entropy(tokens, n=3)
    repetition_ratio = max(
        _ngram_repetition_ratio(tokens, n=2),
        _ngram_repetition_ratio(tokens, n=3),
    )

    # Thresholds: calibrated on typical text
    if len(tokens) >= 6 and repetition_ratio >= 0.25:
        return WorkloadType.REPETITIVE
    if entropy < 1.5:
        return WorkloadType.REPETITIVE
    elif entropy > 4.0:
        return WorkloadType.DIVERSE

    return WorkloadType.UNKNOWN


def classify_features(text: str) -> dict[str, float]:
    """Extract classification features from text.

    Returns a dict with entropy, code_ratio, keyword densities, etc.
    Useful for debugging and for upgrading to ML-based classification.
    """
    if not text:
        return {
            "entropy_3gram": 0.0,
            "entropy_2gram": 0.0,
            "code_ratio": 0.0,
            "code_keyword_density": 0.0,
            "instruction_density": 0.0,
            "word_count": 0.0,
            "avg_word_length": 0.0,
        }

    tokens = re.findall(r'\b\w+\b', text.lower())
    return {
        "entropy_3gram": _ngram_entropy(tokens, n=3),
        "entropy_2gram": _ngram_entropy(tokens, n=2),
        "repetition_3gram": _ngram_repetition_ratio(tokens, n=3),
        "repetition_2gram": _ngram_repetition_ratio(tokens, n=2),
        "code_ratio": _code_ratio(text),
        "code_keyword_density": _keyword_density(text, _CODE_KEYWORDS),
        "instruction_density": _keyword_density(text, _INSTRUCTION_KEYWORDS),
        "word_count": len(tokens),
        "avg_word_length": sum(len(t) for t in tokens) / max(len(tokens), 1),
    }
