"""Workload classifier for auto-speculative selection."""

from __future__ import annotations

import math
import re
from collections import Counter
from enum import Enum


class WorkloadType(str, Enum):
    CODE = "code"
    REPETITIVE = "repetitive"
    DIVERSE = "diverse"
    INSTRUCTION = "instruction"
    UNKNOWN = "unknown"


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

_CODE_MARKERS = {"```", ">>>", "... ", "$ ", "# ", "//", "/*", "*/"}


def _ngram_entropy(tokens: list[str], n: int = 3) -> float:
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
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def _code_ratio(text: str) -> float:
    if not text:
        return 0.0

    lines = text.split("\n")
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker in stripped for marker in _CODE_MARKERS):
            code_lines += 1
            continue
        if any(kw in text.lower() for kw in _CODE_KEYWORDS):
            code_lines += 1
            continue
        if stripped.endswith((";", "{", "}", ")", "]", ",")):
            code_lines += 1

    return code_lines / max(len(lines), 1)


def _keyword_density(text: str, keywords: set[str]) -> float:
    text_lower = text.lower()
    if not text_lower:
        return 0.0
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches / len(keywords)


def classify(text: str) -> WorkloadType:
    if not text:
        return WorkloadType.UNKNOWN

    tokens = re.findall(r'\b\w+\b', text.lower())

    code_ratio = _code_ratio(text)
    code_keyword_density = _keyword_density(text, _CODE_KEYWORDS)

    if code_ratio > 0.3 or code_keyword_density > 0.15:
        return WorkloadType.CODE

    instruction_density = _keyword_density(text, _INSTRUCTION_KEYWORDS)
    if instruction_density > 0.1 or _keyword_density(text, {"please", "can you", "how to", "what is"}) > 0.5:
        return WorkloadType.INSTRUCTION

    entropy = _ngram_entropy(tokens, n=3)
    repetition_ratio = max(
        _ngram_repetition_ratio(tokens, n=2),
        _ngram_repetition_ratio(tokens, n=3),
    )

    if len(tokens) >= 6 and repetition_ratio >= 0.25:
        return WorkloadType.REPETITIVE
    if entropy < 1.5:
        return WorkloadType.REPETITIVE
    elif entropy > 4.0:
        return WorkloadType.DIVERSE

    return WorkloadType.UNKNOWN


def classify_features(text: str) -> dict[str, float]:
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
