"""Output hash computation and registry for reproducibility.

Provides a mechanism to compute SHA-256 hashes of model outputs
(tokens, logits, intermediate hidden states) at each generation step,
enabling deterministic comparison across runs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch


@dataclass
class GenerationOutput:
    """Captures all outputs from a single generation run.

    Attributes:
        token_ids: The generated token ID sequence (including prompt).
        text: The decoded output text (excluding prompt).
        step_logits: List of logit tensors, one per generation step.
        step_hidden_states: Optional list of hidden state tensors.
        model_name: Model used for generation.
        temperature: Sampling temperature used.
        prompt: Input prompt.
    """

    token_ids: list[int]
    text: str
    step_logits: list[torch.Tensor] = field(default_factory=list)
    step_hidden_states: list[torch.Tensor] | None = None
    model_name: str = ""
    temperature: float = 0.0
    prompt: str = ""


def compute_output_hash(
    tensor: torch.Tensor,
    include_shape: bool = True,
) -> str:
    """Compute a SHA-256 hash of a tensor's values.

    Uses the raw byte representation for determinism. The hash is stable
    across identical tensors regardless of device or strides.

    Args:
        tensor: Any torch tensor.
        include_shape: If True, incorporates the shape into the hash
            to distinguish e.g. ``[1, 3]`` from ``[3, 1]``.

    Returns:
        Hex SHA-256 digest (64 chars).
    """
    h = hashlib.sha256()
    if include_shape:
        h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    # Move to CPU, make contiguous, hash the underlying bytes
    arr = tensor.detach().cpu().contiguous()
    h.update(arr.numpy(force=True).tobytes())
    return h.hexdigest()


def compute_text_hash(text: str) -> str:
    """Compute a SHA-256 hash of output text.

    Args:
        text: Decoded output text.

    Returns:
        Hex SHA-256 digest (64 chars).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_token_ids_hash(token_ids: Sequence[int]) -> str:
    """Compute a SHA-256 hash of a token ID sequence.

    Args:
        token_ids: List of token IDs.

    Returns:
        Hex SHA-256 digest (64 chars).
    """
    h = hashlib.sha256()
    h.update(json.dumps(list(token_ids), separators=(",", ":")).encode())
    return h.hexdigest()


class OutputHashRegistry:
    """Stores and compares output hashes from reference and distributed runs.

    Useful for CI pipelines and regression testing:
      1. Run reference (single-node) and store hashes.
      2. Run distributed and store hashes.
      3. Compare — any mismatch indicates drift.

    Usage:
        registry = OutputHashRegistry()

        # After reference run
        registry.store_reference("prompt-1", token_ids, text_hash)

        # After distributed run
        registry.store_candidate("prompt-1", token_ids, text_hash)

        # Compare
        results = registry.compare_all()
    """

    def __init__(self) -> None:
        self._reference: dict[str, dict[str, Any]] = {}
        self._candidate: dict[str, dict[str, Any]] = {}
        self._created_at = time.time()

    def store_reference(
        self,
        key: str,
        output: GenerationOutput,
    ) -> None:
        """Store reference (single-node) output hashes for a prompt.

        Args:
            key: Unique identifier (typically the prompt or request ID).
            output: GenerationOutput from the reference run.
        """
        self._reference[key] = self._extract_hashes(output)

    def store_candidate(
        self,
        key: str,
        output: GenerationOutput,
    ) -> None:
        """Store candidate (distributed) output hashes for a prompt.

        Args:
            key: Same identifier used in ``store_reference``.
            output: GenerationOutput from the distributed run.
        """
        self._candidate[key] = self._extract_hashes(output)

    def compare(self, key: str) -> dict[str, bool]:
        """Compare reference and candidate hashes for a single prompt.

        Args:
            key: The prompt identifier.

        Returns:
            Dict mapping metric names to ``True`` if they match.
        """
        ref = self._reference.get(key, {})
        cand = self._candidate.get(key, {})
        all_keys = set(ref) | set(cand)
        results: dict[str, bool] = {}
        for k in all_keys:
            if k not in ref:
                results[k] = False
            elif k not in cand:
                results[k] = False
            else:
                results[k] = ref[k] == cand[k]
        return results

    def compare_all(self) -> dict[str, dict[str, bool]]:
        """Compare all stored reference vs candidate hashes.

        Returns:
            Dict mapping prompt keys to their comparison results.
        """
        all_keys = set(self._reference) | set(self._candidate)
        return {k: self.compare(k) for k in all_keys}

    def summary(self) -> dict[str, Any]:
        """Return a summary of all comparisons.

        Returns:
            Dict with counts of passing and failing prompts.
        """
        comparisons = self.compare_all()
        total = len(comparisons)
        passed = sum(
            1 for c in comparisons.values() if all(c.values())
        )
        failed = total - passed
        return {
            "total_prompts": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / max(total, 1), 4),
            "comparisons": comparisons,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the registry to a JSON-compatible dict."""
        return {
            "created_at": self._created_at,
            "reference": {k: self._serialize_hashes(v) for k, v in self._reference.items()},
            "candidate": {k: self._serialize_hashes(v) for k, v in self._candidate.items()},
        }

    @staticmethod
    def _extract_hashes(output: GenerationOutput) -> dict[str, str]:
        hashes: dict[str, str] = {
            "token_ids": compute_token_ids_hash(output.token_ids),
            "text": compute_text_hash(output.text),
        }
        for i, logits in enumerate(output.step_logits):
            hashes[f"logits_step_{i}"] = compute_output_hash(logits)
        if output.step_hidden_states:
            for i, hs in enumerate(output.step_hidden_states):
                hashes[f"hidden_step_{i}"] = compute_output_hash(hs)
        return hashes

    @staticmethod
    def _serialize_hashes(hashes: dict[str, str]) -> dict[str, str]:
        return hashes
