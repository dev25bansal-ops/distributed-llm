"""Regression tests for CRITICAL fix C4:

CI installed a non-existent ``testing`` extra (``pip install -e ".[dev,testing]"``)
while ``pyproject.toml`` only defined ``dev``/``self-hosted``/``backends``/etc.
pip silently warns (not errors), so ``transformers``/``torch``/``vllm`` were
never installed in CI -> the 80% coverage gate was measured over a truncated
import graph, and the 104 ML-dependent test modules errored permanently.

Fix: define a real ``testing`` extra providing the ML dependencies
(``torch``/``transformers``/...) that CI needs.

These tests parse ``pyproject.toml`` with stdlib ``tomllib`` and assert the
extra exists and pulls in the ML dependencies that were missing in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ``tomllib`` is stdlib only from Python 3.11; the project supports 3.10
# (requires-python = ">=3.10"), where a bare ``import tomllib`` fails collection
# for the whole module. Fall back to the ``tomli`` backport on 3.10.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 path
    tomllib = pytest.importorskip(
        "tomli", reason="tomllib requires Python >=3.11; install tomli on 3.10"
    )

REPO_ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/ -> repo root


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_testing_extra_is_defined():
    """C4: the ``testing`` extra must exist so CI's ``.[dev,testing]`` resolves."""
    assert "testing" in _extras(), "pyproject.toml missing the 'testing' extra"


# The ML runtime the ``testing`` extra must guarantee. Asserted on the actual
# requirement names rather than on a self-referential extras entry, because the
# extra was later inlined (torch/transformers/... listed directly) instead of
# aggregating ``self-hosted``/``backends`` by name.
_ML_DEPS_REQUIRED = ("torch", "transformers", "accelerate", "safetensors",
                     "huggingface-hub", "sentencepiece")


def _extra_closure(name: str, extras: dict[str, list[str]], _seen: frozenset[str] = frozenset()) -> set[str]:
    """Names a requirement list pulls in, following ``pkg[extra]`` self-references."""
    reqs: set[str] = set()
    for raw in extras.get(name, []):
        reqs.add(raw)
        # An entry like "backends[server]" or "self-hosted" references another extra.
        head = raw.split("[", 1)[0].split(">=", 1)[0].split("==", 1)[0].strip()
        if head in extras and head not in _seen:
            reqs |= _extra_closure(head, extras, _seen | {name})
    return reqs


def test_testing_extra_aggregates_ml_deps():
    """The testing extra must pull in the ML deps that were missing in CI."""
    extras = _extras()
    closure = " ".join(sorted(_extra_closure("testing", extras))).lower()
    for dep in _ML_DEPS_REQUIRED:
        assert dep in closure, f"testing extra must provide {dep} (found: {closure})"


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="tomllib based check; ML import skipped on minimal envs",
)
def test_ml_deps_importable_when_testing_extra_installed():
    """If the testing extra is installed, torch/transformers must import."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("torch/transformers not installed in this environment (expected on minimal venv)")
    assert True
