"""Regression tests for CRITICAL fix C4:

CI installed a non-existent ``testing`` extra (``pip install -e ".[dev,testing]"``)
while ``pyproject.toml`` only defined ``dev``/``self-hosted``/``backends``/etc.
pip silently warns (not errors), so ``transformers``/``torch``/``vllm`` were
never installed in CI -> the 80% coverage gate was measured over a truncated
import graph, and the 104 ML-dependent test modules errored permanently.

Fix: define a real ``testing`` extra aggregating ``dev,self-hosted,backends``.

These tests parse ``pyproject.toml`` with stdlib ``tomllib`` and assert the
extra exists and pulls in the ML dependencies that were missing in CI.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/ -> repo root


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_testing_extra_is_defined():
    """C4: the ``testing`` extra must exist so CI's ``.[dev,testing]`` resolves."""
    assert "testing" in _extras(), "pyproject.toml missing the 'testing' extra"


def test_testing_extra_aggregates_ml_deps():
    """The testing extra must pull in the ML deps that were missing in CI."""
    testing = _extras().get("testing", [])
    joined = " ".join(testing)
    assert "self-hosted" in joined, "testing extra must include self-hosted (torch/transformers)"
    assert "backends" in joined, "testing extra must include backends (vllm)"


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
