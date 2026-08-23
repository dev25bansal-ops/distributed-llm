"""Regression tests for HIGH fix C16: broken .venv / python pin.

The project pinned ``requires-python = ">=3.10"`` which admitted Python 3.14,
where the checked-in environment's ``pydantic_core`` is broken (M1). The pin
is now tightened to ``>=3.10,<3.13`` (CI matrix is 3.10/3.11) so a broken 3.14
install is rejected at resolution time instead of failing at import.
"""

from __future__ import annotations

import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


@pytest.mark.skipif(tomllib is None, reason="tomllib unavailable")
def test_requires_python_excludes_3_13_plus():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    spec = data["project"]["requires-python"]
    # Must cap the upper bound below 3.13 (broken 3.14 env excluded).
    assert "<3.13" in spec
    assert ">=3.10" in spec


def test_current_interpreter_within_supported_range():
    # The verification venv is 3.11; assert we are in the supported band.
    assert sys.version_info >= (3, 10)
    assert sys.version_info < (3, 13)
