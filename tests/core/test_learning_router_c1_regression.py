"""Regression tests for CRITICAL fix C1:

``learning_router.py`` used ``random.random()`` / ``random.choice()`` and
``_random.sample()`` without importing ``random`` -> ``NameError`` on the
exploration / replay code paths (~15% of routes). The module-level
``import random`` / ``_random = random`` added by the fix must make these
primitives available, and the module must import without error.

We exercise the exact primitives the previously-broken lines referenced
(``random.random``, ``random.choice``, ``_random.sample``) so the bug cannot
regress.
"""

from __future__ import annotations

import random

import pytest

from distllm.core import learning_router as lr


def test_module_imports_without_nameerror():
    """Importing the module must succeed (the missing `import random` is fixed)."""
    assert lr is not None
    # The alias used by the replay-buffer sampler must exist.
    assert hasattr(lr, "_random"), "learning_router missing _random alias"


def test_random_primitives_available():
    """The primitives referenced by the previously-broken lines must work."""
    # line 232: random.random()
    assert isinstance(lr.random.random(), float)
    # line 235: random.choice(...)
    assert lr.random.choice(["a", "b", "c"]) in {"a", "b", "c"}
    # line 629: _random.sample(self._replay_buffer, k)
    sample = lr._random.sample([1, 2, 3, 4, 5], 3)
    assert len(sample) == 3
    assert set(sample).issubset({1, 2, 3, 4, 5})
