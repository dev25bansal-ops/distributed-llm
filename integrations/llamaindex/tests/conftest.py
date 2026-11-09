"""Pytest configuration for the distllm-llamaindex test suite.

Forces this checkout's integration source onto ``sys.path`` ahead of
site-packages: an editable install of ``distllm_llamaindex`` may point at a
different checkout (the main repo, when the suite runs from a git worktree),
and site-packages precedes ``PYTHONPATH`` so the shadowing would otherwise go
unnoticed while the tests silently exercise stale code.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))            # integrations/llamaindex/tests
_INTEGRATIONS_DIR = os.path.dirname(os.path.dirname(_HERE))  # integrations/
_THIS_SRC = os.path.join(os.path.dirname(_INTEGRATIONS_DIR), "llamaindex", "src")

if os.path.isdir(_THIS_SRC) and sys.path[0] != _THIS_SRC:
    sys.path.insert(0, _THIS_SRC)
if _INTEGRATIONS_DIR not in sys.path:
    sys.path.insert(1, _INTEGRATIONS_DIR)
