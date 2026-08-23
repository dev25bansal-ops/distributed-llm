"""Test configuration: ensure ``_common`` is importable for the tools provider."""

import sys
from pathlib import Path

# The tools provider imports ``_common.base_tool_provider``; make the
# ``integrations`` directory importable without a live install.
_INTEGRATIONS = Path(__file__).resolve().parents[2]
if str(_INTEGRATIONS) not in sys.path:
    sys.path.insert(0, str(_INTEGRATIONS))
