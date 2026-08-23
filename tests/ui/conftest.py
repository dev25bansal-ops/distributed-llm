"""Shared fixtures for distllm.ui tests.

Usage pattern follows established test conventions:
see tests/core/test_health_manager.py, tests/core/test_node_recovery.py, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._import_helper import bootstrap_fake_packages, load_module

if TYPE_CHECKING:
    from collections.abc import Generator

# Bootstrap fake packages to avoid circular import chains.
# (distllm/ui/app.py does not itself import any distllm subpackage,
#  but bootstrap_fake_packages is idempotent and mirrors the standard
#  test convention.)
bootstrap_fake_packages()

# Load the module under test once at session scope.
_ui_mod = load_module("distllm/ui/app.py")


@pytest.fixture(scope="session")
def ui_app_module():
    """Return the loaded ``distllm.ui.app`` module.

    Use this to access any symbol needed across tests::

        def test_something(ui_app_module):
            assert ui_app_module.API_URL == "http://localhost:8000"
    """
    return _ui_mod
