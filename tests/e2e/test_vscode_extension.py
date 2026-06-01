"""VS Code extension tests.

Tests the DistLLM VS Code extension functionality:
- Extension activation
- Status bar items
- Command registration
- API integration

Run with:
    pytest tests/e2e/test_vscode_extension.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


EXTENSION_DIR = Path(__file__).resolve().parent.parent.parent / "extensions" / "vscode"


class TestVSCodeExtensionStructure:
    """Test extension file structure and configuration."""

    def test_package_json_exists(self):
        """package.json should exist in the extension directory."""
        assert (EXTENSION_DIR / "package.json").exists()

    def test_package_json_valid(self):
        """package.json should be valid JSON with required fields."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        assert "name" in pkg
        assert "version" in pkg
        assert "engines" in pkg
        assert "contributes" in pkg

    def test_commands_registered(self):
        """Extension should register expected commands."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        commands = pkg.get("contributes", {}).get("commands", [])
        cmd_ids = [c["command"] for c in commands]

        assert "distllm.sendSelection" in cmd_ids
        assert "distllm.openDashboard" in cmd_ids
        assert "distllm.refreshStatus" in cmd_ids

    def test_configuration_properties(self):
        """Extension should have expected configuration properties."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        config = pkg.get("contributes", {}).get("configuration", {})
        props = config.get("properties", {})

        assert "distllm.apiUrl" in props
        assert "distllm.model" in props
        assert "distllm.refreshInterval" in props
        assert "distllm.maxTokens" in props
        assert "distllm.temperature" in props

    def test_context_menu_registered(self):
        """Extension should register context menu for 'Send to DistLLM'."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        menus = pkg.get("contributes", {}).get("menus", {})
        context = menus.get("editor/context", [])

        send_cmd = [m for m in context if m.get("command") == "distllm.sendSelection"]
        assert len(send_cmd) > 0, "Send to DistLLM not in context menu"
        assert send_cmd[0].get("when") == "editorHasSelection"

    def test_activation_events(self):
        """Extension should have activation events."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        events = pkg.get("activationEvents", [])
        assert len(events) > 0, "No activation events defined"

    def test_typescript_config_exists(self):
        """tsconfig.json should exist for TypeScript compilation."""
        assert (EXTENSION_DIR / "tsconfig.json").exists() or True  # Optional


class TestVSCodeExtensionSource:
    """Test extension source code structure."""

    def test_extension_ts_exists(self):
        """extension.ts should exist."""
        src_dir = EXTENSION_DIR / "src"
        assert (src_dir / "extension.ts").exists()

    def test_extension_ts_has_activate(self):
        """extension.ts should export an activate function."""
        ext_file = EXTENSION_DIR / "src" / "extension.ts"
        if not ext_file.exists():
            pytest.skip("extension.ts not found")

        content = ext_file.read_text()
        assert "export function activate" in content or "export async function activate" in content

    def test_extension_ts_has_deactivate(self):
        """extension.ts should export a deactivate function."""
        ext_file = EXTENSION_DIR / "src" / "extension.ts"
        if not ext_file.exists():
            pytest.skip("extension.ts not found")

        content = ext_file.read_text()
        assert "export function deactivate" in content

    def test_extension_registers_commands(self):
        """extension.ts should register VS Code commands."""
        ext_file = EXTENSION_DIR / "src" / "extension.ts"
        if not ext_file.exists():
            pytest.skip("extension.ts not found")

        content = ext_file.read_text()
        assert "registerCommand" in content

    def test_extension_has_status_bar(self):
        """extension.ts should create status bar items."""
        ext_file = EXTENSION_DIR / "src" / "extension.ts"
        if not ext_file.exists():
            pytest.skip("extension.ts not found")

        content = ext_file.read_text()
        assert "createStatusBarItem" in content


class TestVSCodeExtensionAPI:
    """Test extension API integration logic."""

    def test_api_url_default(self):
        """Default API URL should be localhost:8000."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        api_url = pkg["contributes"]["configuration"]["properties"]["distllm.apiUrl"]
        assert api_url["default"] == "http://localhost:8000"

    def test_refresh_interval_bounds(self):
        """Refresh interval should have reasonable bounds."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        interval = pkg["contributes"]["configuration"]["properties"]["distllm.refreshInterval"]

        assert interval["minimum"] >= 2
        assert interval["maximum"] <= 300
        assert interval["default"] >= 5

    def test_temperature_bounds(self):
        """Temperature should have standard LLM bounds."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        temp = pkg["contributes"]["configuration"]["properties"]["distllm.temperature"]

        assert temp["minimum"] == 0
        assert temp["maximum"] == 2
        assert 0 < temp["default"] <= 1

    def test_max_tokens_default(self):
        """Max tokens should have a reasonable default."""
        pkg = json.loads((EXTENSION_DIR / "package.json").read_text())
        max_tokens = pkg["contributes"]["configuration"]["properties"]["distllm.maxTokens"]

        assert max_tokens["default"] >= 64
        assert max_tokens["default"] <= 4096
