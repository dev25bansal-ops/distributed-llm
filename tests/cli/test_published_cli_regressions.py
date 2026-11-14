"""Regressions found by running the published package on a real GPU host.

Every test here corresponds to something a user saw in their terminal:

* `distllm --version` exited with "No such option: --version" (there was no
  root callback at all).
* The MAC address column rendered `e6💿9f:23:1d:69` — Rich substituted an emoji
  for `:cd:` and the address became wrong, and it also crashed any cp1252
  console with UnicodeEncodeError.
* A bare `distllm model` said "Missing command." instead of listing its
  subcommands.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.text import Text
from typer.testing import CliRunner

from distllm.cli.main import app


class TestVersionFlag:
    def test_long_flag_prints_a_version(self):
        result = CliRunner().invoke(app, ["--version"])
        assert result.exit_code == 0, result.output
        assert "distllm" in result.output
        assert any(ch.isdigit() for ch in result.output)

    def test_short_flag_works(self):
        result = CliRunner().invoke(app, ["-V"])
        assert result.exit_code == 0, result.output

    def test_help_still_lists_groups(self):
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        for group in ("model", "cluster", "benchmark", "pool"):
            assert group in result.output


class TestMacAddressRendering:
    """A MAC must survive Rich's console markup pass unchanged."""

    MAC = "e6:cd:9f:23:1d:69"

    def test_text_object_does_not_mangle_mac(self):
        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=120).print(Text(self.MAC))
        assert buf.getvalue().strip() == self.MAC

    def test_plain_string_would_be_mangled(self):
        """Documents the bug this guards against."""
        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=120, emoji=True).print(self.MAC)
        assert buf.getvalue().strip() != self.MAC, (
            "Rich no longer substitutes emoji here; the Text() workaround in "
            "devices.py may be removable"
        )

    def test_devices_mac_is_a_text_object(self):
        """The table cell must be a Text, not a bare str."""
        from distllm.cli import devices

        source = devices.__file__
        text = open(source, encoding="utf-8").read()
        assert "Text(mac)" in text, "devices.py must render the MAC via Text()"


class TestBareGroupsShowHelp:
    @pytest.mark.parametrize(
        "group", ["model", "cluster", "benchmark", "config", "security", "system"]
    )
    def test_group_lists_subcommands_when_invoked_alone(self, group):
        result = CliRunner().invoke(app, [group])
        # no_args_is_help makes Typer print help; exit code is 2 by convention.
        assert "Missing command" not in result.output
        assert "Usage" in result.output or "Commands" in result.output