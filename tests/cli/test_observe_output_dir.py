"""Regression: ``distllm system observe`` must write to a runtime location.

Defect: ``_find_project_root`` walked up from ``src/distllm/cli/observe.py``
looking for the first directory containing ``deploy/``.  ``src/distllm/deploy/``
is itself a package (``webhook_server.py`` lives there), so the search stopped
inside the installed source tree and ``system observe`` wrote Grafana
provisioning into ``src/distllm/deploy/grafana/``.  That is a source directory:
it pollutes the package, can land in an installed wheel's read-only path, and
is wiped by a reinstall.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from distllm.cli import observe


def test_generated_files_do_not_land_in_source_tree(tmp_path, monkeypatch):
    """The output directory must never be inside the importable package."""
    import distllm

    package_dir = Path(distllm.__file__).resolve().parent

    target = observe._observability_output_dir()
    resolved = target.resolve()

    assert package_dir not in resolved.parents, (
        f"observability output {resolved} is inside the installed package "
        f"({package_dir})"
    )
    assert "src" not in resolved.parts or "distllm" not in resolved.parts, (
        f"observability output {resolved} looks like a source path"
    )


def test_output_dir_is_under_user_state_directory(monkeypatch, tmp_path):
    """The default location is ``~/.distllm/observability``."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    # Path.home() on Windows honours USERPROFILE; make it deterministic.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    target = observe._observability_output_dir()
    assert target == fake_home / ".distllm" / "observability", target


def test_output_dir_honours_explicit_override(monkeypatch, tmp_path):
    """An explicit ``DISTLLM_OBSERVE_DIR`` wins over the default."""
    override = tmp_path / "custom-observability"
    monkeypatch.setenv("DISTLLM_OBSERVE_DIR", str(override))

    target = observe._observability_output_dir()
    assert target == override, target


def test_run_observe_writes_only_under_output_dir(tmp_path, monkeypatch):
    """End-to-end: the files written are exactly the provisioning set, under
    the output dir, and nothing is created in the package tree."""
    import distllm

    package_dir = Path(distllm.__file__).resolve().parent
    out = tmp_path / "obs"
    monkeypatch.setenv("DISTLLM_OBSERVE_DIR", str(out))

    # Stop run_observe from starting a real HTTP server or blocking forever.
    monkeypatch.setattr(observe, "_start_metrics_server", lambda *a, **k: None)

    import builtins

    real_sleep = observe.time.sleep

    def _stop_after_first_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(observe.time, "sleep", _stop_after_first_sleep)

    observe.run_observe(metrics_only=False, console=_NullConsole())

    grafana = out / "grafana"
    assert (grafana / "distllm-dashboard.json").is_file()
    assert (grafana / "datasources.yaml").is_file()
    assert (grafana / "provisioning" / "dashboards.yaml").is_file()

    # Nothing new appeared under the package directory.
    created = [p for p in package_dir.rglob("*") if p.is_dir() and "grafana" in p.name]
    assert created == [], created


class _NullConsole:
    """Console stand-in that discards all output."""

    def print(self, *args, **kwargs):
        pass