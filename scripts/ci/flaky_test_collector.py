#!/usr/bin/env python3
"""Pytest plugin that records flaky / skipped / xfailed tests as JSON.

Why a plugin and not just junit XML?
-------------------------------------
``pytest-rerunfailures`` retries a failed test but only surfaces the rerun
in the *terminal summary* (``.R.sx``).  The standard ``--junitxml`` report
flattens a re-run test to a single ``<testcase>`` with no marker that it
needed a retry, so a flaky test that eventually passed is indistinguishable
from a genuinely stable test in XML.

This plugin hooks ``pytest_runtest_logreport`` and classifies each test by
its final outcome:

* **flaky**       -- at least one attempt had ``outcome == "rerun"`` AND the
                    test ultimately PASSED.  (A test that failed all retries
                    is a real failure and is *not* counted as flaky here.)
* **skipped**     -- ``outcome == "skipped"`` and NOT an xfail.
* **xfailed**     -- expected-to-fail (``report.wasxfail`` is set).

The report is written at ``pytest_sessionfinish`` so it captures the whole
suite, even if ``--junitxml`` is also requested.  Load it with
``-p flaky_test_collector`` (after putting ``scripts/ci`` on ``PYTHONPATH``)
or wire it into a runner that passes ``-p no:cacheprovider -p flaky_test_collector``.

Usage::

    PYTHONPATH=scripts/ci python -m pytest tests \\
        -p no:cacheprovider -p flaky_test_collector \\
        --flaky-report flaky_report.json --junitxml=junit.xml

The emitted JSON schema::

    {
      "flaky_tests":  ["tests/foo.py::test_bar", ...],
      "flaky_count":  1,
      "skipped_tests": ["tests/foo.py::test_skip", ...],
      "skipped_count": 1,
      "xfailed_tests": ["tests/foo.py::test_xfail", ...],
      "xfailed_count": 1
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def pytest_addoption(parser: "pytest.Parser") -> None:
    parser.addoption(
        "--flaky-report",
        action="store",
        default="flaky_report.json",
        help="Path to write the flaky/skip/xfail JSON report (default: flaky_report.json).",
    )


class FlakyCollector:
    """Collect flaky / skipped / xfailed nodeids across the session."""

    def pytest_runtest_logreport(self, report: "pytest.TestReport") -> None:
        # Re-run attempts carry outcome == "rerun" on the intermediate (failed)
        # attempt.  We record that a rerun happened for this nodeid.
        if report.when == "call" and getattr(report, "outcome", None) == "rerun":
            self._rerun_nodeids.add(report.nodeid)
            return
        if report.when != "call":
            return
        nodeid = report.nodeid
        if getattr(report, "skipped", False):
            if getattr(report, "wasxfail", None) is not None:
                self._xfailed_nodeids.add(nodeid)
            else:
                self._skipped_nodeids.add(nodeid)
        elif getattr(report, "passed", False):
            # A test that passed on some attempt after a rerun is flaky.
            if nodeid in self._rerun_nodeids:
                self._flaky_nodeids.add(nodeid)

    def pytest_sessionfinish(self, session: "pytest.Session") -> None:
        data = {
            "flaky_tests": sorted(self._flaky_nodeids),
            "flaky_count": len(self._flaky_nodeids),
            "skipped_tests": sorted(self._skipped_nodeids),
            "skipped_count": len(self._skipped_nodeids),
            "xfailed_tests": sorted(self._xfailed_nodeids),
            "xfailed_count": len(self._xfailed_nodeids),
        }
        out = Path(session.config.getoption("--flaky-report"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2) + "\n")

    def __init__(self) -> None:
        self._rerun_nodeids: set[str] = set()
        self._flaky_nodeids: set[str] = set()
        self._skipped_nodeids: set[str] = set()
        self._xfailed_nodeids: set[str] = set()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: "pytest.Config") -> None:
    # Register unless a FlakyCollector instance is already registered
    # (re-import / double-configure safe).
    for plugin in config.pluginmanager.get_plugins():
        if isinstance(plugin, FlakyCollector):
            return
    config.pluginmanager.register(FlakyCollector())
