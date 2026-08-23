"""Regression test: Ensure no pickle.load calls exist in production code.

pickle.load is a remote-code-execution (RCE) risk when used with untrusted
data. This test enforces that no module under src/ calls pickle.load or
import pickle, preventing accidental reintroduction of this vulnerability.

See TASK-001 in TASKS.md for context.
"""

from __future__ import annotations

import pathlib

import pytest


PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def _is_generated_protobuf(path: pathlib.Path) -> bool:
    """Skip protobuf-generated files (_pb2.py, _pb2_grpc.py, etc.)."""
    name = path.name
    return name.endswith("_pb2.py") or "_pb2_" in name or name.endswith("_pb2_grpc.py")


def _is_restricted_pickle_zone(path: pathlib.Path) -> bool:
    """Files that use pickle in a SAFE, restricted manner:

    * ``dist/dist/zero_copy.py`` — RestrictedUnpickler that allows only torch's
      exact storage-reduction globals for TRUSTED CUDA IPC (no RCE surface).
    * ``core/advanced_scheduling/disaggregated.py`` — pickles KV tensors with
      ``pickle.dumps`` only (serialize, never ``loads``) for trusted same-cluster
      KV transfer over gRPC. No deserialization of untrusted input occurs.
    """
    return path.name in {"zero_copy.py", "disaggregated.py"}


def _rel(path: pathlib.Path) -> pathlib.Path:
    """Return a display-relative path; fall back to absolute when the file
    is outside PROJECT_ROOT (e.g. the meta-test's temp smuggler file)."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _iter_source_files() -> list[pathlib.Path]:
    """Yield all .py files under src/, excluding generated protobuf."""
    if not SRC_DIR.is_dir():
        return []
    return sorted(
        f for f in SRC_DIR.rglob("*.py") if not _is_generated_protobuf(f)
    )


class TestNoPickleLoad:
    """Scan src/ for pickle.load calls - fail on any occurrence."""

    @pytest.mark.security
    def test_no_pickle_load_in_source(self) -> None:
        """Fail if any pickle.load, pickle.loads, or pickle.Unpickler call is found under src/."""
        if not SRC_DIR.is_dir():
            pytest.skip("src/ directory not found")

        violations: list[str] = []
        for py_file in _iter_source_files():
            if _is_restricted_pickle_zone(py_file):
                continue  # zero_copy.py uses a RestrictedUnpickler (documented)
            content = py_file.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                # Skip comments and docstrings
                if stripped.startswith("#"):
                    continue
                if (
                    "pickle.load" in stripped
                    or "pickle.loads" in stripped
                    or "pickle.Unpickler" in stripped
                ):
                    violations.append(
                        f"  {_rel(py_file)}:{i}: {stripped}"
                    )

        assert not violations, (
            f"Found {len(violations)} pickle.load/loads/Unpickler call(s) in src/:\n"
            + "\n".join(violations)
            + "\n\nUse JSON, msgpack, or safetensors instead. "
            "See TASK-001 in TASKS.md."
        )

    @pytest.mark.security
    def test_no_import_pickle_in_source(self) -> None:
        """Fail if any 'import pickle' or 'from pickle' is found under src/.

        Even importing pickle is a risk signal - it suggests serialization
        logic that should use a safe alternative.
        """
        if not SRC_DIR.is_dir():
            pytest.skip("src/ directory not found")

        violations: list[str] = []
        for py_file in _iter_source_files():
            if _is_restricted_pickle_zone(py_file):
                continue  # zero_copy.py RestrictedUnpickler (documented)
            content = py_file.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith(("import pickle", "from pickle")):
                    violations.append(
                        f"  {_rel(py_file)}:{i}: {stripped}"
                    )

        assert not violations, (
            f"Found {len(violations)} pickle import(s) in src/:\n"
            + "\n".join(violations)
            + "\n\nPickle is a security risk (RCE via deserialization). "
            "Use JSON, msgpack, or safetensors instead."
        )

    @pytest.mark.security
    def test_no_nosec_suppressing_pickle_warnings(self) -> None:
        """Ensure no # nosec or # noqa annotations hide pickle usage.

        A comment like ``pickle.load(data)  # nosec`` suppresses bandit
        warnings. This test flags any such annotation for human review.
        """
        if not SRC_DIR.is_dir():
            pytest.skip("src/ directory not found")

        suspects: list[str] = []
        for py_file in _iter_source_files():
            content = py_file.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.splitlines(), start=1):
                if "pickle" in line and ("# nosec" in line or "# noqa" in line):
                    suspects.append(
                        f"  {_rel(py_file)}:{i}: {line.strip()}"
                    )

        assert not suspects, (
            f"Found {len(suspects)} pickle line(s) with suppressed "
            f"linting in src/:\n" + "\n".join(suspects)
            + "\n\nIf justified, remove this test and document "
            "the exception in SECURITY_HARDENING.md."
        )

    @pytest.mark.security
    def test_pickle_regression_works(self, tmp_path, monkeypatch) -> None:
        """Sanity check: test SHOULD fail if pickle.load is added to src/.

        This is a meta-test that proves the test above actually catches violations.
        """
        # Create a fake src file with pickle.load
        fake_src = tmp_path / "src" / "distllm" / "_test_pickle_smuggler.py"
        fake_src.parent.mkdir(parents=True)
        fake_src.write_text("import pickle\npickle.load(open('x', 'rb'))\n")

        # Monkeypatch SRC_DIR to point at our fake
        monkeypatch.setattr(
            "tests.security.test_no_pickle.SRC_DIR", tmp_path / "src"
        )

        # Run test_no_pickle_load_in_source - it should FAIL (assertion error)
        with pytest.raises(AssertionError, match="pickle.load"):
            TestNoPickleLoad().test_no_pickle_load_in_source()

        # Run test_no_import_pickle_in_source - it should also FAIL
        with pytest.raises(AssertionError, match="pickle import"):
            TestNoPickleLoad().test_no_import_pickle_in_source()
