"""Regression tests for M5: FileSecretBackend strict perms re-applied every write.

Previously the FileSecretBackend only chmod'd the secrets file once at
creation time (``_ensure_file``). A file that already existed with loose
permissions (e.g. 0o644) would never be tightened on a subsequent write, and
writes used a plain ``open(..., "w")`` (non-atomic, no perm re-application).

This test must FAIL against the buggy code and PASS after the fix:

- A write must re-apply restrictive file perms (0o600) every time, not just
  at creation.  We verify this OS-independently by intercepting ``os.chmod``
  and asserting it is called with the restrictive mode on *every* write.
- On POSIX the effective file mode must be exactly 0o600 and the directory
  exactly 0o700 (group/other bits must be clear).
- If the file is manually loosened to 0o644 and then written again (POSIX),
  the permissions must be re-tightened to 0o600.
- Writes must be atomic (temp file + rename) and leave no leftover temp files.
"""

from __future__ import annotations

import os
import stat

import pytest

from distllm.core.secret_manager import FileSecretBackend

_IS_POSIX = os.name != "nt"


def _mode(path: str) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_write_reapplies_secure_chmod_every_time(monkeypatch, tmp_path):
    """The restrictive chmod must be invoked on every write (not just creation).

    On the buggy code, ``put`` -> ``_write`` calls plain ``open`` and never
    chmod's, so after the file already exists this assertion would fail.
    """
    calls = []

    def fake_chmod(path, mode):
        calls.append((path, mode))

    monkeypatch.setattr(os, "chmod", fake_chmod)

    path = tmp_path / "secrets.json"
    backend = FileSecretBackend(str(path))
    calls.clear()  # ignore creation-time chmods

    backend.put("api_key", "first-value")

    file_calls = [(p, m) for (p, m) in calls if os.path.basename(p) == "secrets.json"]
    assert file_calls, "os.chmod was never called on the secrets file during put()"
    # The final chmod on the file must be the restrictive 0o600 owner bits.
    _, final_mode = file_calls[-1]
    assert final_mode & stat.S_IRUSR and final_mode & stat.S_IWUSR
    assert not (final_mode & (stat.S_IRGRP | stat.S_IWGRP |
                              stat.S_IROTH | stat.S_IWOTH))


def test_perm_reapplied_on_each_subsequent_write(monkeypatch, tmp_path):
    """Re-application must happen on *each* write, not just the first."""
    calls = []
    monkeypatch.setattr(os, "chmod", lambda p, m: calls.append((p, m)))

    path = tmp_path / "secrets.json"
    backend = FileSecretBackend(str(path))
    calls.clear()

    backend.put("k1", "v1")
    backend.put("k2", "v2")
    backend.put("k3", "v3")

    file_chmod_count = sum(1 for (p, _) in calls
                           if os.path.basename(p) == "secrets.json")
    # One re-tighten per put (creation-time chmod excluded via clear()).
    assert file_chmod_count >= 3, f"expected chmod on each write, got {file_chmod_count}"


@pytest.mark.skipif(not _IS_POSIX, reason="exact 0o600/0o700 only enforced on POSIX")
def test_write_applies_exact_strict_perms(tmp_path):
    path = tmp_path / "secrets.json"
    backend = FileSecretBackend(str(path))
    backend.put("api_key", "super-secret-value")

    mode = _mode(str(path))
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


@pytest.mark.skipif(not _IS_POSIX, reason="exact 0o700 only enforced on POSIX")
def test_dir_is_tightened_to_700(tmp_path):
    path = tmp_path / "nested" / "secrets.json"
    backend = FileSecretBackend(str(path))
    backend.put("k", "v")

    dirmode = _mode(str(path.parent))
    assert dirmode == 0o700, f"expected 0o700, got {oct(dirmode)}"


@pytest.mark.skipif(not _IS_POSIX, reason="loosen/re-tighten only meaningful on POSIX")
def test_loosened_perms_re_tightened_on_write(tmp_path):
    path = tmp_path / "secrets.json"
    backend = FileSecretBackend(str(path))
    backend.put("api_key", "first-value")

    # Simulate a permissions regression: someone loosened the file to 0o644.
    os.chmod(str(path), 0o644)
    assert _mode(str(path)) == 0o644

    # A subsequent write must re-tighten permissions to 0o600.
    backend.put("api_key", "second-value")
    mode = _mode(str(path))
    assert mode == 0o600, f"perms not re-tightened: {oct(mode)}"

    # Stored value must round-trip via get().
    assert backend.get("api_key") == "second-value"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "secrets.json"
    backend = FileSecretBackend(str(path))
    backend.put("k", "v")
    leftover = [p.name for p in tmp_path.iterdir()
                if p.name.startswith(".secrets_") and p.name.endswith(".tmp")]
    assert not leftover, f"leftover temp files: {leftover}"
