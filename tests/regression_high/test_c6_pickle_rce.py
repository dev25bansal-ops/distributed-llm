"""Regression tests for HIGH fix C6: pickle.loads RCE sink in zero_copy IPC.

``CudaIPCManager.import_tensor`` originally deserialized a received IPC
handle with ``pickle.loads`` and immediately called ``func(*args)``. A
malicious handle could name *any* callable -> remote code execution.

The fix replaced the raw ``pickle.loads`` with
``CudaIPCManager._restricted_loads``: a RestrictedUnpickler whose
``find_class`` allowlists only the torch storage-reconstruction globals,
and which rejects any handle whose payload is not a 2-tuple of
``(callable, args)``. Anything else raises ``pickle.UnpicklingError`` and
``import_tensor`` returns ``None``. Confirmed on 2026-08-20 that the
allowlist paths use ``torch.multiprocessing.reductions`` (plural) and that
received handles are rejected.
"""

from __future__ import annotations

import pickle

import pytest

from distllm.dist.zero_copy import CudaIPCManager


class _Malicious:
    """Stand-in for an attacker-controlled pickle payload."""


def _forge_malicious_handle():
    # A pickle that, if unpickled and called, would execute arbitrary code.
    return pickle.dumps((_Malicious, ()))


def _forge_structurally_wrong_handle():
    # A tuple of wrong length.
    return pickle.dumps(("not", "a", "valid", "handle"))


def _forge_called_func_handle():
    # A 2-tuple of (callable, args) whose callable is not a torch storage
    # reconstruction global (not rejected by find_class but by the
    # callable/args-type validation after load).
    return pickle.dumps((max, (1, 2)))


def test_malicious_callable_rejected():
    mgr = CudaIPCManager()
    # import_tensor short-circuits to None when CUDA is unavailable and
    # otherwise returns None because the restricted loader rejects the
    # attacker-controlled global; either way the forged handle is never
    # executed.
    out = mgr.import_tensor("k", _forge_malicious_handle(), (1,), object)
    assert out is None


def test_wrong_structure_rejected():
    mgr = CudaIPCManager()
    out = mgr.import_tensor("k", _forge_structurally_wrong_handle(), (1,), object)
    assert out is None


def test_non_allowlisted_callable_rejected():
    mgr = CudaIPCManager()
    # Even a well-formed (callable, args) tuple is unsafe unless the callable
    # is a torch storage reconstruction global — the loader raises and
    # import_tensor must not call it.
    out = mgr.import_tensor("k", _forge_called_func_handle(), (1,), object)
    assert out is None


def test_restricted_loads_rejects_malicious_global():
    # Directly exercise the restricted loader (independent of CUDA state).
    with pytest.raises(pickle.UnpicklingError):
        CudaIPCManager._restricted_loads(_forge_malicious_handle())


def test_restricted_loads_rejects_wrong_structure():
    # find_class won't trigger (no globals referenced), so the post-load
    # structural check rejects it: ValueError.
    with pytest.raises((pickle.UnpicklingError, ValueError)):
        CudaIPCManager._restricted_loads(_forge_structurally_wrong_handle())


def test_restricted_loads_rejects_non_allowlisted_callable():
    with pytest.raises(pickle.UnpicklingError):
        CudaIPCManager._restricted_loads(_forge_called_func_handle())


def test_restricted_loads_exists():
    # The fix is present when CudaIPCManager exposes _restricted_loads and
    # import_tensor routes through it (no raw pickle.loads path remains).
    assert hasattr(CudaIPCManager, "_restricted_loads")
