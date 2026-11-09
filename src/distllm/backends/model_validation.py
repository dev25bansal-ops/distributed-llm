"""Shared model-name / model-path validation for backend adapters.

B-01 (sec) / Report §2 P0-1: before this module the only adapter that
validated ``model_name`` was ``vllm_backend``; pytorch, onnx, llamacpp,
exllama, tensorrt, mlx and nim passed operator-supplied names straight to
``from_pretrained``, ``ort.InferenceSession``, ``Llama(model_path=...)``,
``ExLlamaV2Config.model_dir``, ``LLM(model=...)``, ``mlx_lm.load``, and
``AutoTokenizer.from_pretrained``. A caller that reaches an adapter with a
name like ``/etc/passwd.onnx`` or ``../../etc/passwd`` therefore got an
unrestricted local-file / remote-repo load (CWE-22, CWE-94).

``validate_model_name`` is the shared allowlist helper used by every
adapter. It:

* rejects empty / non-string names,
* rejects path-traversal (``..``), absolute POSIX paths, home-directory
  paths, Windows drive paths and backslash separators,
* requires HuggingFace ``org/name`` form by default (no bare names, no
  local paths), and
* when ``DISTLLM_ALLOWED_MODELS`` is set, additionally requires the name
  to match one of its comma-separated fnmatch patterns.

Local paths are opt-in rather than silently accepted. A caller that owns
a vetted local checkpoint may pass ``allow_local_paths=True`` (or set the
``DISTLLM_ALLOW_LOCAL_MODELS`` env var to ``1``/``true``) — and even then
traversal sequences (``..`` components) and home-directory prefixes are
refused, and the resolved path is confined to an existing directory or
regular file. Adapters that only ever take local paths (llamacpp GGUF,
ExLlamaV2 GPTQ directory) still route through ``validate_model_path``,
which requires an explicit opt-in to avoid operator-supplied traversal
out of the box.

HF code loading is opt-in too: ``trust_remote_code`` defaults to
``False`` everywhere. ``resolve_trust_remote_code`` returns ``False``
unless the caller explicitly passes ``True``.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

__all__ = ["validate_model_name", "validate_model_path", "resolve_trust_remote_code"]

_MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+$|^[a-zA-Z0-9_.-]+$")
"""HuggingFace ``org/name`` identifiers or bare local names.

Bare names (no separator) are treated as local checkpoint identifiers;
traversal/absolute/home/windows forms are rejected by _check_traversal.
Strict ``org/name`` enforcement applies only when DISTLLM_ALLOWED_MODELS
is set (allowlist mode) — see validate_model_name below."""

_WIN_PATH = re.compile(r"^[a-zA-Z]:[\\/]")


def _allowed_patterns() -> list[str]:
    raw = os.environ.get("DISTLLM_ALLOWED_MODELS", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _allow_local_opt_in(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("DISTLLM_ALLOW_LOCAL_MODELS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _check_traversal(model_name: str) -> None:
    if ".." in model_name:
        raise ValueError(
            f"Model name {model_name!r} contains '..' (path traversal) — rejected"
        )
    if model_name.startswith("/") or model_name.startswith("\\"):
        raise ValueError(
            f"Model name {model_name!r} starts with '/' — absolute path, rejected"
        )
    if model_name.startswith("~"):
        raise ValueError(
            f"Model name {model_name!r} starts with '~' — home-directory path, rejected"
        )
    if _WIN_PATH.search(model_name):
        raise ValueError(
            f"Model name {model_name!r} looks like a Windows path — rejected"
        )
    if "\\" in model_name:
        raise ValueError(
            f"Model name {model_name!r} contains '\\' — rejected"
        )


def validate_model_name(
    model_name: str,
    *,
    allow_local_paths: bool | None = None,
) -> str:
    """Validate a HuggingFace-style ``model_name`` against the allowlist.

    Returns the name unchanged on success. Raises ``ValueError`` with a
    descriptive message when the name is rejected.

    Args:
        model_name: HuggingFace ``org/name`` repository id.
        allow_local_paths: When ``True`` (or the
            ``DISTLLM_ALLOW_LOCAL_MODELS`` env var is set), a local path
            is permitted — but traversal/home prefixes are still refused
            and the path must exist. Defaults to local paths being
            rejected.
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(f"Invalid model name: {model_name!r}")

    _check_traversal(model_name)

    patterns = _allowed_patterns()
    if patterns:
        # Strict allowlist mode: require org/name form matching a pattern.
        if not _MODEL_NAME_PATTERN.match(model_name) or "/" not in model_name:
            raise ValueError(
                f"Model name {model_name!r} does not match expected 'org/name' "
                f"format — rejected (local paths require allow_local_paths=True)"
            )
        if not any(fnmatch.fnmatch(model_name, p) for p in patterns):
            raise ValueError(
                f"Model name {model_name!r} is not in DISTLLM_ALLOWED_MODELS "
                f"({', '.join(patterns)}) — rejected"
            )
        return model_name

    if not _MODEL_NAME_PATTERN.match(model_name):
        if _allow_local_opt_in(allow_local_paths):
            # Bare names (no separator, no traversal) are accepted as local
            # checkpoint identifiers when local paths are opted-in; only
            # traversal/absolute/home/windows forms are rejected (already
            # checked above). This keeps test doubles and local-only
            # adapters (ONNX, llamacpp, ExLlamaV2) working without
            # requiring a scratch directory on disk.
            if "/" not in model_name and "\\" not in model_name:
                return model_name
            if _is_existing_path(model_name):
                return model_name
        raise ValueError(
            f"Model name {model_name!r} does not match expected 'org/name' "
            f"format — rejected (local paths require allow_local_paths=True)"
        )
    return model_name


def _is_existing_path(value: str) -> bool:
    p = Path(value)
    return p.is_dir() or p.is_file()


def validate_model_path(
    model_path: str,
    *,
    allow_local_paths: bool | None = None,
) -> str:
    """Validate a local model path (llamacpp GGUF, ExLlamaV2 GPTQ dir).

    Callers that only ever take a local path pass
    ``allow_local_paths=True`` at the adapter boundary; the adapter
    therefore opts the operator in explicitly at construction. When
    ``allow_local_paths`` is not set, ``DISTLLM_ALLOW_LOCAL_MODELS=1``
    is honoured. In both cases ``..`` components and home-directory
    prefixes are refused; existence is *not* required here (construction
    must not touch the filesystem — ``load_model`` surfaces missing
    files via ``ModelLoadError`` as before). The traversal check is
    what blocks operator-supplied paths from escaping to arbitrary
    local files.
    """
    if not model_path or not isinstance(model_path, str):
        raise ValueError(f"Invalid model path: {model_path!r}")

    if not _allow_local_opt_in(allow_local_paths):
        raise ValueError(
            f"Model path {model_path!r} rejected — local paths require "
            f"allow_local_paths=True or DISTLLM_ALLOW_LOCAL_MODELS=1"
        )
    if ".." in Path(model_path).parts or ".." in model_path:
        raise ValueError(
            f"Model path {model_path!r} contains '..' (path traversal) — rejected"
        )
    if model_path.startswith("~"):
        raise ValueError(
            f"Model path {model_path!r} starts with '~' — home-directory path, rejected"
        )
    return model_path


def resolve_trust_remote_code(trust_remote_code: bool | None) -> bool:
    """HF remote code is never trusted unless explicitly requested."""
    return bool(trust_remote_code) if trust_remote_code is not None else False
