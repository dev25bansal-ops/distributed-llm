"""Input validation utilities for API endpoints."""

import os
from pathlib import Path


# Allowed base directories for adapter storage (can be extended via config)
ALLOWED_ADAPTER_BASES = [
    Path("/app/adapters"),
    Path("./adapters"),
]


def validate_adapter_path(raw_path: str) -> Path:
    """Validate and resolve an adapter path, preventing path traversal.

    Resolves symlinks via os.path.realpath() before validation to prevent
    symlink-based directory traversal attacks.  Re-validates the resolved
    path against allowed base directories.

    Args:
        raw_path: User-supplied path string.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If the path is invalid, absolute outside allowed dirs,
                    or contains traversal sequences.
    """
    if not raw_path or not raw_path.strip():
        raise ValueError("Adapter path cannot be empty")

    # Check for traversal in the raw input before any resolution
    if ".." in Path(raw_path).parts:
        raise ValueError("Adapter path cannot contain parent directory traversal (..)")

    # Resolve symlinks and relative paths to get the real filesystem path
    # This prevents symlink-based traversal where a symlink points outside allowed dirs
    resolved = Path(os.path.realpath(raw_path))

    # For absolute paths, verify the *real* resolved path is within allowed base directories
    if resolved.is_absolute():
        for base in ALLOWED_ADAPTER_BASES:
            resolved_base = Path(os.path.realpath(base))
            try:
                resolved.relative_to(resolved_base)
                return resolved
            except ValueError:
                continue
        # Not within any allowed base
        allowed = ", ".join(str(b) for b in ALLOWED_ADAPTER_BASES)
        raise ValueError(
            f"Absolute adapter path must be within one of: {allowed}"
        )

    # Relative paths are allowed (resolved against cwd)
    return resolved
