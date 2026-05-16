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

    # Normalize using pathlib (handles ../, ..\, double slashes, etc.)
    resolved = Path(raw_path).resolve()

    # Check for traversal after resolution
    if ".." in Path(raw_path).parts:
        raise ValueError("Adapter path cannot contain parent directory traversal (..)")

    # For absolute paths, verify they are within allowed base directories
    if resolved.is_absolute():
        for base in ALLOWED_ADAPTER_BASES:
            resolved_base = base.resolve()
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
