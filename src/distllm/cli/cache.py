"""CLI command: distllm cache inspect <token_hash>.

Looks up a token_hash in the local KV cache and reports its metadata.

A token_hash is the prefix-hash stored by the KVCacheManager.  Operators
use this to debug cache hit rates, evict polluted entries, or verify
that a specific prompt-prefix is being cached across the cluster.
"""

from __future__ import annotations

import os
from typing import Any


def run_cache_inspect(token_hash: str) -> dict[str, Any] | None:
    """Inspect a token_hash in the local KV cache.

    Returns a dict with the requested metadata, or ``None`` if no entry
    is found in any of the registered KV caches.

    The CLI subcommand discovers KV caches by:
      1. Asking the active ``g.coordinator`` (when the CLI is run
         in-process against the API server's coordinator).
      2. Falling back to scanning a process-wide registry of
         ``KVCacheManager`` instances (useful in CLI mode that runs
         standalone and creates its own manager).

    Args:
        token_hash: The prefix hash to look up.  Treated case-insensitively.

    Returns:
        ``None`` if no matching entry exists; otherwise a dict containing:
        ``model_id``, ``layer_index``, ``sequence_id``, ``block_count``,
        ``last_access_ts`` and a few debugging fields like
        ``created_at`` and ``prefix_hash`` for completeness.
    """
    norm = (token_hash or "").strip().lower()
    if not norm:
        raise ValueError("token_hash must be a non-empty string")

    # Strategy 1: ask the running coordinator (in-process API server).
    try:
        from distllm.api.api_state import g

        coord = getattr(g, "coordinator", None)
        if coord is not None:
            for manager_attr in (
                "_prefix_cache_manager",
                "prefix_cache_manager",
                "_kv_cache_manager",
                "kv_cache_manager",
            ):
                manager = getattr(coord, manager_attr, None)
                if manager is not None:
                    entry = _inspect_manager(manager, norm)
                    if entry is not None:
                        return entry
    except ImportError:
        pass

    # Strategy 2: walk every KVCacheManager we've seen.  The CLI process
    # builds its own manager when it instantiates a local partitioner, so
    # we look it up via a module-level registry.
    registry = _MANAGER_REGISTRY
    for manager in list(registry):
        entry = _inspect_manager(manager, norm)
        if entry is not None:
            return entry

    # No entry found anywhere.
    return None


def register_cache_manager(manager: Any) -> None:
    """Register a KVCacheManager with the CLI's in-process registry.

    Standalone CLI workflows (e.g. ``distllm cache inspect`` invoked from
    a worker startup script) call this once after constructing a manager
    so subsequent ``run_cache_inspect`` calls can find the entry.
    """
    if manager not in _MANAGER_REGISTRY:
        _MANAGER_REGISTRY.append(manager)


def _inspect_manager(manager: Any, token_hash: str) -> dict[str, Any] | None:
    """Search one KVCacheManager for *token_hash*.

    The manager exposes:
      * ``caches: dict[request_id, KVCache]`` — keyed by request id.
      * ``_metadata: dict[request_id, dict]`` — per-entry metadata
        including ``last_accessed`` (monotonic-style float seconds).
      * ``_request_prefix: dict[request_id, str]`` — request -> prefix_hash.

    The CLI matches the caller-supplied hash against the prefix map and
    reports the first match.
    """
    request_prefix = getattr(manager, "_request_prefix", {}) or {}
    metadata = getattr(manager, "_metadata", {}) or {}
    caches = getattr(manager, "caches", {}) or {}

    # Build the inverse index: prefix_hash -> request_id(s).
    for request_id, prefix_hash in request_prefix.items():
        if not prefix_hash:
            continue
        if prefix_hash.lower() == token_hash:
            meta = metadata.get(request_id, {}) or {}
            cache = caches.get(request_id)
            return {
                "model_id": _guess_model_id(manager, meta),
                "layer_index": 0,
                "sequence_id": str(request_id),
                "block_count": _count_blocks(cache),
                "last_access_ts": float(meta.get("last_accessed", 0.0)),
                "created_at": float(meta.get("created_at", 0.0)),
                "prefix_hash": prefix_hash,
                "request_id": str(request_id),
            }
    return None


def _guess_model_id(manager: Any, meta: dict[str, Any]) -> str:
    """Best-effort model id from manager state or metadata."""
    for attr in ("model_id", "model_name"):
        if hasattr(manager, attr):
            return str(getattr(manager, attr))
    return str(meta.get("model_id", "unknown"))


def _count_blocks(cache: Any) -> int:
    """Count blocks in a KVCache.  Returns 0 if the cache is missing."""
    if cache is None:
        return 0
    layers = getattr(cache, "cache", None)
    if layers is None:
        return 0
    if isinstance(layers, dict):
        return len(layers)
    return len(layers)


# Module-level registry of KVCacheManagers that opted-in via
# register_cache_manager().  Process-local; resets on restart.
_MANAGER_REGISTRY: list[Any] = []


def print_cache_inspect(
    token_hash: str,
    entry: dict[str, Any] | None,
    console,
) -> None:
    """Pretty-print the result of ``run_cache_inspect`` on *console*."""
    from rich.table import Table

    console.print(f"[bold]Cache entry for token_hash=[/bold] [cyan]{token_hash}[/cyan]")
    if entry is None:
        console.print("  [yellow](not found in any local KV cache)[/yellow]")
        return

    table = Table(title="KV Cache Entry")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("model_id", str(entry.get("model_id", "?")))
    table.add_row("layer_index", str(entry.get("layer_index", 0)))
    table.add_row("sequence_id", str(entry.get("sequence_id", "?")))
    table.add_row("block_count", str(entry.get("block_count", 0)))
    last_access = entry.get("last_access_ts", 0.0)
    if last_access > 0:
        import datetime as _dt

        ts = _dt.datetime.fromtimestamp(last_access).isoformat(timespec="seconds")
        table.add_row("last_access_ts", ts)
    else:
        table.add_row("last_access_ts", "n/a")
    console.print(table)
