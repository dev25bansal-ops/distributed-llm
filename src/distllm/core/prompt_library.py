r"""Version-controlled prompt template library with SQLite persistence,
CLI tooling, REST API, and A/B testing integration.

Prompt Version Model
--------------------
Each prompt is identified by ``(name, version)``.  Creating a new version
auto-increments the integer version for that name.  A SHA-256 hash of the
template + sorted variables provides integrity verification.

PromptRepository
----------------
SQLite-backed CRUD with thread-safe access.  Supports ``create``, ``get``,
``list``, ``diff``, and ``delete`` operations.

AB Test Integration
-------------------
The :func:`ab_test` function wraps the existing :class:`ABTestCoordinator`
to transparently select between two prompt versions and record metrics.

CLI Usage
---------
::

    # List prompts
    python -m distllm.cli.main prompt-library list
    # Get specific version
    python -m distllm.cli.main prompt-library get --name my_prompt --version 2
    # Create new version
    python -m distllm.cli.main prompt-library create --name my_prompt --tags general code
    # Diff two versions
    python -m distllm.cli.main prompt-library diff --name my_prompt --v1 1 --v2 2

API Routes
----------
- ``GET  /api/v1/prompts`` — list prompt versions
- ``POST /api/v1/prompts`` — create a new prompt version
- ``GET  /api/v1/prompts/{name}`` — get a specific prompt version
- ``GET  /api/v1/prompts/{name}/diff?v1=1&v2=2`` — compare two versions
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from distllm.core.ab_test_coordinator import ABTestCoordinator

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = os.environ.get("DISTLLM_PROMPT_LIBRARY_DB", "prompt_library.db")

# ---------------------------------------------------------------------------
# PromptVersion
# ---------------------------------------------------------------------------


@dataclass
class PromptVersion:
    """A single versioned snapshot of a prompt template.

    Attributes:
        id: Auto-increment row ID from the backing store.
        name: Logical prompt name (multiple versions share one name).
        template: The prompt template text.
        version: Monotonically-increasing version number for this name.
        created_at: Unix timestamp of creation.
        variables: Ordered list of template variable names.
        tags: Free-form tags for discovery and filtering.
        hash: SHA-256 fingerprint of the template + sorted variables.
    """

    id: int = 0
    name: str = ""
    template: str = ""
    version: int = 1
    created_at: float = 0.0
    variables: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    hash: str = ""


# ---------------------------------------------------------------------------
# PromptRepository
# ---------------------------------------------------------------------------


class PromptRepository:
    """Version-controlled prompt template library backed by SQLite.

    Thread-safe for concurrent access from the coordinator, API handlers,
    and CLI commands.
    """

    def __init__(self, db_path: str = "") -> None:
        self._db_path = Path(db_path or DEFAULT_DB_PATH)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._initialize()

    # ── Connection management ────────────────────────────────────────────

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _initialize(self) -> None:
        conn = self._connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL,
                template        TEXT    NOT NULL,
                version         INTEGER NOT NULL,
                variables_json  TEXT    NOT NULL DEFAULT '[]',
                tags_json       TEXT    NOT NULL DEFAULT '[]',
                hash            TEXT    NOT NULL DEFAULT '',
                created_at      REAL    NOT NULL
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_prompts_name_version
            ON prompts(name, version)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompts_name
            ON prompts(name)
        """)
        conn.commit()
        logger.debug("Prompt library initialised at {}", self._db_path)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ── Hashing ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(template: str, variables: list[str]) -> str:
        """SHA-256 fingerprint of template + sorted variables."""
        raw = json.dumps(
            {"template": template, "variables": sorted(variables)},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ── CRUD operations ──────────────────────────────────────────────────

    def create(
        self,
        name: str,
        template: str,
        variables: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> PromptVersion:
        """Create a new version of a prompt.

        The version number is auto-incremented from the highest existing
        version for this prompt name.

        Returns:
            The newly created PromptVersion.
        """
        variables = variables or []
        tags = tags or []

        with self._lock:
            conn = self._connection()

            cursor = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM prompts WHERE name = ?",
                (name,),
            )
            next_version = (cursor.fetchone()[0] or 0) + 1

            now = time.time()
            hash_val = self._compute_hash(template, variables)

            cursor = conn.execute(
                """INSERT INTO prompts
                   (name, template, version, variables_json, tags_json, hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    template,
                    next_version,
                    json.dumps(variables),
                    json.dumps(tags),
                    hash_val,
                    now,
                ),
            )
            conn.commit()

            prompt_id = cursor.lastrowid
            logger.info("Created prompt '{}' version {} (id={})", name, next_version, prompt_id)

            return PromptVersion(
                id=prompt_id,
                name=name,
                template=template,
                version=next_version,
                created_at=now,
                variables=variables,
                tags=tags,
                hash=hash_val,
            )

    def get(self, name: str, version: int | None = None) -> PromptVersion | None:
        """Retrieve a prompt by name and optional version.

        Args:
            name: The prompt name.
            version: Specific version number.  *None* returns the latest.

        Returns:
            PromptVersion if found, otherwise *None*.
        """
        with self._lock:
            conn = self._connection()

            if version is not None:
                cursor = conn.execute(
                    "SELECT * FROM prompts WHERE name = ? AND version = ?",
                    (name, version),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM prompts WHERE name = ? ORDER BY version DESC LIMIT 1",
                    (name,),
                )

            row = cursor.fetchone()
            if row is None:
                return None

            return self._row_to_version(row)

    def list(
        self,
        name: str | None = None,
        tag: str | None = None,
    ) -> list[PromptVersion]:
        """List prompt versions with optional filtering.

        When *name* is provided all versions of that prompt are returned.
        When *tag* is provided results are filtered by tag
        (case-insensitive substring match).  Both filters may be combined.

        Returns:
            List of matching PromptVersion instances.
        """
        with self._lock:
            conn = self._connection()
            params: list[Any] = []
            clauses: list[str] = []

            if name is not None:
                clauses.append("name = ?")
                params.append(name)

            if tag is not None:
                clauses.append("tags_json LIKE ?")
                params.append(f"%{tag}%")

            where = ""
            if clauses:
                where = "WHERE " + " AND ".join(clauses)

            cursor = conn.execute(
                f"SELECT * FROM prompts {where} ORDER BY name, version DESC",
                params,
            )

            return [self._row_to_version(row) for row in cursor.fetchall()]

    def diff(self, name: str, v1: int, v2: int) -> dict[str, Any]:
        """Compare two versions of a prompt.

        Args:
            name: Prompt name.
            v1: First version number.
            v2: Second version number.

        Returns:
            Dictionary describing changes between the two versions.

        Raises:
            ValueError: If either version does not exist.
        """
        p1 = self.get(name, version=v1)
        p2 = self.get(name, version=v2)

        if p1 is None:
            raise ValueError(f"Prompt '{name}' version {v1} not found")
        if p2 is None:
            raise ValueError(f"Prompt '{name}' version {v2} not found")

        changes: dict[str, Any] = {
            "version_a": v1,
            "version_b": v2,
            "created_at_a": p1.created_at,
            "created_at_b": p2.created_at,
        }

        if p1.name != p2.name:
            changes["name"] = {"old": p1.name, "new": p2.name}

        if p1.template != p2.template:
            changes["template"] = {
                "changed": True,
                "old_length": len(p1.template),
                "new_length": len(p2.template),
            }

        if p1.variables != p2.variables:
            changes["variables"] = {"old": p1.variables, "new": p2.variables}

        if p1.tags != p2.tags:
            changes["tags"] = {"old": p1.tags, "new": p2.tags}

        if p1.hash != p2.hash:
            changes["hash"] = {"old": p1.hash, "new": p2.hash}

        return changes

    def delete(self, name: str) -> bool:
        """Delete all versions of a prompt by name.

        Returns:
            True if at least one row was deleted, False if the name
            did not exist.
        """
        with self._lock:
            conn = self._connection()
            cursor = conn.execute("DELETE FROM prompts WHERE name = ?", (name,))
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(
                    "Deleted prompt '{}' ({} version(s))", name, cursor.rowcount,
                )
            else:
                logger.warning("Prompt '{}' not found for deletion", name)
            return deleted

    # ── Row mapping ──────────────────────────────────────────────────────

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> PromptVersion:
        return PromptVersion(
            id=row["id"],
            name=row["name"],
            template=row["template"],
            version=row["version"],
            created_at=row["created_at"],
            variables=json.loads(row["variables_json"]),
            tags=json.loads(row["tags_json"]),
            hash=row["hash"],
        )


# ---------------------------------------------------------------------------
# AB Test Integration
# ---------------------------------------------------------------------------

_ab_coordinator = ABTestCoordinator(
    stable_version="stable",
    min_samples=50,
    significance_level=0.05,
    auto_promote=False,
)


def ab_test(
    repo: PromptRepository,
    prompt_name: str,
    version_a: int = 1,
    version_b: int | None = None,
    split_b: float = 0.5,
    user_id: str = "",
) -> str:
    """Select a prompt version via A/B testing and record the result.

    Uses the existing :class:`ABTestCoordinator` to manage traffic splits
    and record metrics for downstream statistical analysis.

    Args:
        repo: A PromptRepository instance.
        prompt_name: Name of the prompt to test.
        version_a: Control version (default: 1).
        version_b: Candidate version (default: latest version).
        split_b: Fraction of traffic to send to version_b (0.0 to 1.0).
        user_id: Optional user ID for consistent hashing (same user always
            sees the same version).

    Returns:
        The selected prompt version's template string.
    """
    # Resolve version_b to the latest if not specified.
    if version_b is None:
        all_versions = repo.list(name=prompt_name)
        if len(all_versions) < 2:
            p = repo.get(prompt_name, version=version_a)
            if p is None:
                raise ValueError(f"Prompt '{prompt_name}' version {version_a} not found")
            return p.template
        version_b = max(v.version for v in all_versions)

    p_a = repo.get(prompt_name, version=version_a)
    if p_a is None:
        raise ValueError(f"Prompt '{prompt_name}' version {version_a} not found")

    p_b = repo.get(prompt_name, version=version_b)
    if p_b is None:
        raise ValueError(f"Prompt '{prompt_name}' version {version_b} not found")

    label_a = f"{prompt_name}_v{version_a}"
    label_b = f"{prompt_name}_v{version_b}"

    _ab_coordinator.register_version(label_a)
    _ab_coordinator.register_version(label_b)
    _ab_coordinator.set_traffic_split({
        label_a: round((1.0 - split_b) * 100, 1),
        label_b: round(split_b * 100, 1),
    })

    selected = _ab_coordinator.select_version(user_id=user_id)
    selected_version = version_b if selected == label_b else version_a

    _ab_coordinator.record_result(selected, {
        "prompt_name": prompt_name,
        "version": selected_version,
    })

    return p_b.template if selected == label_b else p_a.template


# ---------------------------------------------------------------------------
# Decorator form of AB test
# ---------------------------------------------------------------------------


def ab_test_decorator(
    repo: PromptRepository,
    prompt_name: str,
    version_a: int = 1,
    version_b: int | None = None,
    split_b: float = 0.5,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Decorator that A/B tests two prompt versions.

    The wrapped function receives the selected prompt template as its
    first positional argument::

        @ab_test_decorator(repo, "my_prompt", version_a=1, version_b=2)
        def render(template: str, **kwargs: Any) -> str:
            return template.format(**kwargs)

    Args:
        repo: A PromptRepository instance.
        prompt_name: Name of the prompt to test.
        version_a: Control version (default: 1).
        version_b: Candidate version (default: latest).
        split_b: Fraction of traffic to send to version_b (0.0 to 1.0).

    Returns:
        A decorator that wraps a callable.
    """

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        def wrapper(*args: Any, **kwargs: Any) -> str:
            template = ab_test(
                repo=repo,
                prompt_name=prompt_name,
                version_a=version_a,
                version_b=version_b,
                split_b=split_b,
            )
            return func(template, *args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


def _get_repo() -> PromptRepository:
    db_path = os.environ.get("DISTLLM_PROMPT_LIBRARY_DB", "prompt_library.db")
    return PromptRepository(db_path=db_path)


def prompt_library_list(
    name: str | None = None,
    tag: str | None = None,
    json_output: bool = False,
) -> None:
    """List prompt library entries."""
    from rich.console import Console
    from rich.table import Table

    repo = _get_repo()
    results = repo.list(name=name, tag=tag)

    console = Console()

    if json_output:
        console.print_json(json.dumps(
            [
                {
                    "id": v.id,
                    "name": v.name,
                    "version": v.version,
                    "tags": v.tags,
                    "created_at": v.created_at,
                }
                for v in results
            ],
        ))
        return

    if not results:
        console.print("[yellow]No prompts found.[/yellow]")
        return

    table = Table(title=f"Prompt Library ({len(results)} entries)")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Tags")
    table.add_column("Created")

    for v in results:
        tags_str = ", ".join(v.tags[:3])
        if len(v.tags) > 3:
            tags_str += "..."
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(v.created_at))
        table.add_row(str(v.id), v.name, str(v.version), tags_str, created)

    console.print(table)


def prompt_library_get(
    name: str,
    version: int | None = None,
    json_output: bool = False,
) -> None:
    """Show full details of a specific prompt version."""
    from rich.console import Console

    repo = _get_repo()
    prompt = repo.get(name, version=version)

    console = Console()

    if prompt is None:
        version_str = f" v{version}" if version else ""
        console.print(f"[red]Prompt '{name}'{version_str} not found.[/red]")
        return

    if json_output:
        console.print_json(json.dumps({
            "id": prompt.id,
            "name": prompt.name,
            "version": prompt.version,
            "template": prompt.template,
            "variables": prompt.variables,
            "tags": prompt.tags,
            "hash": prompt.hash,
            "created_at": prompt.created_at,
        }))
        return

    console.print(f"[bold]Name:[/] {prompt.name}")
    console.print(f"[bold]Version:[/] {prompt.version}")
    console.print(f"[bold]ID:[/] {prompt.id}")
    console.print(f"[bold]Hash:[/] {prompt.hash}")
    console.print(f"[bold]Tags:[/] {', '.join(prompt.tags)}")
    console.print(f"[bold]Variables:[/] {', '.join(prompt.variables)}")
    console.print(
        f"[bold]Created:[/] {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(prompt.created_at))}",
    )
    console.print("[bold]Template:[/]")
    console.print(prompt.template)


def prompt_library_create(
    name: str,
    template: str = "",
    template_file: str | None = None,
    variables: list[str] | None = None,
    tags: list[str] | None = None,
) -> None:
    """Create a new prompt version.

    Provide the template inline via *template* or read it from a file
    via *template_file*.  If neither is provided, stdin is read.
    """
    from rich.console import Console

    repo = _get_repo()
    console = Console()

    resolved_template: str = template.strip()

    if not resolved_template and template_file:
        try:
            with open(template_file, encoding="utf-8") as f:
                resolved_template = f.read()
        except Exception as e:
            console.print(f"[red]Failed to read template file: {e}[/red]")
            return

    if not resolved_template:
        console.print(
            "[yellow]Enter template text (Ctrl+D or Ctrl+Z then Enter to finish):[/yellow]",
        )
        import sys
        resolved_template = sys.stdin.read().strip()

    if not resolved_template:
        console.print("[red]Template cannot be empty.[/red]")
        return

    prompt = repo.create(
        name=name,
        template=resolved_template,
        variables=variables or [],
        tags=tags or [],
    )
    console.print(
        f"[green]Created prompt '{name}' version {prompt.version} (id={prompt.id})[/green]",
    )


def prompt_library_diff(
    name: str,
    v1: int,
    v2: int,
    json_output: bool = False,
) -> None:
    """Show differences between two prompt versions."""
    from rich.console import Console

    repo = _get_repo()
    console = Console()

    try:
        changes = repo.diff(name, v1, v2)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return

    if json_output:
        console.print_json(json.dumps(changes, default=str))
        return

    if not changes:
        console.print(
            f"[yellow]No differences between version {v1} and version {v2} of '{name}'.[/yellow]",
        )
        return

    console.print(f"[bold]Diff:[/] {name} v{v1} -> v{v2}")

    for field_name, change in changes.items():
        if field_name in ("version_a", "version_b", "created_at_a", "created_at_b"):
            continue
        if isinstance(change, dict) and "changed" in change:
            console.print(f"  [red]{field_name}[/red]: changed")
            console.print(
                f"    old length: {change.get('old_length', '?')}, "
                f"new length: {change.get('new_length', '?')}",
            )
        elif isinstance(change, dict) and "old" in change:
            console.print(f"  [cyan]{field_name}:[/]")
            console.print(f"    [red]- {change['old']!r}[/red]")
            console.print(f"    [green]+ {change['new']!r}[/green]")


# ---------------------------------------------------------------------------
# API Router  (optional — requires FastAPI)
# ---------------------------------------------------------------------------

HAS_FASTAPI = False
router: Any = None

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True

    class _PromptCreateRequest(BaseModel):
        name: str = Field(..., description="Prompt name")
        template: str = Field(..., description="Prompt template text")
        variables: list[str] = Field(
            default_factory=list, description="Template variable names",
        )
        tags: list[str] = Field(
            default_factory=list, description="Categorisation tags",
        )

    class _PromptResponse(BaseModel):
        id: int
        name: str
        template: str
        version: int
        variables: list[str]
        tags: list[str]
        hash: str
        created_at: float

    class _PromptSummary(BaseModel):
        id: int
        name: str
        version: int
        tags: list[str]
        created_at: float

    router = APIRouter(prefix="/api/v1/prompts", tags=["prompt-library"])

    @router.get("", response_model=list[_PromptSummary])
    async def list_prompts_api(
        name: str | None = Query(None, description="Filter by prompt name"),
        tag: str | None = Query(None, description="Filter by tag (substring)"),
    ):
        """List prompt versions, optionally filtered by name or tag."""
        repo = _get_repo()
        results = repo.list(name=name, tag=tag)
        return [
            _PromptSummary(
                id=v.id,
                name=v.name,
                version=v.version,
                tags=v.tags,
                created_at=v.created_at,
            )
            for v in results
        ]

    @router.post("", response_model=_PromptResponse, status_code=201)
    async def create_prompt_api(body: _PromptCreateRequest):
        """Create a new prompt version for the given name."""
        repo = _get_repo()
        prompt = repo.create(
            name=body.name,
            template=body.template,
            variables=body.variables,
            tags=body.tags,
        )
        return _PromptResponse(
            id=prompt.id,
            name=prompt.name,
            template=prompt.template,
            version=prompt.version,
            variables=prompt.variables,
            tags=prompt.tags,
            hash=prompt.hash,
            created_at=prompt.created_at,
        )

    @router.get("/{name}", response_model=_PromptResponse)
    async def get_prompt_api(
        name: str,
        version: int | None = Query(
            None, description="Specific version (default: latest)",
        ),
    ):
        """Get a prompt by name and optional version."""
        repo = _get_repo()
        prompt = repo.get(name, version=version)
        if prompt is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt '{name}' version {version or 'latest'} not found",
            )
        return _PromptResponse(
            id=prompt.id,
            name=prompt.name,
            template=prompt.template,
            version=prompt.version,
            variables=prompt.variables,
            tags=prompt.tags,
            hash=prompt.hash,
            created_at=prompt.created_at,
        )

    @router.get("/{name}/diff")
    async def diff_prompt_api(
        name: str,
        v1: int = Query(..., description="First version to compare"),
        v2: int = Query(..., description="Second version to compare"),
    ):
        """Compare two versions of a prompt and return the differences."""
        repo = _get_repo()
        try:
            return repo.diff(name, v1, v2)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

except ImportError:
    pass


__all__ = [
    "PromptVersion",
    "PromptRepository",
    "ab_test",
    "ab_test_decorator",
    "prompt_library_list",
    "prompt_library_get",
    "prompt_library_create",
    "prompt_library_diff",
    "HAS_FASTAPI",
    "router",
]
