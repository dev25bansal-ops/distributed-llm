"""Consolidated import helpers for the distllm test suite.

Provides single-source versions of ``make_fake_package`` and ``load_module``
that were previously duplicated across a dozen conftest files and test modules.

Usage in a conftest or test module::

    from tests._import_helper import SRC_DIR, bootstrap_fake_packages, load_module

    # Bootstrap standard fake packages (must run before any ``from distllm.…``
    # import that would trigger the circular import chain).
    bootstrap_fake_packages()

    # Load a source module directly, bypassing ``distllm/__init__.py``.
    _coord_mod = load_module("distllm/core/coordinator.py")
    Coordinator = _coord_mod.Coordinator
"""

from __future__ import annotations

import importlib.util
import string
import sys
import types
from pathlib import Path

SRC_DIR: Path = Path(__file__).resolve().parent.parent / "src"

# The standard set of fake packages needed by all conftest files.
# Only fake packages whose ``__init__.py`` triggers circular imports
# (core -> dist -> models -> core, etc.).  Safe packages like
# ``distllm.errors``, ``distllm.config``, ``distllm.dist.pipeline`` etc.
# import directly without issues and should NOT be faked.
#
# To determine if a package needs faking:
#   $ cd src && python -c "from distllm.X.Y import SomeSymbol"
# If that works, the package is safe.  If it triggers a circular import
# error, it needs to be faked.
_STANDARD_FAKE_PACKAGES: tuple[str, ...] = (
    # The top-level ``distllm`` package is intentionally NOT faked.  Faking it
    # substituted an empty stub for the real package, which made
    # ``distllm.config.settings`` see a half-initialized ``distllm.config``
    # during its own import (``config/__init__.py`` eagerly re-exports
    # settings) and raised ``ImportError: cannot import name
    # 'DistLLMSettings'`` for every test that collected through the helper.
    # The circular import chain this stub once guarded against was eliminated
    # by the dist-layer audit; the real top-level package imports cleanly.
    # NOTE: distllm.api and its sub-packages are intentionally NOT in this list
    # so that API tests can import the real app via from distllm.api import app.
    # Only fake packages that are transitively needed for load_module() to work.
    # Core: coordinator.py import chain reaches dist.node_registrar which
    # imports from models.partitioner -- broken lazy import.
    "distllm.core",
    # Core sub-packages with __init__.py that creates circular imports.
    "distllm.core.structured_output",
    # NOTE: distllm.dist is intentionally NOT faked — using the real package,
    # whose submodules (e.g. dist.partition) resolve their real exports (e.g.
    # HardwareAwarePartitioner). Faking distllm.dist left a stub in sys.modules
    # that broke test_partitioner collection under suite ordering (F-007).
    # NOTE: distllm.dist.backends is also omitted; its real __init__ imports
    # cleanly and does not need the fake.
    # Config: __init__.py eager-imports from settings causing circular issues.
    # NOTE: distllm.config is NOT faked — the real package imports cleanly
    # (__init__ re-exports settings; _model/_network/_cache load standalone),
    # and faking it broke `from distllm.config.settings import X` for real-module
    # tests under full-suite collection (F-007).
    # Backends: some modules import from core (circular risk)
    "distllm.backends",
    # Models: __init__.py has the get_model_info lazy-import bug
    "distllm.models",
)


def make_fake_package(name: str, path: Path) -> types.ModuleType:
    """Create a fake package in ``sys.modules`` to avoid ``__init__.py`` loading.

    This prevents the circular import chain (observability → core → api → …)
    from being triggered at module-import time in test contexts.

    Args:
        name: Fully-qualified package name (e.g. ``"distllm.core"``).
        path: Filesystem path corresponding to *name*.

    Returns:
        The created fake module.
    """
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    mod.__package__ = name
    sys.modules.setdefault(name, mod)
    return mod


def bootstrap_fake_packages(
    extra_packages: dict[str, Path] | None = None,
    src_dir: Path | None = None,
) -> None:
    """Inject all standard fake packages into ``sys.modules``.

    Call this **once** at module load time, before any ``from distllm.…``
    import statement.

    After injecting fake packages, also applies known source import-bug
    workarounds so that ``load_module`` calls succeed.

    Args:
        extra_packages: Optional mapping of *additional* package names
            to filesystem paths.
        src_dir: Override the source directory (defaults to ``SRC_DIR``).
    """
    src = src_dir or SRC_DIR

    # Historical note: this function used to inject fake stub packages for
    # ``distllm``, ``distllm.core``, ``distllm.backends``, ``distllm.models``
    # to dodge circular imports that no longer exist — the dist-layer audit
    # and refactors fixed the underlying source bugs, and every one of those
    # packages now imports cleanly on its own (verified directly).  The stubs
    # were actively harmful: faking top-level ``distllm`` made
    # ``distllm.config.settings`` hit a half-initialized ``distllm.config``
    # (``ImportError: cannot import name 'DistLLMSettings'``), and faking
    # ``distllm.core.structured_output`` hid its real ``JSONSchemaConstraint``
    # export.  We therefore no longer fake anything; the call sites remain
    # valid and harmless.
    if extra_packages:
        for pkg_name, pkg_path in extra_packages.items():
            make_fake_package(pkg_name, pkg_path)

    # F-007: ensure the real settings module is present.  (Re)loading it via
    # ``spec_from_file_location`` here used to trigger a circular import —
    # settings.py imports ``distllm.config._model``, whose import of
    # ``distllm.config`` re-enters ``config/__init__.py`` and re-exports from
    # the still-partial settings module.  A plain dotted import lets the
    # package init order resolve first, so we use that instead.
    _real_cfg = sys.modules.get("distllm.config.settings")
    if _real_cfg is None or getattr(_real_cfg, "__file__", None) is None:
        try:
            import distllm.config.settings  # noqa: F401
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # KNOWN SOURCE IMPORT BUG WORKAROUNDS                                #
    #                                                                     #
    # These pre-load modules whose badly-broken import chains the normal
    # ``load_module`` path cannot resolve.  When the source bug is fixed
    # these stanzas can be removed.                                       #
    # ------------------------------------------------------------------ #

    _apply_source_bug_workarounds(src)


def _apply_source_bug_workarounds(src: Path) -> None:
    """Apply fixes for known source-code import bugs.

    Bug: ``get_model_info`` / ``partition_model_across_nodes`` /
    ``partition_model_gpu_aware`` are defined in
    ``distllm.models.partition_planner`` but ``node_registrar.py`` and
    ``cluster_manager.py`` import them from ``distllm.models.partitioner``.
    When ``distllm.models`` is faked the lazy-init in ``models/__init__.py``
    never runs, so these symbols are missing.

    Fix: pre-load ``partition_planner.py`` and inject its symbols onto a
    synthetic ``distllm.models.partitioner`` module.
    """
    if "distllm.models" not in sys.modules:
        return
    models_mod = sys.modules["distllm.models"]

    if "distllm.models.partitioner" in sys.modules:
        return  # already set up

    # Locate the partition_planner.py file
    pkg_path = getattr(models_mod, "__path__", [str(src / "distllm" / "models")])[0]
    planner_path = Path(pkg_path) / "partition_planner.py"
    if not planner_path.exists():
        return  # source layout differs, skip

    # Load partition_planner via normal importlib
    planner_dotted = "distllm.models.partition_planner"
    planner_spec = importlib.util.spec_from_file_location(
        planner_dotted, str(planner_path), submodule_search_locations=[]
    )
    if planner_spec is None or planner_spec.loader is None:
        return
    planner_mod = importlib.util.module_from_spec(planner_spec)
    sys.modules[planner_dotted] = planner_mod
    planner_spec.loader.exec_module(planner_mod)

    # Load the real partitioner.py first, then patch on the missing symbols
    # that actually live in partition_planner.py.
    part_path = Path(pkg_path) / "partitioner.py"
    if not part_path.exists():
        return
    part_dotted = "distllm.models.partitioner"
    part_spec = importlib.util.spec_from_file_location(
        part_dotted, str(part_path), submodule_search_locations=[]
    )
    if part_spec and part_spec.loader:
        part_mod = importlib.util.module_from_spec(part_spec)
        sys.modules[part_dotted] = part_mod
        part_spec.loader.exec_module(part_mod)

        # Patch missing symbols that live in partition_planner.py
        _MISSING = ("get_model_info", "partition_model_across_nodes", "partition_model_gpu_aware")
        for name in _MISSING:
            if not hasattr(part_mod, name) and hasattr(planner_mod, name):
                setattr(part_mod, name, getattr(planner_mod, name))

    # (distllm.config is no longer faked — its real __init__ eagerly imports
    # settings, so no config workaround is needed and none must run.)

    # --- structured_output workaround ---
    # When faked, symbols from __init__.py (JSONSchemaConstraint etc.) are
    # unavailable.  Some are from sub-modules (loaded below), but
    # JSONSchemaConstraint and validate_structured_output are defined
    # directly in __init__.py.  We create a minimal stub for each.
    if "distllm.core.structured_output" in sys.modules:
        _so_pkg = sys.modules["distllm.core.structured_output"]

        # JSONSchemaConstraint is defined in __init__.py as a real class.
        # We stub it with a minimal version that tests can construct.
        if not hasattr(_so_pkg, "JSONSchemaConstraint"):
            class _JSONSchemaConstraintStub:
                """Stub replacement for the real JSONSchemaConstraint.
                Used when distllm.core.structured_output.__init__.py is
                bypassed by the fake-package mechanism.

                Implements a basic JSON state machine that matches the
                real ``JSONSchemaConstraint`` so that ``validate_token``
                (which calls ``_valid_next_chars()``) tests real behavior.
                """
                def __init__(self, schema=None):
                    self.schema = schema
                    self._state = "object_start"
                    self._stack: list[str] = []
                    self._in_string = False
                    self._escape_next = False
                    self._generated = ""
                def update(self, token_str: str) -> None:
                    self._generated += token_str
                    for ch in token_str:
                        self._state = self._transition(self._state, ch)
                def _transition(self, state: str, char: str) -> str:
                    if self._escape_next:
                        self._escape_next = False
                        return state
                    if self._in_string:
                        if char == '\\':
                            self._escape_next = True
                            return state
                        if char == '"':
                            self._in_string = False
                            return "after_key" if state == "in_string_key" else "after_value"
                        return state
                    if char == '"':
                        self._in_string = True
                        return "in_string_key" if state in ("object_start", "after_open_brace", "after_comma") else "in_string"
                    if char == '{':
                        return "after_open_brace"
                    if char == '}':
                        return self._stack.pop() if self._stack else "done"
                    if char == ':':
                        return "after_colon"
                    if char == ',':
                        return "after_comma" if state in ("after_value", "after_array_value") else state
                    if char == '[':
                        return "array_start"
                    if char == ']':
                        return self._stack.pop() if self._stack else "done"
                    if state in ("after_colon", "array_start", "after_array_comma") and char in 'tfn-0123456789':
                        return "after_value"
                    if state == "in_number":
                        return "in_number" if char in '0123456789.eE+-' else "after_value"
                    return state
                def _valid_next_chars(self) -> set[str]:
                    if self._in_string and not self._escape_next:
                        return set(string.printable) - {'\n', '\r', '\x0b', '\x0c'}
                    _transitions: dict[str, set[str]] = {
                        "object_start": {'"', '}'},
                        "after_open_brace": {'"', '}'},
                        "after_key": {':'},
                        "after_colon": {'"', '{', '[', 't', 'f', 'n', '-',
                                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
                        "after_value": {',', '}'},
                        "after_comma": {'"'},
                        "array_start": {']', '"', '{', '[', 't', 'f', 'n', '-',
                                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
                        "after_array_value": {',', ']'},
                        "after_array_comma": {'"', '{', '[', 't', 'f', 'n', '-',
                                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'},
                        "done": set(),
                    }
                    return _transitions.get(self._state, {'"', '}'})
                def get_logits_mask(self, vocab_size=1, tokenizer=None):
                    import torch
                    return torch.ones(vocab_size, dtype=torch.bool)
                def is_complete(self):
                    def _check_state() -> bool:
                        s = self._state
                        return s == "done" or (s in ("after_value", "after_open_brace") and not self._stack)
                    return _check_state()
                @classmethod
                def from_response_format(cls, response_format, tokenizer=None):
                    return cls(schema=response_format.get("schema", {}))
            _so_pkg.JSONSchemaConstraint = _JSONSchemaConstraintStub

        if not hasattr(_so_pkg, "validate_structured_output"):
            def _validate_structured_output(text, schema=None):
                return text
            _so_pkg.validate_structured_output = _validate_structured_output

        # Also load and merge exports from sub-modules
        _so_dir = getattr(_so_pkg, "__path__", [str(src / "distllm" / "core" / "structured_output")])[0]
        _so_submodules = [
            ("config", ["StructuredOutputConfig"]),
            ("engine", ["StructuredOutputEngine", "GenerationResult", "RepairConfig", "RepairTrajectory"]),
            ("validator", ["SchemaValidator", "OutputRepairer", "ValidationResult", "RepairResult"]),
            ("streaming", ["BufferedAccumulator", "PartialJSONParser", "StructuredStreamHandler", "PartialResult"]),
        ]
        for _sub, _symbols in _so_submodules:
            _sub_path = Path(_so_dir) / f"{_sub}.py"
            if not _sub_path.exists():
                continue
            _sub_dotted = f"distllm.core.structured_output.{_sub}"
            if _sub_dotted not in sys.modules:
                _sub_spec = importlib.util.spec_from_file_location(
                    _sub_dotted, str(_sub_path), submodule_search_locations=[]
                )
                if _sub_spec and _sub_spec.loader:
                    _sub_mod = importlib.util.module_from_spec(_sub_spec)
                    sys.modules[_sub_dotted] = _sub_mod
                    _sub_spec.loader.exec_module(_sub_mod)
            _sub_mod = sys.modules[_sub_dotted]
            for _sym in _symbols:
                if hasattr(_sub_mod, _sym) and not hasattr(_so_pkg, _sym):
                    setattr(_so_pkg, _sym, getattr(_sub_mod, _sym))


def load_module(
    rel_path: str,
    src_dir: Path | None = None,
    package_override: str | None = None,
) -> types.ModuleType:
    """Load a source module directly from its file path, bypassing ``__init__.py``.

    Args:
        rel_path: Relative path from the source root (e.g. ``"distllm/core/coordinator.py"``).
        src_dir: Source directory root (defaults to ``SRC_DIR``).
        package_override: Optional dotted module name override (e.g.
            ``"distllm.core.coordinator"``).  If omitted, the dotted name
            is derived from *rel_path*.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If *rel_path* does not exist under *src_dir*.
        ImportError: If the module cannot be loaded.
    """
    src = src_dir or SRC_DIR
    filepath = (src / rel_path).resolve()

    if not filepath.exists():
        raise FileNotFoundError(f"Source file not found: {filepath}")

    if package_override:
        dotted = package_override
    else:
        rel = filepath.relative_to(src)
        parts = list(rel.parent.parts) + [filepath.stem]
        if parts[0] == "distllm":
            dotted = ".".join(parts)
        else:
            dotted = "distllm." + ".".join(parts)

    if dotted in sys.modules:
        return sys.modules[dotted]

    spec = importlib.util.spec_from_file_location(
        dotted, str(filepath), submodule_search_locations=[]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {filepath}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # F-007: a failed exec leaves a PARTIAL module in sys.modules that
        # poisons later real imports (e.g. a half-loaded distllm.config.settings
        # missing CachePersistenceSettings).  Remove it so a later test can load
        # the real module cleanly.
        sys.modules.pop(dotted, None)
        raise
    return mod


def unload_module(name: str) -> None:
    """Remove a module from ``sys.modules`` to allow a clean reimport.

    Args:
        name: Dotted module name to remove (e.g. ``"distllm.core.coordinator"``).
    """
    sys.modules.pop(name, None)
    # Also remove submodules
    for key in list(sys.modules.keys()):
        if key.startswith(f"{name}."):
            sys.modules.pop(key, None)
