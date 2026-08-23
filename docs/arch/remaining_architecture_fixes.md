# Remaining Architecture Fix Plans

**Date**: 2026-07-18
**Target**: Issues HI-22, HI-26, HI-27, HI-28, HI-29 from comprehensive analysis

## HI-22: Circular dependency chain across three layers

**Status**: Requires structural refactoring (1-2 weeks)

**Affected chain**:
```
backends/pytorch_backend.py → models/partitioner.py
  → dist/fsdp.py → dist/worker.py → models/partitioner.py (cycle!)
```

**Four copies of broken lazy-import machinery**: Already fixed in HI-23 — all 4 packages (`core/__init__.py`, `dist/__init__.py`, `models/__init__.py`, `backends/__init__.py`) now use the shared `LazyImporter` from `utils/lazy_imports.py`.

**Remaining work**:
1. Extract shared types (`PartitionProtocol`, `FSDPConfig`, etc.) into `core/protocols.py` (already exists as a scaffold)
2. Move the `ModelsPartitioner` import out of `pytorch_backend.py` — pass it via dependency injection from the Coordinator instead
3. In `dist/fsdp.py`, import `FSDPShard` via the `LazyImporter` from `dist/__init__.py` rather than directly from `dist.worker`

**Recommended approach for team**:
```
Week 1: Extract shared types into core/protocols.py
  - Move FSDPConfig, ShardMetadata into protocols.py
  - Define IPartitioner protocol for models.partitioner
  
Week 2: Wire via Coordinator DI
  - Coordinator._init_backend() passes partitioner to backend constructor
  - Remove direct import chain between backends → models
```

## HI-26: Backward compatibility re-exports in server.py

**Status**: Cleanup plan ready (2-3 days)

**Affected area**: `api/server.py` re-exports internal model classes and private streaming helpers at module level.

**Action items**:
1. Add `__all__` to `server.py` that **only** exports public API symbols (FastAPI app, lifespan, state)
2. Add `import warnings` + `DeprecationWarning` with `stacklevel=2` for each removed re-export
3. Update internal imports across `api/routes/*.py` to import from source modules, not from `server.py`
4. Remove dead re-exports after one release cycle

**Search to find re-exports**:
```bash
# Find everything server.py exports that isn't its own
grep -n "^from\|^import" src/distllm/api/server.py
# Then check what's used via server.* from other modules
```

## HI-27: 8+ subsystems typed as Any in Coordinator

**Status**: Type-annotation plan ready (1 week)

**Affected area**: `core/coordinator.py` — at least 8 optional subsystem attributes typed as `Any`.

**Fix approach** (add proper types in __init__):

```python
# BEFORE:
self._semantic_cache: Any = None
self._smart_router: Any = None
self._disaggregated_scheduler: Any = None
self._carbon_engine: Any = None
self._cost_tracker: Any = None
self._arbitrage_engine: Any = None
self._hot_swap_mgr: Any = None
self._enable_ha: bool = False

# AFTER (using TYPE_CHECKING imports):
if TYPE_CHECKING:
    from distllm.core.semantic_cache import SemanticCache
    from distllm.core.smart_router import SmartRouter
    ...

class Coordinator:
    def __init__(self, ...):
        self._semantic_cache: SemanticCache | None = None
        self._smart_router: SmartRouter | None = None
        ...
```

**Steps**:
1. Add `TYPE_CHECKING` imports for all subsystem types
2. Replace `Any` with the proper optional types
3. Replace `hasattr()`/`getattr()` patterns with explicit `is None` checks
4. Run mypy to validate and fix cascading type errors

## HI-28: Dual config class hierarchy

**Status**: Migration plan ready (1-2 weeks)

**Affected area**: `config/` directory — 18 files with dual hierarchy (pydantic Settings + legacy Config classes + CoordinatorConfig wrapper).

**Migration plan**:

Phase 1 (2-3 days): Audit and document
- List all config fields across all 3 hierarchies
- Identify duplicates and conflicts

Phase 2 (3-5 days): Consolidate to pydantic Settings
- Remove legacy Config classes from `loader.py`
- Move all CoordinatorConfig fields into the new Settings classes
- Single inheritance tree from `BaseSettings`

Phase 3 (2-3 days): Update callers
- Replace scattered `os.environ.get("DISTLLM_*")` calls with config object access
- Remove CoordinatorConfig wrapper — Coordinator reads Settings directly

## HI-29: Bare except blocks silently swallow errors

**Status**: Systematic audit plan ready (2-3 weeks)

**Scope**: 40+ bare `except` blocks across `api/`, `backends/`, `cloud/`, `cli/`, `core/`, `dashboard/`, `dist/`, `health/`, `plugins/`, `security/`, `terraform/`.

**Fix strategy by severity**:

| Pattern | Fix | Effort |
|---------|-----|--------|
| `except: pass` | Log at minimum (logger.warning or exception) | 2-3 days |
| `except Exception: pass` | Add logging with context | 2-3 days |
| `except Exception:` with no logging | Add `logger.exception(...)` | 1-2 days |
| `except:` with comment explaining why | Acceptable — add `# noqa` and document rationale | 0.5 days |

**Quick wins** (multi-file search + replace):
```bash
# Find bare excepts
rg "except\s*\:" src/distllm --include '*.py'
# Find bare except Exception
rg "except\s+Exception\s*:" src/distllm --include '*.py'
```

**Automated enforcement**:
- Add ruff rule: `[tool.ruff.lint] select = ["E722"]` (bare except)
- Add ruff rule: `[tool.ruff.lint] select = ["BLE001"]` (blind except)
