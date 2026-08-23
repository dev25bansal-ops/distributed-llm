# Coordinator God-Object Decomposition Plan

**Date**: 2026-07-18
**Target**: `src/distllm/core/coordinator.py` (1634 lines, ~55 methods, 15+ subsystems)
**Estimated effort**: 2-3 weeks for full decomposition
**Risk**: Medium — coordinator is the central orchestrator; extraction must be staged with backward-compatible interfaces.

## Current State

Coordinator is a single class that directly instantiates and manages:

| # | Subsystem | Lines | Methods | Extraction Target |
|---|-----------|-------|---------|-------------------|
| 1 | High-Availability / State Replication | ~60 | 6 | `CoordinatorHA` |
| 2 | Hot-swap model loading | ~55 | 5 | `HotSwapManager` |
| 3 | Adaptive compression | ~50 | 1 | Reuse existing |
| 4 | Defragmentation | ~60 | 6 | `MemoryDefragmenter` |
| 5 | Graceful degradation | ~50 | 1 | Reuse existing |
| 6 | Scheduling hints + routing | ~40 | 3 | Into `ModelRouter` |
| 7 | Node callbacks (drain/mark_dead/redistribute/recover) | ~100 | 4 | Into `NodeLifecycleManager` |
| 8 | Request generation (sync + async) | ~110 | 4 | Into `RequestHandler` |
| 9 | Subsystem lifecycle (start/stop/_start_subsystem) | ~200 | 4 | Into `SubsystemManager` |
| 10 | Config + init wiring | ~200 | 14 | Into `CoordinatorConfigurator` |
| 11 | Metrics + health + cleanup | ~80 | 4 | Into `MetricsCollector` |
| 12 | Properties (nodes, order, scheduler, etc.) | ~30 | 5 | Keep in facade |
| 13 | Public API surface (generate, health, etc.) | ~250 | ~10 | Keep as thin facade |

## Decomposition Strategy

### Phase 1: Extract leaf subsystems (no Coordinator dependency)

These can be extracted to standalone files without touching Coordinator internals:

1. **`CoordinatorHA`** → `coordinator_ha.py`
   - Move: `enable_ha()`, `is_leader()`, `ha_status()`, `state_snapshot()`, `apply_state_snapshot()`, `_start_state_replication()`, `_replication_loop()`, `set_replication_peers()`
   - Interface: `CoordinatorHA(config, on_state_change_cb)`
   - Coordinator delegates: `self._ha = CoordinatorHA(...)` then `self._ha.enable_ha(...)`

2. **`NodeLifecycleManager`** → `node_lifecycle.py`
   - Move: `_on_node_drain()`, `_on_node_mark_dead()`, `_on_node_redistribute()`, `_on_node_recover()`
   - Interface: accepts pipeline and resource manager references
   - Coordinator callback registration simplified

3. **`MetricsCollector`** → `coordinator_metrics.py`
   - Move: `record_metric()`, `get_metrics()`, `_cleanup_stale_results()`, `health_check()`
   - Interface: `MetricsCollector(stats: dict)`

**Effort**: 3-5 days. Low risk — no behavioral changes.

### Phase 2: Extract mid-layer subsystems (depend on Coordinator but not vice versa)

4. **`RequestHandler`** → `request_handler.py`
   - Move: `generate()`, `generate_async()`, `wait_for_result()`, `_cleanup_stale_results()`
   - Interface: `RequestHandler(coordinator: "CoordinatorShim", batch_scheduler, ...)`
   - Need to pass a shim with just the methods the request handler needs

5. **`SubsystemManager`** → `subsystem_manager.py`
   - Move: `_start_subsystem()`, `start()`, `stop()`, `_register_subsystems()`
   - Interface: `SubsystemManager(coordinator, config)`
   - Owns the startup ordering and teardown logic

**Effort**: 5-7 days. Medium risk — need careful interface extraction.

### Phase 3: Extract top-level subsystems (tightly coupled)

6. **`CoordinatorConfigurator`** → `coordinator_configurator.py`
   - Move: `init_*` methods, `auto_setup()`, `manual_register()`, `_init_adaptive_batching()`, `init_model_router()`, etc.
   - Interface: `CoordinatorConfigurator(config) -> Coordinator` (builder pattern)

7. **Coordinator facade** → slim to ~200 lines
   - Keep: `__init__()`, `start()`, `stop()`, public property accessors
   - All subsystem calls go through the extracted managers
   - Use delegation pattern (self._ha, self._requests, self._lifecycle, etc.)

**Effort**: 5-7 days. High risk — tight coupling with config system.

### Phase 4: Resolve circular dependency chain

The circular chain is:
```
backends/pytorch_backend.py
  → models/partitioner.py
    → dist/fsdp.py
      → dist/worker.py
        → models/partitioner.py  (cycle!)
```

**Fix options**:
- **Option A**: Extract the shared types into a `protocols.py` module that has no imports
- **Option B**: Use dependency inversion — define abstract interfaces in `protocols.py` that each layer imports
- **Option C**: Move the ModelsPartitioner reference into the coordinator (it's the orchestrator that should know about both)

**Recommended**: Option A + targeted Option C. Extract `PartitionProtocol`, `WorkerProtocol`, `FSDPProtocol` into `core/protocols.py`. Coordinator does the wiring.

**Effort**: 1-2 weeks. Risk: high — affects 4+ files across layers.

## Testing Strategy

Each extraction phase should be validated by:

1. **Refactoring tests**: Create a test that instantiates the OLD Coordinator, calls all public methods, and verifies no AttributeError. Run it before and after each extraction phase.
2. **Unit tests for new classes**: Each extracted class gets its own test file with 80%+ coverage.
3. **Regression suite**: Run `tests/core/test_coordinator*.py` after each phase to verify no behavioral change.

## Backward Compatibility

During all 4 phases, the `Coordinator` class must retain its current public API surface:
- `generate()`, `generate_async()`, `wait_for_result()`, `stop()`, `start()`, `health_check()`, `get_metrics()`, `list_models()`
- Properties: `nodes`, `node_order`, `scheduler`

Internal method renames (`_on_node_*` → `self._lifecycle.on_node_*`) are acceptable within same release cycle.

## Files to Create (in order)

1. `src/distllm/core/coordinator_ha.py`
2. `src/distllm/core/node_lifecycle.py`
3. `src/distllm/core/coordinator_metrics.py`
4. `src/distllm/core/request_handler.py`
5. `src/distllm/core/subsystem_manager.py`
6. `src/distllm/core/coordinator_configurator.py`
7. `src/distllm/core/protocols.py` (if circular dep resolution is needed)
