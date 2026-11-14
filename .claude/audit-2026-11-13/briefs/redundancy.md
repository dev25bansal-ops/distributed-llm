# Fix brief: redundancy

Audit findings for the files in your exclusive scope. For EACH finding:
reproduce-or-refute first (run the check command or read the cited code
carefully), then either FIX it or record an explicit refutation. Never
leave a finding unaddressed. After fixing, extend or add a test in the
named test file and run it. Do NOT commit; leave changes in the tree.

## [high] src/distllm/dist/pipeline/orchestrator.py:190
**Summary**: The redundancy config knob is inert end-to-end: coordinator wires config.redundancy into PipelineOrchestrator._redundancy, but no code in src ever reads it, and the only class that implements redundant fan-out (dist/redundant.py RedundantExecutor) is never constructed anywhere in production src (only exported lazily).
**Failure scenario**: Operator starts the coordinator with redundancy=2 (--redundancy, help text 'Redundant peers per stage') expecting fault-tolerant stage execution; run_pipeline runs each stage exactly once with no fan-out and no failover — the feature silently never engages. Probe with two same-range replicas shows redundancy=1 and redundancy=2 behave identically (2 sequential calls, same output). Supporting grep: `_redundancy` appears only as a write at orchestrator.py:190 and inside redundant.py itself; `RedundantExecutor(` has no non-test, non-docstring instantiation in src. StrategySelector.select_strategy (dist/pipeline/strategy.py:97) also has no production caller — tests only.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python -c "import sys,types; sys.path.insert(0,'src'); import torch; from distllm.dist.pipeline.orchestrator import PipelineOrchestrator as P; calls=[]
for red in (1,2):
    calls.clear(); p=P(redundancy=red)
    p.register_node('n1','h1',1,0,15); p.register_node('n2','h2',1,0,15)
    mk=lambda tag:(lambda hidden_states,kv_cache,request_id:(calls.append(tag) or hidden_states+1))
    p._nodes['n1'].client=types.SimpleNamespace(forward=mk('n1'))
    p._nodes['n2'].client=types.SimpleNamespace(forward=mk('n2'))
    out=p.run_pipeline(torch.zeros(1,2,dtype=torch.long),{},'r')
    print('redundancy=',red,'calls=',len(calls),'output=',out[0,0].item())"  (observed: redundancy= 1 calls= 2 / redundancy= 2 calls= 2, identical)`
**Reproduced by auditor**: True

## [high] src/distllm/dist/redundant.py:113
**Summary**: RedundantExecutor._run_redundant treats every entry of node_order as a sequential stage, so when two replicas cover the same layer range the hidden state is pushed through the same layers twice (duplicate forward application) — corrupting the output.
**Failure scenario**: With replication nodes 'a' and 'b' both covering layers 0-15 and redundancy=2, the orchestrator node_order contains both; the stage loop (line 112) runs once per node, each time fanning out to the same {a,b} candidate set (line 113 finds peers by identical start/end layer). The first stage applies hidden+1 and the second stage applies the SAME layer range again, so input 0 yields output 2.0 (4 forward calls) instead of the single-pass 1.0. Wrong inference results whenever redundancy>1 is actually engaged. The same probe also shows fan-out is sequential (not concurrent — a slow primary blocks, contradicting 'uses the first response') and timeout_s=0.05 is never enforced (wall time 1.2s; `self._timeout` set at line 40 is read nowhere).
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python -c "import sys; sys.path.insert(0,'src'); import torch; import distllm.dist.node_client as nc; from distllm.dist.redundant import RedundantExecutor; from distllm.dist.pipeline.orchestrator import PipelineOrchestrator as P
calls=[]
nc.forward_request=lambda host,port,hidden_states,kv_cache=None,request_id='',**kw:(calls.append(host) or hidden_states+1)
p=P(); p.register_node('a','10.0.0.1',1,0,15); p.register_node('b','10.0.0.2',1,0,15)
out=RedundantExecutor(p,redundancy=2).run_pipeline(torch.zeros(1,4),{},'r')
print('forward calls=',len(calls),'output=',out[0,0].item())"  (observed: forward calls= 4 output= 2.0; expected one pass = 1.0)`
**Reproduced by auditor**: True

## [high] src/distllm/dist/redundant_executor.py:1320
**Summary**: RedundantExecutor.run_redundant silently returns a success-shaped RedundantResult(result=None, replica_id='', failures=0) when the timeout expires before any replica completes, instead of raising as the docstring promises ('Raises RuntimeError: If all replicas failed').
**Failure scenario**: Call run_redundant with n_replicas=2, timeout_s=0.05, and a task that sleeps 0.4s: no exception is raised; the caller receives RedundantResult with result=None and failures=0, and the INFO log falsely reports 'task completed via  in 0.0ms (0/2 replicas failed)'. The guard at line 1320 only raises when best_result is None AND failures > 0, so the timeout case (zero completions examined) falls through to the success return at line 1335. A caller checking only failures==0 concludes success and propagates a None payload downstream.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python -c "import sys,time; sys.path.insert(0,'src'); from distllm.dist.redundant_executor import RedundantExecutor as R; r=R().run_redundant(lambda rid,**kw: time.sleep(0.4), n_replicas=2, timeout_s=0.05); print('no exception; result=',repr(r.result),'failures=',r.failures,'replica_id=',repr(r.replica_id))"  (observed: no exception; result= None failures= 0 replica_id= '')`
**Reproduced by auditor**: True

## [high] src/distllm/dist/redundant_executor.py:1303
**Summary**: RedundantExecutor.run_redundant inspects only the first-completed future (FIRST_COMPLETED + break), so when the fastest replica fails and a healthy backup replica is still running, it raises 'All N redundant replicas failed' — defeating the failover the redundancy exists to provide.
**Failure scenario**: replica-0 raises RuntimeError('replica down') immediately; replica-1 is healthy but needs 0.15s. wait(FIRST_COMPLETED) returns only replica-0's future; the loop at 1303 sees its exception, sets failures=1, exits; best_result stays None and the guard at 1320 raises RuntimeError('All 2 redundant replicas failed') even though the healthy replica would have succeeded 150ms later. The whole purpose of redundant execution (survive a failed replica) fails in the most common real failure shape (fast connection refusal on the primary).
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python -c "import sys,time; sys.path.insert(0,'src'); from distllm.dist.redundant_executor import RedundantExecutor as R
def f(rid,**kw):
    if rid=='replica-0': raise RuntimeError('replica down')
    time.sleep(0.15); return 'healthy-result'
try: print('returned:', R().run_redundant(f, n_replicas=2, timeout_s=5).result)
except RuntimeError as e: print('raised:', e)"  (observed: raised: All 2 redundant replicas failed (1 failures, timeout=5s))`
**Reproduced by auditor**: True

## [high] src/distllm/dist/redundant_executor.py:737
**Summary**: IncrementalRecovery.save_delta without base_params records zero deltas forever — the documented no-base mode is a silent no-op, making recover_to impossible.
**Failure scenario**: The docstring documents base_params=None as a supported mode ('the delta is computed against the parameter's value at the last recorded step (or the checkpoint if no prior delta exists)'). In the code, the first occurrence sets base=current (line 737, comment 'delta = 0 effectively'), the zero delta is skipped at line 745, so no previous delta ever exists; every subsequent call also falls into base=current and records nothing. After two documented save_delta(1, w=5.0) / save_delta(2, w=7.0) calls, delta_count=0 and latest_step=0; a later recover_to(2, params) raises 'target_step 2 exceeds latest step 0' — the incremental-recovery feature is unusable in its documented default mode.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python -c "import sys; sys.path.insert(0,'src'); import torch; from distllm.dist.redundant_executor import IncrementalRecovery as I; r=I(); r.save_delta(1,{'w':torch.full((4,),5.0)}); r.save_delta(2,{'w':torch.full((4,),7.0)}); print('delta_count=',r.delta_count,'latest_step=',r.latest_step)"  (observed: delta_count= 0 latest_step= 0)`
**Reproduced by auditor**: True

## [med] src/distllm/dist/redundant_executor.py:1408
**Summary**: RedundantExecutor.recover_node trivially reports success while silently skipping the promised NCCL state-replication phase: the default NCCLStateReplicator is constructed with auto_init=False, so is_initialized is always False and the state_tensors branch is skipped with no warning, returning success=True and replication_size_bytes=0.
**Failure scenario**: Caller invokes the documented usage executor.recover_node('node-3', state_tensors={'kv': ...}) expecting KV-cache state to be replicated to the recovered node; phase 3 guard at line 1408 requires self._replicator.is_initialized, but the default replicator (line 1234, auto_init=False) never initialized, so the loop is skipped silently; the report claims success=True, replication_size_bytes=0, error=''. Even a bare recover_node('node-3') (the docstring's own example) returns success=True with steps_replayed=0, weights_synced=0 — a no-op recovery indistinguishable from a real one except by inspecting zeroed counters.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python -c "import sys; sys.path.insert(0,'src'); import torch; from distllm.dist.redundant_executor import RedundantExecutor as R; rep=R().recover_node('node-3',state_tensors={'kv':torch.zeros(8)}); print('success=',rep.success,'replication_size_bytes=',rep.replication_size_bytes,'error=',repr(rep.error))"  (observed: success= True replication_size_bytes= 0 error= '')`
**Reproduced by auditor**: True

## [med] src/distllm/dist/redundant_executor.py:733
**Summary**: IncrementalRecovery mixed mode (base_params given once, then omitted) silently computes wrong deltas via _estimate_previous_value's 'approximation', so recover_to returns silently with corrupted parameters.
**Failure scenario**: save_delta(1, w=15.0, base_params=ck{w:10.0}) records +5 correctly; save_delta(2, w=17.0) with no base_params sets base=_estimate_previous_value(name, 1) which returns the SUM of stored deltas (+5) rather than the previous absolute value (15.0), so it records delta=17-5=12 instead of +2. recover_to(2, ck) then applies +5 then +12 and returns w=27.0 where the true value is 17.0 — no exception, silently wrong recovered model/parameter state.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && python -c "import sys; sys.path.insert(0,'src'); import torch; from distllm.dist.redundant_executor import IncrementalRecovery as I; r=I(); ck={'w':torch.full((4,),10.0)}; r.save_delta(1,{'w':torch.full((4,),15.0)},base_params=ck); r.save_delta(2,{'w':torch.full((4,),17.0)}); out=r.recover_to(2,dict(ck)); print('recovered w[0]=',out['w'][0].item(),'expected=17.0')"  (observed: recovered w[0]= 27.0 expected=17.0)`
**Reproduced by auditor**: True
