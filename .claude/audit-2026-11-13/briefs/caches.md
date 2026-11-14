# Fix brief: caches

Audit findings for the files in your exclusive scope. For EACH finding:
reproduce-or-refute first (run the check command or read the cited code
carefully), then either FIX it or record an explicit refutation. Never
leave a finding unaddressed. After fixing, extend or add a test in the
named test file and run it. Do NOT commit; leave changes in the tree.

## [high] src/distllm/core/advanced_scheduling/tiered_store.py:355
**Summary**: TieredMemoryPool._put_l3's eviction loop pops entries without subtracting from _l3_used, so the L3 byte counter grows without bound, every put deletes all existing L3 entries, and get() counts a hit before verifying the NVMe load (popping on failure without decrementing).
**Failure scenario**: TieredMemoryPool(nvme_max_gb=0.00001 => 10000B): five 4000B put(tier=COLD) calls -> l3_used reports 20000 while exactly one 4000B file remains on disk (4000B real vs 10000B capacity) and k0..k3.bin were all deleted even though two entries (8000B) fit. Because used_bytes never decreases, each put wipes the entire L3 tier first. Also get() increments the COLD 'hits' counter (line 224) before _load_from_nvme; if the file is gone it pops the entry (line 229) without subtracting -> measured l3_used stays 4000 with 0 entries and a served miss is counted as a hit.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && PYTHONPATH=src python .tmp-cache-audit/repro_tiered_store.py`
**Reproduced by auditor**: True

## [high] src/distllm/core/cache_manager.py:347
**Summary**: store_prefix() calls _tier_store() without self._lock (docstring at line 216 mandates the lock), so concurrent stores crash with RuntimeError and re-storing an existing prefix double-counts used_bytes.
**Failure scenario**: CacheManager(prefix_cache_enabled=False, gpu_cache_mb=1). 4 threads x 400 store_prefix() calls with unique keys: after the gpu tier fills, _tier_store's eviction loop iterates tier["entries"].items() while another thread inserts into the same dict -> uncaught 'RuntimeError: dictionary keys changed during iteration' at cache_manager.py:222, raised inside the scheduler step (batch_scheduler.py:1105 / step_processor.py:97 have no try/except around it on this path). Separately, storing the SAME prefix 99 times reports used_bytes=99000 while 1 entry of 1000 real bytes exists (line 235 adds size unconditionally), which then evicts live entries as 'phantom' usage grows. Reproduced twice (crash) and deterministically (99000/1000 drift).
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && PYTHONPATH=src python .tmp-cache-audit/repro_race2.py ; PYTHONPATH=src python .tmp-cache-audit/repro_cache_manager.py`
**Reproduced by auditor**: True

## [high] src/distllm/core/cache_manager.py:189
**Summary**: Tier promotion in _tier_lookup() subtracts the promoted entry's size from the source tier unconditionally after a pop that may already have been evicted by the nested _tier_store() demotion, under-reporting used_bytes and leaving the entry duplicated across tiers.
**Failure scenario**: gpu tier (budget 3000) holds G1(2500); cpu holds P(1000, earlier demoted). lookup_prefix(P): _tier_store('gpu', P) runs first (line 183) and its eviction loop demotes G1 to cpu; during that nested _tier_store('cpu') the cpu tier evicts P (oldest) to ssd and subtracts it; then lines 188-189 pop P from cpu (no-op) and subtract 1000 again. Measured: cpu used_bytes=1500 vs 2500 real, and P simultaneously present in gpu and ssd (2x real memory). Repeating this interleaving drives used_bytes downward with no clamp, so the tier admits data past its budget.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && PYTHONPATH=src python .tmp-cache-audit/repro_promote2.py`
**Reproduced by auditor**: True

## [high] src/distllm/dist/prefix_cache.py:275
**Summary**: PrefixCache.evict(), the max_entries popitem loop in store() (line 249), and clear() (line 279+) remove entries without decrementing _total_memory_bytes, leaving phantom bytes that evict live entries which fit the budget.
**Failure scenario**: PrefixCache(min_prefix_len=4, memory_budget_bytes=20000): store 9000B entry, evict() it -> entries=0 but memory reported=9000. Then store two fresh 9000B entries (real 18000, budget 20000): reported hits 27000, _evict_until_fit destroys one live entry -> lookup returns 0 for one of the two prefixes that both fit. Also: max_entries=2 with four 5000B stores reports 20000 bytes for 10000 real; clear() leaves 19000 reported with 0 entries. evict() is exercised by shipped tests (tests/dist/test_prefix_cache.py:139) and reaches production via CacheManager.prefix_cache.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && PYTHONPATH=src python .tmp-cache-audit/repro_pc_final.py`
**Reproduced by auditor**: True

## [high] src/distllm/dist/prefix_cache.py:243
**Summary**: PrefixCache.store() overwrite path replaces kv_data without adjusting _total_memory_bytes, under-reporting memory (here 9x) and silently exceeding the configured budget.
**Failure scenario**: PrefixCache(min_prefix_len=4, memory_budget_bytes=1<<20): store a 1000B payload for tokens T, then store a 9000B payload for the same T. stats() reports prefix_cache_memory_bytes=1000 while the cache actually holds 9000 bytes. Repeat with growing payloads and _total_memory_bytes never approaches the real footprint, so _evict_until_fit never runs and the operator-set budget is violated without bound.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && PYTHONPATH=src python .tmp-cache-audit/repro_pc_overwrite.py`
**Reproduced by auditor**: True

## [high] src/distllm/api/semantic_cache.py:329
**Summary**: SemanticCacheMiddleware (the dedup layer's request-path sibling) returns cached completions keyed by prompt text alone — no credential/tenant/model scope — so any authenticated key can be served another key's cached output, and cache hits bypass QuotaMiddleware and route-level RBAC below it.
**Failure scenario**: With DISTLLM_SEMANTIC_CACHE=1 (documented opt-in, server.py:664), key-a's earlier POST of prompt P with header X-Semantic-Cache: 1 stores its completion via _cache.add(prompt, resp_body.decode()) (semantic_cache.py:343). key-b (different, lower-privileged credential) POSTs the identical P with X-Semantic-Cache: 0.99: search() (line 328) matches by prompt-embedding only and dispatch returns key-a's exact response bytes with 200/X-Cache: HIT (line 331); the route, its role checks, and QuotaMiddleware (built-stack index 17, inner to SemanticCache at index 15) never run — cross-credential completion disclosure plus unmetered serving. Direction is reversible: an admin or any other tenant asking the same (or >=threshold-similar) prompt receives the low-privilege key's poisoned completion.
**Check command**: `python D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper/.claude/tmp_dedup_audit/repro_real_crosskey.py  (prints: key-b request -> 200 X-Cache: HIT {"marker": "completion-generated-for-key-a"})`
**Reproduced by auditor**: True

## [med] src/distllm/core/advanced_scheduling/tiered_store.py:429
**Summary**: _demote_to_l3() writes to NVMe and increments _l3_used without ever checking _l3_capacity, so directory-cache demotions blow through the configured L3 quota.
**Failure scenario**: TieredMemoryPool(cpu_memory_gb=0.000005 => 5000B L2, nvme_max_gb=0.000001 => 1000B L3): 20 warm puts of 400B fill L2; each L2 eviction demotes to L3 unchecked. Measured l3_used=3200 with 8 files on disk versus a 1000-byte L3 capacity - the NVMe budget is exceeded by 3x and grows with load; only _put_l3 (which is itself broken, see above) ever evicts.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && PYTHONPATH=src python .tmp-cache-audit/repro_race.py  # see 'Case A' output`
**Reproduced by auditor**: True

## [med] src/distllm/core/cache_manager.py:578
**Summary**: _tier_stats miss counters are never incremented anywhere in lookup_with_gossip(), so get_tier_stats() reports 0 misses for any number of actual misses and its hit/miss telemetry disagrees with what was served.
**Failure scenario**: CacheManager with empty caches: 50 lookup_with_gossip() calls that all return None leave every tier's 'misses' at 0 ({'local': {'hits': 0, 'misses': 0}, ...}), while the same method does increment 'hits' on success (lines 538/552/563/572). Any hit-rate computed from get_tier_stats() (the counters the class documents as per-tier metrics) is wrong.
**Check command**: `cd D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper && PYTHONPATH=src python .tmp-cache-audit/repro_tier_stats.py`
**Reproduced by auditor**: True

## [med] src/distllm/core/coordinator_config.py:195
**Summary**: CoordinatorConfig.from_settings only copies 8 fields from DistLLMSettings; explicitly configured prefix-cache settings (and the other unlisted fields) are silently replaced by CoordinatorConfig defaults (enabled=True,max_entries=5000 -> False,256).
**Failure scenario**: A user sets prefix_cache: {enabled: true, max_entries: 5000} in config.yaml; distllm coordinator builds CoordinatorConfig.from_settings(settings), which never passes prefix_cache_enabled/prefix_cache_max_entries, so the result reports enabled=False / 256. No error or warning; worse, nothing in src consumes either the settings or the CoordinatorConfig fields (coordinator.py:114 builds CacheManager() with no arguments, so prefix caching is off regardless).
**Check command**: `python -c "import sys;sys.path.insert(0,'src');from distllm.config.settings import DistLLMSettings;from distllm.core.coordinator_config import CoordinatorConfig;s=DistLLMSettings(prefix_cache={'enabled':True,'max_entries':5000});c=CoordinatorConfig.from_settings(s);print(c.prefix_cache_enabled,c.prefix_cache_max_entries)"  # False 256`
**Reproduced by auditor**: True

## [med] src/distllm/core/scheduler/budget_computer.py:165
**Summary**: BatchScheduler.__init__ (batch_scheduler.py:101) replaces the explicitly passed max_tokens_per_batch with compute_dynamic_budget, which silently clamps it to 131072 (and can shrink it further from PagedAttention pool state), so an explicit 200000 budget is never honored or reported back.
**Failure scenario**: config.yaml batching.max_tokens_per_batch=200000 flows via CoordinatorConfig into BatchScheduler(max_tokens_per_batch=200000); the constructed scheduler's max_tokens_per_batch attribute reads 131072 and adjust_budget() re-derives the same clamped value. With a PagedAttention manager attached, adaptive pool-derived logic may shrink the caller's budget arbitrarily (e.g. 1440 tokens in tests) with only a debug log. The canonical config value the operator set is silently not what runs.
**Check command**: `python -c "import sys;sys.path.insert(0,'src');from distllm.core.batch_scheduler import BatchScheduler;print(BatchScheduler(max_tokens_per_batch=200000).max_tokens_per_batch)"  # 131072`
**Reproduced by auditor**: True

## [med] src/distllm/core/semantic_cache.py:210
**Summary**: SemanticCache's adaptive-threshold update clamps an explicitly passed similarity_threshold into the hard-coded defaults band [0.80, 0.99], silently raising 0.5 -> 0.8 on the 10th semantic hit (same class as the fixed dist/prefix_cache.py bug: explicit value replaced by adaptive logic).
**Failure scenario**: SemanticCache(similarity_threshold=0.5, adaptive_threshold=True): after 10 lookups that semantically hit, _update_threshold() computes threshold 0.5049 then applies max(0.80, min(0.99, x)) = 0.80. A later lookup whose true cosine similarity is 0.6 now misses, although the caller explicitly configured 0.5 to accept it; the cache silently serves fewer hits and no log/warning is emitted.
**Check command**: `python -c "import sys;sys.path.insert(0,'src');from distllm.core.semantic_cache import SemanticCache;c=SemanticCache(similarity_threshold=0.5,adaptive_threshold=True);c.hit_rate=0.75;c.store('Q',response='R',embedding=[1.0]*16,scope='t');[c.lookup('other',embedding=[0.95]*16,scope='t') for _ in range(11)];print('threshold',c._threshold)"  # prints ~0.80, caller asked 0.5`
**Reproduced by auditor**: True

## [med] src/distllm/core/semantic_cache.py:217
**Summary**: The same adaptive-threshold path reads self.hit_rate, which does not exist (hit_rate is only a stats() dict key), so every SemanticCache lookup with adaptive_threshold=True raises AttributeError on the 10th semantic hit and the request fails.
**Failure scenario**: SemanticCache(similarity_threshold=0.5, adaptive_threshold=True) used normally: the 10th lookup that finds a semantic match raises AttributeError('SemanticCache' object has no attribute 'hit_rate') from _update_threshold -> _find_similar -> lookup. The public adaptive_threshold flag is therefore unusable, and the exception surfaces inside a cache lookup on the request path. Same missing attribute at line 257 (GP branch).
**Check command**: `python -c "import sys;sys.path.insert(0,'src');from distllm.core.semantic_cache import SemanticCache;c=SemanticCache(similarity_threshold=0.5,adaptive_threshold=True);c.store('Q',response='R',embedding=[1.0]*16,scope='t');[c.lookup('other',embedding=[0.95]*16,scope='t') for _ in range(11)]"  # raises AttributeError on ~10th iteration`
**Reproduced by auditor**: True

## [med] src/distllm/dist/attention.py:247
**Summary**: BlockPool._expand_pool grows an explicitly passed num_blocks (doubling, min +64 blocks) whenever auto_expand is true, and PagedAttentionManager does not expose auto_expand, so an explicit GPU block budget is silently exceeded on demand.
**Failure scenario**: BlockPool(num_blocks=4, block_size=2, ...) allocates 4 blocks, then the 5th allocate_block() silently grows the pool to 68 blocks (and on GPU that allocates a fresh tensor of pool size). inference_engine.py:233 and hybrid_cache.py:108 construct PagedAttentionManager, whose signature has no auto_expand parameter, so even an operator setting DISTLLM_PAGED_NUM_BLOCKS=1024 cannot make 1024 a hard memory ceiling — the pool silently expands beyond it under load. This is the same shape as the fixed prefix_cache memory_budget bug.
**Check command**: `python -c "import sys;sys.path.insert(0,'src');import torch;from distllm.dist.attention import BlockPool;p=BlockPool(num_blocks=4,block_size=2,num_layers=1,num_heads=1,head_dim=2,dtype=torch.float16,device='cpu');[p.allocate_block() for _ in range(5)];print(p.num_blocks)"  # 68`
**Reproduced by auditor**: True

## [med] src/distllm/api/semantic_cache.py:343
**Summary**: Semantic cache key ignores model, temperature, and all generation parameters (store/search use prompt text only), so a request for one model/parameter set is served the completion cached for a different one.
**Failure scenario**: With the flag on, a client POSTs model-expensive, temperature 0.0 with X-Semantic-Cache: 0.99 for prompt text P already cached from an earlier request with model-cheap, temperature 1.0: the middleware returns the cheap model's output as the response to the expensive-model request (200, X-Cache: HIT) without ever executing the route, so the wrong model's completion is served under the requested model name.
**Check command**: `python D:/distributed-llm/.claude/worktrees/bubbly-pondering-whisper/.claude/tmp_dedup_audit/repro_model_blind.py  (prints: model-expensive,temp=0.0 -> 200 X-Cache: HIT {"marker": "output-of-model-cheap-temp=1.0"})`
**Reproduced by auditor**: True

## [med] src/distllm/core/cross_model_prefix_sharing.py:115
**Summary**: store() keeps the caller's token_ids list by reference (no copy), so if the caller mutates a reused token buffer after storing, the entry's token_ids no longer match its prefix_hash key and the sibling-branch guard (line 178) compares against the mutated list, returning KV computed for a different request's prompt.
**Failure scenario**: A serving loop reuses one token buffer across requests: it calls store('x', t, kv) with t=[1,2], then reuses the buffer as t[:]=[9,8] for the next request. A sibling model 'y' (same base) looks up [9,8]: the guard entry.token_ids != token_ids is False because entry.token_ids was mutated to [9,8], so lookup returns the KV stored for [1,2] tagged source_model='x' — the exact 'cache hit returns another request's tokens' condition. Verified: python -I -c "import sys; sys.path.insert(0, r'D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper\src'); from distllm.core.cross_model_prefix_sharing import CrossModelPrefixSharing as C; s=C(default_ttl=3600); s.register_model('b'); s.register_model('x', base_model='b', shared_layers=4); s.register_model('y', base_model='b', shared_layers=4); t=[1,2]; s.store('x', t, 'KV-FOR-1-2'); t[:]=[9,8]; e=s.lookup('y',[9,8]); print('returned for prompt B:', e.token_ids, e.kv_data, 'from', e.source_model)" prints: returned for prompt B: [9, 8] KV-FOR-1-2 from x. cluster_predictive_prefetcher.observe() forwards the caller's list unchanged into store(), so the same pattern applies via that path too.
**Check command**: `python -I -c "import sys; sys.path.insert(0, r'D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper\src'); from distllm.core.cross_model_prefix_sharing import CrossModelPrefixSharing as C; s=C(default_ttl=3600); s.register_model('b'); s.register_model('x', base_model='b', shared_layers=4); s.register_model('y', base_model='b', shared_layers=4); t=[1,2]; s.store('x', t, 'KV-FOR-1-2'); t[:]=[9,8]; e=s.lookup('y',[9,8]); print('returned for prompt B:', e.token_ids, e.kv_data, 'from', e.source_model)"`
**Reproduced by auditor**: True

## [low] src/distllm/core/coordinator_config.py:207
**Summary**: The from_settings override loop silently drops unknown keys (typo'd field names become no-ops) and bypasses all pydantic validation by setattr, so out-of-range/invalid explicit values are accepted.
**Failure scenario**: from_settings(settings, max_batchsize=999) silently ignores the typo (max_batch_size stays 32) with no error. from_settings(settings, port=99999, dtype='int4') returns a CoordinatorConfig reporting port=99999 and dtype='int4' even though the Field constraints (le=65535) and the dtype validator forbid them; the direct constructor CoordinatorConfig(port=99999) correctly raises ValidationError, so the bypass is specific to this path.
**Check command**: `python -c "import sys;sys.path.insert(0,'src');from distllm.config.settings import DistLLMSettings;from distllm.core.coordinator_config import CoordinatorConfig;c=CoordinatorConfig.from_settings(DistLLMSettings(),max_batchsize=999,port=99999,dtype='int4');print(c.max_batch_size,c.port,c.dtype)"  # 32 99999 int4`
**Reproduced by auditor**: True

## [low] src/distllm/core/kv_cache.py:692
**Summary**: KVCache.update silently doubles an explicitly passed max_seq_len on overflow (new_max = max(max_seq_len, needed) * 2), even though the docstring frames growth as the fallback for when max_seq_len is NOT set.
**Failure scenario**: KVCache(max_seq_len=8) pre-allocates an 8-token buffer; appending a 9th+ token replaces _max_seq_len with 24 and reallocates a 24-token buffer instead of enforcing the caller's pre-allocation bound. Code that passes max_seq_len as a memory cap (the buffer is per-layer per-request) gets 3x the memory silently; there is no warning and no way to observe the replacement except reading the private attribute.
**Check command**: `python -c "import sys;sys.path.insert(0,'src');import torch;from distllm.core.kv_cache import KVCache;c=KVCache(max_seq_len=8,num_layers=1);c.init_cache(1,1,1,4,'cpu');c.update(0,torch.randn(1,1,8,4),torch.randn(1,1,8,4));c.update(0,torch.randn(1,1,4,4),torch.randn(1,1,4,4));print(c._max_seq_len)"  # 24`
**Reproduced by auditor**: True

## [low] src/distllm/dist/async_pipeline.py:380
**Summary**: AsyncPipelineEngine discards an explicitly configured num_warmup_micro_batches: the first forward() overwrites self._warmup with min(num_stages-1, num_micro_b), so the setting (and ASYNC_PIPELINE_NUM_WARMUP_MICRO_BATCHES env var) never affects the schedule or the reported stats.
**Failure scenario**: AsyncPipelineConfig(num_stages=3, num_micro_batches=2, num_warmup_micro_batches=7) is accepted (frozen BaseSettings) and _warmup=7 at init, but after one forward() _warmup==2 and stats()['warmup'] reports 2. The configured warmup is replaced by derived logic without a warning, and nothing in _run_1f1b ever consulted the configured value, so the knob is a silent no-op.
**Check command**: `python -c "
import sys; sys.path.insert(0,'src')
import torch
from distllm.dist.async_pipeline import AsyncPipelineEngine, AsyncPipelineConfig
class S:
    _backward_fn = None
    def initialize(self): pass
    def run_micro_batch(self, h, mb_id=None): return h
    def wait(self): pass
e = AsyncPipelineEngine(AsyncPipelineConfig(num_stages=3, num_micro_batches=2, num_warmup_micro_batches=7))
[e.add_stage(S()) for _ in range(3)]
e.forward(torch.zeros(4, 8))
print('warmup =', e._warmup)
"  # configured 7 -> prints 2`
**Reproduced by auditor**: True

## [low] src/distllm/core/cluster_predictive_prefetcher.py:117
**Summary**: observe() persists cross-model entries with kv_data=None, and warm_start() then reports them as CROSS_MODEL hits with hit=True even though no KV payload exists; the result object carries no KV at all, so the cross-model 'warm start' can never actually deliver reusable KV.
**Failure scenario**: A node observes prefix [1,2] for model 'x' (the only store path used by the prefetcher passes kv_data=None). A request for the same prefix on sibling 'y' calls warm_start and receives CROSS_MODEL/hit=True, so a caller that trusts the hit to skip prefill/KV computation gets a warm signal with nothing to reuse; if real KV had been stored, warm_start still discards it (WarmStartResult has no kv field). Verified: check command prints 'warm_start: WarmStartSource.CROSS_MODEL hit= True reported model: x'.
**Check command**: `python -I -c "import sys; sys.path.insert(0, r'D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper\src'); from distllm.core.cross_model_prefix_sharing import CrossModelPrefixSharing as C; from distllm.core.cluster_predictive_prefetcher import ClusterPredictivePrefetcher as P; s=C(default_ttl=3600); s.register_model('b'); s.register_model('x', base_model='b', shared_layers=4); s.register_model('y', base_model='b', shared_layers=4); pf=P(node_id='n1', cross_model_sharing=s); pf.observe([1,2], model_id='x'); r=pf.warm_start([1,2], model_id='y'); print('warm_start:', r.source, 'hit=', r.hit, 'reported model:', r.model_id)"`
**Reproduced by auditor**: True

## [low] src/distllm/core/cross_model_prefix_sharing.py:182
**Summary**: lookup() computes shared = min(entry.shared_layers, model.shared_layers) only for the >0 compatibility check and returns the entry unchanged, so a returned entry can report more shared layers than the querying model actually shares.
**Failure scenario**: Model 'big' (shared_layers=80) stores KV; 'small' (shared_layers=70, same base) looks it up for the same tokens and receives the entry with entry.shared_layers==80. A consumer that uses entry.shared_layers to decide how deep into the returned KV it may read would attend over KV produced by layers 70-79 of a different model — wrong attention values in generation. Verified: the check command prints 'model shares 70 layers; entry reports 80'. Latent today: no production consumer exists, but the returned entry is the only channel the class exposes for slice depth.
**Check command**: `python -I -c "import sys; sys.path.insert(0, r'D:\distributed-llm\.claude\worktrees\bubbly-pondering-whisper\src'); from distllm.core.cross_model_prefix_sharing import CrossModelPrefixSharing as C; s=C(default_ttl=3600); s.register_model('big', base_model='base', shared_layers=80); s.register_model('small', base_model='base', shared_layers=70); s.store('big',[5,6],'KV-80'); e=s.lookup('small',[5,6]); print('model shares 70 layers; entry reports', e.shared_layers)"`
**Reproduced by auditor**: True
