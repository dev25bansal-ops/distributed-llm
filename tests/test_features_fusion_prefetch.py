"""Regression tests — roadmap items 6 (spec + structured-output fusion)
and 7 (cluster-wide predictive KV-cache prefetch).

Kept fast: no real subprocess/GPU; uses fakes + CPU torch only.
"""

import torch

from distllm.core.grammar_constrained_draft import (
    GrammarConstrainedDraftPolicy,
    mask_draft_logits,
)
from distllm.core.cluster_predictive_prefetcher import (
    ClusterPredictivePrefetcher,
    WarmStartSource,
)
from distllm.core.coordinator_metrics import MetricsManager


# ── Item 6: speculative + structured-output fusion ──

def test_mask_draft_logits_zeroes_forbidden():
    logits = torch.tensor([[1.0, 5.0, 9.0, 2.0]])
    # Only token ids 1 and 3 are grammar-valid.
    mask = torch.tensor([False, True, False, True])
    out = mask_draft_logits(logits, mask)
    assert torch.isinf(out[0, 0]) and out[0, 0] < 0   # 0 forbidden
    assert torch.isinf(out[0, 2]) and out[0, 2] < 0   # 2 forbidden
    assert out[0, 1] == 5.0                            # 1 allowed (unchanged)
    assert out[0, 3] == 2.0                            # 3 allowed


def test_mask_draft_logits_pads_short_mask():
    logits = torch.tensor([[1.0, 5.0, 9.0, 2.0, 7.0]])
    mask = torch.tensor([True, True])  # only covers ids 0,1
    out = mask_draft_logits(logits, mask, vocab_size=5)
    # ids 2,3,4 beyond the mask are forbidden.
    assert out[0, 2] < 0 and out[0, 3] < 0 and out[0, 4] < 0
    assert out[0, 0] == 1.0 and out[0, 1] == 5.0


class _FakeGrammar:
    """Grammar whose allowed set is configurable per call."""

    def __init__(self, allowed_ids):
        self._allowed = set(allowed_ids)
        self._vocab = max(allowed_ids) + 1 if allowed_ids else 1

    def get_logits_mask(self, vocab_size, tokenizer, device):
        m = torch.zeros(vocab_size, dtype=torch.bool)
        for i in self._allowed:
            if i < vocab_size:
                m[i] = True
        return m


class _FakeTokenizer:
    vocab_size = 10
    eos_token_id = None


def test_grammar_policy_masks_draft_logits():
    policy = GrammarConstrainedDraftPolicy(_FakeGrammar([2, 4, 7]), _FakeTokenizer())
    logits = torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]])
    out = policy.mask_fn(logits)
    # Only 2,4,7 keep finite logits.
    allowed = [i for i in (2, 4, 7) if out[0, i] == 3.0]
    assert allowed == [2, 4, 7]
    # Everything else zeroed.
    assert torch.isinf(out[0, 0]) and out[0, 0] < 0


def test_speculative_decoder_accepts_grammar_param():
    # Importability + wiring: constructing with grammar sets the mask fn.
    from distllm.core.speculative_decoder import SpeculativeDecoder

    def _fwd(ids, **kw):
        return torch.zeros(ids.shape[0], ids.shape[1], 10)

    dec = SpeculativeDecoder(
        target_forward=_fwd, draft_forward=_fwd, grammar=_FakeGrammar([0, 1]),
        tokenizer=_FakeTokenizer(), device="cpu",
    )
    assert dec._grammar_mask_fn is not None
    assert dec._grammar_policy is not None


# ── Item 7: cluster-wide predictive prefetch ──

class _FakeCache:
    def __init__(self, store):
        self._store = store  # dict token-tuple -> kv
    def lookup(self, token_ids):
        key = tuple(token_ids)
        if key in self._store:
            return len(key), self._store[key]
        return 0, None


class _FakeGossip:
    def __init__(self, index):
        self._index = index  # prefix_hash -> list[(node, _, _)]
    def discover_prefix(self, prefix_hash):
        return [n for n, _, _ in self._index.get(prefix_hash, [])]


class _FakeCrossModel:
    def __init__(self, entries):
        self._entries = entries  # (model_id, token_ids) -> source_model
    def store(self, model_id, token_ids, kv_data=None):
        pass
    def lookup(self, model_id, token_ids):
        key = (model_id, tuple(token_ids))
        src = self._entries.get(key)
        if src is None:
            return None
        class _E: pass
        e = _E(); e.source_model = src
        return e


def _mk_prefetcher(gpu=None, cpu=None, gossip=None, cross=None, metrics=None):
    return ClusterPredictivePrefetcher(
        node_id="node-A",
        local_gpu_cache=gpu, local_cpu_cache=cpu,
        gossip_bridge=gossip, cross_model_sharing=cross, metrics=metrics,
    )


def test_warm_start_local_gpu_hit():
    gpu = _FakeCache({(1, 2, 3): "KV"})
    pf = _mk_prefetcher(gpu=gpu)
    res = pf.warm_start([1, 2, 3])
    assert res.source is WarmStartSource.LOCAL_GPU
    assert res.hit is True
    assert res.latency_ms < 50  # sub-ms-to-tens-of-ms, no network


def test_warm_start_falls_to_cpu():
    cpu = _FakeCache({(7, 8): "KV"})
    pf = _mk_prefetcher(cpu=cpu)
    res = pf.warm_start([7, 8])
    assert res.source is WarmStartSource.LOCAL_CPU


def test_warm_start_cross_node_via_gossip():
    # Prefix not local; gossip says node-B has it.
    gossip = _FakeGossip({"abc": [("node-B", 0, 0)]})
    pf = _mk_prefetcher(gossip=gossip)
    # Force the same hash the prefetcher computes for [1,2,3].
    h = ClusterPredictivePrefetcher._hash([1, 2, 3])
    gossip._index = {h: [("node-B", 0, 0)]}
    res = pf.warm_start([1, 2, 3])
    assert res.source is WarmStartSource.CROSS_NODE
    assert res.node_id == "node-B"
    assert "node-B" in res.replica_nodes


def test_warm_start_cross_model_variant():
    cross = _FakeCrossModel({("llama-instruct", (5, 6)): "llama-base"})
    pf = _mk_prefetcher(cross=cross)
    res = pf.warm_start([5, 6], model_id="llama-instruct")
    assert res.source is WarmStartSource.CROSS_MODEL
    assert res.model_id == "llama-base"


def test_warm_start_miss_records_observation():
    metrics = MetricsManager()
    pf = _mk_prefetcher(metrics=metrics)
    res = pf.warm_start([9, 9, 9, 9])
    assert res.source is WarmStartSource.MISS
    assert res.hit is False
    # observe() must not raise even with a partial pipeline.
    pf.observe([9, 9, 9, 9], model_id="m")
    assert metrics.get().get("kv_warm_starts_total", 0) >= 1


def test_on_local_cache_store_advertises_to_gossip():
    advertised = {}
    class _G:
        def discover_prefix(self, h): return []
        def on_cache_store(self, prefix_hash, node_id, size_bytes=0):
            advertised[prefix_hash] = node_id
    pf = _mk_prefetcher(gossip=_G())
    pf.on_local_cache_store([1, 2, 3])
    assert advertised  # hash advertised to gossip bridge
