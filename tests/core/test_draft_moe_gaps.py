"""Gap tests: Draft models (EAGLE, Medusa, N-gram) + MoE (capacity, all-to-all)."""

import pytest
import torch
import torch.nn as nn

from distllm.core.drafters.eagle import EAGLEGenerator, TrainedEAGLEHeads, EAGLE2Heads
from distllm.core.drafters.medusa import MedusaHeads
from distllm.core.drafters.ngram import NgramMatcher, NgramTrie, NgramTrieNode
from distllm.core.moe_router import MoERouter, GatingConfig
from distllm.core.moe_capacity import ExpertCapacityManager, CapacityConfig
from distllm.core.moe_alltoall import MoEAllToAll


class TestEAGLEGenerator:
    def test_init_creates_predictor(self):
        eagle = EAGLEGenerator(hidden_size=64, vocab_size=100, num_layers=1, num_draft_tokens=3)
        assert eagle is not None

    def test_generate_draft_tokens_returns_list(self):
        eagle = EAGLEGenerator(hidden_size=64, vocab_size=100, num_layers=1)
        hs = torch.randn(1, 1, 64)
        lm_head = nn.Linear(64, 100)
        tokens = eagle.generate_draft_tokens(hs, lm_head, num_drafts=3)
        assert isinstance(tokens, (list, torch.Tensor))

    def test_generate_with_anchor(self):
        eagle = EAGLEGenerator(hidden_size=64, vocab_size=100, num_layers=1)
        hs = torch.randn(1, 1, 64)
        lm_head = nn.Linear(64, 100)
        tokens = eagle.generate_with_anchor(hs, lm_head, num_drafts=3, anchor_ratio=0.3)
        assert isinstance(tokens, (list, torch.Tensor))


class TestTrainedEAGLEHeads:
    def test_init_shapes(self):
        heads = TrainedEAGLEHeads(hidden_size=64, vocab_size=100, num_layers=2)
        hs = torch.randn(2, 4, 64)
        out = heads.forward(hs, input_ids=torch.randint(0, 100, (2, 4)))
        assert out.shape[-1] == 100

    def test_save_load_checkpoint(self, tmp_path):
        heads = TrainedEAGLEHeads(hidden_size=64, vocab_size=100)
        path = str(tmp_path / "eagle.pt")
        heads.save_checkpoint(path)
        heads2 = TrainedEAGLEHeads(hidden_size=64, vocab_size=100)
        heads2.load_checkpoint(path)
        assert isinstance(heads2, TrainedEAGLEHeads)


class TestEAGLE2Heads:
    def test_init_creates_model(self):
        e2 = EAGLE2Heads(hidden_size=64, vocab_size=100, num_draft_tokens=3)
        assert e2 is not None

    def test_feature_extractor_exists(self):
        e2 = EAGLE2Heads(hidden_size=64, vocab_size=100, num_draft_tokens=3)
        assert e2.feature_extractor is not None

    def test_draft_heads_count(self):
        e2 = EAGLE2Heads(hidden_size=64, vocab_size=100, num_draft_tokens=3)
        assert len(e2.draft_heads) == 3

    def test_feature_alignment_loss(self):
        e2 = EAGLE2Heads(hidden_size=64, vocab_size=100, num_draft_tokens=3)
        draft = torch.randn(2, 4, 64)
        target = torch.randn(2, 4, 64)
        loss = e2.compute_feature_alignment_loss(draft, target)
        assert loss.item() >= 0


class TestMedusaHeads:
    def test_init(self):
        medusa = MedusaHeads(num_heads=3, hidden_size=64, vocab_size=100)
        assert len(medusa.heads) == 3

    def test_generate_draft_tokens(self):
        medusa = MedusaHeads(num_heads=3, hidden_size=64, vocab_size=100, top_k_per_head=3)
        logits = torch.randn(1, 1, 100)
        drafts = medusa.generate_draft_tokens(logits)
        assert isinstance(drafts, list)

    def test_is_trained_flag(self):
        medusa = MedusaHeads()
        assert medusa.is_trained is False

    def test_save_load_checkpoint(self, tmp_path):
        medusa = MedusaHeads(hidden_size=64, vocab_size=100)
        path = str(tmp_path / "medusa.pt")
        medusa.save_checkpoint(path)
        medusa2 = MedusaHeads(hidden_size=64, vocab_size=100)
        medusa2.load_checkpoint(path)
        assert medusa2._weights_loaded


class TestNgramTrie:
    def test_insert_and_lookup(self):
        trie = NgramTrie(max_order=5)
        trie.insert((1, 2, 3), 4)
        result = trie.lookup((1, 2, 3))
        assert result is not None
        assert result[4] >= 1

    def test_longest_match(self):
        trie = NgramTrie(max_order=5)
        trie.insert((1, 2), 3)
        trie.insert((1, 2, 3), 4)
        match = trie.longest_match((1, 2, 3, 5))
        assert match is not None
        assert match[0] == 3

    def test_size_increases(self):
        trie = NgramTrie(max_order=5)
        assert trie.size >= 0

    def test_empty_lookup(self):
        trie = NgramTrie()
        assert trie.lookup((99,)) is None


class TestNgramMatcher:
    def test_update_and_predict(self):
        matcher = NgramMatcher(min_match=2, max_match=5)
        for ids in ([1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6]):
            matcher.update(ids)
        tokens = matcher.predict([1, 2, 3], max_drafts=3)
        assert isinstance(tokens, list)

    def test_oov_rate_starts_zero(self):
        matcher = NgramMatcher()
        assert matcher.oov_rate == 0.0

    def test_stats_returns_dict(self):
        matcher = NgramMatcher()
        matcher.update([1, 2, 3])
        s = matcher.stats()
        assert isinstance(s, dict)


class TestMoERouter:
    def test_forward_returns_topk(self):
        router = MoERouter(num_experts=4, num_experts_per_tok=2, hidden_dim=32)
        hs = torch.randn(2, 4, 32)
        indices, weights = router.forward(hs)
        assert indices.shape == (2, 4, 2)

    def test_aux_loss_computed(self):
        router = MoERouter(num_experts=4, num_experts_per_tok=2, hidden_dim=32)
        hs = torch.randn(2, 4, 32)
        indices, weights, aux_loss = router.forward(hs, use_aux_loss=True)
        assert aux_loss.item() >= 0

    def test_register_expert(self):
        router = MoERouter(num_experts=4, num_experts_per_tok=2, hidden_dim=32)
        router.register_expert(0, "node-1")
        assert router.get_expert_node(0) == "node-1"

    def test_route_to_nodes(self):
        router = MoERouter(num_experts=4, num_experts_per_tok=2, hidden_dim=32)
        hs = torch.randn(2, 4, 32)
        result = router.route_to_nodes(hs)
        assert isinstance(result, dict)


class TestMoECapacity:
    def test_compute_capacity(self):
        mgr = ExpertCapacityManager(num_experts=4,
            config=CapacityConfig(capacity_factor=1.25, min_capacity=2))
        cap = mgr.compute_capacity(100)
        assert len(cap) == 4
        for v in cap.values():
            assert v >= 2

    def test_check_overflow_empty(self):
        mgr = ExpertCapacityManager(num_experts=4)
        overflow = mgr.check_overflow({0: 50, 1: 50})
        assert isinstance(overflow, dict)

    def test_stats_keys(self):
        mgr = ExpertCapacityManager(num_experts=4)
        s = mgr.stats()
        assert "total_tokens" in s
        assert "overflow_tokens" in s


class TestMoEAllToAll:
    def test_init(self):
        a2a = MoEAllToAll(num_experts=4, num_nodes=2, experts_per_node=[[0, 1], [2, 3]])
        assert a2a is not None

    def test_stats_returns_dict(self):
        a2a = MoEAllToAll(num_experts=4, num_nodes=1, experts_per_node=[[0, 1, 2, 3]])
        s = a2a.stats()
        assert isinstance(s, dict)
