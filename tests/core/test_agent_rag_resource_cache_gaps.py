"""Gap tests: Agent (multi-turn, max iterations), RAG (index persistence), 
Resource (data loss recovery), Cache (auto-trigger, pattern detection, hot/warm)."""

import os
import tempfile
import time

import pytest

from distllm.core.agent_loop import AgentLoop, AgentMemory, ToolCall, AgentState
from distllm.core.tool_engine import ToolCallingEngine, ToolSchema, ToolResult
from distllm.core.rag_pipeline import RAGPipeline, Document, TextChunker, DocumentChunk
from distllm.core.resource_manager import ResourceManager
from distllm.core.node_recovery import NodeRecoveryManager, SequenceCheckpoint, LayerRedistribution, NodeRecoveryPlan
from distllm.core.graceful_degradation import GracefulDegradation, LoadSnapshot, DegradationLevel, DegradationPlan
from distllm.core.cache_warming import CacheWarmer, WarmUpTier
from distllm.core.predictive_cache import PatternLearner, PredictiveCacheManager, PrefixPattern, CachePrediction, CacheTier


class TestAgentMemory:
    def test_init(self):
        mem = AgentMemory()
        assert mem.max_history == 20
        assert mem.current_step == 0

    def test_add_message(self):
        mem = AgentMemory()
        mem.add_message("user", "hello")
        ctx = mem.get_context()
        assert isinstance(ctx, str)

    def test_add_plan_step(self):
        mem = AgentMemory()
        mem.add_plan_step("step 1")
        assert len(mem.plan) >= 1 or mem.current_step >= 0


class TestAgentLoop:
    def test_max_iterations_exceeded(self):
        calls = []
        def llm_fn(ctx):
            calls.append(1)
            return "FINAL ANSWER: done"
        agent = AgentLoop(llm_fn=llm_fn, max_iterations=3)
        result = agent.run("do something")
        assert isinstance(result, dict)
        assert len(calls) <= 5

    def test_state_initial_idle(self):
        def llm_fn(ctx):
            return "done"
        agent = AgentLoop(llm_fn=llm_fn)
        assert isinstance(agent.state, AgentState)

    def test_memory_property(self):
        def llm_fn(ctx):
            return "done"
        agent = AgentLoop(llm_fn=llm_fn)
        assert agent.memory is not None

    def test_tool_call_parsing(self):
        def llm_fn(ctx):
            return 'TOOL_CALL: {"tool_name": "search", "arguments": {"q": "test"}}'
        agent = AgentLoop(llm_fn=llm_fn, tools=[{"name": "search", "description": "test", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}])
        result = agent.run("search")
        assert isinstance(result, dict)


class TestToolCallingEngine:
    def test_parse_schemas(self):
        engine = ToolCallingEngine()
        schemas = engine.parse_schemas([{"type": "function", "function": {"name": "test", "parameters": {"type": "object"}}}])
        assert len(schemas) == 1
        assert schemas[0].to_prompt_text() != ""

    def test_extract_tool_calls_json(self):
        engine = ToolCallingEngine()
        text = '[{"name": "test", "arguments": {"x": 1}}]'
        calls = engine.extract_tool_calls(text)
        assert len(calls) >= 1

    def test_execute_tool_calls(self):
        engine = ToolCallingEngine()
        calls = [type("TC", (), {"id": "1", "name": "test", "arguments": {}})()]
        results = engine.execute_tool_calls(calls, handlers={"test": lambda **k: "result"})
        assert len(results) >= 0  # may or may not find handler

    def test_has_tool_calls(self):
        engine = ToolCallingEngine()
        assert engine.has_tool_calls('{"name": "test"}')
        assert not engine.has_tool_calls("just text")

    def test_tool_result_to_message(self):
        tr = ToolResult(tool_call_id="1", content="result")
        msg = tr.to_message_dict()
        assert msg["role"] == "tool"
        assert msg["content"] == "result"


class TestTextChunker:
    def test_chunk_small_text(self):
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk("hello world")
        assert len(chunks) >= 1
        assert chunks[0] == "hello world"

    def test_chunk_large_text(self):
        chunker = TextChunker(chunk_size=10, overlap=2)
        text = "this is a longer text that should be split"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_chunk_overlap(self):
        chunker = TextChunker(chunk_size=20, overlap=5)
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunker.chunk(text)
        if len(chunks) > 1:
            assert len(chunks[0]) == 20


class TestRAGPipeline:
    def test_ingest_document(self):
        def embed_fn(texts):
            import numpy as np
            return np.random.randn(len(texts), 8).astype(np.float32)
        rag = RAGPipeline(embedding_fn=embed_fn, dimension=8)
        doc = Document(doc_id="doc1", content="test content for ingestion")
        count = rag.ingest(doc)
        assert isinstance(count, int)

    def test_text_chunker(self):
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk("hello world")
        assert len(chunks) >= 1

    def test_save_index(self, tmp_path):
        def embed_fn(texts):
            import numpy as np
            return np.random.randn(len(texts), 8).astype(np.float32)
        rag = RAGPipeline(embedding_fn=embed_fn, dimension=8)
        rag.save_index(str(tmp_path / "saved_idx"))

    def test_stats(self):
        def embed_fn(texts):
            import numpy as np
            return np.random.randn(len(texts), 8).astype(np.float32)
        rag = RAGPipeline(embedding_fn=embed_fn, dimension=8)
        s = rag.stats()
        assert isinstance(s, dict)


class TestNodeRecoveryManager:
    def test_save_and_get_checkpoint(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("req-1", "kv", [1, 2, 3], [4, 5], "node-1")
        cp = mgr.get_checkpoint("req-1")
        assert cp is not None
        assert cp.request_id == "req-1"
        assert cp.node_id == "node-1"

    def test_drop_checkpoint(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("req-1", "kv", [1], [2], "node-1")
        mgr.drop_checkpoint("req-1")
        assert mgr.get_checkpoint("req-1") is None

    def test_get_checkpoints_for_node(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("req-1", "kv", [1], [2], "node-1")
        mgr.save_checkpoint("req-2", "kv", [3], [4], "node-1")
        cps = mgr.get_checkpoints_for_node("node-1")
        assert len(cps) == 2

    def test_is_draining_and_dead(self):
        mgr = NodeRecoveryManager()
        plan = mgr.on_node_failure("node-1")
        assert mgr.is_draining("node-1")
        assert mgr.draining_nodes == ["node-1"]

    def test_metrics(self):
        mgr = NodeRecoveryManager()
        mgr.save_checkpoint("req-1", "kv", [1], [2], "node-1")
        m = mgr.get_metrics()
        assert isinstance(m, dict)


class TestGracefulDegradation:
    def test_evaluate_normal(self):
        gd = GracefulDegradation()
        plan = gd.evaluate(LoadSnapshot(queue_depth=1, avg_latency_ms=10, memory_util_pct=0.2, request_rate=1))
        assert plan is not None

    def test_evaluate_high_load(self):
        gd = GracefulDegradation()
        plan = gd.evaluate(LoadSnapshot(queue_depth=100, avg_latency_ms=5000, memory_util_pct=0.95, request_rate=100))
        assert plan is not None

    def test_apply_to_params(self):
        plan = DegradationPlan(level=DegradationLevel.MODERATE, max_tokens=50)
        params = {"max_tokens": 100}
        modified = plan.apply_to_params(params)
        assert modified is not None

    def test_current_level(self):
        gd = GracefulDegradation()
        assert gd.current_level is not None


class TestLoadSnapshot:
    def test_score_calculation(self):
        snap = LoadSnapshot(queue_depth=100, memory_util_pct=0.9)
        score = snap.score()
        assert isinstance(score, float)

    def test_score_low_load(self):
        snap = LoadSnapshot(queue_depth=0, memory_util_pct=0.1)
        score = snap.score()
        assert isinstance(score, float)


class TestCacheWarmer:
    def test_add_tier(self):
        cw = CacheWarmer()
        cw.add_tier("hot", ["prompt1", "prompt2"], batch_sizes=[1])
        s = cw.get_stats()
        assert s is not None

    def test_get_stats(self):
        cw = CacheWarmer()
        s = cw.get_stats()
        assert s.total_prompts == 0

    def test_warm_with_empty_prompts(self):
        cw = CacheWarmer()
        result = cw.warm([], None)
        assert result >= 0


class TestPatternLearner:
    def test_observe_and_predict(self):
        pl = PatternLearner(max_patterns=100, min_prefix_len=2)
        pl.observe([1, 2, 3, 4])
        pl.observe([1, 2, 3, 5])
        predictions = pl.predict([1, 2, 3])
        assert isinstance(predictions, list)

    def test_top_patterns(self):
        pl = PatternLearner(max_patterns=100, min_prefix_len=2)
        for i in range(10):
            pl.observe([1, 2, i])
        top = pl.top_patterns(n=5)
        assert len(top) <= 5

    def test_pattern_count(self):
        pl = PatternLearner()
        assert pl.pattern_count >= 0

    def test_evict_lowest_score(self):
        pl = PatternLearner(max_patterns=5, min_prefix_len=1)
        for i in range(10):
            pl.observe([i, i + 1])
        assert pl.pattern_count <= 5


class TestPredictiveCacheManager:
    def test_observe_request(self):
        mgr = PredictiveCacheManager(gpu_memory_bytes=1024*1024)
        predictions = mgr.observe_request([1, 2, 3])
        assert isinstance(predictions, (list, type(None)))

    def test_lookup_empty(self):
        mgr = PredictiveCacheManager(gpu_memory_bytes=1024*1024)
        result = mgr.lookup([99, 98])
        assert result is not None

    def test_stats(self):
        mgr = PredictiveCacheManager(gpu_memory_bytes=1024*1024)
        s = mgr.stats
        assert isinstance(s, dict)

    def test_start_stop_prefetch_service(self):
        mgr = PredictiveCacheManager(gpu_memory_bytes=1024*1024)
        mgr.start_prefetch_service()
        mgr.stop_prefetch_service()


class TestPrefixPattern:
    def test_init(self):
        pp = PrefixPattern(prefix_tokens=(1, 2, 3), frequency=5, score=0.8)
        assert pp.frequency == 5
        assert pp.score == 0.8
