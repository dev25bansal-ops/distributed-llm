"""Tests for the dynamic pipeline executor."""
from distllm.core.pipeline_executor import PipelineStep, PipelineSpec, PlanCompiler, ExecutionPlan, PipelineExecutor


class TestPipelineStep:
    def test_step_creation(self):
        step = PipelineStep(name="embed", step_type="embedding", model="bge-base")
        assert step.name == "embed"
        assert step.step_type == "embedding"
        assert step.model == "bge-base"

    def test_step_with_fallback(self):
        step = PipelineStep(name="llm", step_type="llm", model="llama-70b", fallback="llama-8b")
        assert step.fallback == "llama-8b"


class TestPipelineSpec:
    def test_spec_creation(self):
        steps = [PipelineStep("embed", "embedding", "bge-base"), PipelineStep("llm", "llm", "llama-70b")]
        spec = PipelineSpec(steps=steps, end_to_end_slo_ms=2000.0)
        assert len(spec.steps) == 2


class TestPlanCompiler:
    def test_compile(self):
        compiler = PlanCompiler()
        spec = PipelineSpec(steps=[PipelineStep("llm", "llm", "llama-70b")], end_to_end_slo_ms=5000.0)
        plan = compiler.compile(spec)
        assert isinstance(plan, ExecutionPlan)

    def test_compile_empty(self):
        compiler = PlanCompiler()
        plan = compiler.compile(PipelineSpec(steps=[], end_to_end_slo_ms=1000.0))
        assert len(plan.steps) == 0


class TestPipelineExecutor:
    def test_execute(self):
        compiler = PlanCompiler()
        executor = PipelineExecutor(coordinator_client=object())
        spec = PipelineSpec(steps=[PipelineStep("llm", "llm", "test-model")], end_to_end_slo_ms=5000.0)
        plan = compiler.compile(spec)
        import asyncio
        result = asyncio.run(executor.execute(plan, input_data="Hello"))
        assert isinstance(result, list)
