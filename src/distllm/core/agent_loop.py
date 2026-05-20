"""Agentic framework with tool calling, memory, and planning.

Provides a built-in agent loop (ReAct-style) with:
- Working memory for context retention
- Planning/scheduling capabilities
- Tool composition and chaining
- Self-reflection/correction loop
"""
import time
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    PLANNING = "planning"
    DONE = "done"
    FAILED = "failed"


@dataclass
class AgentMemory:
    """Working memory for the agent."""
    conversation: list[dict] = field(default_factory=list)
    scratchpad: str = ""
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    tool_results: list[dict] = field(default_factory=list)
    max_history: int = 20
    
    def add_message(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content, "timestamp": time.time()})
        if len(self.conversation) > self.max_history:
            self.conversation = self.conversation[-self.max_history:]
    
    def add_plan_step(self, step: str) -> None:
        self.plan.append(step)
    
    def get_context(self) -> str:
        """Get formatted context for the LLM."""
        parts = []
        if self.plan:
            parts.append(f"Plan: {', '.join(self.plan)}")
            if self.current_step < len(self.plan):
                parts.append(f"Current step: {self.plan[self.current_step]}")
        if self.scratchpad:
            parts.append(f"Scratchpad: {self.scratchpad}")
        if self.tool_results:
            last = self.tool_results[-1]
            parts.append(f"Last tool result: {last.get('result', 'N/A')}")
        return "\n\n".join(parts)


@dataclass
class ToolCall:
    """A tool call made by the agent."""
    tool_name: str
    arguments: dict
    result: Any = None
    success: bool = False
    error: str | None = None


class AgentLoop:
    """ReAct-style agent loop with memory and planning.
    
    Usage:
        agent = AgentLoop(llm_fn=generate, tools=[search, calculator])
        result = agent.run("What's the weather in Tokyo?")
    """
    
    def __init__(
        self,
        llm_fn: Callable,
        tools: list[dict] | None = None,
        max_iterations: int = 10,
        reflection_enabled: bool = True,
    ):
        self._llm_fn = llm_fn
        self._tools = {t["name"]: t for t in (tools or [])}
        self._max_iterations = max_iterations
        self._reflection_enabled = reflection_enabled
        self._memory = AgentMemory()
        self._state = AgentState.IDLE
        self._iteration = 0
    
    def run(self, goal: str) -> dict:
        """Run the agent loop to achieve a goal.
        
        Args:
            goal: The goal/task to accomplish.
            
        Returns:
            Result dict with answer, tool_calls, iterations.
        """
        self._state = AgentState.PLANNING
        self._memory.add_message("user", goal)
        self._iteration = 0
        
        # Phase 1: Create plan
        plan = self._create_plan(goal)
        self._memory.plan = plan
        logger.info(f"Agent plan: {plan}")
        
        # Phase 2: Execute plan
        self._state = AgentState.THINKING
        for step_idx, step in enumerate(plan):
            self._memory.current_step = step_idx
            self._state = AgentState.ACTING
            
            result = self._execute_step(step)
            if result.get("failed"):
                if self._reflection_enabled:
                    self._state = AgentState.REFLECTING
                    recovery = self._reflect_and_recover(step, result.get("error", ""))
                    if recovery:
                        result = self._execute_step(step)
                
                if result.get("failed"):
                    self._state = AgentState.FAILED
                    return {
                        "answer": f"Failed at step {step_idx + 1}: {step}",
                        "error": result.get("error"),
                        "tool_calls": self._memory.tool_results,
                        "iterations": self._iteration,
                    }
            
            self._iteration += 1
            if self._iteration >= self._max_iterations:
                break
        
        # Phase 3: Synthesize answer
        self._state = AgentState.THINKING
        answer = self._synthesize_answer(goal)
        self._state = AgentState.DONE
        self._memory.add_message("assistant", answer)
        
        return {
            "answer": answer,
            "tool_calls": self._memory.tool_results,
            "iterations": self._iteration,
            "plan": plan,
        }
    
    def _create_plan(self, goal: str) -> list[str]:
        """Create a plan to achieve the goal."""
        prompt = f"""Break down this goal into clear, actionable steps:

Goal: {goal}

Return a JSON array of steps. Example: ["Search for information", "Analyze results", "Synthesize answer"]
"""
        response = self._llm_fn(prompt)
        try:
            # Extract JSON array from response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        # Fallback: single step
        return [goal]
    
    def _execute_step(self, step: str) -> dict:
        """Execute a single plan step."""
        context = self._memory.get_context()
        prompt = f"""Execute this step:

Step: {step}
Context: {context}

Available tools: {list(self._tools.keys())}

If you need to use a tool, respond with:
TOOL_CALL: <tool_name>(<arguments>)

Otherwise, respond with the step result.
"""
        response = self._llm_fn(prompt)
        
        # Check for tool call
        if "TOOL_CALL:" in response:
            tool_call = self._parse_tool_call(response)
            if tool_call:
                result = self._call_tool(tool_call)
                self._memory.tool_results.append({
                    "tool": tool_call.tool_name,
                    "arguments": tool_call.arguments,
                    "result": result,
                    "success": True,
                })
                self._memory.scratchpad += f"\nStep: {step}\nResult: {result}"
                return {"success": True, "result": result}
        
        self._memory.scratchpad += f"\nStep: {step}\nResult: {response}"
        return {"success": True, "result": response}
    
    def _parse_tool_call(self, response: str) -> ToolCall | None:
        """Parse a tool call from LLM response."""
        try:
            start = response.index("TOOL_CALL: ") + len("TOOL_CALL: ")
            call_str = response[start:].strip()
            
            # Parse tool_name(args)
            paren_idx = call_str.index("(")
            tool_name = call_str[:paren_idx]
            args_str = call_str[paren_idx + 1:call_str.rindex(")")]
            
            arguments = json.loads(args_str) if args_str.strip() != "{}" else {}
            return ToolCall(tool_name=tool_name, arguments=arguments)
        except (ValueError, json.JSONDecodeError):
            return None
    
    def _call_tool(self, tool_call: ToolCall) -> Any:
        """Execute a tool call."""
        tool = self._tools.get(tool_call.tool_name)
        if tool is None:
            return f"Error: Unknown tool '{tool_call.tool_name}'"
        
        try:
            result = tool["handler"](**tool_call.arguments)
            tool_call.success = True
            tool_call.result = result
            return result
        except Exception as e:
            tool_call.success = False
            tool_call.error = str(e)
            return f"Error: {e}"
    
    def _reflect_and_recover(self, step: str, error: str) -> bool:
        """Reflect on failure and attempt recovery."""
        prompt = f"""The following step failed:

Step: {step}
Error: {error}

Suggest a recovery strategy. Respond with "RETRY" to retry or "SKIP" to skip.
"""
        response = self._llm_fn(prompt).strip().upper()
        return "RETRY" in response
    
    def _synthesize_answer(self, goal: str) -> str:
        """Synthesize a final answer from collected information."""
        context = self._memory.get_context()
        prompt = f"""Based on the information gathered, answer the original goal:

Goal: {goal}

Context: {context}

Provide a clear, concise answer.
"""
        return self._llm_fn(prompt)
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    @property
    def memory(self) -> AgentMemory:
        return self._memory
