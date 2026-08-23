#!/bin/bash
cd /d/distributed-llm

echo "=== STEP 1: First batch ==="
python -m pytest tests/core/test_kv_cache.py tests/core/test_speculative_decoder.py tests/core/test_event_bus.py -q --tb=short > /d/distributed-llm/out1.txt 2>&1
echo "EXIT: $?" >> /d/distributed-llm/out1.txt

echo "=== STEP 2: Second batch ==="
python -m pytest tests/core/test_agentic_router.py tests/core/test_autonomous_healer.py tests/core/test_prompt_library.py tests/api/test_prompt_injection.py -q --tb=short > /d/distributed-llm/out2.txt 2>&1
echo "EXIT: $?" >> /d/distributed-llm/out2.txt

echo "=== STEP 3: Imports ==="
python -c "import sys; sys.path.insert(0,'src'); from distllm.core.agentic_router import AgenticRouter; from distllm.core.autonomous_healer import AutonomousHealer; from distllm.core.prompt_library import PromptRepository; from distllm.api.prompt_injection import PromptInjectionMiddleware, FastInjectionClassifier; print('IMPORTS OK')" > /d/distributed-llm/out3.txt 2>&1
echo "EXIT: $?" >> /d/distributed-llm/out3.txt

echo "DONE" > /d/distributed-llm/runner_done.txt
