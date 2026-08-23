@echo off
cd /d D:\distributed-llm
echo === STEP 1: First batch of tests ===
python -m pytest tests/core/test_kv_cache.py tests/core/test_speculative_decoder.py tests/core/test_event_bus.py -q --tb=short
set RESULT1=%ERRORLEVEL%
echo.

echo === STEP 2: Second batch of tests ===
python -m pytest tests/core/test_agentic_router.py tests/core/test_autonomous_healer.py tests/core/test_prompt_library.py tests/api/test_prompt_injection.py -q --tb=short
set RESULT2=%ERRORLEVEL%
echo.

echo === STEP 3: Import check ===
python -c "import sys; sys.path.insert(0,'src'); from distllm.core.agentic_router import AgenticRouter; from distllm.core.autonomous_healer import AutonomousHealer; from distllm.core.prompt_library import PromptRepository; from distllm.api.prompt_injection import PromptInjectionMiddleware, FastInjectionClassifier; print('IMPORTS OK')"
set RESULT3=%ERRORLEVEL%
echo.

echo === SUMMARY ===
if %RESULT1%==0 (echo STEP 1: PASS) else (echo STEP 1: FAIL)
if %RESULT2%==0 (echo STEP 2: PASS) else (echo STEP 2: FAIL)
if %RESULT3%==0 (echo STEP 3: PASS) else (echo STEP 3: FAIL)
