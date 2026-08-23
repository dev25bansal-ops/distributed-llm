Set-Location D:/distributed-llm

# Step 1
Write-Output "=== STEP 1 ==="
$r1 = & python -m pytest tests/core/test_kv_cache.py tests/core/test_speculative_decoder.py tests/core/test_event_bus.py -q --tb=short 2>&1
$r1 | Out-String | Set-Content step1_output.txt
$exit1 = $LASTEXITCODE
Add-Content step1_output.txt "EXIT_CODE: $exit1"

# Step 2
Write-Output "=== STEP 2 ==="
$r2 = & python -m pytest tests/core/test_agentic_router.py tests/core/test_autonomous_healer.py tests/core/test_prompt_library.py tests/api/test_prompt_injection.py -q --tb=short 2>&1
$r2 | Out-String | Set-Content step2_output.txt
$exit2 = $LASTEXITCODE
Add-Content step2_output.txt "EXIT_CODE: $exit2"

# Step 3
Write-Output "=== STEP 3 ==="
$r3 = & python -c "import sys; sys.path.insert(0,'src'); from distllm.core.agentic_router import AgenticRouter; from distllm.core.autonomous_healer import AutonomousHealer; from distllm.core.prompt_library import PromptRepository; from distllm.api.prompt_injection import PromptInjectionMiddleware, FastInjectionClassifier; print('IMPORTS OK')" 2>&1
$r3 | Out-String | Set-Content step3_output.txt
$exit3 = $LASTEXITCODE
Add-Content step3_output.txt "EXIT_CODE: $exit3"

# Summary
$summary = @"
STEP1: $(if ($exit1 -eq 0) { 'PASS' } else { 'FAIL' })
STEP2: $(if ($exit2 -eq 0) { 'PASS' } else { 'FAIL' })
STEP3: $(if ($exit3 -eq 0) { 'PASS' } else { 'FAIL' })
"@
$summary | Out-String | Set-Content summary.txt
Write-Output $summary
