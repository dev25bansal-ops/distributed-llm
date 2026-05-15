@echo off
echo ============================================
echo Distributed LLM - REST API Server
echo ============================================
echo.

set MODEL=%1
if "%MODEL%"=="" set MODEL=roneneldan/TinyStories-1M

echo Model: %MODEL%
echo API: http://localhost:8000
echo.
echo Endpoints:
echo   GET  /v1/models
echo   POST /v1/chat/completions
echo   POST /v1/completions
echo   GET  /health
echo.

distllm-api --model %MODEL% --local --port 8000

pause
