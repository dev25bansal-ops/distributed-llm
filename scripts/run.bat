@echo off
echo ============================================
echo Distributed LLM - Production Server
echo ============================================
echo.

set MODEL=%1
if "%MODEL%"=="" set MODEL=microsoft/Phi-3-mini-4k-instruct

echo Model: %MODEL%
echo.
echo Choose mode:
echo   1. Local (full model on this machine)
echo   2. API Server (OpenAI-compatible REST API)
echo   3. Coordinator + Workers (distributed)
echo.

set /p MODE="Enter mode (1/2/3): "

if "%MODE%"=="1" (
    echo.
    echo Starting local mode...
    distllm --model %MODEL% --local --chat
) else if "%MODE%"=="2" (
    echo.
    echo Starting API server on port 8000...
    distllm-api --model %MODEL% --local --port 8000
) else if "%MODE%"=="3" (
    echo.
    echo Starting distributed mode...
    echo.
    echo Open new terminals for each node:
    echo.
    echo Terminal 1 (Worker 0):
    echo   distllm-node --node-id node_0 --model %MODEL% --start-layer 0 --end-layer 3 --total-layers 8 --port 50051
    echo.
    echo Terminal 2 (Worker 1):
    echo   distllm-node --node-id node_1 --model %MODEL% --start-layer 4 --end-layer 7 --total-layers 8 --port 50052
    echo.
    echo Terminal 3 (Coordinator):
    echo   distllm --model %MODEL% --nodes localhost:50051:0:3 localhost:50052:4:7 --total-layers 8
    echo.
    pause
) else (
    echo Invalid mode.
    pause
)
