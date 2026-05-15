@echo off
echo ============================================
echo Distributed LLM - Single Machine Test
echo ============================================
echo.

REM Set defaults
set MODEL=%1
if "%MODEL%"=="" set MODEL=roneneldan/TinyStories-1M

echo Model: %MODEL%
echo.

REM Start worker 1 (layers 0-3 for 8-layer models)
start "Worker Node 0" distllm-node --node-id node_0 --model %MODEL% --start-layer 0 --end-layer 3 --total-layers 8 --port 50051

timeout /t 5 /nobreak >nul

REM Start worker 2 (layers 4-7 for 8-layer models)
start "Worker Node 1" distllm-node --node-id node_1 --model %MODEL% --start-layer 4 --end-layer 7 --total-layers 8 --port 50052

timeout /t 5 /nobreak >nul

REM Start coordinator
start "Coordinator" distllm --model %MODEL% --nodes localhost:50051:0:3 localhost:50052:4:7 --total-layers 8

echo.
echo All processes started!
echo - Worker 0: port 50051 (layers 0-3)
echo - Worker 1: port 50052 (layers 4-7)
echo - Coordinator: port 50050
echo.
pause
