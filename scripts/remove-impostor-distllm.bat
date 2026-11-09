@echo off
REM Removes the impostor "distllm 1.0.2" package from system Python.
REM
REM That package is NOT this project. It installs distllm/cli.py into
REM C:\Python314\Lib\site-packages and a distllm.exe into C:\Python314\Scripts,
REM which sits EARLIER on PATH than the real CLI's install directory. The result
REM is that `distllm` runs the impostor and dies with:
REM     TypeError: 'module' object is not callable
REM
REM The real CLI (distributed-llm) is installed under the user site-packages and
REM is unaffected by this script. Run this as Administrator.

echo === Removing impostor package ===
py -3.14 -m pip uninstall -y distllm

echo.
echo === Removing leftover launchers from C:\Python314\Scripts ===
if exist "C:\Python314\Scripts\distllm.exe"           del /f /q "C:\Python314\Scripts\distllm.exe"
if exist "C:\Python314\Scripts\distllm-api.exe"       del /f /q "C:\Python314\Scripts\distllm-api.exe"
if exist "C:\Python314\Scripts\distllm-node.exe"      del /f /q "C:\Python314\Scripts\distllm-node.exe"
if exist "C:\Python314\Scripts\distllm-coordinator.exe" del /f /q "C:\Python314\Scripts\distllm-coordinator.exe"

echo.
echo === Removing leftover package directories ===
if exist "C:\Python314\Lib\site-packages\distllm"               rmdir /s /q "C:\Python314\Lib\site-packages\distllm"
if exist "C:\Python314\Lib\site-packages\distllm-1.0.2.dist-info" rmdir /s /q "C:\Python314\Lib\site-packages\distllm-1.0.2.dist-info"

echo.
echo === Verifying ===
where distllm
distllm --version

echo.
echo If the line above says "distllm 0.6.2.2", you are done.
pause
