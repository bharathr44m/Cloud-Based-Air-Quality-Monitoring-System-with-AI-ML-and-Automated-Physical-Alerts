@echo off
title Air Quality ML Backend Launcher
echo ====================================================================
echo Starting Air Quality ML Anomaly Backend (FastAPI)...
echo ====================================================================

cd /d "%~dp0ml_backend"

set "PY_EXE="
if exist ".venv-win\Scripts\python.exe" (
    set "PY_EXE=.venv-win\Scripts\python.exe"
    set "ACT_BAT=.venv-win\Scripts\activate.bat"
) else if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
    set "ACT_BAT=venv\Scripts\activate.bat"
) else (
    echo Error: Virtual environment not found in ml_backend\.venv-win or ml_backend\venv.
    echo Please create virtual environment and install dependencies.
    pause
    exit /b 1
)

echo Activating Python virtual environment...
call "%ACT_BAT%"

echo Starting FastAPI server on http://localhost:8001 ...
"%PY_EXE%" -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

pause
