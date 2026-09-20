@echo off
echo ==============================================================================
echo  Cross-Modal Jailbreak Detection (CMJD) v1.0 - Full Stack Launcher
echo ==============================================================================

cd /d "%~dp0\.."

echo [*] Starting FastAPI Backend on http://127.0.0.1:8001...
start "CMJD-Backend" cmd /k "call .\venv\Scripts\activate && python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001"

echo [*] Waiting 4 seconds for model initialization...
timeout /t 4 /nobreak >nul

echo [*] Starting React 19 Frontend on http://127.0.0.1:5173...
start "CMJD-Frontend" cmd /k "cd frontend && call npm.cmd run dev"

echo ==============================================================================
echo  CMJD System Online! 
echo  Access Web Dashboard at: http://127.0.0.1:5173/
echo  API Documentation at:    http://127.0.0.1:8001/docs
echo ==============================================================================
pause
