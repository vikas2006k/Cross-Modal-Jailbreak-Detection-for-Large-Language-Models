#!/bin/bash
# ==============================================================================
# Cross-Modal Jailbreak Detection (CMJD) v1.0 - Linux / macOS Launcher
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR" || exit 1

echo "[*] Starting FastAPI Backend on http://127.0.0.1:8001..."
source venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 &
BACKEND_PID=$!

echo "[*] Waiting 4 seconds for model initialization..."
sleep 4

echo "[*] Starting React 19 Frontend on http://127.0.0.1:5173..."
cd frontend || exit 1
npm run dev &
FRONTEND_PID=$!

echo "=============================================================================="
echo " CMJD System Online!"
echo " Web Dashboard:   http://127.0.0.1:5173/"
echo " API Docs:        http://127.0.0.1:8001/docs"
echo " PIDs:            Backend ($BACKEND_PID), Frontend ($FRONTEND_PID)"
echo "=============================================================================="

wait
