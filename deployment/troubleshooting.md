# Deployment Troubleshooting & FAQ Guide

Solutions to common issues encountered during local development, Docker builds, and production deployment.

---

## 1. Port Conflicts (Address Already in Use)

### Symptoms
- Error: `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8001): only one usage of each socket address`
- Error: `Port 5173 is already in use`

### Solutions
- **Windows (PowerShell)**:
  ```powershell
  # Find process occupying port 8001
  netstat -ano | findstr :8001
  # Terminate process by PID (replace 12345 with PID from above)
  taskkill /PID 12345 /F

  # For port 5173
  netstat -ano | findstr :5173
  taskkill /PID <PID> /F
  ```
- **Linux / macOS**:
  ```bash
  sudo lsof -i :8001 | awk 'NR>1 {print $2}' | xargs kill -9
  sudo lsof -i :5173 | awk 'NR>1 {print $2}' | xargs kill -9
  ```

---

## 2. EasyOCR Model Download Failures or Delays

### Symptoms
- Backend hangs during first request with: `Downloading detection model, please wait...`
- `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>`

### Solutions
1. EasyOCR downloads its CRAFT and recognition weights to `~/.EasyOCR/model/` on first use.
2. In restricted or air-gapped networks, pre-download the weights:
   ```bash
   python -c "import easyocr; easyocr.Reader(['en'])"
   ```
3. If SSL certificate errors occur on macOS:
   ```bash
   # Run Python's certificate install script
   /Applications/Python\ 3.10/Install\ Certificates.command
   ```

---

## 3. Windows PowerShell Execution Policy Restrictions

### Symptoms
- Error: `.\venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`

### Solutions
Run PowerShell with process-level policy bypass:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
Or execute CLI commands using `.cmd` extensions directly:
```powershell
cmd /c "npm.cmd run build"
```

---

## 4. CUDA Out of Memory (OOM) Errors

### Symptoms
- `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB`

### Solutions
1. Force CPU fallback by setting environment variable:
   ```bash
   export DEVICE=cpu
   # On Windows PowerShell:
   $env:DEVICE="cpu"
   ```
2. Decrease max sequence length in `backend/config.py` from 128 to 64.
3. In `fusion_engine/inference.py`, ensure tensors are executed inside `with torch.no_grad():` and invoke `torch.cuda.empty_cache()` between batches.

---

## 5. React Frontend: CORS Policy Rejection

### Symptoms
- Browser console error: `Access to XMLHttpRequest at 'http://127.0.0.1:8001/predict' from origin 'http://127.0.0.1:5173' has been blocked by CORS policy`

### Solutions
1. Confirm that `backend/app.py` has `CORSMiddleware` configured with:
   ```python
   allow_origins=["*"]
   ```
2. Verify that the backend is actively running on port 8001 (`http://127.0.0.1:8001/health`).
3. Ensure no trailing slashes differ between request URLs (e.g. use `/predict` without trailing slash).

---

## 6. Docker: Container Fails Healthcheck

### Symptoms
- `docker compose ps` reports status `unhealthy` for `cmjd-backend`.

### Solutions
1. Inspect container logs:
   ```bash
   docker compose -f docker/docker-compose.yml logs cmjd-backend
   ```
2. If OpenCV raises `ImportError: libGL.so.1: cannot open shared object file`, ensure `libgl1` and `libglib2.0-0` are included in `apt-get install` inside `Dockerfile.backend`.
3. Verify that the container's internal healthcheck uses `http://127.0.0.1:8001/health`.
