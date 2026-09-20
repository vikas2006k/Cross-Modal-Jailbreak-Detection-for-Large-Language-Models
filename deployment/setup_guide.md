# Comprehensive Local Setup & Execution Guide

This guide details how to install, configure, and run the Cross-Modal Jailbreak Detection (CMJD) system from scratch on **Windows**, **Linux (Ubuntu/Debian)**, and **macOS**.

---

## 1. System Requirements

| Specification | Minimum Requirement | Recommended Production |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Ubuntu 22.04 LTS |
| **Python** | 3.10 or 3.11 | 3.10.12 |
| **Node.js** | v18.0.0+ | v20 LTS / v22 LTS |
| **RAM** | 8 GB | 16 GB+ |
| **GPU (Optional)** | None (Runs on CPU) | NVIDIA GPU with 8GB+ VRAM & CUDA 11.8 / 12.1 |
| **Free Disk Space** | 6 GB | 15 GB |

---

## 2. Windows Installation (PowerShell)

### Step 1: Clone Repository
```powershell
git clone https://github.com/vikas2006k/Cross-Modal-Jailbreak-Detection-for-Large-Language-Models.git
cd "Cross-Modal-Jailbreak-Detection-for-Large-Language-Models"
```

### Step 2: Set Up Python Virtual Environment
```powershell
# Create venv
python -m venv venv

# Activate venv (PowerShell execution policy workaround if needed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install Python requirements
pip install -r requirements.txt
```

### Step 3: Install Frontend Node Dependencies
```powershell
cd frontend
npm.cmd install
cd ..
```

### Step 4: Run the Complete System
Open two terminal windows:

**Terminal 1 — Backend Service:**
```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
```
*Verify: Open `http://127.0.0.1:8001/health` in your browser. Expected output: `{"status":"healthy", "model_loaded":true}`.*

**Terminal 2 — Frontend Dev Server:**
```powershell
cd frontend
npm.cmd run dev
```
*Open `http://127.0.0.1:5173/` in your browser.*

---

## 3. Linux Installation (Ubuntu / Debian)

### Step 1: System Packages
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git curl build-essential libgl1-mesa-glx libglib2.0-0
```

### Step 2: Install Node.js (via NodeSource)
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### Step 3: Clone & Virtual Environment
```bash
git clone https://github.com/vikas2006k/Cross-Modal-Jailbreak-Detection-for-Large-Language-Models.git
cd Cross-Modal-Jailbreak-Detection-for-Large-Language-Models

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Frontend Installation & Launch
```bash
cd frontend && npm install && cd ..

# Launch backend in background
source venv/bin/activate
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8001 &

# Launch frontend
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

---

## 4. macOS Installation (Apple Silicon M1/M2/M3 & Intel)

### Step 1: Install Homebrew Dependencies
```bash
brew install python@3.10 node git
```

### Step 2: Clone & Virtual Environment
```bash
git clone https://github.com/vikas2006k/Cross-Modal-Jailbreak-Detection-for-Large-Language-Models.git
cd Cross-Modal-Jailbreak-Detection-for-Large-Language-Models

python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
*(PyTorch will automatically utilize Apple Silicon Metal Performance Shaders - MPS if available).*

### Step 3: Run Services
```bash
# Terminal 1:
source venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001

# Terminal 2:
cd frontend
npm install
npm run dev
```

---

## 5. Verification Checklist

1. **Backend Health Check**:
   ```bash
   curl http://127.0.0.1:8001/health
   ```
2. **Text Inference Test**:
   ```bash
   curl -X POST http://127.0.0.1:8001/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello world"}'
   ```
3. **Frontend Production Build Check**:
   ```bash
   cd frontend
   npm run build
   ```
