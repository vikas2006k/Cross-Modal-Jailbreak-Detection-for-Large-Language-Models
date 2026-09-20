# Cross-Modal Jailbreak Detection (CMJD) — Release Package v1.0

**Release Tag**: `v1.0-release`  
**Date**: September 20, 2026  
**License**: MIT License  
**Author**: Vikas K.  

---

## Release Directory Index

This release package consolidates all artifacts, research documentation, hackathon presentations, and deployment assets for the **Cross-Modal Jailbreak Detection** project.

```text
project_release/
├── paper/                          # Part 1: IEEE Research Publication
│   ├── ieee_paper.md               # 19-section full conference paper draft
│   ├── figures/                    # 62 high-resolution 300 DPI figures
│   └── tables/                     # Formatted LaTeX & Markdown ablation tables
├── presentation/                   # Part 2: SAP Hackathon Pitch
│   ├── sap_hackathon_presentation.md# 16-slide professional pitch deck
│   └── speaker_notes.md            # Word-for-word spoken narration script
├── README.md                       # Part 3: GitHub Master Showcase
├── demo/                           # Part 4: Demo Scripts & Judge Q&A
│   ├── 5_minute_demo_script.md     # 5-minute executive pitch script
│   ├── 10_minute_demo_script.md    # 10-minute technical deep-dive script
│   └── judge_QA.md                 # 25 tough judge Q&A responses
├── docs/                           # Part 5: Viva & Placement Preparation
│   ├── viva_questions.md           # 30 core defense questions & answers
│   ├── architecture_questions.md   # 25 system architecture questions & answers
│   ├── ml_questions.md             # 25 deep learning & vision theory questions
│   └── deployment_questions.md     # 25 DevOps & deployment questions & answers
└── deployment/                     # Part 6: Deployment & Infrastructure
    ├── setup_guide.md              # Zero-to-hero local setup guide
    ├── docker_guide.md             # Containerization & Kubernetes guide
    ├── troubleshooting.md          # FAQ & issue troubleshooting
    └── docker/                     # Production Dockerfiles & docker-compose.yml
```

---

## 1-Command Startup Scripts

### Windows (`start_all.bat`)
```bat
@echo off
echo Starting Cross-Modal Jailbreak Detection (CMJD) v1.0...

start "CMJD Backend" cmd /k ".\venv\Scripts\activate && python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001"
timeout /t 3 /nobreak >nul
start "CMJD Frontend" cmd /k "cd frontend && npm run dev"

echo System launching at http://127.0.0.1:5173/
```

### Linux / macOS (`start_all.sh`)
```bash
#!/bin/bash
echo "Starting Cross-Modal Jailbreak Detection (CMJD) v1.0..."

source venv/bin/activate
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 &
BACKEND_PID=$!

sleep 3
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "System running at http://127.0.0.1:5173/ (PIDs: $BACKEND_PID, $FRONTEND_PID)"
```

---

## Key Performance Summary

- **Architecture**: DistilBERT (66M) + OpenAI CLIP ViT-B/32 (Frozen) + EasyOCR CRAFT + Adaptive Gated Fusion
- **CMJD-10K Text Benchmark**: **99.20% Accuracy**, **0.9996 ROC-AUC**, **13.1ms Latency**
- **CMJD-Vision Benchmark**: **99.67% Accuracy**, **1.0000 ROC-AUC**
- **IEEE Multimodal Challenge Benchmark ($N=120$)**:
  - **Accuracy**: **98.33%** (118/120 correct)
  - **Recall**: **100.00%** (0 false negatives)
  - **Precision**: **96.15%**
  - **$F_1$-Score**: **98.04%**
  - **ROC-AUC**: **0.9854**
  - **False Positive Rate**: **2.86%** (reduced from 40.00% unimodal text)
- **Model Checkpoints**: All checkpoints preserved without retraining.
- **Frontend Dashboard**: React 19 + TypeScript + Tailwind CSS v4 running on `127.0.0.1:5173`.
- **Backend API**: FastAPI running on `127.0.0.1:8001`.

---

## Release Readiness Confirmation

Version 1.0 Release Candidate: **READY FOR DEPLOYMENT, SUBMISSION & EVALUATION**.
