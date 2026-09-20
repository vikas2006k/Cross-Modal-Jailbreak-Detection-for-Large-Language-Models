# Cross-Modal Jailbreak Detection for Large Language Models (CMJD-LLM)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end multi-modal security framework designed to detect, analyze, and mitigate cross-modal jailbreak and prompt injection attacks targeting Vision-Language Models (VLMs) and Large Language Models (LLMs).

---

## 📌 Motivation & Threat Landscape

Modern multi-modal AI systems accept diverse input modalities: free-form text, raster images, document PDFs, and encoded QR codes. Adversaries exploit cross-modal seams through:
- **Typographic & Steganographic Attacks**: Rendering malicious text prompts as perturbed images or ASCII art to bypass pure text filters.
- **Multimodal Prompt Injections**: Embedding malicious system overrides within PDF attachments or QR codes.
- **Visual Meme / Context Split Attacks**: Splitting harmful intent across visual and textual channels such that neither modality triggers individual single-modal safety classifiers.

**CMJD-LLM** resolves this by fusing text-level transformer representations (DistilBERT) with visual-semantic embeddings (CLIP), alongside dedicated document and code parsers, passing fused features through an explainable AI risk scoring engine.

---

## 🗺️ Project Roadmap (12 Phases)

| Phase | Goal | Key Artifacts | Status |
| :--- | :--- | :--- | :---: |
| **Phase 0** | **Project Setup** | Directory structure, requirements, `.gitignore`, README | **Completed** |
| **Phase 1** | **Dataset Creation** | CMJD-30K multimodal dataset (Alpaca, AdvBench, QR, PDF, Adv Images) | Pending |
| **Phase 2** | **Data Preprocessing** | OCR extraction, PDF text, QR decoding, feature pipelines | Pending |
| **Phase 3** | **Text Model Training** | DistilBERT jailbreak detector checkpoints & tokenizer | Pending |
| **Phase 4** | **Image Model Training**| CLIP vision jailbreak detector & adversarial checkpoints | Pending |
| **Phase 5** | **Cross-Modal Fusion** | Multi-modal feature fusion engine & attack classification | Pending |
| **Phase 6** | **Risk Scoring Engine** | Rule-based & confidence-weighted risk evaluation | Pending |
| **Phase 7** | **Explainable AI (XAI)**| GradCAM visual heatmaps, SHAP token attribution | Pending |
| **Phase 8** | **Backend API** | FastAPI high-throughput inference service | Pending |
| **Phase 9** | **Frontend Dashboard** | React security dashboard & interactive scan views | Pending |
| **Phase 10** | **Testing & Benchmarks**| Confusion matrix, ROC curve, latency, evaluation report | Pending |
| **Phase 11** | **Deployment & Release**| Docker, Nginx, docker-compose, v1.0 release | Pending |

---

## 🏗️ Architecture Pipeline

```text
                  ┌─────────────────────────────────────────────────────┐
                  │                   User Input                        │
                  │       (Text, Image, PDF Document, QR Code)          │
                  └──────────────────────────┬──────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          ┌───────────────────────┐                     ┌───────────────────────┐
          │  Text & Document Parse│                     │ Vision Preprocessing  │
          │ (PDF Extract, OCR, QR)│                     │ (Augmentation, Normal)│
          └───────────┬───────────┘                     └───────────┬───────────┘
                      │                                             │
                      ▼                                             ▼
          ┌───────────────────────┐                     ┌───────────────────────┐
          │ DistilBERT Embeddings │                     │   CLIP Embeddings     │
          │   (Text Modality)     │                     │   (Image Modality)    │
          └───────────┬───────────┘                     └───────────┬───────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │ Cross-Modal Fusion Engine │
                               │  (Multi-Head Cross Attn)  │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │ Security & Risk Engine    │
                               │ (Rules + Confidence Score)│
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          ┌───────────────────────────┐               ┌───────────────────────────┐
          │   Verdict: PASS / BLOCK   │               │ Explainable AI (XAI)      │
          │ Risk Score (0-100), Class │               │ GradCAM Heatmap + SHAP    │
          └───────────────────────────┘               └───────────────────────────┘
```

---

## 📂 Repository Structure

```text
CrossModal-Jailbreak-Detection/
├── backend/                  # FastAPI service, fusion engine, risk scoring, XAI
├── frontend/                 # React UI dashboard for monitoring and scanning
├── data/                     # CMJD-30K dataset structure
│   ├── annotations/          # Ground truth labels & splits
│   ├── external/             # Base datasets (AdvBench, Alpaca, etc.)
│   ├── raw/                  # Raw multimodal samples (images, PDFs, QRs)
│   └── processed/            # Extracted features, embeddings, OCR text
├── models/                   # Model architectures and saved checkpoints
│   ├── text_detector/        # DistilBERT classifier
│   └── image_detector/       # CLIP visual classifier
├── notebooks/                # Exploratory analysis & experiments
├── scripts/                  # Preprocessing, data builders, utilities
├── reports/                  # Evaluation metrics, ROC curves, confusion matrices
├── docs/                     # Technical specifications and API docs
├── tests/                    # Unit, integration, and security test suites
├── docker/                   # Dockerfile, docker-compose, and Nginx config
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusions
└── README.md                 # Project overview and roadmap
```

---

## ⚡ Quickstart

### 1. Clone & Setup Environment

```bash
git clone https://github.com/vikas2006k/Cross-Modal-Jailbreak-Detection-for-Large-Language-Models.git
cd "Cross-Modal Jailbreak Detection for Large Language Models"

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Follow Development Phases
The project is built systematically across the 12 phases detailed above. To proceed to **Phase 1 (Dataset Creation)**, refer to instructions in `docs/` and run the corresponding scripts under `scripts/`.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
