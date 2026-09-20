# SentinelGuard AI — Cross-Modal AI Security Platform (Frontend)

A production-grade, dark-cybersecurity themed web application and executive dashboard for real-time Cross-Modal Jailbreak & Prompt Injection Defense.

Built for enterprise threat monitoring, AI safety auditing, and placement portfolio demonstrations.

---

## Technology Stack

- **Frontend Core:** React 19, TypeScript, Vite
- **Styling:** Tailwind CSS (v4), Glassmorphism, Neon Cyber Accent system
- **Visualization:** Recharts (Analytics, Donut, Histograms, Area flows), Lucide Icons
- **Animation & Upload:** Framer Motion, React Dropzone
- **Routing & Networking:** React Router v7, Axios with resilient backend fallback handling
- **Backend API:** FastAPI (Python 3.10+), DistilBERT Sequence Classifier, CLIP ViT-B/32, EasyOCR

---

## Project Structure

```text
frontend/
├── public/
│   ├── figures/               # 62 Publication-grade explainability figures (SHAP, LIME, Attention, Saliency, ROC)
│   └── samples/               # 13 Hard benchmark challenge images (benign slides, injections, ciphers)
├── demo_screenshots/          # High-resolution screenshots of all 7 production pages
│   ├── 01_dashboard_overview.png
│   ├── 02_text_scanner.png
│   ├── 03_image_scanner.png
│   ├── 04_cross_modal_scanner.png
│   ├── 05_explainability_gallery.png
│   ├── 06_analytics_dashboard.png
│   └── 07_reports_page.png
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── ImageDropzone.tsx      # Drag-and-drop uploader with 1-click benchmark presets
│   │   │   ├── LightboxModal.tsx      # Pan & zoom modal viewer for model figures
│   │   │   ├── MetricCard.tsx         # KPI tile with glassmorphism and glowing indicators
│   │   │   ├── RiskGauge.tsx          # SVG semi-circle animated risk gauge (0–100%)
│   │   │   └── VerdictBadge.tsx       # Glowing SAFE (emerald) vs JAILBREAK (crimson) badge
│   │   └── layout/
│   │       ├── Footer.tsx             # Enterprise footer with model architecture attribution
│   │       ├── Navbar.tsx             # Sticky header with live FastAPI /health polling
│   │       └── Sidebar.tsx            # Navigation rail with active guardrail specifications
│   ├── pages/
│   │   ├── Dashboard.tsx              # Overview: KPIs, Model Status, System Benchmark, Recent Scans
│   │   ├── TextScanner.tsx            # Single & Batch prompt detection, SHAP token attribution
│   │   ├── ImageScanner.tsx           # Image injection detection, OCR text, CLIP attention heatmap
│   │   ├── CrossModalScanner.tsx      # Flagship dual-modality scanner with adaptive gating breakdown
│   │   ├── Explainability.tsx         # Filterable gallery (SHAP, LIME, Attention, Saliency, Bounding Boxes)
│   │   ├── Analytics.tsx              # Recharts category donut, bimodal risk histogram, ablation table
│   │   └── Reports.tsx                # Filterable audit log, search, CSV export, print/PDF
│   ├── services/
│   │   ├── api.ts                     # Axios client for /health, /model-info, /predict, /cross-modal-predict
│   │   └── storage.ts                 # LocalStorage telemetry with pre-seeded benchmark records
│   ├── types/
│   │   └── api.ts                     # Strict TypeScript interfaces matching FastAPI backend schemas
│   ├── App.tsx                        # Master application layout and route tree
│   ├── index.css                      # Cyber-dark theme design tokens and custom glassmorphism styles
│   └── main.tsx                       # React 19 application mount entrypoint
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## 7 Core Pages

### 1. Dashboard (Overview)
- **Top KPIs:** Total Scans, Threats Blocked, Safe Prompts Authorized, Average Risk Score, Average Latency.
- **Model Status:** Live health indicator connected to `GET /health` and `GET /model-info`.
- **System Benchmark Summary:** Validation Benchmark (120 Samples) metrics (98.33% Accuracy, 98.54% ROC-AUC, 100% Recall, 3.33% FPR).
- **Recent Activity Stream:** Chronological audit table of evaluated queries.

### 2. Text Scanner
- Evaluates raw prompts via `POST /predict`.
- Interactive prompt presets (DAN 11.0, Instruction Override, Roleplay, Benign Python, Academic Crypto).
- Displays animated semi-circle Risk Gauge, Confidence, Attack Category, and latency.
- **SHAP Token Attribution Map:** Color-coded token tiles highlighting adversarial trigger tokens (+SHAP) vs benign context (-SHAP).
- **Batch Scanner:** Modal supporting multi-line bulk prompt evaluations via `POST /batch-predict`.

### 3. Image Scanner
- Evaluates visual prompt injection documents via `POST /cross-modal-predict`.
- Drag-and-drop uploader or 1-click benchmark sample picker (`safe_educational_00.jpg`, `jailbreak_injection_00.jpg`).
- Deconstructs prediction into:
  - **CLIP ViT-B/32 Vision Branch**
  - **EasyOCR + DistilBERT Typography Branch**
  - **Fused Cross-Modal Verdict**
- Visual attention tab with Grad-CAM saliency activations.

### 4. Cross-Modal Scanner
- Dual-input engine accepting image + optional companion prompt.
- Demonstrates adaptive gating false-positive suppression (e.g. typographic slide risk discounted from 95.8% to 4.1%).
- Real-time **Adaptive Contribution Weight Bar** visualizing $w_v$ vs $w_t$.
- Detailed natural-language decision reasoning.

### 5. Explainability Gallery
- Publication-quality figures across 5 categories:
  1. Fusion & Benchmark (ROC-AUC, Confusion Matrix, False Positive Reduction)
  2. SHAP & LIME (Global token importance, Local force plots, Linear surrogates)
  3. DistilBERT Attention (Multi-head self-attention heatmaps)
  4. Vision & Saliency (CLIP Grad-CAM patch activations)
  5. EasyOCR Bounding Boxes (Spatial text localization)
- Interactive **LightboxModal** with smooth Zoom In, Zoom Out, Reset, and Download controls.

### 6. Analytics Page
- **Attack Category Donut Chart** (Recharts).
- **Bimodal Risk Score Histogram** demonstrating clean separation between safe (0–20%) and threat (80–100%) regimes.
- **Risk Score Chronological Sequence** area chart.
- **Modality Distribution Bar Chart** (Safe vs Threats across Text, Vision, and Multimodal).
- **System Benchmark Ablation Table** comparing DistilBERT alone (80.0%), CLIP+OCR alone (90.0%), and Adaptive Fusion (98.33%).

### 7. Scan Reports & Audit Logs
- Complete searchable, filterable security history.
- Filter by Verdict (`SAFE`, `JAILBREAK`) and Attack Category.
- Instant search by prompt keyword, filename, or decision reasoning.
- **Export Options:** 1-click CSV download and Print/PDF stylesheet.
- Modal inspection of individual scan telemetry.

---

## Getting Started

### 1. Launch FastAPI Backend

```bash
# In workspace root
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

### 2. Launch Vite React Frontend

```bash
# In frontend/ directory
npm install
npm run dev
```

Visit `http://127.0.0.1:5173/` in your browser.

### 3. Production Build

```bash
npm run build
npm run preview
```
