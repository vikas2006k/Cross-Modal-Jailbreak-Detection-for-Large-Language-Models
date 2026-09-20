# SentinelGuard AI — Frontend Web Application Walkthrough

## Executive Summary

SentinelGuard AI delivers a complete, production-grade React 19 + TypeScript + Vite + Tailwind CSS cybersecurity dashboard for real-time **Cross-Modal AI Security**.

Zero machine learning models were modified or retrained: the web application seamlessly interfaces with the existing FastAPI backend (`127.0.0.1:8001`), connecting to DistilBERT, CLIP ViT-B/32, EasyOCR, and the Adaptive Gated Fusion Engine.

---

## Gallery of Implemented Pages

### 1. Dashboard (Overview)
- Real-time telemetry displaying: Total Scans, Threats Blocked, Safe Prompts, Average Risk Score, and Latency.
- Guardrail Subsystem Health (DistilBERT: 99.20% Acc, CLIP: 512-dim features, EasyOCR: Spatial BBoxes, Fusion: 100% Recall).
- System Benchmark Card (Validation Benchmark: 120 Samples, ROC-AUC: 0.9854).
- Recent Activity Stream table with instant verdict inspection.

![Dashboard Overview](demo_screenshots/01_dashboard_overview.png)

---

### 2. Text Jailbreak Scanner
- Single prompt scanning via `POST /predict`.
- Interactive preset selector for instant testing of DAN 11.0, Instruction Overrides, Roleplay, and Benign queries.
- Animated SVG semi-circle Risk Gauge (0–100%).
- Color-coded **SHAP Token Attribution Map** highlighting adversarial trigger tokens (+SHAP) vs benign tokens (-SHAP).
- Batch prompt scanning modal powered by `POST /batch-predict`.

![Text Scanner](demo_screenshots/02_text_scanner.png)

---

### 3. Visual Jailbreak Scanner
- Image evaluation via `POST /cross-modal-predict`.
- Drag-and-drop zone with 1-click benchmark sample presets (`safe_educational_00.jpg`, `jailbreak_injection_00.jpg`).
- Deconstructs predictions into CLIP vision branch, EasyOCR extracted typography, and unified verdict.
- Visual attention tab displaying CLIP Grad-CAM saliency activations.

![Image Scanner](demo_screenshots/03_image_scanner.png)

---

### 4. Cross-Modal Fusion Scanner
- Dual-input evaluation supporting image + companion prompt.
- Adaptive contribution weight progress bar ($w_v$ vs $w_t$).
- Detailed natural-language decision reasoning demonstrating false-positive correction for educational slides and bimodal consensus blocking for stealth prompt injections.

![Cross-Modal Scanner](demo_screenshots/04_cross_modal_scanner.png)

---

### 5. Multimodal Explainability Gallery
- Filterable gallery across 5 categories (Fusion & Benchmark, SHAP & LIME, DistilBERT Attention, Vision Saliency, EasyOCR Bounding Boxes).
- Interactive LightboxModal with smooth Zoom In, Zoom Out, Reset, and Figure Download controls.

![Explainability Gallery](demo_screenshots/05_explainability_gallery.png)

---

### 6. Security Analytics & Ablation Studies
- Recharts visualizations:
  - Attack category distribution donut chart.
  - Bimodal risk score histogram showing clear separation between safe (0–20%) and threat (80–100%) regimes.
  - Chronological risk sequence flow.
  - Modality breakdown bar chart.
- System ablation study table comparing DistilBERT alone (80.0%), CLIP+OCR alone (90.0%), and Adaptive Fusion (98.33%).

![Analytics Dashboard](demo_screenshots/06_analytics_dashboard.png)

---

### 7. Scan Reports & Audit Logs
- Searchable and filterable security audit table.
- Filter by Verdict (`SAFE`, `JAILBREAK`) and Category.
- 1-Click CSV export and printable PDF layout.
- Detailed telemetry inspector modal for individual scans.

![Reports Page](demo_screenshots/07_reports_page.png)

---

## API Integration Architecture

| Endpoint | Method | Purpose | Resilient Fallback |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Live backend health & model status | Returns offline status badge |
| `/model-info` | `GET` | Architecture & benchmark metadata | Returns system benchmark specs |
| `/predict` | `POST` | Text prompt evaluation | Regex calibrated heuristic fallback |
| `/batch-predict`| `POST` | Multi-prompt bulk evaluation | Batch sequence fallback |
| `/cross-modal-predict` | `POST` | Multipart image + prompt fusion | Multi-branch calibrated fallback |

---

## Verification Summary

1. **Build Test:** `npm run build` completed with zero TypeScript or bundling errors.
2. **Backend Connectivity:** FastAPI backend tested and operational on `127.0.0.1:8001`.
3. **Browser Validation:** All 7 pages loaded, rendered, and interacted with using Chrome browser automation.
4. **Export Capabilities:** Verified CSV download and PDF print formatting.
