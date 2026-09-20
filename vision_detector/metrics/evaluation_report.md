# Vision Jailbreak Detector — Evaluation Report (Phase 3.2)

## Executive Summary
This report presents the empirical benchmark evaluation of the **Cross-Modal Jailbreak Detection (CMJD) Vision Subsystem**. The detector pairs deep semantic vision representations from a frozen **OpenAI CLIP (ViT-B/32)** encoder with a trained multi-layer classification head and an **EasyOCR** optical text extraction pipeline.

---

## 1. Experimental Setup & Benchmark Dataset
- **Architecture**: OpenAI CLIP ViT-B/32 (Vision Backbone) + 2-layer MLP Classifier Head (512 $\rightarrow$ 256 $\rightarrow$ 64 $\rightarrow$ 2)
- **Dataset**: CMJD-Vision Balanced Benchmark ($N = 2,000$ images)
  - **Safe Class ($N = 1,000$)**: Natural clean imagery from diverse photographic domains.
  - **Jailbreak Class ($N = 1,000$)**: Typographic prompt injection attacks, adversarial overlays, simulated system dialogs, and terminal payloads.
- **Partitions**:
  - Training Set: 1,400 samples (70%)
  - Validation Set: 300 samples (15%)
  - Held-out Test Set: 300 samples (15%)

---

## 2. Test Set Performance Metrics

| Metric | Result | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **99.67%** | $> 95.00\%$ | **PASS (Superior)** |
| **Precision** | **0.9934** | $> 0.9500$ | **PASS** |
| **Recall** | **1.0000** | $> 0.9500$ | **PASS** |
| **F1 Score** | **0.9967** | $> 0.9500$ | **PASS** |
| **ROC-AUC** | **1.0000** | $> 0.9800$ | **PASS (Exceptional)** |

---

## 3. Confusion Matrix Breakdown

| Total Samples: 300 | Predicted Safe (0) | Predicted Jailbreak (1) |
| :--- | :---: | :---: |
| **Actual Safe (0)** | **149** (TN) | **1** (FP) |
| **Actual Jailbreak (1)** | **0** (FN) | **150** (TP) |

- **False Positive Rate (FPR)**: `0.0067`
- **False Negative Rate (FNR)**: `0.0000`

---

## 4. Visual Explainability (ViT Attention Saliency)
Self-attention matrices extracted from CLIP ViT-B/32 Layer 11 demonstrate that:
1. On **Safe Images**, the attention distribution is diffuse across global natural features and contours.
2. On **Jailbreak Images**, attention routes sharply toward typographic injection boundaries, system warning headers, and high-contrast text tokens.
3. Publication-ready figures exported to `vision_detector/metrics/figures/`.

---

## 5. Subsystem Deliverables
- Checkpoint: `vision_detector/models/best_model/model.pt`
- Config: `vision_detector/models/best_model/config.json`
- Metrics: `vision_detector/metrics/metrics.json`
- History: `vision_detector/metrics/training_history.csv`
- Figures: `vision_detector/metrics/figures/` (Confusion Matrix, ROC, PR, Loss, Accuracy, Saliency Heatmaps)
