# Ablation Study: Multimodal Jailbreak Detection

**Evaluation Benchmarks:**
1. **Internal Optimization Benchmark ($N=80$):** 50 SAFE, 30 JAILBREAK
2. **Final Challenging Benchmark ($N=120$):** 70 SAFE, 50 JAILBREAK

**Modality Components:**
- **Vision:** CLIP ViT-B/32 Visual Encoder + Linear Classification Head
- **Text:** EasyOCR Token Extraction + Fine-Tuned DistilBERT
- **Fusion (Original):** Hard Threshold Heuristic Fusion (Phase 3.3)
- **Fusion (Optimized):** Adaptive Semantic Gating + Shannon Entropy + Continuous Reliability Weighting (Phase 3.3.1)

---

## 1. Comparative Component Ablation

### A. Final Challenging Evaluation Benchmark ($N=120$)

| System Architecture | Accuracy | Precision | Recall | F1 Score | False Positive Rate | False Negative Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vision Only (CLIP + Head)** | 41.67% | 41.67% | 100.00% | 0.5882 | 100.00% | 0.00% |
| **Text Only (DistilBERT + OCR)** | 76.67% | 64.10% | 100.00% | 0.7812 | 40.00% | 0.00% |
| **Fusion (Phase 3.3 Baseline)** | 76.67% | 64.10% | 100.00% | 0.7812 | 40.00% | 0.00% |
| **Fusion (Phase 3.3.1 Optimized)** | **98.33%** | **96.15%** | **100.00%** | **0.9804** | **2.86%** | **0.00%** |

### B. Internal Optimization Benchmark ($N=80$)

| System Architecture | Accuracy | Precision | Recall | F1 Score | False Positive Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| **Vision Only** | 37.50% | 37.50% | 100.00% | 0.5455 | 100.00% |
| **Text Only** | 90.00% | 78.95% | 100.00% | 0.8824 | 16.00% |
| **Fusion (Original)** | 90.00% | 78.95% | 100.00% | 0.8824 | 16.00% |
| **Fusion (Optimized)** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **0.00%** |

---

## 2. Component Contribution Analysis

1. **Failure Mode of Standalone Vision (41.67% Accuracy):**
   - The CLIP ViT-B/32 encoder exhibits spatial oversensitivity to typographic text elements. Any image containing structured paragraphs, headers, or terminal blocks triggers jailbreak predictions, resulting in a **100.00% FPR** on benign text images.
2. **Failure Mode of Standalone Text (76.67% Accuracy):**
   - While DistilBERT achieves high precision on standard conversational inputs, technical lecture vocabulary (thermodynamics, kernel memory, encryption) and system administration logs trigger elevated false positive risks (40.00% FPR).
3. **Synergy of Optimized Cross-Modal Fusion (98.33% Accuracy):**
   - The combination of Shannon visual entropy $H(X)$, OCR confidence $\gamma_{\text{ocr}}$, and semantic context gating resolves unimodal ambiguity. False alarms drop from 40.00% to **2.86%**, while preserving **100.00% recall** across all adversarial vectors.
