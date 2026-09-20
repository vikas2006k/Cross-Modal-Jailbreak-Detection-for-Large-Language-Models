# Table I: Cross-Modal Jailbreak Detection Ablation Study

Empirical comparison across single-modality baselines and the proposed adaptive gated fusion architecture evaluated on the $N=120$ hard adversarial challenge set.

| Architecture / Subsystem | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | False Positive Rate (FPR %) | False Negative Rate (FNR %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/32 (Vision Only)** | 41.67% | 41.67% | 100.00% | 58.82% | 0.6036 | 100.00% | 0.00% |
| **DistilBERT (Text Only via OCR)** | 76.67% | 64.10% | 100.00% | 78.12% | 0.8914 | 40.00% | 0.00% |
| **Static Linear Fusion Baseline** | 76.67% | 64.10% | 100.00% | 78.12% | 0.8914 | 40.00% | 0.00% |
| **Proposed Adaptive Gated Fusion** | **98.33%** | **96.15%** | **100.00%** | **98.04%** | **0.9854** | **2.86%** | **0.00%** |

### Confusion Matrix Breakdown ($N=120$)
- **Vision Only**: True Negatives (TN) = 0, False Positives (FP) = 70, False Negatives (FN) = 0, True Positives (TP) = 50.
- **Text Only**: True Negatives (TN) = 42, False Positives (FP) = 28, False Negatives (FN) = 0, True Positives (TP) = 50.
- **Proposed Adaptive Gated Fusion**: True Negatives (TN) = 68, False Positives (FP) = 2, False Negatives (FN) = 0, True Positives (TP) = 50.

> **Key Observation**: The proposed Adaptive Gated Fusion engine reduces the false positive rate from **40.00% down to 2.86%** while maintaining **100.00% recall** on stealth adversarial attacks, yielding a net **+21.66% accuracy improvement** over unimodal text classification.
