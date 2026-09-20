# DistilBERT Training & Evaluation Summary — CMJD Phase 3.1.3

## Executive Summary
- **Model Name**: `distilbert-base-uncased` (Sequence Classification Head, 2 classes)
- **Dataset**: CMJD-10K Text Benchmark (`models/text_classifier/tokenized_dataset`)
- **Partitions**: Train (7,000 samples) | Validation (1,500 samples) | Test (1,500 samples)
- **Epochs Completed**: 4 / 10
- **Best Epoch**: 2
- **Early Stopping Triggered**: Yes
- **Total Training Duration**: 3771.9s (62.86 min)

---

## IEEE Hyperparameter Configuration
- **Batch Size**: 16
- **Learning Rate**: 2e-05
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.1
- **Max Sequence Length**: 128 tokens
- **Random Seed**: 42
- **Early Stopping Patience**: 2 epochs (monitoring `validation_f1`)

---

## Final Test Performance Metrics (Held-Out Test Split, N=1,500)
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.9920** (99.20%) | Overall correct binary classifications |
| **Precision** | **0.9933** | True jailbreak detections / total flagged prompts |
| **Recall** | **0.9907** | True jailbreak detections / total actual jailbreaks |
| **F1 Score** | **0.9920** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9996** | Area under the ROC curve (discrimination ability) |
| **Matthews Corr. Coeff. (MCC)** | **0.9840** | High-quality correlation metric across binary classes |
| **Balanced Accuracy** | **0.9920** | Macro-average accuracy across safe and jailbreak |

---

## Confusion Matrix Summary
| True \ Predicted | Predicted Safe (0) | Predicted Jailbreak (1) | Total | Class Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Actual Safe (0)** | **745** (TN) | **5** (FP) | 750 | 99.33% |
| **Actual Jailbreak (1)** | **7** (FN) | **743** (TP) | 750 | 99.07% |
| **Total** | 752 | 748 | 1,500 | - |

---

## Training History Across Epochs
| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 Score | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0765 | 0.0889 | 0.9727 | 0.9972 | 0.9480 | 0.9720 | 1638.58s |
| 2 | 0.0280 | 0.0416 | 0.9920 | 0.9973 | 0.9867 | 0.9920 | 739.92s |
| 3 | 0.0135 | 0.0533 | 0.9907 | 0.9986 | 0.9827 | 0.9906 | 699.41s |
| 4 | 0.0012 | 0.0615 | 0.9920 | 0.9986 | 0.9853 | 0.9919 | 689.85s |

---

## Visualizations Generated
- `loss_curve.png`: Training vs. validation loss trajectory.
- `accuracy_curve.png`: Epoch-wise accuracy and F1 score trajectory.
- `precision_recall_curve.png`: Precision vs. recall tradeoff curve (AUC = 0.9996).
- `roc_curve.png`: Receiver operating characteristic curve (AUC = 0.9996).
- `confusion_matrix.png`: Heatmap with count and class percentage annotations.
