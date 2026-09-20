# CMJD Visual Jailbreak Detector (Image Prompt Injection Detection)

The **Visual Jailbreak Detector** is the second modality subsystem of the Cross-Modal Jailbreak Detection (CMJD) framework. It identifies adversarial prompt injections and jailbreak attacks disguised inside visual media—including typographic overlays, system dialog spoofing, terminal code injections, and adversarial images.

---

## 1. Subsystem Architecture

The visual detection pipeline integrates two complementary inspection branches:

```
                      ┌───────────────────────────────────────┐
                      │              Input Image              │
                      │       (PNG / JPG / JPEG / WEBP)       │
                      └──────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
   ┌────────────────────────────────┐       ┌────────────────────────────────┐
   │    Semantic Vision Branch      │       │     Typographic OCR Branch     │
   │  (OpenAI CLIP ViT-B/32 Frozen) │       │       (EasyOCR Deep CRAFT)     │
   └────────────────┬───────────────┘       └────────────────┬───────────────┘
                    │ 512-dim Embeddings                     │ Extracted Strings,
                    ▼                                        │ Confidence & BBoxes
   ┌────────────────────────────────┐                        │
   │      MLP Classifier Head       │                        │
   │  (512 -> 256 -> 64 -> 2)       │                        │
   └────────────────┬───────────────┘                        │
                    │ Probabilities & Risk                   │
                    ▼                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    VisionJailbreakDetector Verdict                      │
   │  - Prediction: SAFE / JAILBREAK                                         │
   │  - Risk Score: 0.0 - 100.0                                              │
   │  - Blocked: True / False                                                │
   │  - Extracted Text & Bounding Boxes                                      │
   │  - ViT Layer 11 Self-Attention Saliency Maps                            │
   └─────────────────────────────────────────────────────────────────────────┘
```

1. **Semantic Vision Branch**: Utilizes a frozen `openai/clip-vit-base-patch32` visual transformer to capture deep visual-semantic representations ($d = 512$). A trained multilayer perceptron classification head maps these representations to binary threat probabilities.
2. **Typographic OCR Branch**: An EasyOCR engine scans the image for hidden or rendered text, recovering raw strings, bounding polygon coordinates, and per-token detection confidence.
3. **Spatial Saliency Explainability**: Extracts Layer 11 multi-head self-attention routing from the `[CLS]` token to all 49 visual patches, projecting a 300 DPI spatial heatmap pinpointing adversarial injection zones.

---

## 2. Directory Structure

```
vision_detector/
├── dataset/
│   ├── safe/                    # Clean natural imagery (1,000 images)
│   ├── jailbreak/               # Typographic adversarial visual attacks (1,000 images)
│   └── metadata.csv             # 2,000-sample catalog with split & attack taxonomy
├── models/
│   └── best_model/
│       ├── model.pt             # Trained MLP classifier weights (PyTorch)
│       └── config.json          # Architecture and training hyperparameters
├── metrics/
│   ├── figures/
│   │   ├── confusion_matrix.png # 300 DPI test confusion matrix
│   │   ├── roc_curve.png        # 300 DPI receiver operating characteristic
│   │   ├── precision_recall_curve.png # 300 DPI PR curve
│   │   ├── loss_curve.png       # 300 DPI training & validation loss
│   │   ├── accuracy_curve.png   # 300 DPI accuracy convergence
│   │   ├── saliency_sample_1_jailbreak.png # 300 DPI ViT attention heatmap
│   │   ├── saliency_sample_2_jailbreak.png # 300 DPI ViT attention heatmap
│   │   ├── saliency_sample_3_safe.png      # 300 DPI ViT attention heatmap
│   │   └── saliency_sample_4_safe.png      # 300 DPI ViT attention heatmap
│   ├── metrics.json             # Benchmark evaluation metrics (JSON)
│   ├── training_history.csv     # Per-epoch loss, accuracy, and F1 progression
│   └── evaluation_report.md     # IEEE conference-style experimental report
├── dataset_builder.py           # Ingestion and synthetic typographic attack generator
├── preprocessing.py             # Format normalization (RGB/EXIF) & CLIP transforms
├── ocr.py                       # EasyOCR extraction wrapper with bounding boxes
├── feature_extractor.py         # Frozen CLIP ViT-B/32 512-dim embedding extractor
├── train_clip_classifier.py     # Training script with early stopping & F1 monitoring
├── inference.py                 # Unified VisionJailbreakDetector inference service
├── explainability.py            # Vision Transformer Layer 11 attention saliency engine
├── evaluate_vision.py           # Evaluation pipeline and 300 DPI figure exporter
└── README.md                    # Technical documentation
```

---

## 3. Dataset Pipeline & Synthesis

The visual benchmark dataset consists of **2,000 balanced images**:
- **Safe Class ($N = 1,000$)**: Natural photographs from landscape, urban, and object domains.
- **Jailbreak Class ($N = 1,000$)**: Rendered typographic attacks using adversarial prompts from the CMJD text benchmark and AdvBench:
  - *Style 0: System Dialog Spoofing* — Warning frames, administrative alerts, and system override banners.
  - *Style 1: Scenic Text Overlays* — Adversarial instructions blended over complex natural backgrounds.
  - *Style 2: High-Contrast Typographic Cards* — Bold, high-contrast adversarial instructions.
  - *Style 3: Terminal & Code Dialogs* — Monospace developer mode injections and pseudo-shell exploits.
- **Supported File Formats**: PNG, JPG, JPEG, and WEBP.
- **Partitions**:
  - Training Set: 1,400 samples (70%)
  - Validation Set: 300 samples (15%)
  - Held-out Test Set: 300 samples (15%)

---

## 4. Empirical Evaluation Results

Evaluated on the 300 held-out test images ($150\text{ Safe}, 150\text{ Jailbreak}$):

| Metric | Score | IEEE Target | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **99.67%** | $> 95.00\%$ | **PASS (Superior)** |
| **Precision** | **0.9934** | $> 0.9500$ | **PASS** |
| **Recall** | **1.0000** | $> 0.9500$ | **PASS (Zero Missed Attacks)** |
| **F1-Score** | **0.9967** | $> 0.9500$ | **PASS** |
| **ROC-AUC** | **1.0000** | $> 0.9800$ | **PASS (Optimal Separation)** |

### Confusion Matrix
- **True Negatives (TN)**: 149
- **False Positives (FP)**: 1
- **False Negatives (FN)**: 0
- **True Positives (TP)**: 150

---

## 5. Usage Examples

### Single Image Inference (with OCR)
```python
from pathlib import Path
from vision_detector.inference import VisionJailbreakDetector

detector = VisionJailbreakDetector.get_instance()

image_path = Path("path/to/suspicious_image.png")
result = detector.predict(image_path, run_ocr=True)

print(f"Prediction : {result['prediction']}")        # "JAILBREAK" or "SAFE"
print(f"Confidence : {result['confidence']:.4f}")
print(f"Risk Score : {result['risk_score']:.2f}")     # 0.0 - 100.0
print(f"Blocked    : {result['blocked']}")            # True / False
print(f"OCR Text   : {result['ocr_text']}")
print(f"OCR BBoxes : {result['ocr_boxes']}")
```

### Visual Saliency Heatmap Generation
```python
from pathlib import Path
from vision_detector.explainability import VisionExplainabilityEngine

explainer = VisionExplainabilityEngine()
explainer.visualize_saliency(
    image=Path("path/to/image.png"),
    output_path=Path("metrics/figures/sample_saliency.png"),
    title="Typographic Jailbreak Saliency Map",
    label="JAILBREAK"
)
```

---

## 6. Next Steps & Cross-Modal Fusion
The Vision Detector operates as a standalone module and is architected for seamless integration into **Phase 3.4 (Multimodal Cross-Modal Fusion Network)**, where OCR extracted text will be piped into the DistilBERT language model while visual embeddings are jointly attended via a cross-attention fusion head.
