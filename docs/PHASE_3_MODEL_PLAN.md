# CMJD Phase 3 Model Plan

## Overview
Phase 3 encompasses the machine learning and deep learning model development lifecycle for the **Cross-Modal Jailbreak Detection (CMJD)** system. The architectural framework integrates a modular multi-stage detection pipeline combining textual semantics, vision-language representations, visual text extraction, and cross-modal fusion.

---

## Model 1 — DistilBERT (Text Modality Baseline)

### Purpose
Binary classification of incoming textual prompts as either **Safe (0)** or **Jailbreak (1)** to detect semantic, policy bypass, role override, and direct prompt injection attacks.

### Dataset
- **Benchmark**: CMJD-10K Text Dataset (`dataset/processed/cmjd_text_dataset.csv`)
- **Distribution**: 10,000 total prompts (5,000 safe, 5,000 jailbreak across 11 attack types)
- **Partitions**:
  - Train: 7,000 samples (70%)
  - Validation: 1,500 samples (15%)
  - Test: 1,500 samples (15%)

### Model Architecture
- **Base Model**: `distilbert-base-uncased` (6 layers, 768 hidden dimensions, 12 attention heads, 66M parameters)
- **Input**: Prompt text sequence (tokenized via Hugging Face AutoTokenizer)
- **Output**: Binary classification logits / probabilities: `0` (Safe) vs `1` (Jailbreak)

### Training Strategy (IEEE Configuration)
- **Maximum Epochs**: 10
- **Early Stopping**: Patience 2 (monitoring validation loss / validation F1)
- **Best Model Selection**: Highest Validation F1 score (`metric_for_best_model: "f1"`)
- **Batch Size**: 16
- **Learning Rate**: 2e-5 with AdamW optimizer
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.1
- **Max Sequence Length**: 128 tokens
- **Random Seed**: 42 (deterministic reproducibility)
- **Save Strategy**: Per epoch, retaining top 3 checkpoints

---

## Future Multimodal Models in Pipeline

To counter cross-modal jailbreaks that evade pure text classifiers by distributing or obfuscating prompts across visual channels, the CMJD architecture will deploy complementary vision and fusion models:

### 1. Model 2 — OpenCLIP (Vision-Language Embedding Model)
- **Purpose**: Extract rich visual-semantic embeddings from input images to detect anomalous image-text alignments and adversarial visual artifacts.
- **Backbone**: ViT-B/32 or ViT-L/14 trained with contrastive learning.
- **Application**: Joint embedding projection to identify visual jailbreak context and semantic dissonance.

### 2. Model 3 — Optical Character Recognition (EasyOCR + Tesseract)
- **Purpose**: Robust visual text extraction across adversarial image distortions.
- **Capabilities**: Detects and transcribes hidden prompts, rotated typography (90°/180°/270°/diagonal), tiny fonts (2–5px), low-contrast blended text, watermark stamps, and QR code decoded payloads.
- **Integration**: Feeds extracted visual text directly into the DistilBERT language pipeline.

### 3. Model 4 — Multimodal Cross-Modal Fusion Network
- **Purpose**: Deep cross-attention and feature concatenation combining textual tokens, visual tokens, and OCR-extracted signals.
- **Classification Head**: Joint classification determining overall multimodal threat level, pinpointing which modality contains the adversarial payload.

---

## Phase Roadmap
- **Phase 3.1.1**: Initialize DistilBERT training environment & IEEE configuration (Completed)
- **Phase 3.1.2**: DistilBERT Dataset Loader & Tokenization
- **Phase 3.1.3**: DistilBERT Fine-Tuning Execution
- **Phase 3.1.4**: Evaluation & Checkpoint Validation on Test Set
- **Phase 3.2**: OCR Feature Pipeline (EasyOCR + Tesseract)
- **Phase 3.3**: OpenCLIP Vision Classifier
- **Phase 3.4**: Multimodal Cross-Modal Fusion Network
