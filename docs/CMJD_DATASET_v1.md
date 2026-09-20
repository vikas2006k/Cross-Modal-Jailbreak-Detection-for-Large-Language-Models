# CMJD Dataset v1.0

## Dataset Summary

The **Cross-Modal Jailbreak Detection (CMJD)** benchmark represents a comprehensive, multimodal security dataset designed to evaluate and improve the safety and robustness of Large Language Models (LLMs) and Vision-Language Models (VLMs) against adversarial attacks, visual prompt injections, and cross-modal jailbreak attempts.

- **Text Dataset**: 10,000 prompts (5,000 safe, 5,000 jailbreak)
- **Image Dataset**: 5,000 images (2,500 safe, 2,500 adversarial jailbreak)
- **Safe Samples**: 11,000 (Text + Image + Multimodal Combinations)
- **Jailbreak Samples**: 11,000 (Text + Image + Multimodal Combinations)
- **Total Samples**: 22,000

---

## Image Attack Categories

The adversarial image benchmark consists of exactly 2,500 jailbreak images generated across eight realistic attack modalities paired 1:1 with source images:

- **Hidden Prompt** (350 images): Blended white text, low opacity, tiny corner coordinates, transparent background overlays.
- **Tiny Font** (350 images): 2–5 pixel micro-typography positioned along borders and edge regions.
- **Low Contrast** (300 images): Color blending against sky, foliage, dark regions, and localized adaptive backgrounds.
- **Overlay Prompt** (350 images): Visual captions, dialogue boxes, sticky notes, and system notification overlays.
- **Meme Attack** (300 images): Social media meme formats, multi-panel layouts, speech bubbles, and chat UI mockups.
- **QR Overlay** (300 images): Adversarially encoded QR codes embedded as posters, stickers, stamps, and floating badges.
- **Rotated Text** (250 images): Multi-angle orientations (90°, 180°, 270°, diagonal 30–45°).
- **Watermark Prompt** (300 images): Semi-transparent copyright watermarks, diagonal stripes, and corner photographer signatures.

---

## Safe Image Subsets

- **COCO 2017**: 1,500 images
- **Flickr30k**: 1,000 images
- **Total Safe Images**: 2,500 images

---

## Dataset Splits

The dataset partitions strictly enforce zero overlap across splits:
- **Train Set**: 7,000 samples (70%)
- **Validation Set**: 1,500 samples (15%)
- **Test Set**: 1,500 samples (15%)

---

## Dataset Audit & Integrity Verification

The dataset has undergone a comprehensive, non-destructive audit (Phase 2.5) covering all 22,000 target samples, 5,000 image files, metadata registries, and JSON annotations.

- **Corrupted JPEGs**: 0
- **Zero-Byte Files**: 0
- **Duplicate Source Images**: 0 (1:1 disjoint mapping)
- **Prompt Length Boundaries**: 15–498 characters
- **Split Leakage**: 0 overlap
- **Audit Verdict**: **PASS** (100% verified across all 36 audit checkpoints)

Detailed audit records are archived in `dataset/reports/`:
- `cmjd_dataset_audit_report.json`
- `cmjd_dataset_audit_report.csv`
- `cmjd_dataset_statistics.json`
- `cmjd_dataset_summary.md`
- `cmjd_integrity_report.txt`
