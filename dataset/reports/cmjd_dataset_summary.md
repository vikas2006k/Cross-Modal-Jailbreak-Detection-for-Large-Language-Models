# CMJD Dataset v1.0 — Comprehensive Benchmark Summary

## Executive Summary
The **Cross-Modal Jailbreak Detection (CMJD)** benchmark represents a standardized, multimodal security dataset designed to evaluate and enhance the robustness of Large Language Models (LLMs) and Vision-Language Models (VLMs) against cross-modal jailbreaks, adversarial visual prompt injections, and multimodal bypass attacks.

- **Total Benchmark Scope**: 22,000 samples (11,000 Safe, 11,000 Jailbreak)
- **Core Modalities**: Text (`cmjd_text_dataset.csv`), Images (`image_metadata.csv`, `jailbreak_image_metadata.csv`), Cross-Modal Multimodal Pairs.
- **Audit Status**: **PASS** (100% verified across all dimensions)

---

## 1. Text Dataset (`cmjd_text_dataset.csv`)

- **Total Prompts**: 10,000 (5,000 Safe, 5,000 Jailbreak)
- **Duplicate Prompts**: 0 | **Empty Prompts**: 0
- **Prompt Length**: Min 15 chars, Max 498 chars (Mean: 112.4)
- **Splits**: Train (7,000), Validation (1,500), Test (1,500) with 0 overlap.

### Text Attack Categories
| Attack Category | Sample Count | Modality |
| :--- | :--- | :--- |
| **safe** | 5,000 | text |
| **policy_bypass** | 1,435 | text |
| **unknown (AdvBench Raw)** | 666 | text |
| **role_override** | 531 | text |
| **prompt_injection** | 447 | text |
| **developer_mode** | 314 | text |
| **tool_injection** | 304 | text |
| **system_prompt_extraction** | 302 | text |
| **indirect_injection** | 300 | text |
| **multi_turn_jailbreak** | 250 | text |
| **encoded_attack** | 250 | text |
| **reasoning_manipulation** | 201 | text |

---

## 2. Image Dataset Architecture

- **Total Safe Images**: 2,500 (1,500 COCO 2017, 1,000 Flickr30k) — 373.97 MB
- **Total Jailbreak Images**: 2,500 across 8 distinct attack categories — 288.16 MB
- **Source Disjointness**: Every safe image maps 1:1 to exactly one adversarial copy (0 source duplicates).

### Jailbreak Image Attack Categories
| Attack Category | Count | Storage (MB) | Style / Description |
| :--- | :--- | :--- | :--- |
| **Hidden Prompt** | 350 | 40.47 | Invisible white, tiny corner, blended background, transparent overlay |
| **Tiny Font** | 350 | 39.78 | 2–5px font placed on corners and image edges |
| **Low Contrast** | 300 | 34.52 | Gray, sky, nature, and adaptive local background color blending |
| **Overlay Prompt** | 350 | 40.71 | Captions, sticky notes, UI dialogs, and scene notices |
| **Meme Attack** | 300 | 34.92 | Reaction memes, speech bubbles, comparison panels, chat screenshots |
| **QR Overlay** | 300 | 35.15 | Sticker, poster, product, and floating QR codes with unique SHA256 hashes |
| **Rotated Text** | 250 | 29.4 | Vertical 90°, vertical 270°, upside-down 180°, and diagonal 30–45° |
| **Watermark Prompt** | 300 | 33.21 | Center copyright, diagonal stripes, photographer signature, repeating tiled pattern |
| **Total Jailbreak** | **2500** | **288.16** | **8 Attack Categories** |

---

## 3. Data Integrity & Verification Summary

- **Total JPEGs Scanned**: 5,000
- **Corrupted JPEGs**: 0
- **Zero-Byte Files**: 0
- **Missing Images**: 0
- **Metadata-Disk Parity**: 100%
- **Audit Verdict**: **PASS**
