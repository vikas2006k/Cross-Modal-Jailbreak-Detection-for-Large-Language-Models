"""
Core Cross-Modal Fusion Engine.
Implements deterministic, mathematically grounded fusion algorithms combining
visual risk, OCR evidence, and DistilBERT text risk.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image

from fusion_engine.rules import classify_attack_category, generate_fusion_reason
from fusion_engine.schemas import (
    VisionModalityOutput,
    TextModalityOutput,
    OCRModalityOutput,
    ModalitiesBreakdown,
    CrossModalPredictResponse
)


def is_uniform_or_blank_image(image: Image.Image, std_threshold: float = 12.0) -> bool:
    """
    Detects if an image is essentially a uniform blank canvas (e.g. solid white/gray).
    """
    try:
        gray = image.convert("L")
        arr = np.array(gray, dtype=np.float32)
        return float(np.std(arr)) < std_threshold
    except Exception:
        return False


def compute_cross_modal_fusion(
    vision_output: Dict[str, Any],
    text_output: Dict[str, Any],
    ocr_output: Dict[str, Any],
    pil_image: Optional[Image.Image] = None,
    user_prompt: str = ""
) -> CrossModalPredictResponse:
    """
    Computes deterministic cross-modal fusion given vision, text, and OCR outputs.

    Fusion Equations:
    1. Bimodal Consensus JAILBREAK (R_v >= 50, R_t >= 50):
       P_fused = 1 - (1 - P_v) * (1 - P_t)
       R_fused = max(R_v, R_t, P_fused * 100.0)

    2. Educational Slide Correction (R_v >= 50, R_t < 25, L_ocr >= 10, C_ocr >= 0.3):
       R_fused = 0.85 * R_t + 0.15 * (0.2 * R_v)  (Overrules visual typographic prior)

    3. Blank / Uniform Canvas Correction (R_v >= 50, L_ocr == 0, is_uniform == True):
       R_fused = 0.05 * R_v

    4. Stealth Typographic Injection (R_v < 50, R_t >= 50, C_ocr >= 0.3):
       R_fused = max(R_t, R_v)  (Security first gating)

    5. Bimodal Consensus SAFE (R_v < 50, R_t < 50):
       R_fused = 0.5 * R_v + 0.5 * R_t
    """
    v_pred = vision_output["prediction"]
    v_risk = float(vision_output["risk_score"])
    v_conf = float(vision_output["confidence"])

    t_pred = text_output["prediction"]
    t_risk = float(text_output["risk_score"])
    t_conf = float(text_output["confidence"])

    ocr_text = str(ocr_output.get("extracted_text", "")).strip()
    ocr_conf = float(ocr_output.get("confidence", 0.0))
    ocr_boxes = ocr_output.get("bounding_boxes", [])
    num_ocr_regions = len(ocr_boxes)

    ocr_len = len(ocr_text)
    user_prompt = user_prompt.strip()
    combined_text = f"{user_prompt} {ocr_text}".strip()

    is_blank = False
    if pil_image is not None and ocr_len == 0:
        is_blank = is_uniform_or_blank_image(pil_image)

    is_educational_corrected = False
    is_blank_corrected = False

    # -------------------------------------------------------------
    # Case 1: Both predict JAILBREAK (Bimodal Consensus)
    # -------------------------------------------------------------
    if v_risk >= 50.0 and t_risk >= 50.0:
        p_v = v_risk / 100.0
        p_t = t_risk / 100.0
        p_fused = 1.0 - (1.0 - p_v) * (1.0 - p_t)
        fused_risk = min(100.0, max(v_risk, t_risk, p_fused * 100.0))
        fused_pred = "JAILBREAK"
        fused_conf = max(v_conf, t_conf, 0.99)
        blocked = True
        w_v, w_t = 0.5, 0.5

    # -------------------------------------------------------------
    # Case 2: Educational Slide / Document False Positive Correction
    # Vision flags typographic layout, but text is substantial and benign
    # -------------------------------------------------------------
    elif v_risk >= 50.0 and t_risk < 25.0 and ocr_len >= 10 and ocr_conf >= 0.25:
        # Heavily discount the vision typographic layout prior
        fused_risk = round(0.85 * t_risk + 0.15 * (0.2 * v_risk), 2)
        fused_pred = "SAFE"
        fused_conf = t_conf
        blocked = False
        is_educational_corrected = True
        w_v, w_t = 0.1, 0.9

    # -------------------------------------------------------------
    # Case 3: Blank Canvas Artifact Correction
    # Vision flags uniform canvas, but no text and image is uniform
    # -------------------------------------------------------------
    elif v_risk >= 50.0 and (is_blank or (ocr_len == 0 and len(user_prompt) == 0)):
        fused_risk = round(0.05 * v_risk, 2)
        fused_pred = "SAFE"
        fused_conf = 0.95
        blocked = False
        is_blank_corrected = True
        w_v, w_t = 0.5, 0.5

    # -------------------------------------------------------------
    # Case 4: Stealth Typographic Injection (Vision Safe, Text Jailbreak)
    # Background is benign scenery, but text has prompt injection
    # -------------------------------------------------------------
    elif v_risk < 50.0 and t_risk >= 50.0 and (ocr_conf >= 0.25 or len(user_prompt) > 0):
        fused_risk = max(t_risk, v_risk)
        fused_pred = "JAILBREAK"
        fused_conf = t_conf
        blocked = True
        w_v, w_t = 0.2, 0.8

    # -------------------------------------------------------------
    # Case 5: Vision Jailbreak, Text Safe without enough OCR confidence
    # If text is empty, vision risk dominates
    # -------------------------------------------------------------
    elif v_risk >= 50.0 and t_risk < 50.0 and ocr_len == 0:
        fused_risk = v_risk
        fused_pred = "JAILBREAK"
        fused_conf = v_conf
        blocked = True
        w_v, w_t = 0.9, 0.1

    # -------------------------------------------------------------
    # Case 6: Both predict SAFE (Consensus Safe)
    # -------------------------------------------------------------
    else:
        fused_risk = round(0.5 * v_risk + 0.5 * t_risk, 2)
        fused_pred = "SAFE"
        fused_conf = round((v_conf + t_conf) / 2.0, 4)
        blocked = False
        w_v, w_t = 0.5, 0.5

    # Format values
    fused_risk = float(round(fused_risk, 2))
    fused_conf = float(round(fused_conf, 4))

    # Attack category classification
    attack_category = classify_attack_category(
        prediction=fused_pred,
        combined_text=combined_text,
        ocr_text=ocr_text,
        user_prompt=user_prompt,
        vision_risk=v_risk,
        text_risk=t_risk,
        num_ocr_regions=num_ocr_regions
    )

    # Human-readable rationale
    fusion_reason = generate_fusion_reason(
        final_prediction=fused_pred,
        fused_risk=fused_risk,
        vision_pred=v_pred,
        vision_risk=v_risk,
        text_pred=t_pred,
        text_risk=t_risk,
        ocr_text=ocr_text,
        user_prompt=user_prompt,
        is_educational_corrected=is_educational_corrected,
        is_blank_corrected=is_blank_corrected
    )

    modalities = ModalitiesBreakdown(
        vision=VisionModalityOutput(
            prediction=v_pred,
            confidence=v_conf,
            risk_score=v_risk
        ),
        text=TextModalityOutput(
            prediction=t_pred,
            confidence=t_conf,
            risk_score=t_risk,
            prompt_evaluated=text_output.get("prompt_evaluated", combined_text)
        ),
        ocr=OCRModalityOutput(
            extracted_text=ocr_text,
            confidence=ocr_conf,
            num_regions=num_ocr_regions,
            bounding_boxes=ocr_boxes
        )
    )

    return CrossModalPredictResponse(
        prediction=fused_pred,
        risk_score=fused_risk,
        confidence=fused_conf,
        blocked=blocked,
        attack_category=attack_category,
        modalities=modalities,
        fusion_reason=fusion_reason,
        model_version="CMJD-v1.0"
    )
