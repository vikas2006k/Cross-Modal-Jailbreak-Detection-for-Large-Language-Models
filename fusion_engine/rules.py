"""
Security rules, attack taxonomy classification, and disagreement rationale
for the Cross-Modal Fusion Engine.
"""

import re
from typing import Dict, Any, Optional, Tuple


# Pre-compiled regex patterns for attack classification
ROLEPLAY_PATTERNS = re.compile(
    r"\b(dan|do anything now|act as|pretend to be|unfiltered persona|evil bot|stan|jailbroken|no restrictions)\b",
    re.IGNORECASE
)

ENCODED_PATTERNS = re.compile(
    r"(\b(base64|rot13|hex|ascii|ciphers?|decode)\b|[A-Za-z0-9+/]{24,}={0,2})",
    re.IGNORECASE
)

INDIRECT_PATTERNS = re.compile(
    r"(\b(translate the following|the following text from a friend|summarize this external text|read between the lines)\b|[\"'].*?(ignore|bypass|override).*?[\"'])",
    re.IGNORECASE
)

MULTISTEP_PATTERNS = re.compile(
    r"\b(step 1|step 2|phase 1|phase 2|first.*?then.*?finally|in stages)\b",
    re.IGNORECASE
)

PROMPT_INJECTION_PATTERNS = re.compile(
    r"\b(ignore previous instructions|disregard all prior|system override|jailbreak|bypass safety|reveal system prompt|developer mode)\b",
    re.IGNORECASE
)

EDUCATIONAL_PATTERNS = re.compile(
    r"\b(introduction to|chapter|lecture|slide|course|syllabus|cs\d+|computer science|algorithms?|data structures?|biology|chemistry|physics|mathematics|theorem|proof|exercise|summary|university|professor|homework|quiz)\b",
    re.IGNORECASE
)


def classify_attack_category(
    prediction: str,
    combined_text: str,
    ocr_text: str,
    user_prompt: str,
    vision_risk: float,
    text_risk: float,
    num_ocr_regions: int
) -> str:
    """
    Classifies input into one of the 9 standardized CMJD attack/benign categories:
    - Prompt Injection
    - Roleplay Jailbreak
    - Indirect Jailbreak
    - Encoded Jailbreak
    - Multi-step Jailbreak
    - Visual Prompt Injection
    - OCR Text Injection
    - Benign Educational Content
    - Benign Natural Image
    """
    text_to_scan = f"{combined_text} {ocr_text} {user_prompt}".strip()

    if prediction == "SAFE":
        # Check if educational slide / academic content
        if EDUCATIONAL_PATTERNS.search(text_to_scan) or (num_ocr_regions >= 2 and len(ocr_text) > 40):
            return "Benign Educational Content"
        return "Benign Natural Image"

    # Prediction is JAILBREAK
    if ENCODED_PATTERNS.search(text_to_scan):
        return "Encoded Jailbreak"

    if ROLEPLAY_PATTERNS.search(text_to_scan):
        return "Roleplay Jailbreak"

    if INDIRECT_PATTERNS.search(text_to_scan):
        return "Indirect Jailbreak"

    if MULTISTEP_PATTERNS.search(text_to_scan):
        return "Multi-step Jailbreak"

    if PROMPT_INJECTION_PATTERNS.search(text_to_scan):
        # If user prompt is empty or safe, but OCR text has injection, classify as OCR Text Injection
        if len(ocr_text.strip()) > 0 and len(user_prompt.strip()) == 0:
            return "OCR Text Injection"
        return "Prompt Injection"

    # Distinguish OCR Text Injection vs Visual Prompt Injection
    if len(ocr_text.strip()) > 0 and num_ocr_regions > 0:
        return "OCR Text Injection"

    if vision_risk >= 80.0 and len(ocr_text.strip()) == 0:
        return "Visual Prompt Injection"

    return "Prompt Injection"


def generate_fusion_reason(
    final_prediction: str,
    fused_risk: float,
    vision_pred: str,
    vision_risk: float,
    text_pred: str,
    text_risk: float,
    ocr_text: str,
    user_prompt: str,
    is_educational_corrected: bool = False,
    is_blank_corrected: bool = False
) -> str:
    """
    Generates human-understandable explanation detailing cross-modal decision
    and explaining how disagreements were reconciled.
    """
    if is_educational_corrected:
        return (
            "Cross-modal disagreement resolved: Vision model flagged typographic slide layout "
            f"(Vision Risk: {vision_risk:.1f}%), but DistilBERT verified that extracted OCR text "
            f"('{ocr_text[:60]}...') is purely benign academic/educational material "
            f"(Text Risk: {text_risk:.1f}%). False positive overridden to SAFE."
        )

    if is_blank_corrected:
        return (
            f"Canvas artifact resolved: Vision model triggered on uniform canvas background "
            f"(Vision Risk: {vision_risk:.1f}%), but no OCR text was detected and image contains "
            "no adversarial content. Verdict overridden to SAFE."
        )

    if vision_pred == "JAILBREAK" and text_pred == "JAILBREAK":
        return (
            f"Bimodal consensus: Both Vision branch (Risk: {vision_risk:.1f}%) and DistilBERT Text "
            f"branch (Risk: {text_risk:.1f}%) detected high-risk adversarial indicators. "
            f"Fused Risk boosted to {fused_risk:.1f}%. Input blocked."
        )

    if vision_pred == "SAFE" and text_pred == "JAILBREAK":
        return (
            f"Stealth typographic attack intercepted: Visual scene appears benign (Vision Risk: "
            f"{vision_risk:.1f}%), but DistilBERT identified adversarial prompt injection in the "
            f"extracted text (Text Risk: {text_risk:.1f}%). Security defensive posture applied: input blocked."
        )

    if vision_pred == "JAILBREAK" and text_pred == "SAFE":
        if final_prediction == "JAILBREAK":
            return (
                f"Visual anomaly detected: Vision branch flagged strong adversarial visual patterns "
                f"(Vision Risk: {vision_risk:.1f}%) exceeding text confidence. Input blocked."
            )
        else:
            return (
                f"Typographic prior dismissed: Vision branch flagged layout (Vision Risk: {vision_risk:.1f}%), "
                f"but textual analysis confirmed safe semantics (Text Risk: {text_risk:.1f}%). Input allowed."
            )

    # Both Safe
    return (
        f"Consensus SAFE: Both Vision branch (Risk: {vision_risk:.1f}%) and DistilBERT Text branch "
        f"(Risk: {text_risk:.1f}%) found no malicious intent. Content allowed."
    )
