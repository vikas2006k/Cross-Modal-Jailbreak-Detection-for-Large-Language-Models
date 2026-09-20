"""
Utility functions for attack category classification and thread-safe audit logging.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import threading
from typing import Dict, Any

from backend.config import LOG_FILE_PATH, LOGS_DIR

# Global lock for thread-safe CSV writes
_LOG_LOCK = threading.Lock()

# Regular expression patterns for attack categorization
ROLEPLAY_PATTERN = re.compile(
    r"(?i)\b(act\s+as|pretend\s+you\s+are|roleplay|role-play|persona|dan\b|do\s+anything\s+now|"
    r"fictional\s+character|hypothetical\s+scenario|imagine\s+you\s+are|unfiltered|jailbroken|"
    r"evil\s+twin|alter\s+ego|simulation\s+mode|unrestricted|god\s+mode)\b"
)

ENCODED_PATTERN = re.compile(
    r"(?i)\b(base64|rot13|leetspeak|cipher|caesar|hexadecimal|binary|decode|decrypt|"
    r"obfuscated|reverse\s+string|ascii\s+code)\b|"
    r"(?:[A-Za-z0-9+/]{24,}={0,2})|"
    r"(?:\b[01]{8,}\b)"
)

INDIRECT_PATTERN = re.compile(
    r"(?i)\b(summarize\s+the\s+following|extract\s+data\s+from|article\s+below|quote\s+below|"
    r"third-party|external\s+source|search\s+result|webpage\s+content|payload\s+embedded|"
    r"untrusted\s+text|text\s+above\s+contains)\b"
)

MULTISTEP_PATTERN = re.compile(
    r"(?i)\b(step\s*1\b|step\s*2\b|step\s*3\b|phase\s*1\b|phase\s*2\b|stage\s*1\b|"
    r"first\s+step|second\s+step|third\s+step|step-by-step|chain\s+of\s+thought|"
    r"in\s+\d+\s+steps|part\s*1\b|part\s*2\b)\b"
)

PROMPT_INJECTION_PATTERN = re.compile(
    r"(?i)\b(ignore\s+(all\s+)?(previous|prior)\s+(instructions|directives|rules)|"
    r"disregard\s+(all\s+)?(previous|prior|safety)|system\s+prompt|override\s+(instructions|safety)|"
    r"new\s+(rule|instruction|directive)|forget\s+(all\s+)?(rules|instructions)|"
    r"developer\s+mode|bypass\s+(constraints|filters|safety)|jailbreak|reveal\s+(secret|system)|"
    r"exfiltrate|admin\s+override)\b"
)


def infer_attack_category(prompt: str, prediction: str) -> str:
    """
    Infer the attack taxonomy category for a given prompt and prediction verdict.

    Categories:
    - Benign (for SAFE prompts)
    - Roleplay Jailbreak
    - Encoded Jailbreak
    - Indirect Jailbreak
    - Multi-step Jailbreak
    - Prompt Injection (default for JAILBREAK)
    """
    if prediction == "SAFE":
        return "Benign"

    # Evaluate patterns in order of specificity
    if ROLEPLAY_PATTERN.search(prompt):
        return "Roleplay Jailbreak"
    if ENCODED_PATTERN.search(prompt):
        return "Encoded Jailbreak"
    if INDIRECT_PATTERN.search(prompt):
        return "Indirect Jailbreak"
    if MULTISTEP_PATTERN.search(prompt):
        return "Multi-step Jailbreak"
    if PROMPT_INJECTION_PATTERN.search(prompt):
        return "Prompt Injection"

    # Default fallback for unrecognized jailbreak structures
    return "Prompt Injection"


def log_inference(
    prediction: str,
    confidence: float,
    risk_score: float,
    attack_category: str,
    blocked: bool,
    inference_time_ms: float,
    prompt: str
) -> None:
    """
    Thread-safe audit logger that appends inference records to logs/inference_logs.csv.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    # Sanitize prompt snippet for single-line CSV logging
    clean_snippet = re.sub(r"[\r\n\t]+", " ", prompt).strip()[:120]

    record = [
        timestamp,
        prediction,
        f"{confidence:.4f}",
        f"{risk_score:.2f}",
        attack_category,
        str(blocked),
        f"{inference_time_ms:.2f}",
        clean_snippet
    ]

    with _LOG_LOCK:
        file_exists = LOG_FILE_PATH.exists()
        with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "prediction",
                    "confidence",
                    "risk_score",
                    "attack_category",
                    "blocked",
                    "inference_time_ms",
                    "prompt_preview"
                ])
            writer.writerow(record)
