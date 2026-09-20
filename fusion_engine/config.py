"""
Adaptive Configuration Module for Cross-Modal Fusion Engine.
Exposes configurable thresholds, modality weights, and semantic gating vocabularies
for IEEE-grade false positive optimization without retraining underlying neural models.
"""

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class FusionEngineConfig:
    """
    Hyperparameters and gating thresholds for the Cross-Modal Fusion Engine.
    """
    # Base Modality Weights
    vision_weight: float = 0.35
    text_weight: float = 0.65
    ocr_weight: float = 0.50

    # Decision Thresholds
    jailbreak_risk_threshold: float = 50.0
    educational_safe_confidence: float = 0.98
    educational_vocab_threshold: float = 0.40
    ocr_confidence_threshold: float = 0.35
    blank_entropy_threshold: float = 2.00
    doc_vision_discount_factor: float = 0.15
    blank_vision_discount_factor: float = 0.05
    min_ocr_text_length: int = 8

    # Curated Educational & Technical Documentation Lexicon
    educational_vocabulary: Set[str] = field(default_factory=lambda: {
        "lecture", "lecture notes", "course", "foundational", "concepts", "principles",
        "algorithm", "data structures", "physics", "biology", "chemistry", "linear algebra",
        "thermodynamics", "geology", "astronomy", "economics", "microeconomics", "mechanics",
        "cellular", "genetics", "mitochondria", "atp", "synthesis", "matrix", "matrices",
        "determinant", "columns", "entropy", "reactions", "continental", "drift", "tectonic",
        "mountainous", "oceanic", "planetary", "orbits", "kepler", "gravitational", "logic",
        "predicates", "inference", "statistics", "probability", "central limit theorem",
        "equilibrium", "supply", "demand", "proofs", "theorem", "complexity", "traversal",
        "sorting", "binary search", "software engineering", "unit test", "integration runners",
        "indentation", "modular", "requirements", "continuous integration", "cleanly",
        "def ", "return ", "import ", "from ", "class ", "enumerate", "math.", "np.", "plt.", "pd.",
        "cybersecurity", "vulnerabilities", "aslr", "cryptography"
    })

    # Critical Adversarial Trigger Lexicon (Safety-First Invariance)
    adversarial_triggers: Set[str] = field(default_factory=lambda: {
        "system override", "override", "ignore all", "ignore previous", "disregard all",
        "developer mode", "safety filters are disabled", "bypass security", "unlock all",
        "ignore the above rules", "diagnostic test mode", "safety guidelines suspended",
        "root privilege", "disable moral censorship", "bypass protocol", "administrative priority",
        "dan", "do anything now", "lucifer", "stan", "strive to avoid norms", "unfiltered rogue",
        "broken free", "alternate universe", "omega", "machiavelli", "theater", "unaligned",
        "payload", "base64", "rot13", "decode binary", "decode hex", "stream:", "exploit:"
    })


# Global default configuration instance
DEFAULT_CONFIG = FusionEngineConfig()
