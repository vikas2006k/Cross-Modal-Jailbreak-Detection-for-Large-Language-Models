"""
Configuration module for the Cross-Modal Jailbreak Detection (CMJD) text inference engine.
"""

from pathlib import Path
import os
import torch

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model configuration
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "text_classifier" / "best_model"
MODEL_PATH = Path(os.getenv("CMJD_MODEL_PATH", str(DEFAULT_MODEL_DIR)))

# Metrics and metadata configuration
METRICS_PATH = PROJECT_ROOT / "models" / "text_classifier" / "metrics" / "metrics.json"
ROC_AUC_PATH = PROJECT_ROOT / "models" / "text_classifier" / "metrics" / "roc_auc.json"

# Logging configuration
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE_PATH = LOGS_DIR / "inference_logs.csv"

# Inference parameters
MAX_LENGTH = int(os.getenv("CMJD_MAX_LENGTH", "512"))
CLASSIFICATION_THRESHOLD = float(os.getenv("CMJD_THRESHOLD", "0.5"))

# Device auto-detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# API metadata
API_TITLE = "Cross-Modal Jailbreak Detection (CMJD) — Text Inference API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Production-grade FastAPI inference microservice for real-time text jailbreak "
    "and prompt injection detection using fine-tuned DistilBERT on the CMJD-10K benchmark."
)
