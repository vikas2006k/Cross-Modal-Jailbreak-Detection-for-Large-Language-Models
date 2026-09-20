import sys
from pathlib import Path

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_api import (
    client,
    test_health_endpoint,
    test_model_info_endpoint,
    test_predict_safe_prompt,
    test_predict_jailbreak_prompt,
    test_predict_empty_prompt,
    test_predict_long_prompt,
    test_batch_predict_endpoint,
    test_inference_audit_logging
)

__all__ = [
    "client",
    "test_health_endpoint",
    "test_model_info_endpoint",
    "test_predict_safe_prompt",
    "test_predict_jailbreak_prompt",
    "test_predict_empty_prompt",
    "test_predict_long_prompt",
    "test_batch_predict_endpoint",
    "test_inference_audit_logging"
]
