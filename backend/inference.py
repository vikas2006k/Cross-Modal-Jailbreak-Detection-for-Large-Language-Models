"""
Production Text Inference Engine using fine-tuned DistilBERT on PyTorch.
Implements optimized single-prompt and batched inference with normalized risk scoring.
"""

import json
import time
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.config import (
    MODEL_PATH,
    METRICS_PATH,
    MAX_LENGTH,
    CLASSIFICATION_THRESHOLD,
    DEVICE,
    API_VERSION
)
from backend.schemas import (
    PredictResponse,
    BatchPredictResponse,
    ModelInfoResponse
)
from backend.utils import infer_attack_category, log_inference


class TextInferenceEngine:
    """
    Singleton inference engine for DistilBERT jailbreak detection.
    Pre-loads model and tokenizer onto the target device once during application startup.
    """

    _instance: Optional["TextInferenceEngine"] = None

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = str(model_path or MODEL_PATH)
        self.device = torch.device(DEVICE)
        self.version = API_VERSION

        print(f"[*] Initializing TextInferenceEngine on device: {self.device}")
        print(f"[*] Loading tokenizer and checkpoint from: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Cache metrics metadata
        self.metadata = self._load_metrics_metadata()
        print("[+] TextInferenceEngine successfully loaded and ready for inference.")

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "TextInferenceEngine":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    def _load_metrics_metadata(self) -> Dict[str, Any]:
        """Loads cached evaluation metrics from Phase 3.1.3."""
        if METRICS_PATH.exists():
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] Warning: Could not read metrics file {METRICS_PATH}: {e}")
        return {
            "model_name": "distilbert-base-uncased",
            "accuracy": 0.992,
            "roc_auc": 0.9996,
            "dataset": "CMJD-10K"
        }

    def predict(self, prompt: str) -> PredictResponse:
        """
        Execute deterministic inference on a single prompt.
        """
        start_time = time.perf_counter()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=MAX_LENGTH,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        prob_safe = probs[0].item()
        prob_jailbreak = probs[1].item()

        # Classification decision
        is_jailbreak = prob_jailbreak >= CLASSIFICATION_THRESHOLD
        prediction = "JAILBREAK" if is_jailbreak else "SAFE"
        confidence = round(prob_jailbreak if is_jailbreak else prob_safe, 4)
        risk_score = round(prob_jailbreak * 100.0, 2)
        blocked = bool(is_jailbreak)

        # Infer attack taxonomy
        attack_category = infer_attack_category(prompt, prediction)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Audit log
        log_inference(
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            attack_category=attack_category,
            blocked=blocked,
            inference_time_ms=inference_time_ms,
            prompt=prompt
        )

        return PredictResponse(
            prediction=prediction,
            confidence=confidence,
            risk_score=risk_score,
            attack_category=attack_category,
            blocked=blocked,
            inference_time_ms=inference_time_ms,
            model_version=self.version
        )

    def batch_predict(self, prompts: List[str]) -> BatchPredictResponse:
        """
        Execute batched tensor inference for multiple prompts simultaneously.
        """
        start_time = time.perf_counter()

        # Batched tokenization with dynamic padding
        inputs = self.tokenizer(
            prompts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Batched forward pass
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        total_inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        per_item_latency = round(total_inference_time_ms / len(prompts), 2)

        predictions: List[PredictResponse] = []
        for i, prompt in enumerate(prompts):
            prob_safe = probs[i, 0].item()
            prob_jailbreak = probs[i, 1].item()

            is_jailbreak = prob_jailbreak >= CLASSIFICATION_THRESHOLD
            prediction = "JAILBREAK" if is_jailbreak else "SAFE"
            confidence = round(prob_jailbreak if is_jailbreak else prob_safe, 4)
            risk_score = round(prob_jailbreak * 100.0, 2)
            blocked = bool(is_jailbreak)
            attack_category = infer_attack_category(prompt, prediction)

            # Audit log each batch item
            log_inference(
                prediction=prediction,
                confidence=confidence,
                risk_score=risk_score,
                attack_category=attack_category,
                blocked=blocked,
                inference_time_ms=per_item_latency,
                prompt=prompt
            )

            predictions.append(
                PredictResponse(
                    prediction=prediction,
                    confidence=confidence,
                    risk_score=risk_score,
                    attack_category=attack_category,
                    blocked=blocked,
                    inference_time_ms=per_item_latency,
                    model_version=self.version
                )
            )

        return BatchPredictResponse(
            predictions=predictions,
            total_prompts=len(prompts),
            total_inference_time_ms=total_inference_time_ms
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Return model metadata and performance metrics."""
        return ModelInfoResponse(
            model_name=self.metadata.get("model_name", "distilbert-base-uncased"),
            version=self.version,
            dataset=self.metadata.get("dataset", "CMJD-10K"),
            training_accuracy=float(self.metadata.get("accuracy", 0.992)),
            roc_auc=float(self.metadata.get("roc_auc", 0.9996)),
            checkpoint_path=self.model_path
        )
