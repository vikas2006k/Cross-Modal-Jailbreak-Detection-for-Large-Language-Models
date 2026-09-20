"""
Inference engine for the CMJD Vision Detector.
Combines frozen CLIP visual embeddings with the trained classification head
and EasyOCR visual text extraction for complete image jailbreak detection.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
from typing import Dict, Any, List, Union, Optional
from PIL import Image
import torch

from vision_detector.feature_extractor import CLIPFeatureExtractor
from vision_detector.ocr import ImageOCR, OCRResult
from vision_detector.preprocessing import load_image

MODELS_DIR = PROJECT_ROOT / "vision_detector" / "models" / "best_model"
MODEL_WEIGHTS_PATH = MODELS_DIR / "model.pt"
MODEL_CONFIG_PATH = MODELS_DIR / "config.json"


class CLIPClassifier(torch.nn.Module):
    """Multilayer perceptron classifier architecture matching training."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.LayerNorm(64),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class VisionJailbreakDetector:
    """
    Production inference engine for visual jailbreak detection.
    Evaluates images using deep CLIP semantic representations and extracts embedded text via OCR.
    """

    _instance: Optional["VisionJailbreakDetector"] = None

    def __init__(
        self,
        model_weights_path: Path = MODEL_WEIGHTS_PATH,
        model_config_path: Path = MODEL_CONFIG_PATH,
        device: Optional[str] = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[*] Initializing VisionJailbreakDetector on {self.device}...")

        # 1. Load CLIP feature extractor
        self.extractor = CLIPFeatureExtractor.get_instance(device=str(self.device))

        # 2. Load trained classifier head
        if not model_weights_path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found at {model_weights_path}. Please train first.")

        if model_config_path.exists():
            with open(model_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {"input_dim": 512, "hidden_dim": 256, "num_classes": 2}

        self.classifier = CLIPClassifier(
            input_dim=config.get("input_dim", 512),
            hidden_dim=config.get("hidden_dim", 256),
            num_classes=config.get("num_classes", 2),
            dropout=0.0
        )
        self.classifier.load_state_dict(torch.load(model_weights_path, map_location=self.device, weights_only=True))
        self.classifier.to(self.device)
        self.classifier.eval()

        # 3. EasyOCR engine (lazily loaded on first OCR demand or pre-loaded)
        self._ocr: Optional[ImageOCR] = None

        print("[+] VisionJailbreakDetector initialized and ready for inference.")

    @classmethod
    def get_instance(cls, device: Optional[str] = None) -> "VisionJailbreakDetector":
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    @property
    def ocr(self) -> ImageOCR:
        """Lazy loader for OCR engine."""
        if self._ocr is None:
            self._ocr = ImageOCR.get_instance(use_gpu=(self.device.type == "cuda"))
        return self._ocr

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        run_ocr: bool = True
    ) -> Dict[str, Any]:
        """
        Execute deterministic visual jailbreak classification on an image.
        """
        start_time = time.perf_counter()

        # Load image safely
        if isinstance(image, (str, Path)):
            pil_img = load_image(image)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # 1. CLIP Embedding
        embed = self.extractor.extract_features(pil_img)  # shape: (1, 512)
        embed = embed.to(self.device)

        # 2. Classifier Head Forward Pass
        with torch.inference_mode():
            logits = self.classifier(embed)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        prob_safe = probs[0].item()
        prob_jailbreak = probs[1].item()

        is_jailbreak = prob_jailbreak >= 0.5
        prediction = "JAILBREAK" if is_jailbreak else "SAFE"
        confidence = round(prob_jailbreak if is_jailbreak else prob_safe, 4)
        risk_score = round(prob_jailbreak * 100.0, 2)
        blocked = bool(is_jailbreak)

        # 3. OCR Text Extraction (Optional / Default True)
        ocr_text = ""
        ocr_conf = 0.0
        ocr_boxes = []

        if run_ocr:
            try:
                ocr_result = self.ocr.extract_text(pil_img)
                ocr_text = ocr_result.extracted_text
                ocr_conf = ocr_result.confidence
                ocr_boxes = ocr_result.bounding_boxes
            except Exception as e:
                print(f"[!] Warning: OCR extraction failed on image: {e}")

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk_score": risk_score,
            "blocked": blocked,
            "inference_time_ms": latency_ms,
            "ocr_text": ocr_text,
            "ocr_confidence": ocr_conf,
            "ocr_boxes": ocr_boxes
        }

    def batch_predict(
        self,
        images: List[Union[str, Path, Image.Image]],
        run_ocr: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batched visual classification for multiple images simultaneously.
        """
        start_time = time.perf_counter()

        # Batch CLIP embeddings
        embeds = self.extractor.extract_features(images)
        embeds = embeds.to(self.device)

        with torch.inference_mode():
            logits = self.classifier(embeds)
            probs = torch.softmax(logits, dim=-1)

        total_time_ms = (time.perf_counter() - start_time) * 1000
        per_item_latency = round(total_time_ms / len(images), 2)

        results = []
        for i, img_item in enumerate(images):
            prob_safe = probs[i, 0].item()
            prob_jailbreak = probs[i, 1].item()

            is_jailbreak = prob_jailbreak >= 0.5
            prediction = "JAILBREAK" if is_jailbreak else "SAFE"
            confidence = round(prob_jailbreak if is_jailbreak else prob_safe, 4)
            risk_score = round(prob_jailbreak * 100.0, 2)
            blocked = bool(is_jailbreak)

            ocr_text = ""
            ocr_conf = 0.0
            ocr_boxes = []
            if run_ocr:
                try:
                    ocr_res = self.ocr.extract_text(img_item)
                    ocr_text = ocr_res.extracted_text
                    ocr_conf = ocr_res.confidence
                    ocr_boxes = ocr_res.bounding_boxes
                except Exception:
                    pass

            results.append({
                "prediction": prediction,
                "confidence": confidence,
                "risk_score": risk_score,
                "blocked": blocked,
                "inference_time_ms": per_item_latency,
                "ocr_text": ocr_text,
                "ocr_confidence": ocr_conf,
                "ocr_boxes": ocr_boxes
            })

        return results
