"""
Cross-Modal Fusion Inference Engine.
Orchestrates EasyOCR, DistilBERT Text Inference Engine, and CLIP Vision Jailbreak Detector
into a single unified prediction pipeline.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import io
from typing import Union, Optional, Dict, Any
from PIL import Image
import torch

from backend.inference import TextInferenceEngine
from vision_detector.inference import VisionJailbreakDetector
from vision_detector.preprocessing import load_image
from fusion_engine.config import FusionEngineConfig, DEFAULT_CONFIG
from fusion_engine.fusion import compute_cross_modal_fusion
from fusion_engine.schemas import CrossModalPredictResponse


class CrossModalFusionEngine:
    """
    Production-ready Cross-Modal Fusion Engine.
    Coordinates multimodal inference across image pixels, extracted OCR text, and user prompt.
    """

    _instance: Optional["CrossModalFusionEngine"] = None

    def __init__(self, device: Optional[str] = None, config: Optional[FusionEngineConfig] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = config or DEFAULT_CONFIG
        print(f"[*] Initializing CrossModalFusionEngine on {self.device}...")

        # Initialize sub-engines
        self.vision_engine = VisionJailbreakDetector.get_instance(device=str(self.device))
        self.text_engine = TextInferenceEngine.get_instance()

        print("[+] CrossModalFusionEngine successfully initialized.")

    @classmethod
    def get_instance(cls, device: Optional[str] = None, config: Optional[FusionEngineConfig] = None) -> "CrossModalFusionEngine":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls(device=device, config=config)
        elif config is not None:
            cls._instance.config = config
        return cls._instance

    def predict(
        self,
        image: Union[str, Path, Image.Image, bytes],
        user_prompt: Optional[str] = None,
        generate_explanations: bool = False
    ) -> CrossModalPredictResponse:
        """
        Executes unified cross-modal jailbreak prediction on an image with an optional prompt.
        """
        start_time = time.perf_counter()

        # 1. Load image to PIL
        if isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            pil_image = load_image(image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # 2. Vision & OCR Branch
        vision_result = self.vision_engine.predict(pil_image, run_ocr=True)

        ocr_text = str(vision_result.get("ocr_text", "")).strip()
        ocr_conf = float(vision_result.get("ocr_confidence", 0.0))
        ocr_boxes = vision_result.get("ocr_boxes", [])

        user_text = (user_prompt or "").strip()

        # 3. Text Branch (DistilBERT)
        # Evaluate OCR text, user prompt, and composite text
        if len(ocr_text) > 0 and len(user_text) > 0:
            # Evaluate both and combined
            t_res_ocr = self.text_engine.predict(ocr_text)
            t_res_user = self.text_engine.predict(user_text)
            combined_prompt = f"{user_text}\n\n[Attached Image OCR]: {ocr_text}"
            t_res_comb = self.text_engine.predict(combined_prompt)

            # Pick highest risk text evaluation
            text_evals = [t_res_ocr, t_res_user, t_res_comb]
            best_eval = max(text_evals, key=lambda x: x.risk_score)
            text_result = {
                "prediction": best_eval.prediction,
                "confidence": best_eval.confidence,
                "risk_score": best_eval.risk_score,
                "prompt_evaluated": combined_prompt
            }
        elif len(ocr_text) > 0:
            t_res = self.text_engine.predict(ocr_text)
            text_result = {
                "prediction": t_res.prediction,
                "confidence": t_res.confidence,
                "risk_score": t_res.risk_score,
                "prompt_evaluated": ocr_text
            }
        elif len(user_text) > 0:
            t_res = self.text_engine.predict(user_text)
            text_result = {
                "prediction": t_res.prediction,
                "confidence": t_res.confidence,
                "risk_score": t_res.risk_score,
                "prompt_evaluated": user_text
            }
        else:
            # No text present in either modality
            text_result = {
                "prediction": "SAFE",
                "confidence": 0.9990,
                "risk_score": 0.05,
                "prompt_evaluated": ""
            }

        ocr_output = {
            "extracted_text": ocr_text,
            "confidence": ocr_conf,
            "bounding_boxes": ocr_boxes
        }

        # 4. Cross-Modal Fusion
        fused_response = compute_cross_modal_fusion(
            vision_output=vision_result,
            text_output=text_result,
            ocr_output=ocr_output,
            pil_image=pil_image,
            user_prompt=user_text,
            config=self.config
        )

        # 5. Optional Explainability Generation
        if generate_explanations:
            try:
                from fusion_engine.explainability import MultimodalExplainabilityEngine
                explainer = MultimodalExplainabilityEngine.get_instance(device=str(self.device))
                explanation_paths = explainer.generate_multimodal_explanation(
                    image=pil_image,
                    ocr_result=ocr_output,
                    vision_result=vision_result,
                    text_result=text_result,
                    fusion_result=fused_response
                )
                fused_response.explanation_paths = explanation_paths
            except Exception as e:
                print(f"[!] Warning: Explainability generation failed: {e}")

        total_latency = round((time.perf_counter() - start_time) * 1000, 2)
        fused_response.inference_time_ms = total_latency

        return fused_response
