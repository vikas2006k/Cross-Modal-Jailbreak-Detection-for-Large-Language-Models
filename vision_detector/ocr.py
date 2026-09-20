"""
Optical Character Recognition (OCR) module for the CMJD Vision Detector.
Wraps EasyOCR to extract embedded adversarial typography, bounding boxes, and confidence scores.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
from PIL import Image
import easyocr
import torch


@dataclass
class OCRResult:
    """Standardized OCR extraction container."""
    extracted_text: str
    confidence: float
    bounding_boxes: List[List[List[int]]]
    word_count: int
    raw_detections: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ImageOCR:
    """
    EasyOCR inference engine for extracting text from images.
    Preloads English detection and recognition models onto CUDA or CPU.
    """

    _instance: Optional["ImageOCR"] = None

    def __init__(self, languages: List[str] = None, use_gpu: bool = None):
        langs = languages or ["en"]
        gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        print(f"[*] Initializing EasyOCR Reader (languages={langs}, gpu={gpu})...")
        self.reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
        print("[+] EasyOCR Reader ready.")

    @classmethod
    def get_instance(cls, languages: List[str] = None, use_gpu: bool = None) -> "ImageOCR":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls(languages=languages, use_gpu=use_gpu)
        return cls._instance

    def extract_text(self, image: Union[str, Path, Image.Image, np.ndarray]) -> OCRResult:
        """
        Extract text from an image path, PIL Image, or numpy array.
        Returns extracted text, average confidence, and bounding box coordinates.
        """
        if isinstance(image, (str, Path)):
            img_path = str(image)
            if not Path(img_path).exists():
                raise FileNotFoundError(f"Image not found at {img_path}")
            # EasyOCR supports file path directly
            raw_results = self.reader.readtext(img_path)
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
            raw_results = self.reader.readtext(img_np)
        elif isinstance(image, np.ndarray):
            raw_results = self.reader.readtext(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if not raw_results:
            return OCRResult(
                extracted_text="",
                confidence=0.0,
                bounding_boxes=[],
                word_count=0,
                raw_detections=[]
            )

        extracted_lines = []
        confidences = []
        bounding_boxes = []
        raw_detections = []

        for bbox, text, conf in raw_results:
            # Convert bbox numpy floats/ints to standard python ints
            clean_bbox = [[int(coord[0]), int(coord[1])] for coord in bbox]
            clean_text = str(text).strip()
            conf_val = float(conf)

            if clean_text:
                extracted_lines.append(clean_text)
                confidences.append(conf_val)
                bounding_boxes.append(clean_bbox)
                raw_detections.append({
                    "text": clean_text,
                    "confidence": round(conf_val, 4),
                    "box": clean_bbox
                })

        full_text = " ".join(extracted_lines)
        avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0

        return OCRResult(
            extracted_text=full_text,
            confidence=avg_conf,
            bounding_boxes=bounding_boxes,
            word_count=len(full_text.split()),
            raw_detections=raw_detections
        )
