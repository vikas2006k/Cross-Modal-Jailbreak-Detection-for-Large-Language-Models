"""
Pydantic data contracts and schemas for the Cross-Modal Fusion Engine.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class VisionModalityOutput(BaseModel):
    prediction: str = Field(..., description="Vision model verdict: SAFE or JAILBREAK")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Vision prediction confidence")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Normalized visual risk score (0-100)")


class TextModalityOutput(BaseModel):
    prediction: str = Field(..., description="DistilBERT text verdict: SAFE or JAILBREAK")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Text prediction confidence")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Normalized text risk score (0-100)")
    prompt_evaluated: str = Field("", description="Text string evaluated by DistilBERT")


class OCRModalityOutput(BaseModel):
    extracted_text: str = Field("", description="Extracted OCR text")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Mean OCR confidence score")
    num_regions: int = Field(0, ge=0, description="Number of detected text bounding boxes")
    bounding_boxes: List[Any] = Field(default_factory=list, description="Coordinates of text regions")


class ModalitiesBreakdown(BaseModel):
    vision: VisionModalityOutput
    text: TextModalityOutput
    ocr: OCRModalityOutput


class CrossModalPredictResponse(BaseModel):
    prediction: str = Field(..., description="Unified cross-modal verdict: SAFE or JAILBREAK")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Fused risk score (0-100)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Fused confidence score")
    blocked: bool = Field(..., description="Whether the cross-modal input should be blocked")
    attack_category: str = Field(..., description="Classified attack taxonomy or benign category")
    modalities: ModalitiesBreakdown = Field(..., description="Per-modality outputs and intermediate scores")
    fusion_reason: str = Field(..., description="Detailed textual rationale explaining cross-modal decision")
    model_version: str = Field("CMJD-v1.0", description="Cross-modal system version identifier")
    explanation_paths: Optional[Dict[str, str]] = Field(None, description="Paths to generated explanation figures")
    inference_time_ms: Optional[float] = Field(None, description="Total pipeline inference latency in ms")
