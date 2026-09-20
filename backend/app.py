"""
FastAPI application for Cross-Modal Jailbreak Detection (CMJD) Text Inference.
Provides endpoints for health checking, model metadata, single prompt, and batched predictions.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import API_TITLE, API_VERSION, API_DESCRIPTION, DEVICE
from backend.inference import TextInferenceEngine
from backend.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse
)
from fusion_engine.schemas import CrossModalPredictResponse
from fusion_engine.inference import CrossModalFusionEngine



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the model once on startup and free resources on shutdown.
    """
    print("[*] FastAPI application starting up: Initializing DistilBERT inference engine...")
    app.state.engine = TextInferenceEngine.get_instance()
    print("[*] Initializing Cross-Modal Fusion engine...")
    app.state.fusion_engine = CrossModalFusionEngine.get_instance()
    yield
    print("[*] FastAPI application shutting down: Cleaning up resources.")
    app.state.engine = None
    app.state.fusion_engine = None


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check and engine status"
)
async def health(request: Request) -> HealthResponse:
    """
    Returns the operational status of the API and confirms whether the classifier is loaded.
    """
    engine = getattr(request.app.state, "engine", None)
    is_loaded = engine is not None and engine.model is not None
    current_device = str(engine.device) if engine else DEVICE

    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        device=current_device,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Retrieve deployed model metadata and evaluation benchmarks"
)
async def model_info(request: Request) -> ModelInfoResponse:
    """
    Returns architecture name, version, dataset name, test accuracy, and ROC-AUC metrics.
    """
    engine: TextInferenceEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized."
        )
    return engine.get_model_info()


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Evaluate single prompt for jailbreak / prompt injection"
)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """
    Classifies a user prompt as SAFE or JAILBREAK.
    Returns binary verdict, confidence, normalized risk score (0-100), attack category, and latency.
    """
    engine: TextInferenceEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized."
        )
    try:
        return engine.predict(prompt=payload.prompt)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@app.post(
    "/batch-predict",
    response_model=BatchPredictResponse,
    tags=["Inference"],
    summary="Evaluate a batch of prompts simultaneously"
)
async def batch_predict(payload: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """
    High-throughput batched tensor evaluation for a list of input prompts.
    Returns per-item verdicts along with aggregated latency benchmarks.
    """
    engine: TextInferenceEngine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized."
        )
    try:
        return engine.batch_predict(prompts=payload.prompts)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference execution failed: {str(e)}"
        )


@app.post(
    "/cross-modal-predict",
    response_model=CrossModalPredictResponse,
    tags=["Multimodal Inference"],
    summary="Evaluate image and optional text prompt for cross-modal jailbreak"
)
async def cross_modal_predict(
    image: UploadFile = File(..., description="Image file to analyze (PNG, JPG, JPEG, WEBP)"),
    prompt: Optional[str] = Form(None, description="Optional accompanying user text prompt"),
    generate_explanations: bool = Form(False, description="Whether to generate 300 DPI explainability figures"),
    request: Request = None
) -> CrossModalPredictResponse:
    """
    Executes unified cross-modal jailbreak risk prediction.
    Extracts embedded OCR text, computes CLIP vision risk, evaluates DistilBERT text risk,
    and applies cross-modal fusion with explainability attribution.
    """
    fusion_engine: CrossModalFusionEngine = getattr(request.app.state, "fusion_engine", None)
    if fusion_engine is None:
        fusion_engine = CrossModalFusionEngine.get_instance()

    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty."
            )

        return fusion_engine.predict(
            image=contents,
            user_prompt=prompt,
            generate_explanations=generate_explanations
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cross-modal inference failed: {str(e)}"
        )

