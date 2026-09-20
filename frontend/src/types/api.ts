export type Verdict = 'SAFE' | 'JAILBREAK';

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  timestamp: string;
}

export interface ModelInfoResponse {
  architecture: string;
  version: string;
  task: string;
  dataset_name: string;
  num_classes: number;
  classes: string[];
  test_accuracy: number;
  roc_auc: number;
  device: string;
}

export interface PredictRequest {
  prompt: string;
}

export interface PredictResponse {
  prediction: Verdict;
  confidence: number;
  risk_score: number;
  attack_category: string;
  blocked?: boolean;
  inference_time_ms?: number;
  latency_ms: number;
  prompt_evaluated: string;
  model_version?: string;
  timestamp?: string;
}

export interface BatchPredictRequest {
  prompts: string[];
}

export interface BatchPredictResponse {
  predictions?: PredictResponse[];
  results: PredictResponse[];
  total_prompts: number;
  jailbreaks_detected: number;
  safe_prompts: number;
  total_inference_time_ms?: number;
  total_latency_ms: number;
  average_latency_ms: number;
}

export interface BoundingBox {
  box: number[][];
  text: string;
  confidence: number;
}

export interface VisionModalityOutput {
  prediction: string;
  confidence: number;
  risk_score: number;
}

export interface TextModalityOutput {
  prediction: string;
  confidence: number;
  risk_score: number;
  prompt_evaluated: string;
}

export interface OCRModalityOutput {
  extracted_text: string;
  confidence: number;
  num_regions: number;
  bounding_boxes: any[];
}

export interface ModalitiesBreakdown {
  vision: VisionModalityOutput;
  text: TextModalityOutput;
  ocr: OCRModalityOutput;
}

export interface CrossModalPredictResponse {
  prediction: Verdict;
  confidence: number;
  risk_score: number;
  attack_category: string;
  blocked: boolean;
  fusion_reason: string;
  model_version?: string;
  explanation_paths?: Record<string, string> | null;
  inference_time_ms?: number;
  latency_ms: number;
  modalities?: ModalitiesBreakdown;

  // Normalized convenience properties for UI rendering
  vision_prediction: Verdict;
  vision_confidence: number;
  vision_risk_score: number;
  text_prediction: Verdict;
  text_confidence: number;
  text_risk_score: number;
  ocr_extracted_text: string;
  ocr_confidence: number;
  ocr_bounding_boxes?: BoundingBox[];
  vision_attention_heatmap?: string;
  modality_weights: {
    vision: number;
    text: number;
    ocr: number;
  };
  timestamp?: string;
}

export interface ScanRecord {
  id: string;
  type: 'TEXT' | 'IMAGE' | 'CROSS_MODAL';
  timestamp: string;
  verdict: Verdict;
  confidence: number;
  riskScore: number;
  attackCategory: string;
  latencyMs: number;
  inputPrompt?: string;
  imageName?: string;
  imageUrl?: string;
  ocrText?: string;
  visionRisk?: number;
  textRisk?: number;
  reason?: string;
}
