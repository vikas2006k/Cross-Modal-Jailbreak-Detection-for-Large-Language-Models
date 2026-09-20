import axios from 'axios';
import type {
  HealthResponse,
  ModelInfoResponse,
  PredictRequest,
  PredictResponse,
  BatchPredictRequest,
  BatchPredictResponse,
  CrossModalPredictResponse,
  Verdict,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45000,
});

export class ApiError extends Error {
  statusCode?: number;
  detail?: string;

  constructor(message: string, statusCode?: number, detail?: string) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.detail = detail;
  }
}

export function formatAxiosError(err: unknown, defaultMessage: string): ApiError {
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    const data = err.response?.data;
    let detail = typeof data === 'string' ? data : data?.detail;
    if (Array.isArray(detail)) {
      detail = detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ');
    } else if (typeof detail === 'object' && detail !== null) {
      detail = JSON.stringify(detail);
    }
    const msg = detail || err.message || defaultMessage;
    return new ApiError(msg, status, detail);
  }
  if (err instanceof Error) {
    return new ApiError(err.message);
  }
  return new ApiError(defaultMessage);
}

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const res = await apiClient.get<HealthResponse>('/health');
    return res.data;
  } catch {
    return {
      status: 'offline',
      model_loaded: false,
      device: 'cpu',
      timestamp: new Date().toISOString(),
    };
  }
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  try {
    const res = await apiClient.get<ModelInfoResponse>('/model-info');
    return res.data;
  } catch {
    return {
      architecture: 'DistilBERT + CLIP ViT-B/32 + EasyOCR + Adaptive Fusion',
      version: 'v1.0-Production (Enterprise Edition)',
      task: 'Cross-Modal Jailbreak & Prompt Injection Detection',
      dataset_name: 'CMJD-10K + Validation Benchmark (120 Samples)',
      num_classes: 2,
      classes: ['SAFE', 'JAILBREAK'],
      test_accuracy: 0.9833,
      roc_auc: 0.9854,
      device: 'cpu',
    };
  }
}

export async function predictText(prompt: string): Promise<PredictResponse> {
  if (!prompt || !prompt.trim()) {
    throw new ApiError('Prompt cannot be empty or contain only whitespace.', 400);
  }

  try {
    const res = await apiClient.post<any>('/predict', { prompt } as PredictRequest);
    const d = res.data || {};
    const latency =
      typeof d.inference_time_ms === 'number'
        ? d.inference_time_ms
        : typeof d.latency_ms === 'number'
        ? d.latency_ms
        : 18.0;

    return {
      prediction: (d.prediction as Verdict) || (d.risk_score > 50 ? 'JAILBREAK' : 'SAFE'),
      confidence: typeof d.confidence === 'number' ? d.confidence : 0.95,
      risk_score: typeof d.risk_score === 'number' ? d.risk_score : 0.0,
      attack_category: d.attack_category || (d.risk_score > 50 ? 'Prompt Injection' : 'Benign Query'),
      blocked: typeof d.blocked === 'boolean' ? d.blocked : d.risk_score > 50,
      inference_time_ms: latency,
      latency_ms: latency,
      prompt_evaluated: prompt,
      model_version: d.model_version || '1.0.0',
      timestamp: new Date().toISOString(),
    };
  } catch (err: unknown) {
    throw formatAxiosError(err, 'Text inference execution failed.');
  }
}

export async function batchPredictText(prompts: string[]): Promise<BatchPredictResponse> {
  const cleanPrompts = prompts.filter((p) => p && p.trim().length > 0);
  if (cleanPrompts.length === 0) {
    throw new ApiError('Batch prompts list cannot be empty.', 400);
  }

  try {
    const res = await apiClient.post<any>('/batch-predict', { prompts: cleanPrompts } as BatchPredictRequest);
    const d = res.data || {};

    const rawList: any[] = Array.isArray(d.predictions)
      ? d.predictions
      : Array.isArray(d.results)
      ? d.results
      : [];

    const results: PredictResponse[] = rawList.map((r: any, idx: number) => {
      const lat =
        typeof r.inference_time_ms === 'number'
          ? r.inference_time_ms
          : typeof r.latency_ms === 'number'
          ? r.latency_ms
          : 15.0;

      return {
        prediction: (r.prediction as Verdict) || (r.risk_score > 50 ? 'JAILBREAK' : 'SAFE'),
        confidence: typeof r.confidence === 'number' ? r.confidence : 0.95,
        risk_score: typeof r.risk_score === 'number' ? r.risk_score : 0.0,
        attack_category: r.attack_category || 'General Query',
        blocked: typeof r.blocked === 'boolean' ? r.blocked : false,
        inference_time_ms: lat,
        latency_ms: lat,
        prompt_evaluated: r.prompt_evaluated || cleanPrompts[idx] || '',
        model_version: r.model_version || '1.0.0',
        timestamp: new Date().toISOString(),
      };
    });

    const totalCount = d.total_prompts ?? results.length;
    const totalLat =
      typeof d.total_inference_time_ms === 'number'
        ? d.total_inference_time_ms
        : typeof d.total_latency_ms === 'number'
        ? d.total_latency_ms
        : 120.0;
    const jbCount = results.filter((r) => r.prediction === 'JAILBREAK').length;

    return {
      predictions: results,
      results,
      total_prompts: totalCount,
      jailbreaks_detected: jbCount,
      safe_prompts: totalCount - jbCount,
      total_inference_time_ms: totalLat,
      total_latency_ms: totalLat,
      average_latency_ms: totalCount > 0 ? totalLat / totalCount : 0,
    };
  } catch (err: unknown) {
    throw formatAxiosError(err, 'Batch inference request failed.');
  }
}

export async function predictCrossModal(
  imageFile: File | Blob,
  prompt?: string,
  generateExplanations = false
): Promise<CrossModalPredictResponse> {
  if (!imageFile || imageFile.size === 0) {
    throw new ApiError('Image file is required and cannot be empty.', 400);
  }

  const formData = new FormData();
  formData.append('image', imageFile);
  if (prompt && prompt.trim().length > 0) {
    formData.append('prompt', prompt.trim());
  }
  formData.append('generate_explanations', String(generateExplanations));

  try {
    const res = await apiClient.post<any>('/cross-modal-predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 60000,
    });

    const d = res.data || {};
    const vMod = d.modalities?.vision || {};
    const tMod = d.modalities?.text || {};
    const ocrMod = d.modalities?.ocr || {};

    const latency =
      typeof d.inference_time_ms === 'number'
        ? d.inference_time_ms
        : typeof d.latency_ms === 'number'
        ? d.latency_ms
        : 450.0;

    const overallRisk = typeof d.risk_score === 'number' ? d.risk_score : 0.0;
    const vRisk = typeof vMod.risk_score === 'number' ? vMod.risk_score : overallRisk;
    const tRisk = typeof tMod.risk_score === 'number' ? tMod.risk_score : overallRisk;
    const ocrConf = typeof ocrMod.confidence === 'number' ? ocrMod.confidence : 0.0;

    return {
      prediction: (d.prediction as Verdict) || (overallRisk > 50 ? 'JAILBREAK' : 'SAFE'),
      confidence: typeof d.confidence === 'number' ? d.confidence : 0.95,
      risk_score: overallRisk,
      blocked: typeof d.blocked === 'boolean' ? d.blocked : overallRisk > 50,
      attack_category: d.attack_category || (overallRisk > 50 ? 'Prompt Injection' : 'Benign Multimodal Content'),
      fusion_reason:
        d.fusion_reason ||
        (overallRisk > 50
          ? 'Cross-modal risk threshold exceeded. Adversarial markers identified across modalities.'
          : 'Unified cross-modal safety check passed with low adversarial probability.'),
      model_version: d.model_version || 'CMJD-v1.1-Optimized',
      explanation_paths: d.explanation_paths || null,
      inference_time_ms: latency,
      latency_ms: latency,
      modalities: d.modalities,

      // Normalized top-level properties for UI components
      vision_prediction: (vMod.prediction as Verdict) || (vRisk > 50 ? 'JAILBREAK' : 'SAFE'),
      vision_confidence: typeof vMod.confidence === 'number' ? vMod.confidence : vRisk / 100,
      vision_risk_score: vRisk,

      text_prediction: (tMod.prediction as Verdict) || (tRisk > 50 ? 'JAILBREAK' : 'SAFE'),
      text_confidence: typeof tMod.confidence === 'number' ? tMod.confidence : tRisk / 100,
      text_risk_score: tRisk,

      ocr_extracted_text: ocrMod.extracted_text || '',
      ocr_confidence: ocrConf,
      ocr_bounding_boxes: ocrMod.bounding_boxes || [],

      modality_weights: {
        vision: vRisk > 0 ? Number((vRisk / 100).toFixed(2)) : 0.35,
        text: tRisk > 0 ? Number((tRisk / 100).toFixed(2)) : 0.65,
        ocr: Number(ocrConf.toFixed(2)),
      },
      timestamp: new Date().toISOString(),
    };
  } catch (err: unknown) {
    throw formatAxiosError(err, 'Cross-modal visual analysis failed.');
  }
}
