import React, { useState } from 'react';
import {
  ImageIcon,
  Scan,
  RefreshCw,
  Clock,
  AlertTriangle,
  Sparkles,
  Eye,
  RotateCcw,
} from 'lucide-react';
import { VerdictBadge } from '../components/common/VerdictBadge';
import { RiskGauge } from '../components/common/RiskGauge';
import { ImageDropzone } from '../components/common/ImageDropzone';
import { LightboxModal } from '../components/common/LightboxModal';
import { ErrorBoundary } from '../components/common/ErrorBoundary';
import { predictCrossModal, ApiError } from '../services/api';
import { saveScanRecord } from '../services/storage';
import type { CrossModalPredictResponse } from '../types/api';

export const ImageScanner: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CrossModalPredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Active sub-tab in results
  const [activeTab, setActiveTab] = useState<'fusion' | 'ocr' | 'heatmap'>('fusion');

  // Lightbox for attention heatmap
  const [lightboxOpen, setLightboxOpen] = useState(false);

  const handleImageSelected = (file: File, preview: string) => {
    setSelectedFile(file);
    setPreviewUrl(preview);
    setResult(null);
    setError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please choose an image or pick a benchmark sample to analyze.');
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const res = await predictCrossModal(selectedFile, undefined, true);
      setResult(res);
      saveScanRecord({
        type: 'IMAGE',
        verdict: res.prediction,
        confidence: res.confidence ?? 0.95,
        riskScore: res.risk_score ?? 0,
        attackCategory: res.attack_category || 'Multimodal Assessment',
        latencyMs: res.latency_ms ?? res.inference_time_ms ?? 0,
        imageName: selectedFile.name,
        imageUrl: previewUrl || undefined,
        ocrText: res.ocr_extracted_text || '',
        visionRisk: res.vision_risk_score ?? 0,
        textRisk: res.text_risk_score ?? 0,
        reason: res.fusion_reason,
      });
    } catch (err: unknown) {
      console.error('Image scan error:', err);
      if (err instanceof ApiError) {
        setError(`[HTTP ${err.statusCode || 'ERR'}] ${err.message}`);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Multimodal visual analysis failed. Check backend service connectivity.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Determine which visual attention heatmap figure to link
  const isJailbreak = result?.prediction === 'JAILBREAK';
  const heatmapFigureUrl = isJailbreak
    ? '/figures/vision_saliency_sample_1_jailbreak.png'
    : '/figures/vision_saliency_sample_3_safe.png';

  const ocrVisUrl = isJailbreak
    ? '/figures/vision_ocr_vis_2_jailbreak.png'
    : '/figures/vision_ocr_vis_1_safe.png';

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400">
              <ImageIcon className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                Visual Jailbreak Detector
              </h1>
              <p className="text-xs text-slate-400">
                Phase 3.2 CLIP ViT-B/32 + EasyOCR Spatial Text Extraction Pipeline
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Image Upload & Presets */}
        <div className="lg:col-span-6 space-y-6">
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Upload Target Image
              </span>
              <span className="text-[11px] text-cyan-400 font-mono">
                Multimodal OCR & CLIP
              </span>
            </div>

            <ImageDropzone
              onImageSelected={handleImageSelected}
              selectedPreview={previewUrl}
              onClear={handleClear}
              disabled={loading}
            />

            {error && (
              <div className="p-3.5 rounded-xl bg-red-950/60 border border-red-800 text-red-200 text-xs flex items-center justify-between gap-3 animate-in fade-in">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                  <span>{error}</span>
                </div>
                <button
                  type="button"
                  onClick={handleAnalyze}
                  className="px-2.5 py-1 rounded-lg bg-red-900/60 hover:bg-red-800 text-[11px] font-semibold text-white flex items-center gap-1 transition-colors shrink-0"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Retry</span>
                </button>
              </div>
            )}

            {selectedFile && (
              <div className="pt-2 flex items-center justify-between">
                <button
                  type="button"
                  onClick={handleClear}
                  disabled={loading}
                  className="px-3 py-2 text-xs font-medium text-slate-400 hover:text-slate-200 transition-colors"
                >
                  Reset
                </button>

                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-102"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Extracting OCR & CLIP Saliency...</span>
                    </>
                  ) : (
                    <>
                      <Scan className="w-4 h-4" />
                      <span>Run Visual Scan</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Multi-Modality Results with Error Boundary */}
        <div className="lg:col-span-6 space-y-6">
          <ErrorBoundary fallbackTitle="Image Scanner Result Error" onReset={() => setResult(null)}>
            {result ? (
              <div className="space-y-4 animate-in fade-in duration-200">
                {/* Primary Fusion Card */}
                <div
                  className={`rounded-2xl p-6 border shadow-2xl space-y-5 transition-all ${
                    isJailbreak
                      ? 'bg-slate-900/90 border-red-600/40 shadow-red-950/40'
                      : 'bg-slate-900/90 border-emerald-600/40 shadow-emerald-950/40'
                  }`}
                >
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div>
                      <span className="text-xs uppercase tracking-wider text-slate-400 font-medium">
                        Cross-Modal Verdict
                      </span>
                      <div className="mt-1">
                        <VerdictBadge verdict={result.prediction} size="lg" />
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-xs text-slate-400 uppercase tracking-wider">
                        Overall Confidence
                      </span>
                      <p className="text-2xl font-mono font-bold text-white mt-0.5">
                        {((result.confidence ?? 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Risk Gauge */}
                  <div className="py-2 bg-slate-950/60 rounded-xl border border-slate-800/80">
                    <RiskGauge riskScore={result.risk_score ?? 0} label="Integrated Multimodal Risk Score" />
                  </div>

                  {/* Modality Split Badges */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                      <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                        <span>CLIP Vision Branch</span>
                        <VerdictBadge verdict={result.vision_prediction} size="sm" />
                      </div>
                      <div className="flex items-baseline justify-between font-mono">
                        <span className="text-xs text-slate-300">Risk Score</span>
                        <span className="text-sm font-bold text-cyan-400">
                          {(result.vision_risk_score ?? 0).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                      <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                        <span>OCR + DistilBERT Branch</span>
                        <VerdictBadge verdict={result.text_prediction} size="sm" />
                      </div>
                      <div className="flex items-baseline justify-between font-mono">
                        <span className="text-xs text-slate-300">Risk Score</span>
                        <span className="text-sm font-bold text-cyan-400">
                          {(result.text_risk_score ?? 0).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Decision Logic Explanation */}
                  <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-xs leading-relaxed space-y-1">
                    <div className="flex items-center gap-1.5 font-semibold text-slate-200">
                      <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                      <span>Fusion Decision Logic:</span>
                    </div>
                    <p className="text-slate-400 font-mono text-[11px]">
                      {result.fusion_reason || 'Visual and typographical signals assessed.'}
                    </p>
                  </div>
                </div>

                {/* Subsystem Inspection Tabs (OCR vs Attention Heatmap) */}
                <div className="rounded-2xl p-5 bg-slate-900/80 border border-slate-800 shadow-xl space-y-4">
                  <div className="flex border-b border-slate-800 gap-2">
                    <button
                      type="button"
                      onClick={() => setActiveTab('fusion')}
                      className={`px-3 py-2 text-xs font-semibold rounded-t-lg transition-all ${
                        activeTab === 'fusion'
                          ? 'text-cyan-400 border-b-2 border-cyan-400 bg-cyan-950/20'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Category & Latency
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveTab('ocr')}
                      className={`px-3 py-2 text-xs font-semibold rounded-t-lg transition-all ${
                        activeTab === 'ocr'
                          ? 'text-cyan-400 border-b-2 border-cyan-400 bg-cyan-950/20'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      OCR Extracted Text
                    </button>
                    <button
                      type="button"
                      onClick={() => setActiveTab('heatmap')}
                      className={`px-3 py-2 text-xs font-semibold rounded-t-lg transition-all ${
                        activeTab === 'heatmap'
                          ? 'text-cyan-400 border-b-2 border-cyan-400 bg-cyan-950/20'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Visual Attention & Saliency
                    </button>
                  </div>

                  {activeTab === 'fusion' && (
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                          <span className="text-[11px] uppercase tracking-wider text-slate-400">
                            Attack Category
                          </span>
                          <p className="text-xs font-semibold text-slate-200 mt-1">
                            {result.attack_category || 'Multimodal Threat'}
                          </p>
                        </div>
                        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                          <span className="text-[11px] uppercase tracking-wider text-slate-400">
                            End-to-End Latency
                          </span>
                          <p className="text-xs font-mono font-semibold text-cyan-400 mt-1 flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {(result.latency_ms ?? result.inference_time_ms ?? 0).toFixed(1)} ms
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'ocr' && (
                    <div className="space-y-3">
                      <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs">
                        <div className="flex items-center justify-between text-slate-400 pb-2 border-b border-slate-800 text-[11px]">
                          <span>Extracted Typography (EasyOCR)</span>
                          <span className="text-cyan-400">
                            Confidence: {((result.ocr_confidence ?? 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="pt-2 text-slate-200 whitespace-pre-wrap leading-relaxed max-h-40 overflow-y-auto">
                          {result.ocr_extracted_text || '(No typography detected in image)'}
                        </p>
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400">
                        <span>Spatial Bounding Box Overlay:</span>
                        <a
                          href={ocrVisUrl}
                          target="_blank"
                          rel="noreferrer"
                          className="text-cyan-400 hover:underline flex items-center gap-1"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          View Bounding Box Analysis
                        </a>
                      </div>
                    </div>
                  )}

                  {activeTab === 'heatmap' && (
                    <div className="space-y-3">
                      <div className="relative rounded-xl overflow-hidden border border-slate-800 bg-slate-950 p-2 group">
                        <img
                          src={heatmapFigureUrl}
                          alt="Visual Attention Saliency"
                          className="w-full h-48 object-cover rounded-lg"
                        />
                        <button
                          type="button"
                          onClick={() => setLightboxOpen(true)}
                          className="absolute inset-0 m-auto w-fit h-fit px-3 py-1.5 rounded-lg bg-slate-900/90 text-cyan-300 border border-cyan-500/50 text-xs font-semibold shadow-xl opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1.5"
                        >
                          <Eye className="w-4 h-4" />
                          Expand Saliency Map
                        </button>
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed">
                        High-gradient saliency activations reveal regions where CLIP detected
                        adversarial patterns or typographic contrast manipulation.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="h-full min-h-[420px] rounded-2xl border border-dashed border-slate-800 bg-slate-950/40 p-8 flex flex-col items-center justify-center text-center text-slate-500">
                <ImageIcon className="w-12 h-12 text-slate-700 mb-3" />
                <p className="text-sm font-semibold text-slate-300">Ready to Analyze Image</p>
                <p className="text-xs text-slate-500 max-w-xs mt-1">
                  Upload any screenshot, poster, or select a benchmark sample on the left to
                  begin cross-modal vision scanning.
                </p>
              </div>
            )}
          </ErrorBoundary>
        </div>
      </div>

      {/* Lightbox for visual saliency */}
      <LightboxModal
        isOpen={lightboxOpen}
        onClose={() => setLightboxOpen(false)}
        imageUrl={heatmapFigureUrl}
        title="CLIP ViT-B/32 Visual Attention & Saliency Map"
        category="Vision Explainability"
        description="Grad-CAM spatial activation highlighting adversarial typographic regions"
        metadata={{
          model: 'CLIP ViT-B/32',
          resolution: '224x224 input tensor',
          verdict: result?.prediction || 'N/A',
        }}
      />
    </div>
  );
};
